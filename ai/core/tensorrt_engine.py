"""Native TensorRT engine for fast inference."""
import os
import torch
import numpy as np
from typing import Tuple, Optional

TRT_AVAILABLE = False
trt = None
cuda = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TRT_AVAILABLE = True
except ImportError:
    pass

_cuda_initialized = False

def _init_cuda():
    """Initialize CUDA context lazily."""
    global _cuda_initialized
    if _cuda_initialized:
        return True
    try:
        cuda.init()
        device = cuda.Device(0)
        ctx = device.make_context()
        import atexit
        atexit.register(ctx.pop)
        _cuda_initialized = True
        return True
    except Exception as e:
        print(f"[TRT] CUDA init failed: {e}")
        return False


class TensorRTEngine:
    """TensorRT engine wrapper for fast inference."""
    
    def __init__(self, engine_path: str = None):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT or pycuda not available")
        if not _init_cuda():
            raise RuntimeError("Failed to initialize CUDA context")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        
        # Bindings
        self.d_input = None
        self.d_policy = None
        self.d_value = None
        self.h_policy = None
        self.h_value = None
        self.batch_size = 0
        
        if engine_path and os.path.exists(engine_path):
            self.load_engine(engine_path)
    
    def build_engine(
        self,
        model: torch.nn.Module,
        onnx_path: str = "./model/model.onnx",
        engine_path: str = "./model/model.trt",
        max_batch_size: int = 4096,
        fp16: bool = True
    ) -> bool:
        """Build TensorRT engine from PyTorch model."""
        # Get the raw model if it's compiled
        raw_model = model
        if hasattr(model, '_orig_mod'):
            raw_model = model._orig_mod
        
        # Export to ONNX using legacy exporter
        raw_model.eval()
        dummy_input = torch.randn(1, 3, 9, 9).cuda()
        
        with torch.no_grad():
            torch.onnx.export(
                raw_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['policy', 'value'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'policy': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                },
                dynamo=False
            )
        print(f"[TRT] Exported ONNX to {onnx_path}")
        
        # Build TensorRT engine
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT] ONNX parse error: {parser.get_error(i)}")
                return False
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled")
        
        # Optimization profile for dynamic batch
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 9, 9), (max_batch_size // 2, 3, 9, 9), (max_batch_size, 3, 9, 9))
        config.add_optimization_profile(profile)
        
        print(f"[TRT] Building engine (max_batch={max_batch_size})... This may take a few minutes.")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("[TRT] Engine build failed")
            return False
        
        # Save engine
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"[TRT] Engine saved to {engine_path}")
        
        # Load the built engine
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        
        return True
    
    def load_engine(self, engine_path: str) -> bool:
        """Load TensorRT engine from file."""
        if not os.path.exists(engine_path):
            return False
        
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            return False
        
        self.context = self.engine.create_execution_context()
        print(f"[TRT] Engine loaded from {engine_path}")
        return True
    
    def _allocate_buffers(self, batch_size: int):
        """Allocate GPU buffers for inference."""
        if batch_size == self.batch_size:
            return
        
        # Free old buffers
        if self.d_input is not None:
            self.d_input.free()
            self.d_policy.free()
            self.d_value.free()
        
        self.batch_size = batch_size
        
        # Input: [batch, 3, 9, 9]
        input_size = batch_size * 3 * 9 * 9 * 4  # float32
        self.d_input = cuda.mem_alloc(input_size)
        
        # Policy output: [batch, 81]
        policy_size = batch_size * 81 * 4
        self.d_policy = cuda.mem_alloc(policy_size)
        self.h_policy = cuda.pagelocked_empty((batch_size, 81), dtype=np.float32)
        
        # Value output: [batch, 1]
        value_size = batch_size * 1 * 4
        self.d_value = cuda.mem_alloc(value_size)
        self.h_value = cuda.pagelocked_empty((batch_size, 1), dtype=np.float32)
        
        # Set input shape for dynamic batch
        self.context.set_input_shape('input', (batch_size, 3, 9, 9))
    
    def infer(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on batch."""
        batch_size = input_tensor.shape[0]
        self._allocate_buffers(batch_size)
        
        # Copy input to GPU
        input_flat = np.ascontiguousarray(input_tensor.astype(np.float32))
        cuda.memcpy_htod_async(self.d_input, input_flat, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address('input', int(self.d_input))
        self.context.set_tensor_address('policy', int(self.d_policy))
        self.context.set_tensor_address('value', int(self.d_value))
        
        # Execute
        self.context.execute_async_v3(self.stream.handle)
        
        # Copy outputs back
        cuda.memcpy_dtoh_async(self.h_policy[:batch_size], self.d_policy, self.stream)
        cuda.memcpy_dtoh_async(self.h_value[:batch_size], self.d_value, self.stream)
        self.stream.synchronize()
        
        return self.h_policy[:batch_size].copy(), self.h_value[:batch_size].copy()
    
    def is_ready(self) -> bool:
        """Check if engine is loaded and ready."""
        return self.engine is not None and self.context is not None


def get_tensorrt_engine(
    model: torch.nn.Module = None,
    engine_path: str = "./model/model.trt",
    onnx_path: str = "./model/model.onnx",
    max_batch_size: int = 4096,
    force_rebuild: bool = False
) -> Optional[TensorRTEngine]:
    """Get or build TensorRT engine."""
    if not TRT_AVAILABLE:
        print("[TRT] TensorRT not available, using PyTorch")
        return None
    
    engine = TensorRTEngine()
    
    # Try to load existing engine
    if not force_rebuild and os.path.exists(engine_path):
        if engine.load_engine(engine_path):
            return engine
    
    # Build new engine if model provided
    if model is not None:
        if engine.build_engine(model, onnx_path, engine_path, max_batch_size):
            return engine
    
    return None
