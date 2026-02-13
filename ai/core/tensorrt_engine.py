"""Native TensorRT engine for fast inference using torch CUDA streams."""
import os
import torch
import numpy as np
from typing import Tuple, Optional

TRT_AVAILABLE = False
trt = None

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    pass


class TensorRTEngine:
    """TensorRT engine wrapper using PyTorch CUDA memory management."""
    
    def __init__(self, engine_path: str = None):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        # Force PyTorch to fully initialize CUDA context
        torch.cuda.init()
        torch.cuda.set_device(0)
        _ = torch.cuda.current_device()
        torch.cuda.synchronize()
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = torch.cuda.current_stream()
        
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
        dummy_input = torch.randn(1, 7, 9, 9).cuda()
        
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
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 1GB
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled")
        
        # Optimization profile for dynamic batch
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 7, 9, 9), (max_batch_size // 2, 7, 9, 9), (max_batch_size, 7, 9, 9))
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
        """Allocate GPU buffers using PyTorch."""
        if batch_size == self.batch_size:
            return
        
        self.batch_size = batch_size
        
        # Use PyTorch tensors for GPU memory (automatically managed)
        self.d_input = torch.empty((batch_size, 7, 9, 9), dtype=torch.float32, device='cuda')
        self.d_policy = torch.empty((batch_size, 81), dtype=torch.float32, device='cuda')
        self.d_value = torch.empty((batch_size, 1), dtype=torch.float32, device='cuda')
        
        # Set input shape for dynamic batch
        self.context.set_input_shape('input', (batch_size, 7, 9, 9))
    
    def infer(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on batch using PyTorch CUDA."""
        batch_size = input_tensor.shape[0]
        self._allocate_buffers(batch_size)
        
        with torch.cuda.stream(self.stream):
            # Copy input to GPU using PyTorch
            self.d_input.copy_(torch.from_numpy(input_tensor))
            
            # Set tensor addresses
            self.context.set_tensor_address('input', self.d_input.data_ptr())
            self.context.set_tensor_address('policy', self.d_policy.data_ptr())
            self.context.set_tensor_address('value', self.d_value.data_ptr())
            
            # Execute
            self.context.execute_async_v3(self.stream.cuda_stream)
        
        # Synchronize and copy back
        self.stream.synchronize()
        
        return self.d_policy.cpu().numpy(), self.d_value.cpu().numpy()
    
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
