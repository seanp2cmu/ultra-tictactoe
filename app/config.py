"""Global configuration and state for the app"""
import os
import torch

MODEL_DIR = "model"
HF_REPO_ID = "sean2474/ultra-tictactoe-models"

# Global state
models = {}
dtw_calculator = None
baseline_test_stop_flag = False
baseline_test_results = {'current': None}

# Mock spaces.GPU decorator for local runs
if os.environ.get('SPACE_ID'):
    import spaces
else:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            if fn is not None:
                return fn
            def decorator(func):
                return func
            return decorator
    spaces = _MockSpaces()


def get_best_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_model_on_gpu(model_name: str):
    """Move model to best available GPU and initialize TRT if available."""
    if model_name not in models:
        return None
    network = models[model_name]
    device = get_best_device()
    if device.type != 'cpu':
        network.device = device
        network.model = network.model.to(device)
        # Try TensorRT if on CUDA and not already initialized
        if device.type == 'cuda' and network.trt_engine is None:
            network._try_tensorrt()
            if network.trt_engine is None and not network._compiled:
                network._try_compile()
    return network
