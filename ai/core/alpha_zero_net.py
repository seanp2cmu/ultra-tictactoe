"""AlphaZero network wrapper with training capabilities."""
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
import threading
from torch.amp import autocast, GradScaler

from .network import Model
from utils import BoardEncoder
from .tensorrt_engine import get_tensorrt_process_client, export_onnx, TRT_AVAILABLE


class AlphaZeroNet:
    """Network wrapper with optimizer, scheduler, and training methods."""
    
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        model: Optional[Model] = None,
        use_amp: bool = True,
        total_iterations: int = 300
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        if model is None:
            self.model = Model().to(self.device)
        else:
            self.model = model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_iterations,
            eta_min=lr * 0.01
        )
        
        self.scaler = GradScaler() if use_amp and torch.cuda.is_available() else None
        self.predict_lock = threading.Lock()
        self._compiled = False
        self.trt_engine = None
        
        # Try TensorRT first, then torch.compile (only when actually on CUDA)
        if self.device.type == 'cuda':
            self._try_tensorrt()
            if self.trt_engine is None and hasattr(torch, 'compile'):
                self._try_compile()
    
    def predict(self, board_state) -> Tuple[np.ndarray, float]:
        """Thread-safe single board prediction with canonical form."""
        with self.predict_lock:
            # Use BoardEncoder for consistent transformation
            canonical_tensor, inverse_transform = BoardEncoder.to_inference_tensor(board_state)
            
            self.model.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(canonical_tensor).unsqueeze(0).to(self.device)
                
                if self.device.type == 'cuda':
                    with autocast(device_type='cuda'):
                        policy_logits, value = self.model(tensor)
                else:
                    policy_logits, value = self.model(tensor)
                
                policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.cpu().numpy()[0, 0]
            
            # Transform policy back to original orientation
            original_policy = inverse_transform(policy)
            return original_policy, float(value)
    
    def predict_batch(self, board_states) -> Tuple[np.ndarray, np.ndarray]:
        """Thread-safe batch prediction with canonical form."""
        with self.predict_lock:
            # Use BoardEncoder for consistent transformation
            batch_tensor, inverse_fns = BoardEncoder.to_inference_tensor_batch(board_states)
            
            # Use TensorRT if available
            if self.trt_engine is not None and self.trt_engine.is_ready():
                policy_logits, values = self.trt_engine.infer(batch_tensor)
                policy_probs = self._softmax_numpy(policy_logits)
            else:
                self.model.eval()
                with torch.no_grad():
                    tensor = torch.from_numpy(batch_tensor).to(self.device)
                    
                    if self.device.type == 'cuda':
                        with autocast(device_type='cuda'):
                            policy_logits, values = self.model(tensor)
                    else:
                        policy_logits, values = self.model(tensor)
                    
                    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
                    values = values.cpu().numpy()
            
            # Transform each policy back to original orientation
            original_policies = []
            for i, policy in enumerate(policy_probs):
                original_policy = inverse_fns[i](policy)
                original_policies.append(original_policy)
            
            return np.array(original_policies), values
    
    def _softmax_numpy(self, x: np.ndarray) -> np.ndarray:
        """Numpy softmax for TensorRT output."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def train_step(
        self,
        boards: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray
    ) -> Tuple[float, float, float]:
        """Single training step with gradient clipping."""
        self.model.train()
        
        boards_tensor = torch.FloatTensor(boards).to(self.device)
        policies_tensor = torch.FloatTensor(policies).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            with autocast(device_type='cuda'):
                policy_logits, value_preds = self.model(boards_tensor)
                policy_loss = F.cross_entropy(policy_logits, policies_tensor)
                value_loss = F.mse_loss(value_preds.squeeze(-1), values_tensor.squeeze(-1))
                total_loss = policy_loss + value_loss
            
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, value_preds = self.model(boards_tensor)
            policy_loss = F.cross_entropy(policy_logits, policies_tensor)
            value_loss = F.mse_loss(value_preds.squeeze(-1), values_tensor.squeeze(-1))
            total_loss = policy_loss + value_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def step_scheduler(self) -> float:
        """Update learning rate."""
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def save(self, filepath: str, iteration: int = None) -> None:
        """Save model, optimizer, and scheduler states."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_res_blocks': len(self.model.res_blocks),
            'num_channels': self.model.num_channels,
            'iteration': iteration,
        }
        if self.scaler is not None:
            save_dict['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str) -> int:
        """Load model, optimizer, and scheduler states. Returns iteration number."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Handle compiled model state dict (strip _orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            checkpoint['model_state_dict'] = state_dict
        
        if 'num_res_blocks' in checkpoint and 'num_channels' in checkpoint:
            num_res_blocks = checkpoint['num_res_blocks']
            num_channels = checkpoint['num_channels']
        else:
            max_block_idx = -1
            for key in state_dict.keys():
                if key.startswith('res_blocks.'):
                    block_idx = int(key.split('.')[1])
                    max_block_idx = max(max_block_idx, block_idx)
            num_res_blocks = max_block_idx + 1
            num_channels = state_dict['input.0.weight'].shape[0]
        
        if len(self.model.res_blocks) != num_res_blocks or self.model.num_channels != num_channels:
            self.model = Model(num_res_blocks=num_res_blocks, num_channels=num_channels).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay']
            )
        
        # Handle torch.compile state_dict prefix mismatch
        # If model is compiled, keys have "_orig_mod." prefix
        # If checkpoint was saved without compile, keys don't have prefix
        model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(state_dict.keys())
        
        if model_keys and checkpoint_keys:
            model_has_prefix = model_keys[0].startswith('_orig_mod.')
            ckpt_has_prefix = checkpoint_keys[0].startswith('_orig_mod.')
            
            if model_has_prefix and not ckpt_has_prefix:
                # Add prefix to checkpoint keys
                state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
            elif not model_has_prefix and ckpt_has_prefix:
                # Remove prefix from checkpoint keys
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Rebuild TensorRT engine with new weights (only if already on CUDA)
        if self.device.type == 'cuda':
            self._try_tensorrt(force_rebuild=True)
            if self.trt_engine is None and not self._compiled:
                self._try_compile()
        
        return checkpoint.get('iteration', 0)
    
    def _try_tensorrt(self, force_rebuild: bool = False):
        """Try to use TensorRT via a dedicated server process."""
        if not TRT_AVAILABLE:
            return
        
        try:
            # If we already have a live server, just rebuild the engine in-place
            if force_rebuild and self.trt_engine is not None and self.trt_engine.is_ready():
                if export_onnx(self.model):
                    if self.trt_engine.rebuild("./model/model.onnx", "./model/model.trt"):
                        print("[Model] TensorRT engine rebuilt (separate process)")
                        return
                # Rebuild failed – shut down and retry fresh
                self.trt_engine.shutdown()
                self.trt_engine = None
            
            # Fresh start (or retry after failed rebuild)
            if self.trt_engine is not None:
                self.trt_engine.shutdown()
            
            self.trt_engine = get_tensorrt_process_client(
                model=self.model,
                engine_path="./model/model.trt",
                onnx_path="./model/model.onnx",
                max_batch_size=4096,
                force_rebuild=force_rebuild,
            )
            if self.trt_engine is not None:
                print("[Model] Using TensorRT (separate process)")
        except Exception as e:
            print(f"[Model] TensorRT process failed: {e}")
            if self.trt_engine is not None:
                self.trt_engine.shutdown()
            self.trt_engine = None
    
    def sync_trt_weights(self):
        """Rebuild TRT engine after training to sync with updated PyTorch weights.
        
        Only acts if a TRT server process is already running.
        Returns True if rebuild succeeded, False otherwise.
        """
        if self.trt_engine is None or not self.trt_engine.is_ready():
            return False
        try:
            if export_onnx(self.model):
                if self.trt_engine.rebuild("./model/model.onnx", "./model/model.trt"):
                    return True
            # Rebuild failed — fall back to PyTorch
            print("[Model] TRT weight sync failed, falling back to PyTorch")
            self.trt_engine.shutdown()
            self.trt_engine = None
        except Exception as e:
            print(f"[Model] TRT weight sync error: {e}")
            self.shutdown_trt()
        return False
    
    def shutdown_trt(self):
        """Clean up the TensorRT server process."""
        if self.trt_engine is not None:
            self.trt_engine.shutdown()
            self.trt_engine = None
    
    def __del__(self):
        self.shutdown_trt()
    
    def _try_compile(self):
        """Apply torch.compile as fallback."""
        if self._compiled:
            return
        if not torch.cuda.is_available() or not hasattr(torch, 'compile'):
            return
        
        # Use inductor with reduce-overhead mode
        try:
            self.model = torch.compile(self.model, mode='reduce-overhead')
            self._compiled = True
            self._compile_backend = 'inductor'
            print("[Model] Compiled with inductor (reduce-overhead)")
        except Exception:
            self._compile_backend = 'eager'
            print("[Model] Using eager mode (no compilation)")
