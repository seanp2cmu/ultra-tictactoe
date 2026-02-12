"""AlphaZero network wrapper with training capabilities."""
from typing import Dict, Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import threading
from torch.amp import autocast, GradScaler

from .network import Model
from utils import BoardSymmetry


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
        
        # torch.compile for CUDA (PyTorch 2.0+)
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception:
                pass  # compile 실패시 원본 사용
    
    def predict(self, board_state) -> Tuple[np.ndarray, float]:
        """Thread-safe single board prediction with canonical form."""
        with self.predict_lock:
            from game import Board
            
            # Get canonical transform
            boards_arr, completed_arr, transform_idx = BoardSymmetry.get_canonical_with_transform(board_state)
            
            # Create canonical board
            canonical_board = Board()
            for r in range(9):
                for c in range(9):
                    if boards_arr[r, c] != 0:
                        canonical_board.set_cell(r, c, int(boards_arr[r, c]))
            # Handle both Cython BoardCy and Python Board
            if hasattr(canonical_board, 'set_completed_boards_2d'):
                # Cython Board - use setter method
                canonical_board.set_completed_boards_2d(completed_arr.tolist())
            else:
                # Python Board - direct assignment
                canonical_board.completed_boards = completed_arr.tolist()
            canonical_board.current_player = board_state.current_player
            
            # Transform last_move
            if board_state.last_move is not None and transform_idx != 0:
                transforms = BoardSymmetry._build_transforms()
                old_idx = board_state.last_move[0] * 9 + board_state.last_move[1]
                new_idx = np.where(transforms[transform_idx] == old_idx)[0]
                if len(new_idx) > 0:
                    canonical_board.last_move = (new_idx[0] // 9, new_idx[0] % 9)
            else:
                canonical_board.last_move = board_state.last_move
            
            # Predict on canonical board
            policy, value = self.model.predict(canonical_board)
            
            # Transform policy back to original orientation
            original_policy = BoardSymmetry.inverse_transform_policy(policy, transform_idx)
            return original_policy, value
    
    def predict_batch(self, board_states) -> Tuple[np.ndarray, np.ndarray]:
        """Thread-safe batch prediction with canonical form."""
        with self.predict_lock:
            from game import Board
            self.model.eval()
            
            # Process each board to canonical form
            canonical_boards = []
            transform_indices = []
            
            for board_state in board_states:
                boards_arr, completed_arr, transform_idx = BoardSymmetry.get_canonical_with_transform(board_state)
                transform_indices.append(transform_idx)
                
                canonical_board = Board()
                for r in range(9):
                    for c in range(9):
                        if boards_arr[r, c] != 0:
                            canonical_board.set_cell(r, c, int(boards_arr[r, c]))
                # Handle both Cython BoardCy and Python Board
                if hasattr(canonical_board, 'set_completed_boards_2d'):
                    canonical_board.set_completed_boards_2d(completed_arr.tolist())
                else:
                    canonical_board.completed_boards = completed_arr.tolist()
                canonical_board.current_player = board_state.current_player
                
                if board_state.last_move is not None and transform_idx != 0:
                    transforms = BoardSymmetry._build_transforms()
                    old_idx = board_state.last_move[0] * 9 + board_state.last_move[1]
                    new_idx = np.where(transforms[transform_idx] == old_idx)[0]
                    if len(new_idx) > 0:
                        canonical_board.last_move = (new_idx[0] // 9, new_idx[0] % 9)
                else:
                    canonical_board.last_move = board_state.last_move
                
                canonical_boards.append(canonical_board)
            
            with torch.no_grad():
                board_tensors = [
                    self.model._board_to_tensor(board).squeeze(0)
                    for board in canonical_boards
                ]
                batch_tensor = torch.stack(board_tensors).to(self.device)
                
                # CUDA에서 FP16 inference 사용
                if self.device.type == 'cuda':
                    with autocast(device_type='cuda'):
                        policy_logits, values = self.model(batch_tensor)
                else:
                    policy_logits, values = self.model(batch_tensor)
                policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
                
                # Transform each policy back to original orientation
                original_policies = []
                for i, policy in enumerate(policy_probs):
                    original_policy = BoardSymmetry.inverse_transform_policy(policy, transform_indices[i])
                    original_policies.append(original_policy)
                
                return np.array(original_policies), values.cpu().numpy()
    
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
        
        if 'num_res_blocks' in checkpoint and 'num_channels' in checkpoint:
            num_res_blocks = checkpoint['num_res_blocks']
            num_channels = checkpoint['num_channels']
        else:
            state_dict: Dict[str, Tensor] = checkpoint['model_state_dict']
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint.get('iteration', 0)
