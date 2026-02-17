"""Dataset loader for NNUE training."""
import numpy as np
import torch
from torch.utils.data import Dataset

from nnue.core.features import NUM_FEATURES

# Feature offsets (must match features.py)
MY_PIECE = 0
OPP_PIECE = 81
MY_SUB_WON = 162
OPP_SUB_WON = 171
SUB_DRAW = 180
ACTIVE_SUB = 189
ACTIVE_ANY = 198


def batch_board_to_features(boards):
    """Vectorized feature extraction for entire batch.
    
    Converts (N, 92) board arrays → (N, 199) STM and NSTM feature tensors.
    ~50x faster than per-position Python loop.
    
    Args:
        boards: (N, 92) int8 array
        
    Returns:
        (stm_features, nstm_features): each (N, 199) float32
    """
    N = len(boards)
    stm = np.zeros((N, NUM_FEATURES), dtype=np.float32)
    nstm = np.zeros((N, NUM_FEATURES), dtype=np.float32)
    
    cells = boards[:, :81].astype(np.int8)       # (N, 81)
    meta = boards[:, 81:90].astype(np.int8)       # (N, 9)
    active = boards[:, 90].astype(np.int16)       # (N,)
    current_player = boards[:, 91].astype(np.int8) # (N,)
    
    # ── Cell features (vectorized) ──
    # For each cell, determine if it's my piece or opponent's piece
    for idx in range(81):
        col = cells[:, idx]  # (N,)
        occupied = col != 0  # (N,) bool
        is_stm = (col == current_player) & occupied
        is_opp = (~is_stm) & occupied
        
        stm[is_stm, MY_PIECE + idx] = 1.0
        nstm[is_stm, OPP_PIECE + idx] = 1.0
        stm[is_opp, OPP_PIECE + idx] = 1.0
        nstm[is_opp, MY_PIECE + idx] = 1.0
    
    # ── Sub-board status features (vectorized) ──
    for sub in range(9):
        status = meta[:, sub]  # (N,)
        
        is_draw = status == 3
        stm[is_draw, SUB_DRAW + sub] = 1.0
        nstm[is_draw, SUB_DRAW + sub] = 1.0
        
        is_stm_won = (status == current_player) & (status != 0) & (~is_draw)
        stm[is_stm_won, MY_SUB_WON + sub] = 1.0
        nstm[is_stm_won, OPP_SUB_WON + sub] = 1.0
        
        is_opp_won = (status != 0) & (~is_draw) & (~is_stm_won)
        stm[is_opp_won, OPP_SUB_WON + sub] = 1.0
        nstm[is_opp_won, MY_SUB_WON + sub] = 1.0
    
    # ── Active sub-board features (vectorized) ──
    is_any = active < 0
    stm[is_any, ACTIVE_ANY] = 1.0
    nstm[is_any, ACTIVE_ANY] = 1.0
    
    for sub in range(9):
        mask = active == sub
        stm[mask, ACTIVE_SUB + sub] = 1.0
        nstm[mask, ACTIVE_SUB + sub] = 1.0
    
    return stm, nstm


class NNUEDataset(Dataset):
    """PyTorch dataset from .npz training data."""
    
    def __init__(self, boards, values):
        """
        Args:
            boards: (N, 92) int8 array of board states
            values: (N,) float32 array of evaluations
        """
        self.values = values.astype(np.float32)
        
        # Vectorized feature extraction (batch)
        self.stm_features, self.nstm_features = batch_board_to_features(boards)
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.stm_features[idx]),
            torch.from_numpy(self.nstm_features[idx]),
            torch.tensor(self.values[idx], dtype=torch.float32),
        )
    
    @classmethod
    def from_npz(cls, path):
        """Load dataset from .npz file."""
        data = np.load(path)
        return cls(data['boards'], data['values'])
    
    @classmethod
    def train_val_split(cls, boards, values, val_ratio=0.1, seed=42):
        """Split into train/val datasets."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(values))
        val_size = int(len(values) * val_ratio)
        
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        train_ds = cls(boards[train_idx], values[train_idx])
        val_ds = cls(boards[val_idx], values[val_idx])
        return train_ds, val_ds
