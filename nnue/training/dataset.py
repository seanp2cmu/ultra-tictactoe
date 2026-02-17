"""Dataset loader for NNUE training."""
import numpy as np
import torch
from torch.utils.data import Dataset

from nnue.core.features import NUM_FEATURES, board_array_to_features


class NNUEDataset(Dataset):
    """PyTorch dataset from .npz training data."""
    
    def __init__(self, boards, values):
        """
        Args:
            boards: (N, 92) int8 array of board states
            values: (N,) float32 array of evaluations
        """
        self.boards = boards
        self.values = values
        
        # Pre-compute features
        self.stm_features = np.zeros((len(boards), NUM_FEATURES), dtype=np.float32)
        self.nstm_features = np.zeros((len(boards), NUM_FEATURES), dtype=np.float32)
        
        for i in range(len(boards)):
            stm, nstm = board_array_to_features(boards[i])
            self.stm_features[i] = stm
            self.nstm_features[i] = nstm
    
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
