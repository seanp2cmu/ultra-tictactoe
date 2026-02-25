"""Dataset loader for NNUE training with D4 symmetry augmentation."""
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

# ─── D4 symmetry tables (3×3 grid, 8 transformations) ────────────
# Each maps index [0..8] (row-major 3×3) to another index.
_SYM_3x3 = np.array([
    [0,1,2, 3,4,5, 6,7,8],  # identity
    [6,3,0, 7,4,1, 8,5,2],  # rot90
    [8,7,6, 5,4,3, 2,1,0],  # rot180
    [2,5,8, 1,4,7, 0,3,6],  # rot270
    [2,1,0, 5,4,3, 8,7,6],  # flipH
    [6,7,8, 3,4,5, 0,1,2],  # flipV
    [0,3,6, 1,4,7, 2,5,8],  # diagMain
    [8,5,2, 7,4,1, 6,3,0],  # diagAnti
], dtype=np.int32)

def _build_feature_permutations():
    """Build 8 permutation arrays for 199-dim feature vector.
    
    For symmetry s, perm[s] maps feature index i → transformed feature index.
    Cell features (0-80, 81-161): 9×9 board transform (sub + cell both permuted)
    Sub-board features (162-189): 3×3 meta-board transform
    Active features (189-198): 3×3 sub transform, 198 stays
    """
    perms = np.zeros((8, NUM_FEATURES), dtype=np.int32)
    for s in range(8):
        sym = _SYM_3x3[s]
        # Cell features: global_idx = sub_row*27 + cell_row*9 + sub_col*3 + cell_col
        # But stored as row*9+col in 9×9 grid.
        # Transform (r,c) on 9×9: sub and cell both get same 3×3 transform
        for r in range(9):
            for c in range(9):
                sub_r, sub_c = r // 3, c // 3
                cell_r, cell_c = r % 3, c % 3
                new_sub = sym[sub_r * 3 + sub_c]
                new_cell = sym[cell_r * 3 + cell_c]
                new_r = (new_sub // 3) * 3 + (new_cell // 3)
                new_c = (new_sub % 3) * 3 + (new_cell % 3)
                old_idx = r * 9 + c
                new_idx = new_r * 9 + new_c
                perms[s, MY_PIECE + old_idx] = MY_PIECE + new_idx
                perms[s, OPP_PIECE + old_idx] = OPP_PIECE + new_idx
        # Sub-board status features
        for sub in range(9):
            perms[s, MY_SUB_WON + sub] = MY_SUB_WON + sym[sub]
            perms[s, OPP_SUB_WON + sub] = OPP_SUB_WON + sym[sub]
            perms[s, SUB_DRAW + sub] = SUB_DRAW + sym[sub]
        # Active sub features
        for sub in range(9):
            perms[s, ACTIVE_SUB + sub] = ACTIVE_SUB + sym[sub]
        perms[s, ACTIVE_ANY] = ACTIVE_ANY
    return perms

# Precomputed: FEAT_PERMS[s] is a (199,) permutation for symmetry s
FEAT_PERMS = _build_feature_permutations()


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
    """PyTorch dataset with online D4 symmetry augmentation.
    
    When augment=True (training), each __getitem__ applies a random D4 symmetry
    transformation to the feature vector. This makes the model learn
    symmetry-invariant evaluation without increasing dataset size.
    """
    
    def __init__(self, boards, values, augment=False):
        """
        Args:
            boards: (N, 92) int8 array of board states
            values: (N,) float32 array of evaluations
            augment: if True, apply random D4 symmetry each access
        """
        self.values = values.astype(np.float32)
        self.augment = augment
        
        # Vectorized feature extraction (batch)
        self.stm_features, self.nstm_features = batch_board_to_features(boards)
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        stm = self.stm_features[idx]
        nstm = self.nstm_features[idx]
        
        if self.augment:
            # Random D4 symmetry (1-7, skip identity)
            s = np.random.randint(0, 8)
            if s > 0:
                perm = FEAT_PERMS[s]
                stm = stm[perm]
                nstm = nstm[perm]
        
        return (
            torch.from_numpy(stm.copy()),
            torch.from_numpy(nstm.copy()),
            torch.tensor(self.values[idx], dtype=torch.float32),
        )
    
    @classmethod
    def from_npz(cls, path, augment=False):
        """Load dataset from .npz file."""
        data = np.load(path)
        return cls(data['boards'], data['values'], augment=augment)
    
    @classmethod
    def train_val_split(cls, boards, values, val_ratio=0.1, seed=42):
        """Split into train/val datasets. Train gets augment=True."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(values))
        val_size = int(len(values) * val_ratio)
        
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        train_ds = cls(boards[train_idx], values[train_idx], augment=True)
        val_ds = cls(boards[val_idx], values[val_idx], augment=False)
        return train_ds, val_ds
