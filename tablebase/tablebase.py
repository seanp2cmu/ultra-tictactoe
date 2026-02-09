"""Tablebase Storage and Lookup for Ultimate Tic-Tac-Toe

Efficient storage and retrieval of precomputed endgame positions.
"""

import os
from typing import Optional, Tuple, Dict
import numpy as np


class CompactTablebase:
    """
    Memory-efficient tablebase using numpy arrays.
    
    For production use with millions of positions.
    Uses memory-mapped file for large tablebases.
    """
    
    def __init__(self, path: Optional[str] = None):
        self.hashes: Optional[np.ndarray] = None  # uint64 array
        self.results: Optional[np.ndarray] = None  # int8 array
        self.dtws: Optional[np.ndarray] = None  # uint8 array
        self.path = path
        
        if path and os.path.exists(path):
            self.load(path)
    
    def build_from_dict(self, positions: Dict):
        """Build compact representation from position dict.
        
        Handles nested dict: positions[hash] = {constraint: (result, dtw, move)}
        Stores best result per hash (min dtw among winning, or best available).
        """
        # Flatten nested dict: pick best result per hash
        flattened = {}
        for h, constraint_dict in positions.items():
            best_result = -2
            best_dtw = 999
            for constraint, entry in constraint_dict.items():
                if len(entry) == 3:
                    r, d, _ = entry  # (result, dtw, move)
                else:
                    r, d = entry  # (result, dtw)
                # Prefer wins, then draws, then losses; minimize dtw
                if r > best_result or (r == best_result and d < best_dtw):
                    best_result = r
                    best_dtw = d
            flattened[h] = (best_result, best_dtw)
        
        n = len(flattened)
        self.hashes = np.zeros(n, dtype=np.uint64)
        self.results = np.zeros(n, dtype=np.int8)
        self.dtws = np.zeros(n, dtype=np.uint8)
        
        for i, (h, (r, d)) in enumerate(flattened.items()):
            self.hashes[i] = h & 0xFFFFFFFFFFFFFFFF  # Ensure positive
            self.results[i] = r
            self.dtws[i] = min(d, 255)
        
        # Sort by hash for binary search
        sort_idx = np.argsort(self.hashes)
        self.hashes = self.hashes[sort_idx]
        self.results = self.results[sort_idx]
        self.dtws = self.dtws[sort_idx]
    
    def lookup(self, board_hash: int) -> Optional[Tuple[int, int]]:
        """O(log n) lookup using binary search."""
        if self.hashes is None:
            return None
        
        h = board_hash & 0xFFFFFFFFFFFFFFFF
        idx = np.searchsorted(self.hashes, h)
        
        if idx < len(self.hashes) and self.hashes[idx] == h:
            return (int(self.results[idx]), int(self.dtws[idx]))
        
        return None
    
    def save(self, path: str):
        """Save compact tablebase."""
        np.savez_compressed(
            path,
            hashes=self.hashes,
            results=self.results,
            dtws=self.dtws
        )
        print(f"✓ Saved compact tablebase: {len(self.hashes)} positions")
    
    def load(self, path: str):
        """Load compact tablebase."""
        data = np.load(path)
        self.hashes = data['hashes']
        self.results = data['results']
        self.dtws = data['dtws']
        print(f"✓ Loaded compact tablebase: {len(self.hashes)} positions")
    
    def get_size_mb(self) -> float:
        """Get actual memory usage in MB."""
        if self.hashes is None:
            return 0
        total = (self.hashes.nbytes + self.results.nbytes + self.dtws.nbytes)
        return total / (1024 * 1024)
