"""Weighted replay buffer for self-play data."""
from typing import Tuple, Dict, List, Optional
from collections import deque, Counter
import numpy as np


class SelfPlayData:
    """Position-weighted replay buffer."""
    
    def __init__(self, max_size: int = 10000) -> None:
        self.data = deque(maxlen=max_size)
        self.weights = deque(maxlen=max_size)
        self.categories = deque(maxlen=max_size)
        self._probs_cache: Optional[np.ndarray] = None
        self._cache_dirty = True
    
    def _get_weight(self, state: np.ndarray) -> Tuple[float, str]:
        """Calculate position weight based on empty cells."""
        my_plane = state[0]
        opponent_plane = state[1]
        empty_count = np.sum((my_plane == 0) & (opponent_plane == 0))
        
        if empty_count >= 50:
            return 1.0, "opening"
        elif empty_count >= 40:
            return 1.0, "early_mid"
        elif empty_count >= 30:
            return 1.0, "mid"
        elif empty_count >= 26:
            return 1.2, "transition"
        elif empty_count >= 20:
            return 0.8, "near_tablebase"
        elif empty_count >= 10:
            return 0.5, "tablebase"
        else:
            return 0.3, "deep_tablebase"
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float, dtw: Optional[int] = None) -> None:
        """Add training sample with automatic weighting."""
        weight, category = self._get_weight(state)
        self.data.append((state, policy, value, dtw))
        self.weights.append(weight)
        self.categories.append(category)
        self._cache_dirty = True
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """Weighted sampling."""
        if len(self.data) < batch_size:
            batch_indices = list(range(len(self.data)))
        else:
            if self._cache_dirty:
                weights_array = np.array(self.weights)
                total_weight = np.sum(weights_array)
                self._probs_cache = weights_array / total_weight
                self._cache_dirty = False
            
            batch_indices = np.random.choice(
                len(self.data),
                size=batch_size,
                replace=False,
                p=self._probs_cache
            )
        
        batch = [self.data[i] for i in batch_indices]
        states, policies, values, dtws = zip(*batch)
        return np.array(states), np.array(policies), np.array(values), list(dtws)
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        total = len(self.data)
        if total == 0:
            return {}
        
        counter = Counter(self.categories)
        distribution = {cat: count for cat, count in counter.items()}
        avg_weight = sum(self.weights) / len(self.weights)
        
        return {
            'total': total,
            'avg_weight': avg_weight,
            'distribution': distribution
        }
    
    def __len__(self) -> int:
        return len(self.data)
