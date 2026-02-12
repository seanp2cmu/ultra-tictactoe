"""Lc0-style replay buffer with game-aware sampling."""
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
import numpy as np


class SelfPlayData:
    """
    Replay buffer with Lc0-style sampling:
    - Game ID tracking
    - One position per game per batch
    - Age-based weighting (history discounting)
    """
    
    def __init__(self, max_size: int = 2000000, decay_factor: float = 0.97) -> None:
        self.max_size = max_size
        self.decay_factor = decay_factor
        
        # Storage: (state, policy, value, game_id, iteration)
        self.data: List[Tuple] = []
        self.game_to_indices: Dict[int, List[int]] = defaultdict(list)
        
        self._next_game_id = 0
        self._current_iteration = 0
    
    def new_game_id(self) -> int:
        """Get a new unique game ID."""
        game_id = self._next_game_id
        self._next_game_id += 1
        return game_id
    
    def set_iteration(self, iteration: int) -> None:
        """Set current iteration for age weighting."""
        self._current_iteration = iteration
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float, 
            game_id: int, iteration: Optional[int] = None) -> None:
        """Add training sample with game ID."""
        if iteration is None:
            iteration = self._current_iteration
        
        idx = len(self.data)
        self.data.append((state, policy, value, game_id, iteration))
        self.game_to_indices[game_id].append(idx)
        
        # Evict old data if over capacity
        if len(self.data) > self.max_size:
            self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        """Remove oldest 10% of data."""
        remove_count = self.max_size // 10
        self.data = self.data[remove_count:]
        
        # Rebuild game index
        self.game_to_indices.clear()
        for idx, (_, _, _, game_id, _) in enumerate(self.data):
            self.game_to_indices[game_id].append(idx)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Lc0-style sampling: one position per game per batch.
        Uses age-based weighting for game selection.
        Optimized with cached weights.
        """
        if len(self.data) == 0:
            return np.array([]), np.array([]), np.array([]), []
        
        # Cache game weights (rebuild only when iteration changes)
        if not hasattr(self, '_cached_iteration') or self._cached_iteration != self._current_iteration:
            self._rebuild_sample_cache()
        
        if len(self._cached_game_ids) == 0:
            return np.array([]), np.array([]), np.array([]), []
        
        # Sample games using cached probabilities
        selected_games = np.random.choice(
            len(self._cached_game_ids), 
            size=batch_size, 
            replace=True,
            p=self._cached_probs
        )
        
        # One random position per selected game
        batch_indices = []
        for game_idx in selected_games:
            indices = self._cached_indices[game_idx]
            pos_idx = indices[np.random.randint(len(indices))]
            batch_indices.append(pos_idx)
        
        # Batch gather
        states = np.array([self.data[i][0] for i in batch_indices])
        policies = np.array([self.data[i][1] for i in batch_indices])
        values = np.array([self.data[i][2] for i in batch_indices])
        
        return states, policies, values, [None] * len(batch_indices)
    
    def _rebuild_sample_cache(self) -> None:
        """Rebuild cached sampling weights."""
        self._cached_game_ids = list(self.game_to_indices.keys())
        self._cached_indices = [self.game_to_indices[gid] for gid in self._cached_game_ids]
        
        # Calculate weights
        weights = []
        for indices in self._cached_indices:
            if indices:
                _, _, _, _, iteration = self.data[indices[0]]
                age = self._current_iteration - iteration
                weights.append(self.decay_factor ** age)
            else:
                weights.append(0.0)
        
        weights = np.array(weights, dtype=np.float64)
        total = weights.sum()
        self._cached_probs = weights / total if total > 0 else np.ones(len(weights)) / len(weights)
        self._cached_iteration = self._current_iteration
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if len(self.data) == 0:
            return {}
        
        n_games = len(self.game_to_indices)
        avg_positions_per_game = len(self.data) / n_games if n_games > 0 else 0
        
        return {
            'total': len(self.data),
            'games': n_games,
            'avg_positions_per_game': avg_positions_per_game,
            'current_iteration': self._current_iteration
        }
    
    def __len__(self) -> int:
        return len(self.data)
