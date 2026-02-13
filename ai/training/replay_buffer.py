"""Lc0-style replay buffer with game-aware sampling.

Uses pre-allocated numpy ring buffer for O(1) batch sampling via fancy indexing,
instead of per-sample list comprehension + np.array().
"""
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
import numpy as np


class SelfPlayData:
    """
    Replay buffer with Lc0-style sampling:
    - Game ID tracking
    - One position per game per batch
    - Age-based weighting (history discounting)
    
    Storage uses pre-allocated numpy ring buffers (lazy-allocated on first add).
    sample() uses numpy fancy indexing for O(1) batch gather.
    """
    
    def __init__(self, max_size: int = 2000000, decay_factor: float = 0.97) -> None:
        self.max_size = max_size
        self.decay_factor = decay_factor
        
        # Lazy-allocated contiguous ring buffers (set on first add)
        self._states: Optional[np.ndarray] = None   # (max_size, C, H, W)
        self._policies: Optional[np.ndarray] = None  # (max_size, 81)
        self._values = np.zeros(max_size, dtype=np.float32)
        self._game_ids = np.full(max_size, -1, dtype=np.int64)
        self._iterations = np.zeros(max_size, dtype=np.int32)
        
        self._size = 0   # number of valid entries (0..max_size)
        self._head = 0   # next write position (circular)
        
        # Game tracking: game_id → {'iteration': int, 'positions': set of ring indices}
        self._game_info: Dict[int, dict] = {}
        
        self._next_game_id = 0
        self._current_iteration = 0
        self._cache_valid = False
    
    def new_game_id(self) -> int:
        """Get a new unique game ID."""
        game_id = self._next_game_id
        self._next_game_id += 1
        return game_id
    
    def set_iteration(self, iteration: int) -> None:
        """Set current iteration for age weighting."""
        if iteration != self._current_iteration:
            self._current_iteration = iteration
            self._cache_valid = False
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float, 
            game_id: int, iteration: Optional[int] = None) -> None:
        """Add training sample with game ID."""
        if iteration is None:
            iteration = self._current_iteration
        
        # Lazy allocate on first add (avoids upfront multi-GB allocation)
        if self._states is None:
            self._states = np.zeros((self.max_size,) + state.shape, dtype=np.float32)
            self._policies = np.zeros((self.max_size, 81), dtype=np.float32)
        
        pos = self._head
        
        # If overwriting old entry (buffer full), clean up old game tracking
        if self._size == self.max_size:
            old_gid = int(self._game_ids[pos])
            if old_gid >= 0 and old_gid in self._game_info:
                info = self._game_info[old_gid]
                info['positions'].discard(pos)
                if not info['positions']:
                    del self._game_info[old_gid]
        
        # Write to ring buffer
        self._states[pos] = state
        self._policies[pos] = policy
        self._values[pos] = value
        self._game_ids[pos] = game_id
        self._iterations[pos] = iteration
        
        # Update game tracking
        if game_id not in self._game_info:
            self._game_info[game_id] = {'iteration': iteration, 'positions': set()}
        self._game_info[game_id]['positions'].add(pos)
        
        # Advance circular pointer
        self._head = (self._head + 1) % self.max_size
        if self._size < self.max_size:
            self._size += 1
        self._cache_valid = False
    
    def add_batch(self, states: np.ndarray, policies: np.ndarray, 
                  values: np.ndarray, game_ids: np.ndarray,
                  iteration: Optional[int] = None) -> None:
        """Add multiple samples at once (faster than individual add calls)."""
        if iteration is None:
            iteration = self._current_iteration
        
        n = len(states)
        if n == 0:
            return
        
        # Lazy allocate
        if self._states is None:
            self._states = np.zeros((self.max_size,) + states.shape[1:], dtype=np.float32)
            self._policies = np.zeros((self.max_size, 81), dtype=np.float32)
        
        for i in range(n):
            pos = self._head
            
            # Clean up old entry if overwriting
            if self._size == self.max_size:
                old_gid = int(self._game_ids[pos])
                if old_gid >= 0 and old_gid in self._game_info:
                    info = self._game_info[old_gid]
                    info['positions'].discard(pos)
                    if not info['positions']:
                        del self._game_info[old_gid]
            
            # Write
            self._states[pos] = states[i]
            self._policies[pos] = policies[i]
            self._values[pos] = values[i]
            gid = int(game_ids[i])
            self._game_ids[pos] = gid
            self._iterations[pos] = iteration
            
            if gid not in self._game_info:
                self._game_info[gid] = {'iteration': iteration, 'positions': set()}
            self._game_info[gid]['positions'].add(pos)
            
            self._head = (self._head + 1) % self.max_size
            if self._size < self.max_size:
                self._size += 1
        
        self._cache_valid = False
    
    # Backward-compatible property for external access
    @property
    def game_to_indices(self):
        """Backward-compatible game→indices mapping."""
        return {gid: list(info['positions']) for gid, info in self._game_info.items()}
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Lc0-style sampling: one position per game per batch.
        Uses age-based weighting for game selection.
        Batch gather via numpy fancy indexing (O(1) per batch).
        """
        if self._size == 0:
            return np.array([]), np.array([]), np.array([]), []
        
        # Rebuild cache when invalidated (iteration change or new data)
        if not self._cache_valid:
            self._rebuild_sample_cache()
        
        if self._n_games == 0:
            return np.array([]), np.array([]), np.array([]), []
        
        # Sample games using cached probabilities
        selected_games = np.random.choice(
            self._n_games, 
            size=batch_size, 
            replace=True,
            p=self._cached_probs
        )
        
        # One random position per selected game (vectorized where possible)
        batch_indices = np.empty(batch_size, dtype=np.int64)
        for i, game_idx in enumerate(selected_games):
            positions = self._cached_positions[game_idx]
            batch_indices[i] = positions[np.random.randint(len(positions))]
        
        # O(1) batch gather via numpy fancy indexing (the main optimization)
        states = self._states[batch_indices]
        policies = self._policies[batch_indices]
        values = self._values[batch_indices]
        
        return states, policies, values, [None] * batch_size
    
    def _rebuild_sample_cache(self) -> None:
        """Rebuild cached sampling weights from game info."""
        game_ids = list(self._game_info.keys())
        self._n_games = len(game_ids)
        
        if self._n_games == 0:
            self._cache_valid = True
            return
        
        # Build position arrays and weights
        self._cached_positions = []
        weights = np.empty(self._n_games, dtype=np.float64)
        
        for i, gid in enumerate(game_ids):
            info = self._game_info[gid]
            self._cached_positions.append(np.array(list(info['positions']), dtype=np.int64))
            age = max(0, self._current_iteration - info['iteration'])
            weights[i] = self.decay_factor ** age
        
        total = weights.sum()
        self._cached_probs = weights / total if total > 0 else np.ones(self._n_games) / self._n_games
        self._cache_valid = True
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if self._size == 0:
            return {}
        
        n_games = len(self._game_info)
        avg_positions_per_game = self._size / n_games if n_games > 0 else 0
        
        return {
            'total': self._size,
            'games': n_games,
            'avg_positions_per_game': avg_positions_per_game,
            'current_iteration': self._current_iteration
        }
    
    def __len__(self) -> int:
        return self._size
