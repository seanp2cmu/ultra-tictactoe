"""Parallel MCTS for batch inference across multiple games.

Supports double-buffered pipeline: overlaps GPU inference with CPU MCTS work
by splitting games into 2 groups and alternating inference/expansion.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
import time

from game import Board
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from ai.mcts import Node
from ai.mcts.node_cy import select_multi_leaves_cy, expand_backprop_batch_cy, revert_vl_batch_cy


# Timing stats for parallel self-play
_parallel_timing = {
    'total_time': 0.0,
    'network_time': 0.0,
    'mcts_overhead': 0.0,
    'batches': 0,
    'games': 0,
    'moves': 0,
}

def reset_parallel_timing():
    """Reset timing stats (clear in-place to preserve references)."""
    _parallel_timing['total_time'] = 0.0
    _parallel_timing['network_time'] = 0.0
    _parallel_timing['mcts_overhead'] = 0.0
    _parallel_timing['batches'] = 0
    _parallel_timing['games'] = 0
    _parallel_timing['moves'] = 0

def get_parallel_timing():
    return _parallel_timing.copy()


class ParallelMCTS:
    """MCTS that collects leaves from multiple games for batch inference.
    
    Uses double-buffered pipeline to overlap GPU inference with CPU MCTS work.
    """
    
    def __init__(
        self,
        network: AlphaZeroNet,
        num_simulations: int = 100,
        c_puct: float = 1.0,
        dtw_calculator: Optional[DTWCalculator] = None
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dtw_calculator = dtw_calculator
    
    # ── Helper methods (delegated to Cython for speed) ──
    
    def _select_multi_leaves(self, roots, indices, leaves_per_game):
        """Select multiple leaves per game using virtual loss (Cython)."""
        return select_multi_leaves_cy(roots, indices, leaves_per_game, self.c_puct)
    
    @staticmethod
    def _expand_backprop(leaves, policies, values):
        """Expand leaf nodes, backprop values, revert virtual losses (Cython)."""
        n = len(leaves)
        if n == 0:
            return
        values_scaled = np.ascontiguousarray(
            values[:n].ravel(), dtype=np.float32
        )
        policies_f32 = np.ascontiguousarray(policies[:n], dtype=np.float32)
        expand_backprop_batch_cy(leaves, policies_f32, values_scaled)
    
    @staticmethod
    def _revert_virtual_losses(leaves):
        """Revert virtual losses without expand/backprop (Cython)."""
        if leaves:
            revert_vl_batch_cy(leaves)
    
    def _expand_roots(self, roots, games, add_noise):
        """Initial expansion of all root nodes (synchronous)."""
        global _parallel_timing
        boards = [g['board'] for g in games]
        t0 = time.perf_counter()
        policies, _ = self.network.predict_batch(boards)
        _parallel_timing['network_time'] += time.perf_counter() - t0
        
        if add_noise:
            noise_batch = np.random.dirichlet([0.3] * 81, size=len(roots))
            policies = 0.75 * policies + 0.25 * noise_batch
        policies_f32 = policies.astype(np.float32) if policies.dtype != np.float32 else policies
        for i, root in enumerate(roots):
            root.expand_numpy(policies_f32[i])
    
    @staticmethod
    def _select_moves(roots, temperature):
        """Select moves based on visit counts."""
        results = []
        for root in roots:
            visits = np.zeros(81, dtype=np.float32)
            for action, child in root.children.items():
                visits[action] = child.visits
            
            if visits.sum() == 0:
                legal = root.board.get_legal_moves()
                policy = np.zeros(81)
                for action, child in root.children.items():
                    policy[action] = child.prior_prob
                if policy.sum() > 0:
                    policy = policy / policy.sum()
                else:
                    for r, c in legal:
                        policy[r * 9 + c] = 1.0 / len(legal)
                
                if temperature == 0:
                    action = int(np.argmax(policy))
                else:
                    action = int(np.random.choice(81, p=policy))
            elif temperature < 0.01:
                action = int(np.argmax(visits))
                policy = np.zeros(81)
                policy[action] = 1.0
            else:
                # Use log-space to avoid overflow for large visits or small temperature
                with np.errstate(divide='ignore'):
                    log_visits = np.where(visits > 0, np.log(visits), -1e9)
                log_temp = log_visits / temperature
                log_temp -= log_temp.max()  # numerical stability
                visits_temp = np.exp(log_temp)
                total = visits_temp.sum()
                if total == 0 or not np.isfinite(total):
                    policy = np.ones(81) / 81
                else:
                    policy = visits_temp / total
                policy = policy / policy.sum()
                action = np.random.choice(81, p=policy)
            
            results.append((policy, action))
        return results
    
    # ── Main search methods ──
    
    def search_parallel(
        self,
        games: List[Dict],
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Virtual-loss multi-leaf MCTS with double-buffered pipeline.
        
        Instead of 1 leaf per game per simulation (many small batches),
        selects multiple leaves per game using virtual loss, batching them
        into a few large inference calls for much better GPU utilization.
        
        With leaves_per_game=K and N games:
          Old: num_sims × 2 inference calls of ~N/2 boards each
          New: ceil(num_sims/K) × 2 calls of ~N*K/2 boards each → K× fewer calls
        
        Uses async submit/collect for GPU/CPU overlap.
        """
        if not games:
            return []
        
        roots = [Node(game['board']) for game in games]
        global _parallel_timing
        
        # Initial expansion (synchronous — no prior results to overlap)
        self._expand_roots(roots, games, add_noise)
        
        # Determine leaves_per_game: maximize batch size to minimize inference rounds
        # Constrained by TRT max_batch_size (group_size * leaves_per_game <= max_bs)
        n_games = len(roots)
        group_size = (n_games + 1) // 2  # ceil(n/2)
        max_bs = int(self.network._trt_max_bs) if hasattr(self.network, '_trt_max_bs') and self.network._trt_max_bs != float('inf') else 8192
        max_leaves = max(1, max_bs // max(1, group_size))
        leaves_per_game = max(1, min(self.num_simulations, max_leaves))
        num_rounds = max(1, (self.num_simulations + leaves_per_game - 1) // leaves_per_game)
        # Recompute to distribute evenly
        leaves_per_game = max(1, self.num_simulations // num_rounds)
        leftover = self.num_simulations - leaves_per_game * num_rounds
        
        # Split into 2 groups for double buffering
        mid = n_games // 2
        groups = [list(range(0, mid)), list(range(mid, n_games))]
        
        # Pipeline loop with multi-leaf selection
        pending_leaves = None
        pending_handle = None
        pending_submit_time = 0.0
        
        for rnd in range(num_rounds):
            # Extra leaf on first rounds to absorb leftover
            k = leaves_per_game + (1 if rnd < leftover else 0)
            
            for g in range(2):
                if not groups[g]:
                    continue
                
                # 1. Select K leaves per game (virtual loss prevents duplicates)
                leaves, leaf_boards = self._select_multi_leaves(roots, groups[g], k)
                
                # 2. Submit inference (non-blocking)
                t0 = time.perf_counter()
                if leaf_boards:
                    handle = self.network.predict_batch_submit(leaf_boards)
                else:
                    handle = None
                submit_elapsed = time.perf_counter() - t0
                
                # 3. While GPU works: expand+backprop previous batch (CPU overlaps GPU)
                if pending_handle is not None:
                    t_collect = time.perf_counter()
                    policies, values = self.network.predict_batch_collect(pending_handle)
                    _parallel_timing['network_time'] += (time.perf_counter() - t_collect) + pending_submit_time
                    self._expand_backprop(pending_leaves, policies, values)
                    pending_handle = None
                    pending_leaves = None
                
                # 4. Store current handle for next phase
                if handle is not None:
                    pending_leaves = leaves
                    pending_handle = handle
                    pending_submit_time = submit_elapsed
                elif leaves:
                    # No boards to infer (all terminal) — revert virtual losses
                    self._revert_virtual_losses(leaves)
        
        # Process final pending results
        if pending_handle is not None:
            t0 = time.perf_counter()
            policies, values = self.network.predict_batch_collect(pending_handle)
            _parallel_timing['network_time'] += (time.perf_counter() - t0) + pending_submit_time
            self._expand_backprop(pending_leaves, policies, values)
        
        return self._select_moves(roots, temperature)
    
    def search_parallel_sync(
        self,
        games: List[Dict],
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Synchronous MCTS with virtual loss multi-leaf batching (no pipeline).
        Kept for benchmarking comparison against the async pipeline version.
        """
        if not games:
            return []
        
        roots = [Node(game['board']) for game in games]
        global _parallel_timing
        
        self._expand_roots(roots, games, add_noise)
        
        all_indices = list(range(len(roots)))
        n_games = len(roots)
        max_bs = int(self.network._trt_max_bs) if hasattr(self.network, '_trt_max_bs') and self.network._trt_max_bs != float('inf') else 8192
        max_leaves = max(1, max_bs // max(1, n_games))
        leaves_per_game = max(1, min(self.num_simulations, max_leaves))
        num_rounds = max(1, (self.num_simulations + leaves_per_game - 1) // leaves_per_game)
        leaves_per_game = max(1, self.num_simulations // num_rounds)
        leftover = self.num_simulations - leaves_per_game * num_rounds
        
        for rnd in range(num_rounds):
            k = leaves_per_game + (1 if rnd < leftover else 0)
            leaves, leaf_boards = self._select_multi_leaves(roots, all_indices, k)
            
            if not leaf_boards:
                self._revert_virtual_losses(leaves)
                break
            
            t0 = time.perf_counter()
            policies_batch, values_batch = self.network.predict_batch(leaf_boards)
            _parallel_timing['network_time'] += time.perf_counter() - t0
            
            self._expand_backprop(leaves, policies_batch, values_batch)
        
        return self._select_moves(roots, temperature)
