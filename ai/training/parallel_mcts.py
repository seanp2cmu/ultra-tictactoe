"""Parallel MCTS for batch inference across multiple games."""
from typing import List, Tuple, Optional, Dict
import numpy as np
import time

from game import Board
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from ai.mcts import Node


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
    """MCTS that can collect leaves from multiple games for batch inference."""
    
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
    
    def search_parallel(
        self,
        games: List[Dict],
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Run MCTS on multiple games in parallel, sharing GPU inference.
        
        Args:
            games: List of game dicts with 'board' key
            temperature: Temperature for move selection
            add_noise: Add Dirichlet noise at root
            
        Returns:
            List of (policy, action) for each game
        """
        roots = []
        for game in games:
            root = Node(game['board'])
            roots.append(root)
        
        global _parallel_timing
        
        # Initial expansion of all roots
        boards = [g['board'] for g in games]
        t0 = time.perf_counter()
        policies, _ = self.network.predict_batch(boards)
        _parallel_timing['network_time'] += time.perf_counter() - t0
        
        for i, root in enumerate(roots):
            policy = policies[i]
            if add_noise:
                noise = np.random.dirichlet([0.3] * 81)
                policy = 0.75 * policy + 0.25 * noise
            root.expand(dict(enumerate(policy)))
        
        # Run simulations
        sims_per_batch = max(1, self.num_simulations // 10)
        
        for _ in range(0, self.num_simulations, sims_per_batch):
            batch_size = min(sims_per_batch, self.num_simulations)
            
            all_leaves = []
            all_boards = []
            
            for game_idx, root in enumerate(roots):
                for _ in range(batch_size // len(games) + 1):
                    node = root
                    search_path = [node]
                    
                    while node.is_expanded() and not node.is_terminal():
                        _, node = node.select_child(self.c_puct)
                        search_path.append(node)
                    
                    if not node.is_terminal() and not node.is_expanded():
                        all_leaves.append((game_idx, node, search_path))
                        all_boards.append(node.board)
            
            if not all_boards:
                continue
            
            t0 = time.perf_counter()
            policies_batch, values_batch = self.network.predict_batch(all_boards)
            _parallel_timing['network_time'] += time.perf_counter() - t0
            
            for i, (game_idx, node, search_path) in enumerate(all_leaves):
                policy = policies_batch[i]
                value = 2.0 * values_batch[i].item() - 1.0
                
                if self.dtw_calculator and self.dtw_calculator.is_endgame(node.board):
                    cached = self.dtw_calculator.lookup_cache(node.board)
                    if cached is not None:
                        result, _, _ = cached
                        value = float(result)
                
                node.expand(dict(enumerate(policy)))
                
                for path_node in reversed(search_path):
                    path_node.update(value)
                    value = -value
        
        # Select moves
        results = []
        for i, root in enumerate(roots):
            visits = np.array([
                root.children[a].visits if a in root.children else 0
                for a in range(81)
            ])
            
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
            elif temperature == 0:
                action = int(np.argmax(visits))
                policy = np.zeros(81)
                policy[action] = 1.0
            else:
                visits_temp = visits ** (1.0 / temperature)
                total = visits_temp.sum()
                if total == 0:
                    policy = np.ones(81) / 81
                else:
                    policy = visits_temp / total
                policy = policy / policy.sum()
                action = np.random.choice(81, p=policy)
            
            results.append((policy, action))
        
        return results
