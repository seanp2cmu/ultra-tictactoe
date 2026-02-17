"""NNUE training data generator using AlphaZero agent."""
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from game import Board
from ai.core import AlphaZeroNet
from ai.mcts import AlphaZeroAgent
from ai.mcts import Node
from ai.endgame import DTWCalculator
from nnue.config import DataGenConfig
from nnue.data.skipping import PositionFilter


class NNUEDataGenerator:
    """Generate NNUE training data from AlphaZero self-play."""
    
    def __init__(
        self,
        network: AlphaZeroNet,
        config: Optional[DataGenConfig] = None,
        num_simulations: int = 800,
        seed: Optional[int] = None
    ):
        self.network = network
        self.config = config or DataGenConfig()
        self.num_simulations = num_simulations
        self.rng = random.Random(seed)
        
        self.dtw = DTWCalculator(use_cache=True)
        self.agent = AlphaZeroAgent(
            network=network,
            num_simulations=num_simulations,
            temperature=0.1,  # Low temperature for quality data
            dtw_calculator=self.dtw
        )
        self.position_filter = PositionFilter(
            config=self.config,
            dtw_calculator=self.dtw
        )
        
        # Statistics
        self.stats = {
            'games_played': 0,
            'positions_total': 0,
            'positions_saved': 0,
            'positions_skipped': 0,
        }
    
    def generate_game(self) -> List[Dict]:
        """
        Play one self-play game and collect NNUE training data.
        
        Returns:
            List of training samples: {'board': np.ndarray, 'value': float}
        """
        board = Board()
        samples = []
        ply = 0
        
        while board.winner in (None, -1):
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            # MCTS search
            root = self.agent.search(board, add_noise=False)
            
            # Get best child value (continuous eval)
            eval_value = self._get_best_child_value(root)
            
            self.stats['positions_total'] += 1
            
            # Check if position should be saved
            if self.position_filter.should_save(board, ply, eval_value, self.rng):
                sample = {
                    'board': self._board_to_array(board),
                    'value': eval_value,
                    'ply': ply,
                }
                samples.append(sample)
                self.stats['positions_saved'] += 1
            else:
                self.stats['positions_skipped'] += 1
            
            # Select move (use visit counts, low temperature)
            action_probs = np.zeros(81, dtype=np.float32)
            for action, child in root.children.items():
                action_probs[action] = child.visits
            
            if action_probs.sum() == 0:
                break
            
            # Temperature-based selection
            action_probs = action_probs ** (1.0 / 0.1)  # Low temp
            action_probs = action_probs / action_probs.sum()
            action = int(np.random.choice(81, p=action_probs))
            
            row, col = action // 9, action % 9
            if (row, col) not in legal_moves:
                break
            
            board.make_move(row, col)
            ply += 1
        
        self.stats['games_played'] += 1
        return samples
    
    def generate_dataset(
        self,
        num_games: int,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate NNUE training dataset from multiple games.
        
        Args:
            num_games: Number of self-play games
            output_path: Path to save .npz file (optional)
            verbose: Print progress
            
        Returns:
            Dict with 'boards' and 'values' arrays
        """
        all_boards = []
        all_values = []
        
        start_time = time.time()
        
        for i in range(num_games):
            samples = self.generate_game()
            
            for s in samples:
                all_boards.append(s['board'])
                all_values.append(s['value'])
            
            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                saved_rate = self.stats['positions_saved'] / max(1, self.stats['positions_total'])
                print(f"Games: {i+1}/{num_games} | "
                      f"Positions: {self.stats['positions_saved']} ({saved_rate:.1%} kept) | "
                      f"Rate: {rate:.1f} games/s")
        
        dataset = {
            'boards': np.array(all_boards, dtype=np.int8),
            'values': np.array(all_values, dtype=np.float32),
        }
        
        if output_path:
            np.savez_compressed(output_path, **dataset)
            if verbose:
                print(f"Saved {len(all_values)} positions to {output_path}")
        
        return dataset
    
    def _get_best_child_value(self, root: Node) -> float:
        """Get evaluation from best (most visited) child."""
        if not root.children:
            return 0.0
        
        best_child = max(root.children.values(), key=lambda c: c.visits)
        # Negate because child value is from opponent's perspective
        return -best_child.value()
    
    def _board_to_array(self, board: Board) -> np.ndarray:
        """
        Convert board to flat array for storage.
        
        Format: 81 cells + 9 meta-board + 1 active_board + 1 current_player
        Total: 92 int8 values
        """
        arr = np.zeros(92, dtype=np.int8)
        
        # 81 cells
        for r in range(9):
            for c in range(9):
                arr[r * 9 + c] = board.get_cell(r, c)
        
        # 9 meta-board states
        if hasattr(board, 'get_completed_boards_2d'):
            completed = board.get_completed_boards_2d()
        else:
            completed = board.completed_boards
        
        for br in range(3):
            for bc in range(3):
                arr[81 + br * 3 + bc] = completed[br][bc]
        
        # Active board (derived from last_move)
        if board.last_move is not None:
            lr, lc = board.last_move
            target_br, target_bc = lr % 3, lc % 3
            if hasattr(board, 'get_completed_boards_2d'):
                target_completed = completed[target_br][target_bc]
            else:
                target_completed = completed[target_br][target_bc]
            
            if target_completed == 0:
                arr[90] = target_br * 3 + target_bc
            else:
                arr[90] = -1  # Any board allowed
        else:
            arr[90] = -1  # First move, any board
        
        # Current player
        arr[91] = board.current_player
        
        return arr
    
    def print_stats(self):
        """Print generation statistics."""
        print(f"\n=== NNUE Data Generation Stats ===")
        print(f"Games played: {self.stats['games_played']}")
        print(f"Positions total: {self.stats['positions_total']}")
        print(f"Positions saved: {self.stats['positions_saved']}")
        print(f"Positions skipped: {self.stats['positions_skipped']}")
        if self.stats['positions_total'] > 0:
            rate = self.stats['positions_saved'] / self.stats['positions_total']
            print(f"Keep rate: {rate:.1%}")
