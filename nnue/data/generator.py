"""NNUE training data generator using AlphaZero agent."""
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        
        Blends search score with game result:
          target = λ × search_score + (1-λ) × game_result
        where game_result is from the side-to-move's perspective.
        
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
                    'search_eval': eval_value,
                    'stm': board.current_player,  # side to move
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
        
        # Determine game result: +1 = P1 win, -1 = P2 win, 0 = draw/unfinished
        if board.winner == 1:
            game_result_p1 = 1.0
        elif board.winner == 2:
            game_result_p1 = -1.0
        else:
            game_result_p1 = 0.0
        
        # Blend search score with game result for each sample
        lam = self.config.lambda_search
        for s in samples:
            # Game result from side-to-move's perspective
            game_result_stm = game_result_p1 if s['stm'] == 1 else -game_result_p1
            s['value'] = lam * s['search_eval'] + (1.0 - lam) * game_result_stm
        
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
        # Pre-allocate chunks to avoid per-sample list.append overhead
        board_chunks = []
        value_chunks = []
        
        start_time = time.time()
        
        for i in range(num_games):
            samples = self.generate_game()
            
            if samples:
                boards_batch = np.array([s['board'] for s in samples], dtype=np.int8)
                values_batch = np.array([s['value'] for s in samples], dtype=np.float32)
                board_chunks.append(boards_batch)
                value_chunks.append(values_batch)
            
            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                saved = self.stats['positions_saved']
                total = self.stats['positions_total']
                kept = saved / max(1, total)
                print(f"Games: {i+1}/{num_games} | "
                      f"Positions: {saved:,} ({kept:.1%} kept) | "
                      f"Rate: {rate:.1f} games/s")
        
        if board_chunks:
            all_boards = np.concatenate(board_chunks)
            all_values = np.concatenate(value_chunks)
        else:
            all_boards = np.zeros((0, 92), dtype=np.int8)
            all_values = np.zeros(0, dtype=np.float32)
        
        dataset = {
            'boards': all_boards,
            'values': all_values,
        }
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **dataset)
            if verbose:
                elapsed = time.time() - start_time
                print(f"\nSaved {len(all_values):,} positions to {output_path} "
                      f"({elapsed:.1f}s total)")
        
        return dataset
    
    def _get_best_child_value(self, root: Node) -> float:
        """Get evaluation from best (most visited) child."""
        if not root.children:
            return 0.0
        
        best_child = max(root.children.values(), key=lambda c: c.visits)
        # Negate because child value is from opponent's perspective
        return -best_child.value()
    
    def _board_to_array(self, board: Board) -> np.ndarray:
        """Convert board to flat (92,) int8 array for storage."""
        arr = np.zeros(92, dtype=np.int8)
        
        # 81 cells — use board's flat representation if available
        for r in range(9):
            for c in range(9):
                arr[r * 9 + c] = board.get_cell(r, c)
        
        # 9 meta-board states
        completed = board.get_completed_boards_2d()
        for i in range(9):
            arr[81 + i] = completed[i // 3][i % 3]
        
        # Active board
        if board.last_move is not None:
            lr, lc = board.last_move
            target_sub = (lr % 3) * 3 + (lc % 3)
            arr[90] = target_sub if arr[81 + target_sub] == 0 else -1
        else:
            arr[90] = -1
        
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
