"""NNUE training data generator using AlphaZero agent."""
import random
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from game import Board
from ai.core import AlphaZeroNet
from ai.mcts import AlphaZeroAgent
from ai.training.parallel_mcts import ParallelMCTS
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
    
    def generate_dataset(
        self,
        num_games: int,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate NNUE training dataset using parallel GPU-batched self-play.
        
        Runs `batch_size` games simultaneously via ParallelMCTS for high GPU
        utilization. Much faster than sequential generate_game().
        
        Args:
            num_games: Total number of self-play games
            output_path: Path to save .npz file (optional)
            verbose: Print progress
            
        Returns:
            Dict with 'boards' and 'values' arrays
        """
        batch_size = min(num_games, 2048)
        
        mcts = ParallelMCTS(
            network=self.network,
            num_simulations=self.num_simulations,
            c_puct=1.0,
            dtw_calculator=self.dtw,
        )
        
        board_chunks = []
        value_chunks = []
        start_time = time.time()
        games_done = 0
        
        pbar = tqdm(total=num_games, desc="Phase1 DataGen", unit="g",
                    ncols=90, leave=False, disable=not verbose)
        
        while games_done < num_games:
            cur_batch = min(batch_size, num_games - games_done)
            
            # Init batch of games
            active = []
            for _ in range(cur_batch):
                active.append({
                    'board': Board(),
                    'ply': 0,
                    'raw_samples': [],   # (board_arr, search_eval, stm)
                    'done': False,
                })
            
            # Play until all games in batch are done
            while any(not g['done'] for g in active):
                # Collect games that need a move
                pending = [g for g in active if not g['done']]
                if not pending:
                    break
                
                # ParallelMCTS batch search (return roots to extract value)
                games_for_mcts = [{'board': g['board']} for g in pending]
                results, roots = mcts.search_parallel(
                    games_for_mcts, temperature=0.1, add_noise=False,
                    return_roots=True,
                )
                
                for g, (policy, action), root in zip(pending, results, roots):
                    board = g['board']
                    
                    legal_moves = board.get_legal_moves()
                    if not legal_moves:
                        g['done'] = True
                        continue
                    
                    # Extract eval from MCTS root's best child value
                    eval_value = 0.0
                    if root is not None and root.children:
                        best_child = max(root.children.values(),
                                         key=lambda c: c.visits)
                        eval_value = -best_child.value()  # negate: child is opponent
                    
                    self.stats['positions_total'] += 1
                    
                    # Position filtering and collection
                    if self.position_filter.should_save(
                        board, g['ply'], eval_value, self.rng
                    ):
                        g['raw_samples'].append((
                            self._board_to_array(board),
                            eval_value,
                            board.current_player,
                        ))
                        self.stats['positions_saved'] += 1
                    else:
                        self.stats['positions_skipped'] += 1
                    
                    # Make move
                    row, col = action // 9, action % 9
                    if (row, col) not in legal_moves:
                        g['done'] = True
                        continue
                    
                    board.make_move(row, col)
                    g['ply'] += 1
                    
                    if board.winner not in (None, -1):
                        g['done'] = True
            
            # Finalize: blend search_eval with game result
            lam = self.config.lambda_search
            for g in active:
                board = g['board']
                if board.winner == 1:
                    game_result_p1 = 1.0
                elif board.winner == 2:
                    game_result_p1 = -1.0
                else:
                    game_result_p1 = 0.0
                
                if g['raw_samples']:
                    boards_arr = np.array(
                        [s[0] for s in g['raw_samples']], dtype=np.int8
                    )
                    values_arr = np.array([
                        lam * s[1] + (1.0 - lam) * (
                            game_result_p1 if s[2] == 1 else -game_result_p1
                        )
                        for s in g['raw_samples']
                    ], dtype=np.float32)
                    board_chunks.append(boards_arr)
                    value_chunks.append(values_arr)
                
                self.stats['games_played'] += 1
            
            games_done += cur_batch
            pbar.update(cur_batch)
            
            saved = self.stats['positions_saved']
            elapsed = time.time() - start_time
            rate = games_done / elapsed if elapsed > 0 else 0
            pbar.set_postfix_str(f"{saved:,}pos {rate:.1f}g/s")
        
        pbar.close()
        
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
                tqdm.write(f"Saved {len(all_values):,} positions to {output_path} "
                           f"({elapsed:.1f}s, {games_done/elapsed:.1f} g/s)")
        
        return dataset
    
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
