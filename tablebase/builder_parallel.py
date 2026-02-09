"""
Parallel Tablebase Builder using multiprocessing.

Strategy: Parallelize solve within each level.
- Enumerate all boards for level N
- Split into chunks, distribute to workers
- Each worker has READ-ONLY copy of cache (levels 1 to N-1)
- Collect results and merge
"""

import os
import pickle
import time
from collections import defaultdict
from typing import Dict, Set, Optional, List, Tuple
from multiprocessing import Pool, cpu_count

from game.board import Board
from game import USING_CYTHON
from .solver import TablebaseSolver
from .enumerator import PositionEnumerator
from .tablebase import CompactTablebase


def _board_to_state(board) -> dict:
    """Convert board to picklable state dict. Works with both Board and BoardCy."""
    is_cython = hasattr(board, 'get_completed_state')
    
    if is_cython:
        # Cython Board - use getter methods
        cells = [[board.get_cell(r, c) for c in range(9)] for r in range(9)]
        completed = [board.get_completed_state(i) for i in range(9)]
        sub_counts = [board.get_sub_count_pair(i) for i in range(9)]
    else:
        # Pure Python Board
        cells = [list(row) for row in board.cells]
        completed = [board.completed_boards[i//3][i%3] for i in range(9)]
        sub_counts = list(board.sub_counts)
    
    return {
        'cells': cells,
        'completed': completed,
        'sub_counts': sub_counts,
        'x_masks': list(board.x_masks),
        'o_masks': list(board.o_masks),
        'completed_mask': board.completed_mask,
        'current_player': board.current_player,
        'last_move': board.last_move,
        'winner': board.winner,
        'constraint': getattr(board, 'constraint', -1),
    }


def _state_to_board(state: dict) -> Board:
    """Reconstruct board from state dict."""
    board = Board()
    board.cells = [list(row) for row in state['cells']]
    # Convert flat completed list to 3x3
    completed = state['completed']
    board.completed_boards = [[completed[i*3+j] for j in range(3)] for i in range(3)]
    board.sub_counts = list(state['sub_counts'])
    board.x_masks = list(state['x_masks'])
    board.o_masks = list(state['o_masks'])
    board.completed_mask = state['completed_mask']
    board.current_player = state['current_player']
    board.last_move = state['last_move']
    board.winner = state['winner']
    board.constraint = state['constraint']
    return board


def _solve_batch(args) -> List[Tuple[int, int, Tuple]]:
    """Worker function to solve a batch of boards."""
    boards_states, cache, empty_count = args
    
    solver = TablebaseSolver()
    solver.cache = cache
    
    results = []
    for state in boards_states:
        board = _state_to_board(state)
        
        board_hash = solver._hash_board(board)
        result, dtw, best_move = solver.solve(board)
        
        results.append((board_hash, state['constraint'], (result, dtw, best_move)))
    
    return results


class ParallelTablebaseBuilder:
    """Build tablebase using multiple CPU cores."""
    
    def __init__(
        self,
        max_empty: int = 15,
        data_dir: str = 'tablebase/data',
        num_workers: int = None
    ):
        self.max_empty = max_empty
        self.data_dir = data_dir
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        os.makedirs(data_dir, exist_ok=True)
        
        # Main storage
        self.positions: Dict[int, Dict] = {}
        self.seen_hashes: Set[int] = set()
        self.level_positions: Dict[int, Dict[int, Dict]] = {}
        self.level_reached: Dict[int, Set[int]] = {}
        
        # Stats
        self.stats = defaultdict(int)
        
        # Solver for single-threaded operations
        self.solver = TablebaseSolver()
        self.solver.cache = self.positions
        
        # Load existing
        self._load_existing()
        
        print(f"Parallel builder initialized with {self.num_workers} workers")
    
    def _get_level_path(self, level: int) -> str:
        return os.path.join(self.data_dir, f'level_{level}.pkl')
    
    def _get_completed_levels(self) -> Set[int]:
        completed = set()
        for level in range(1, 82):
            if os.path.exists(self._get_level_path(level)):
                completed.add(level)
        return completed
    
    def _load_existing(self):
        completed = self._get_completed_levels()
        if not completed:
            return
        
        print(f"✓ Loading existing levels: {sorted(completed)}")
        
        for level in sorted(completed):
            try:
                with open(self._get_level_path(level), 'rb') as f:
                    data = pickle.load(f)
                
                level_pos = data.get('positions', {})
                level_reached = data.get('reached', set())
                
                self.level_positions[level] = level_pos
                self.level_reached[level] = level_reached
                
                for h, constraints in level_pos.items():
                    self.positions[h] = constraints
                    for c in constraints:
                        self.seen_hashes.add((h << 4) | (c + 1))
                
            except Exception as e:
                print(f"⚠ Failed to load level {level}: {e}")
        
        self.solver.cache = self.positions
        print(f"  Total positions: {len(self.positions)}")
    
    def _save_level(self, level: int):
        if level not in self.level_positions:
            return
        
        path = self._get_level_path(level)
        with open(path, 'wb') as f:
            pickle.dump({
                'positions': self.level_positions[level],
                'reached': self.level_reached.get(level, set())
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _clear_old_levels(self, max_level_to_clear: int, verbose: bool = True):
        """Clear old levels from memory to reduce RAM usage."""
        if max_level_to_clear < 1:
            return
        
        cleared_count = 0
        for level in range(1, max_level_to_clear + 1):
            if level in self.level_positions:
                for h in self.level_positions[level]:
                    if h in self.positions:
                        del self.positions[h]
                del self.level_positions[level]
                cleared_count += 1
        
        if cleared_count > 0 and verbose:
            print(f"  💾 Cleared levels 1-{max_level_to_clear} from RAM")
    
    def build(self, max_positions_per_level: int = None, verbose: bool = True):
        """Build tablebase with parallel processing."""
        start_time = time.time()
        start_count = len(self.positions)
        total_processed = 0
        completed = self._get_completed_levels()
        
        print("=" * 60)
        print(f"Building Tablebase (Parallel): 1 to {self.max_empty} empty cells")
        print(f"Workers: {self.num_workers}")
        print("=" * 60)
        print(f"Starting with {start_count} existing positions")
        if completed:
            print(f"Already completed: {sorted(completed)}")
        
        try:
            for empty_count in range(1, self.max_empty + 1):
                if empty_count in completed:
                    print(f"\n[{empty_count}/{self.max_empty}] Already complete, skipping...")
                    continue
                
                # Initialize level storage
                if empty_count not in self.level_positions:
                    self.level_positions[empty_count] = {}
                if empty_count not in self.level_reached:
                    self.level_reached[empty_count] = set()
                
                print(f"\n{'─' * 60}")
                print(f"[{empty_count}/{self.max_empty}] Building level {empty_count}")
                print(f"{'─' * 60}")
                
                level_start = len(self.positions)
                
                # Step 1: Enumerate all boards for this level
                print("  Enumerating boards...")
                enumerator = PositionEnumerator(empty_cells=empty_count, seen_hashes=self.seen_hashes)
                boards_data = []
                
                for board in enumerator.enumerate(max_positions=max_positions_per_level, show_progress=verbose):
                    # Convert to state dict for multiprocessing
                    boards_data.append(_board_to_state(board))
                
                total_boards = len(boards_data)
                print(f"  Enumerated {total_boards} boards")
                
                if total_boards == 0:
                    continue
                
                # Step 2: Parallel solve
                if self.num_workers > 1 and total_boards > 1000:
                    results = self._parallel_solve(boards_data, empty_count)
                else:
                    # Single-threaded for small batches
                    results = self._sequential_solve(boards_data, empty_count)
                
                # Step 3: Store results
                for board_hash, constraint, (result, dtw, best_move) in results:
                    if board_hash not in self.positions:
                        self.positions[board_hash] = {}
                    self.positions[board_hash][constraint] = (result, dtw, best_move)
                    
                    if board_hash not in self.level_positions[empty_count]:
                        self.level_positions[empty_count][board_hash] = {}
                    self.level_positions[empty_count][board_hash][constraint] = (result, dtw, best_move)
                
                total_processed += total_boards
                
                # Save this level
                self._save_level(empty_count)
                
                level_new = len(self.positions) - level_start
                print(f"  Level {empty_count}: +{level_new} positions (total: {len(self.positions)})")
                
                # Memory optimization: clear old levels from RAM (keep only recent 7)
                self._clear_old_levels(empty_count - 6, verbose)
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        
        finally:
            elapsed = time.time() - start_time
            print("\n" + "=" * 60)
            print("Build Complete!")
            print(f"  Positions processed: {total_processed}")
            print(f"  Total positions: {len(self.positions)}")
            print(f"  Time: {elapsed:.1f}s")
            if total_processed > 0:
                print(f"  Rate: {total_processed/elapsed:.1f} pos/s")
            print("=" * 60)
    
    def _parallel_solve(self, boards_data: List, empty_count: int) -> List:
        """Solve boards in parallel."""
        # Split into chunks
        chunk_size = max(100, len(boards_data) // (self.num_workers * 4))
        chunks = [
            boards_data[i:i + chunk_size]
            for i in range(0, len(boards_data), chunk_size)
        ]
        
        print(f"  Solving {len(boards_data)} boards with {self.num_workers} workers ({len(chunks)} chunks)...")
        
        # Prepare args for workers
        # Each worker gets a READ-ONLY copy of the cache
        cache_copy = dict(self.positions)  # Shallow copy is enough
        args_list = [(chunk, cache_copy, empty_count) for chunk in chunks]
        
        # Run parallel
        results = []
        with Pool(self.num_workers) as pool:
            for chunk_results in pool.imap_unordered(_solve_batch, args_list):
                results.extend(chunk_results)
        
        return results
    
    def _sequential_solve(self, boards_data: List, empty_count: int) -> List:
        """Solve boards sequentially."""
        results = []
        for state in boards_data:
            board = _state_to_board(state)
            
            board_hash = self.solver._hash_board(board)
            result, dtw, best_move = self.solver.solve(board)
            
            results.append((board_hash, state['constraint'], (result, dtw, best_move)))
        
        return results
    
    def export_compact(self, path: str):
        compact = CompactTablebase()
        compact.build_from_dict(self.positions)
        compact.save(path)
        print(f"✓ Exported compact tablebase: {compact.get_size_mb():.2f} MB")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build tablebase (parallel)')
    parser.add_argument('--max-empty', type=int, default=15)
    parser.add_argument('--data-dir', type=str, default='tablebase/data')
    parser.add_argument('--workers', type=int, default=None)
    
    args = parser.parse_args()
    
    builder = ParallelTablebaseBuilder(
        max_empty=args.max_empty,
        data_dir=args.data_dir,
        num_workers=args.workers
    )
    
    builder.build()
    
    compact_path = os.path.join(args.data_dir, 'compact.npz')
    builder.export_compact(compact_path)


if __name__ == '__main__':
    main()
