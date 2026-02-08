"""Systematic Tablebase Builder

Builds tablebase using systematic position enumeration:
1. Enumerate all valid positions with ≤N empty cells
2. Apply D4 symmetry reduction
3. Filter with 4-move backward DFS reachability
4. Solve each position exactly with BFS + memoization
"""

import os
import pickle
import time
from typing import Dict, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm

from game import Board
from .solver import TablebaseSolver, ReachabilityChecker, count_empty
from .enumerator import PositionEnumerator
from .tablebase import Tablebase, CompactTablebase


class TablebaseBuilder:
    """
    Build endgame tablebase using systematic enumeration.
    
    Strategy:
    1. Use PositionEnumerator for systematic generation
    2. D4 symmetry reduction (8x storage savings)
    3. 4-move backward DFS reachability filter
    4. BFS + memoization solver
    """
    
    def __init__(
        self,
        empty_cells: int = 15,
        save_interval: int = 10000,
        save_path: str = 'tablebase/endgame.pkl',
        base_tablebase_path: Optional[str] = None
    ):
        """
        Args:
            empty_cells: Target number of empty playable cells
            save_interval: Save progress every N positions
            save_path: Path to save tablebase
            base_tablebase_path: Path to smaller tablebase for incremental building
        """
        self.empty_cells = empty_cells
        self.save_interval = save_interval
        self.save_path = save_path
        
        # Load base tablebase for incremental building
        base_positions = self._load_base_tablebase(base_tablebase_path)
        
        self.solver = TablebaseSolver(max_depth=30, base_tablebase=base_positions)
        
        # Storage - hash -> (result, dtw, best_move)
        self.positions: Dict[int, Tuple[int, int, Optional[Tuple[int, int]]]] = {}
        self.seen_hashes: Set[int] = set()
        
        # Stats
        self.stats = defaultdict(int)
        
        # Load existing if available
        self._load_existing()
    
    def _load_base_tablebase(self, path: Optional[str]) -> Dict:
        """Load base tablebase for incremental building."""
        if path is None or not os.path.exists(path):
            return {}
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            positions = data.get('positions', {})
            print(f"✓ Loaded base tablebase: {len(positions)} positions")
            return positions
        except Exception as e:
            print(f"⚠ Failed to load base tablebase: {e}")
            return {}
    
    def _load_existing(self):
        """Load existing tablebase if available."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                self.positions = data.get('positions', {})
                self.seen_hashes = set(self.positions.keys())
                self.stats = defaultdict(int, data.get('stats', {}))
                print(f"✓ Loaded existing tablebase: {len(self.positions)} positions")
            except Exception as e:
                print(f"⚠ Failed to load existing: {e}")
    
    def build(self, max_positions: Optional[int] = None, verbose: bool = True):
        """
        Build tablebase using systematic enumeration.
        
        Args:
            max_positions: Stop after this many positions (None = all)
            verbose: Show progress
        """
        start_time = time.time()
        start_count = len(self.positions)
        
        print("=" * 60)
        print(f"Building Tablebase: {self.empty_cells} empty cells")
        print("=" * 60)
        print(f"Starting with {start_count} existing positions")
        
        # Create enumerator
        enumerator = PositionEnumerator(
            empty_cells=self.empty_cells,
            backward_depth=4
        )
        
        positions_processed = 0
        
        try:
            for board in enumerator.enumerate(max_positions=max_positions, show_progress=verbose):
                self._solve_and_store(board)
                positions_processed += 1
                
                # Periodic save
                if positions_processed % self.save_interval == 0:
                    self._save()
                    if verbose:
                        print(f"  Saved: {len(self.positions)} positions")
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        
        finally:
            self._save()
        
        elapsed = time.time() - start_time
        new_count = len(self.positions) - start_count
        
        print(f"\n{'=' * 60}")
        print(f"Build Complete!")
        print(f"  Positions processed: {positions_processed}")
        print(f"  New positions: {new_count}")
        print(f"  Total positions: {len(self.positions)}")
        print(f"  Time: {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Rate: {positions_processed/elapsed:.1f} pos/s")
        print(f"  Solver stats: {dict(self.solver.stats)}")
        print(f"{'=' * 60}")
        
        return self.positions
    
    def _solve_and_store(self, board: Board) -> bool:
        """
        Solve position and store in tablebase.
        
        Returns:
            True if new position was added
        """
        board_hash = self.solver._hash_board(board)
        
        # Skip if already seen
        if board_hash in self.seen_hashes:
            self.stats['duplicates'] += 1
            return False
        
        self.seen_hashes.add(board_hash)
        
        # Solve position
        result, dtw, best_move = self.solver.solve(board)
        
        # Store result
        self.positions[board_hash] = (result, dtw, best_move)
        
        # Update stats
        self.stats['positions_solved'] += 1
        if result == 1:
            self.stats['wins'] += 1
        elif result == -1:
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
        
        return True
    
    def _save(self):
        """Save current tablebase."""
        os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else '.', exist_ok=True)
        
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'stats': dict(self.stats),
                'empty_cells': self.empty_cells
            }, f)
    
    def get_stats(self) -> dict:
        """Get builder statistics."""
        return {
            'total_positions': len(self.positions),
            'empty_cells': self.empty_cells,
            **dict(self.stats)
        }
    
    def export_compact(self, path: str):
        """Export to compact numpy format."""
        compact = CompactTablebase()
        compact.build_from_dict(self.positions)
        compact.save(path)
        print(f"✓ Exported compact tablebase: {compact.get_size_mb():.2f} MB")


def main():
    """Build tablebase from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build endgame tablebase (systematic enumeration)')
    parser.add_argument('--empty', type=int, default=10, help='Number of empty cells')
    parser.add_argument('--output', type=str, default='tablebase/endgame.pkl', help='Output path')
    parser.add_argument('--max', type=int, default=None, help='Max positions to generate')
    parser.add_argument('--base', type=str, default=None, help='Base tablebase for incremental build')
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print(f"Systematic Tablebase Builder")
    print(f"  Empty cells: {args.empty}")
    print(f"  Output: {args.output}")
    if args.base:
        print(f"  Base tablebase: {args.base}")
    print(f"{'=' * 60}\n")
    
    builder = TablebaseBuilder(
        empty_cells=args.empty,
        save_path=args.output,
        base_tablebase_path=args.base
    )
    
    builder.build(max_positions=args.max)
    
    # Export compact version
    compact_path = args.output.replace('.pkl', '.npz')
    builder.export_compact(compact_path)


if __name__ == '__main__':
    main()
