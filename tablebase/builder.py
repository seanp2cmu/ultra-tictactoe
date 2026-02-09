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
from .solver import TablebaseSolver, count_empty
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
    5. Build from 1 empty to max_empty (progressive construction)
    """
    
    def __init__(
        self,
        max_empty: int = 15,
        save_path: str = 'tablebase/endgame.pkl',
        base_tablebase_path: Optional[str] = None
    ):
        """
        Args:
            max_empty: Maximum number of empty playable cells (builds 1 to max_empty)
            save_path: Path to save tablebase
            base_tablebase_path: Path to existing tablebase to continue from
        """
        self.max_empty = max_empty
        self.save_path = save_path
        
        # Load base tablebase for incremental building
        base_positions = self._load_base_tablebase(base_tablebase_path)
        
        self.solver = TablebaseSolver(max_depth=30, base_tablebase=base_positions)
        
        # Storage - hash -> {constraint: (result, dtw, best_move)}
        # Share dict with solver.cache so lookups work during build
        self.positions: Dict[int, Dict[int, Tuple[int, int, Optional[Tuple[int, int]]]]] = {}
        self.solver.cache = self.positions  # Share reference
        self.seen_hashes: Set[Tuple[int, int]] = set()  # (hash, constraint) pairs
        
        # Track which empty counts are complete
        self.completed_empty: Set[int] = set()
        
        # Forward reachability tracking
        self.position_levels: Dict[int, int] = {}  # hash -> empty_count
        self.reached_hashes: Set[int] = set()  # hashes reached from higher levels
        
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
                # Rebuild seen_hashes as (hash, constraint) pairs
                self.seen_hashes = set()
                for h, constraints in self.positions.items():
                    for c in constraints:
                        self.seen_hashes.add((h, c))
                self.stats = defaultdict(int, data.get('stats', {}))
                self.completed_empty = set(data.get('completed_empty', []))
                
                # Share positions with solver cache for child lookups
                self.solver.cache = self.positions
                
                print(f"✓ Loaded existing tablebase: {len(self.positions)} positions")
                if self.completed_empty:
                    print(f"  Completed empty counts: {sorted(self.completed_empty)}")
            except Exception as e:
                print(f"⚠ Failed to load existing: {e}")
    
    def build(self, max_positions_per_level: Optional[int] = None, verbose: bool = True):
        """
        Build tablebase using progressive construction (1 to max_empty).
        
        Args:
            max_positions_per_level: Stop after this many positions per empty count (None = all)
            verbose: Show progress
        """
        start_time = time.time()
        start_count = len(self.positions)
        
        print("=" * 60)
        print(f"Building Tablebase: 1 to {self.max_empty} empty cells")
        print("=" * 60)
        print(f"Starting with {start_count} existing positions")
        if self.completed_empty:
            print(f"Already completed: {sorted(self.completed_empty)}")
        
        total_processed = 0
        
        try:
            # Build from 1 empty to max_empty (progressive order)
            for empty_count in range(1, self.max_empty + 1):
                # Skip if already completed
                if empty_count in self.completed_empty:
                    print(f"\n[{empty_count}/{self.max_empty}] Already complete, skipping...")
                    continue
                
                print(f"\n{'─' * 60}")
                print(f"[{empty_count}/{self.max_empty}] Building positions with {empty_count} empty cells")
                print(f"{'─' * 60}")
                
                level_start = len(self.positions)
                level_processed = 0
                
                # Create enumerator for this empty count
                enumerator = PositionEnumerator(empty_cells=empty_count)
                
                for board in enumerator.enumerate(max_positions=max_positions_per_level, show_progress=verbose):
                    self._solve_and_store(board, empty_count)
                    level_processed += 1
                    total_processed += 1
                
                # Mark this level as complete (save only at level end)
                self.completed_empty.add(empty_count)
                self._save()
                
                level_new = len(self.positions) - level_start
                print(f"  Level {empty_count}: +{level_new} positions (total: {len(self.positions)})")
                
                # Prune unreachable positions from level (empty_count - 6)
                prune_level = empty_count - 6
                if prune_level >= 1:
                    self._prune_unreachable(prune_level, verbose)
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user - progress saved")
        
        finally:
            self._save()
        
        elapsed = time.time() - start_time
        new_count = len(self.positions) - start_count
        
        print(f"\n{'=' * 60}")
        print(f"Build Complete!")
        print(f"  Empty range: 1 to {self.max_empty}")
        print(f"  Completed levels: {sorted(self.completed_empty)}")
        print(f"  Positions processed: {total_processed}")
        print(f"  New positions: {new_count}")
        print(f"  Total positions: {len(self.positions)}")
        print(f"  Time: {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Rate: {total_processed/elapsed:.1f} pos/s")
        print(f"  Solver stats: {dict(self.solver.stats)}")
        print(f"{'=' * 60}")
        
        return self.positions
    
    def _solve_and_store(self, board: Board, empty_count: int) -> bool:
        """
        Solve position and store in tablebase.
        
        Returns:
            True if new position was added
        """
        board_hash = self.solver._hash_board(board)
        constraint = getattr(board, 'constraint', -1)
        
        # Skip if already seen (check hash + constraint pair)
        key = (board_hash, constraint)
        if key in self.seen_hashes:
            self.stats['duplicates'] += 1
            return False
        
        self.seen_hashes.add(key)
        result, dtw, best_move = self.solver.solve(board)
        
        # DTW can be > empty_count in some cases
        if dtw > empty_count:
            self.stats['dtw_exceeded'] += 1
        
        # Store result in nested dict: hash -> {constraint: (result, dtw, move)}
        if board_hash not in self.positions:
            self.positions[board_hash] = {}
        self.positions[board_hash][constraint] = (result, dtw, best_move)
        self.position_levels[board_hash] = empty_count
        
        # Mark child positions as reached (for forward reachability)
        if empty_count >= 2:
            self._mark_children_reached(board)
        
        # Update stats
        self.stats['positions_solved'] += 1
        if result == 1:
            self.stats['wins'] += 1
        elif result == -1:
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
        
        return True
    
    def _mark_children_reached(self, board: Board):
        """Mark all child positions as reached (using make/undo, no clone)."""
        moves = board.get_legal_moves()
        prev_last_move = board.last_move
        prev_winner = board.winner
        
        for r, c in moves:
            sub_r, sub_c = r // 3, c // 3
            prev_completed = board.completed_boards[sub_r][sub_c]
            
            board.make_move(r, c, validate=False)
            child_hash = self.solver._hash_board(board)
            self.reached_hashes.add(child_hash)
            board.undo_move(r, c, prev_completed, prev_winner, prev_last_move)
    
    def _prune_unreachable(self, level: int, verbose: bool = True):
        """Remove unreachable positions from a level."""
        to_delete = []
        for h, lvl in self.position_levels.items():
            if lvl == level and h not in self.reached_hashes:
                to_delete.append(h)
        
        for h in to_delete:
            # Remove all constraints for this hash from seen_hashes
            if h in self.positions:
                for constraint in self.positions[h]:
                    self.seen_hashes.discard((h, constraint))
            del self.positions[h]
            del self.position_levels[h]
        
        if verbose and to_delete:
            print(f"  🗑 Pruned {len(to_delete)} unreachable positions from level {level}")
    
    def _save(self):
        """Save current tablebase."""
        os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else '.', exist_ok=True)
        
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'stats': dict(self.stats),
                'max_empty': self.max_empty,
                'completed_empty': list(self.completed_empty)
            }, f)
    
    def get_stats(self) -> dict:
        """Get builder statistics."""
        return {
            'total_positions': len(self.positions),
            'max_empty': self.max_empty,
            'completed_levels': sorted(self.completed_empty),
            **dict(self.stats)
        }
    
    def export_compact(self, path: str):
        """Export to compact numpy format."""
        compact = CompactTablebase()
        compact.build_from_dict(self.positions)
        compact.save(path)
        print(f"✓ Exported compact tablebase: {compact.get_size_mb():.2f} MB")


def upload_to_hf(file_paths: list, repo_id: str = "sean2474/ultra-tictactoe-models"):
    """Upload tablebase files to HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                print(f"Uploading {filename} to HuggingFace...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"tablebase/{filename}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✓ Uploaded: {filename}")
        
        print(f"✓ All files uploaded to {repo_id}")
    except Exception as e:
        print(f"⚠ HuggingFace upload failed: {e}")


def main():
    """Build tablebase from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build endgame tablebase')
    parser.add_argument('--max-empty', type=int, default=15, help='Maximum empty cells (builds 1 to max)')
    parser.add_argument('--output', type=str, default='tablebase/endgame.pkl', help='Output path')
    parser.add_argument('--max-per-level', type=int, default=None, help='Max positions per level (for testing)')
    parser.add_argument('--base', type=str, default=None, help='Continue from existing tablebase')
    parser.add_argument('--upload-hf', action='store_true', help='Upload to HuggingFace after build')
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print(f"Endgame Tablebase Builder")
    print(f"  Empty range: 1 to {args.max_empty}")
    print(f"  Output: {args.output}")
    if args.base:
        print(f"  Continue from: {args.base}")
    if args.upload_hf:
        print(f"  Upload to HF: enabled")
    print(f"{'=' * 60}\n")
    
    builder = TablebaseBuilder(
        max_empty=args.max_empty,
        save_path=args.output,
        base_tablebase_path=args.base
    )
    
    builder.build(max_positions_per_level=args.max_per_level)
    
    # Export compact version
    compact_path = args.output.replace('.pkl', '.npz')
    builder.export_compact(compact_path)
    
    # Upload to HuggingFace if requested
    if args.upload_hf:
        upload_to_hf([args.output, compact_path])


if __name__ == '__main__':
    main()
