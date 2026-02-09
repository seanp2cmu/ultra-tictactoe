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

from game import Board
from .solver import TablebaseSolver
from .enumerator import PositionEnumerator
from .tablebase import CompactTablebase


class TablebaseBuilder:
    """
    Build endgame tablebase using systematic enumeration.
    
    Strategy:
    1. Use PositionEnumerator for systematic generation
    2. D4 symmetry reduction (8x storage savings)
    3. 4-move backward DFS reachability filter
    4. BFS + memoization solver
    5. Build from 1 empty to max_empty (progressive construction)
    
    Storage: Per-level files in data_dir/
    - level_N.pkl: {'positions': {...}, 'reached': set()}
    """
    
    def __init__(
        self,
        max_empty: int = 15,
        data_dir: str = 'tablebase/data',
        base_tablebase_path: Optional[str] = None
    ):
        """
        Args:
            max_empty: Maximum number of empty playable cells (builds 1 to max_empty)
            data_dir: Directory to save per-level tablebase files
            base_tablebase_path: Path to existing tablebase to continue from
        """
        self.max_empty = max_empty
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Load base tablebase for incremental building
        base_positions = self._load_base_tablebase(base_tablebase_path)
        
        self.solver = TablebaseSolver(max_depth=30, base_tablebase=base_positions)
        
        # Storage - hash -> {constraint: (result, dtw, best_move)}
        # Share dict with solver.cache so lookups work during build
        self.positions: Dict[int, Dict[int, Tuple[int, int, Optional[Tuple[int, int]]]]] = {}
        self.solver.cache = self.positions  # Share reference
        # Pack (hash, constraint) into single int: key = (hash << 4) | (constraint + 1)
        # constraint: -1..8 -> 0..9 (4 bits)
        self.seen_hashes: Set[int] = set()
        
        # Per-level storage: level -> {hash: {constraint: (result, dtw, move)}}
        self.level_positions: Dict[int, Dict[int, Dict]] = {}
        # Per-level reached hashes: level -> set of reached hashes
        self.level_reached: Dict[int, Set[int]] = {}
        
        # Stats
        self.stats = defaultdict(int)
        
        # Load existing levels
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
    
    def _get_level_path(self, level: int) -> str:
        """Get file path for a level."""
        return os.path.join(self.data_dir, f'level_{level}.pkl')
    
    def _get_completed_levels(self) -> Set[int]:
        """Get set of completed levels from existing files."""
        completed = set()
        for level in range(1, 82):  # Max possible levels
            if os.path.exists(self._get_level_path(level)):
                completed.add(level)
        return completed
    
    def _load_existing(self):
        """Load existing per-level tablebase files."""
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
                
                # Merge into main positions dict for solver cache
                for h, constraints in level_pos.items():
                    self.positions[h] = constraints
                    for c in constraints:
                        self.seen_hashes.add((h << 4) | (c + 1))
                
            except Exception as e:
                print(f"⚠ Failed to load level {level}: {e}")
        
        # Share positions with solver cache
        self.solver.cache = self.positions
        print(f"  Total positions: {len(self.positions)}")
    
    def build(self, max_positions_per_level: Optional[int] = None, verbose: bool = True):
        """
        Build tablebase using progressive construction (1 to max_empty).
        
        Args:
            max_positions_per_level: Stop after this many positions per empty count (None = all)
            verbose: Show progress
        """
        start_time = time.time()
        start_count = len(self.positions)
        
        completed = self._get_completed_levels()
        
        print("=" * 60)
        print(f"Building Tablebase: 1 to {self.max_empty} empty cells")
        print("=" * 60)
        print(f"Starting with {start_count} existing positions")
        if completed:
            print(f"Already completed: {sorted(completed)}")
        
        total_processed = 0
        
        try:
            # Build from 1 empty to max_empty (progressive order)
            for empty_count in range(1, self.max_empty + 1):
                # Skip if already completed
                if empty_count in completed:
                    print(f"\n[{empty_count}/{self.max_empty}] Already complete, skipping...")
                    continue
                
                # Initialize level storage
                if empty_count not in self.level_positions:
                    self.level_positions[empty_count] = {}
                if empty_count not in self.level_reached:
                    self.level_reached[empty_count] = set()
                
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
                
                # Save this level
                self._save_level(empty_count)
                
                level_new = len(self.positions) - level_start
                print(f"  Level {empty_count}: +{level_new} positions (total: {len(self.positions)})")
                
                # Prune unreachable positions from level (empty_count - 7)
                # At level N, children go down to N-1, so level 1 is child of level 2
                # Level 8 can safely prune level 1 (7 hops: 8→7→6→5→4→3→2→1)
                prune_level = empty_count - 7
                if prune_level >= 1:
                    self._prune_unreachable(prune_level, verbose)
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user - progress saved")
            # Save current level in progress
            current_level = empty_count
            if current_level in self.level_positions and self.level_positions[current_level]:
                self._save_level(current_level)
        
        elapsed = time.time() - start_time
        new_count = len(self.positions) - start_count
        
        print(f"\n{'=' * 60}")
        print(f"Build Complete!")
        print(f"  Empty range: 1 to {self.max_empty}")
        print(f"  Completed levels: {sorted(self._get_completed_levels())}")
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
        
        # Skip if already seen (packed int key)
        key = (board_hash << 4) | (constraint + 1)
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
        
        # Also store in level-specific dict
        if board_hash not in self.level_positions[empty_count]:
            self.level_positions[empty_count][board_hash] = {}
        self.level_positions[empty_count][board_hash][constraint] = (result, dtw, best_move)
        
        # Mark child positions as reached (for forward reachability)
        if empty_count >= 2:
            self._mark_children_reached(board, empty_count)
        
        # Update stats
        self.stats['positions_solved'] += 1
        if result == 1:
            self.stats['wins'] += 1
        elif result == -1:
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
        
        return True
    
    def _mark_children_reached(self, board: Board, parent_level: int):
        """Mark all child positions as reached (using make/undo, no clone)."""
        child_level = parent_level - 1
        if child_level < 1:
            return
        
        # Initialize reached set for child level if needed
        if child_level not in self.level_reached:
            self.level_reached[child_level] = set()
        
        moves = board.get_legal_moves()
        prev_last_move = board.last_move
        prev_winner = board.winner
        
        for r, c in moves:
            sub_r, sub_c = r // 3, c // 3
            sub_idx = sub_r * 3 + sub_c
            # Compatible with BoardCy (1D) and Board (2D)
            if hasattr(board, 'get_completed_state'):
                prev_completed = board.get_completed_state(sub_idx)
            else:
                prev_completed = board.completed_boards[sub_r][sub_c]
            
            board.make_move(r, c, validate=False)
            child_hash = self.solver._hash_board(board)
            self.level_reached[child_level].add(child_hash)
            board.undo_move(r, c, prev_completed, prev_winner, prev_last_move)
    
    def _prune_unreachable(self, level: int, verbose: bool = True):
        """Remove unreachable positions from a level."""
        if level not in self.level_positions:
            return
        
        reached = self.level_reached.get(level, set())
        to_delete = [h for h in self.level_positions[level] if h not in reached]
        
        for h in to_delete:
            # Remove from seen_hashes
            if h in self.level_positions[level]:
                for constraint in self.level_positions[level][h]:
                    self.seen_hashes.discard((h << 4) | (constraint + 1))
            # Remove from main positions
            if h in self.positions:
                del self.positions[h]
            # Remove from level positions
            del self.level_positions[level][h]
        
        if to_delete:
            # Update level file after pruning
            self._save_level(level)
            if verbose:
                print(f"  🗑 Pruned {len(to_delete)} unreachable positions from level {level}")
        
        # Clear reached set for this level (no longer needed)
        if level in self.level_reached:
            del self.level_reached[level]
    
    def _save_level(self, level: int):
        """Save a single level's positions and reached hashes."""
        if level not in self.level_positions:
            return
        
        path = self._get_level_path(level)
        with open(path, 'wb') as f:
            pickle.dump({
                'positions': self.level_positions[level],
                'reached': self.level_reached.get(level, set())
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_stats(self) -> dict:
        """Get builder statistics."""
        return {
            'total_positions': len(self.positions),
            'max_empty': self.max_empty,
            'completed_levels': sorted(self._get_completed_levels()),
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
    parser.add_argument('--data-dir', type=str, default='tablebase/data', help='Data directory for level files')
    parser.add_argument('--max-per-level', type=int, default=None, help='Max positions per level (for testing)')
    parser.add_argument('--base', type=str, default=None, help='Continue from existing tablebase')
    parser.add_argument('--upload-hf', action='store_true', help='Upload to HuggingFace after build')
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print(f"Endgame Tablebase Builder")
    print(f"  Empty range: 1 to {args.max_empty}")
    print(f"  Data dir: {args.data_dir}")
    if args.base:
        print(f"  Continue from: {args.base}")
    if args.upload_hf:
        print(f"  Upload to HF: enabled")
    print(f"{'=' * 60}\n")
    
    builder = TablebaseBuilder(
        max_empty=args.max_empty,
        data_dir=args.data_dir,
        base_tablebase_path=args.base
    )
    
    builder.build(max_positions_per_level=args.max_per_level)
    
    # Export compact version
    compact_path = os.path.join(args.data_dir, 'compact.npz')
    builder.export_compact(compact_path)
    
    # Upload to HuggingFace if requested
    if args.upload_hf:
        # Collect all level files + compact
        level_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pkl')]
        upload_to_hf(level_files + [compact_path])


if __name__ == '__main__':
    main()
