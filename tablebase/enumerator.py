"""
Position Enumerator for Tablebase Generation

Uses precomputed local_boards and meta_boards for efficient enumeration.
"""

from typing import Generator, Tuple, List, Set
from tqdm import tqdm

from game import Board
from .local_boards import OPEN_BOARDS, _make_key as local_key
from .meta_boards import META_BOARDS, D4_TRANSFORMS, _make_key as meta_key


class PositionEnumerator:
    """Enumerate positions with exactly N empty cells."""
    
    def __init__(self, empty_cells: int = 15):
        self.empty_cells = empty_cells
        self.seen_hashes: Set[Tuple] = set()
        self.stats = {
            'meta_boards': 0,
            'generated': 0,
            'duplicate': 0,
            'yielded': 0
        }
        
        # Load precomputed data once
        self._local_boards = OPEN_BOARDS
        self._meta_boards = self._load_meta_boards()
    
    def _load_meta_boards(self) -> List[Tuple[Tuple[int, ...], int, int]]:
        """Load all meta-boards for this empty_cells count."""
        result = []
        for num_open in range(1, 10):
            key = meta_key(num_open, self.empty_cells)
            result.extend(META_BOARDS.get(key, []))
        return result
    
    def _get_local(self, empty: int, diff: int) -> List[Tuple[int, ...]]:
        """Get local boards from preloaded cache."""
        return self._local_boards.get(local_key(diff, empty), [])
    
    def enumerate(self, max_positions: int = None, show_progress: bool = True) -> Generator[Board, None, None]:
        """Enumerate valid positions."""
        count = 0
        pbar = tqdm(desc="Enumerating") if show_progress else None
        
        # Use preloaded meta-boards
        for meta, min_diff, max_diff in self._meta_boards:
            self.stats['meta_boards'] += 1
            
            # Fill sub-boards (pass seen_hashes for early dedup)
            for board in self._fill_subboards(meta, min_diff, max_diff):
                self.stats['yielded'] += 1
                count += 1
                
                if show_progress:
                    pbar.update(1)
                
                yield board
                
                if max_positions and count >= max_positions:
                    if show_progress:
                        pbar.close()
                    return
        
        if show_progress:
            pbar.close()
    
    def _build_raw_data(self, meta: Tuple[int, ...], open_indices: List[int],
                        empty_dist: Tuple[int, ...], diff_dist: Tuple[int, ...]) -> Tuple:
        """Build raw_data directly from meta+distributions (no Board needed)."""
        raw_data = []
        open_idx = 0
        for sub_idx in range(9):
            state = meta[sub_idx]
            if state != 0:
                raw_data.append((state, 0, 0))
            else:
                empty = empty_dist[open_idx]
                diff = diff_dist[open_idx]
                filled = 9 - empty
                x_count = (filled + diff) // 2
                o_count = (filled - diff) // 2
                raw_data.append((0, x_count, o_count))
                open_idx += 1
        return tuple(raw_data)
    
    def _get_canonical_keys(self, raw_data: Tuple, open_indices: List[int]) -> List[Tuple[int, Tuple]]:
        """Return (constraint, canonical_key) pairs for dedup."""
        seen_canonical = set()
        result = []
        
        for constraint in open_indices:
            candidates = []
            for perm in D4_TRANSFORMS:
                sym_data = tuple(raw_data[perm[i]] for i in range(9))
                sym_constraint = perm.index(constraint)
                candidates.append((sym_data, sym_constraint))
                
                flipped = tuple(
                    (3 - s, 0, 0) if s in (1, 2) else (0, o, x) if s == 0 else (s, 0, 0)
                    for s, x, o in sym_data
                )
                candidates.append((flipped, sym_constraint))
            
            canonical = min(candidates)
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                result.append((constraint, canonical))
        
        return result
    
    def _fill_subboards(self, meta: Tuple[int, ...], min_diff: int, max_diff: int) -> Generator[Board, None, None]:
        """Fill sub-boards using precomputed local boards."""
        open_indices = [i for i, s in enumerate(meta) if s == 0]
        num_open = len(open_indices)
        
        if num_open == 0:
            return
        
        total_filled = 9 * num_open - self.empty_cells
        
        # For each valid diff
        for diff in range(min_diff, max_diff + 1):
            if (total_filled + diff) % 2 != 0:
                continue
            
            x_total = (total_filled + diff) // 2
            o_total = (total_filled - diff) // 2
            
            if x_total < 0 or o_total < 0:
                continue
            
            # Distribute empty cells
            for empty_dist in self._distribute_empty(num_open, self.empty_cells):
                # Distribute diff across sub-boards
                for diff_dist in self._distribute_diff(num_open, diff, empty_dist):
                    # Step 1: Build raw_data WITHOUT creating Board
                    raw_data = self._build_raw_data(meta, open_indices, empty_dist, diff_dist)
                    
                    # Step 2: Get canonical keys, filter already-seen BEFORE Board creation
                    canonical_pairs = []
                    for constraint, key in self._get_canonical_keys(raw_data, open_indices):
                        if key not in self.seen_hashes:
                            self.seen_hashes.add(key)
                            canonical_pairs.append((constraint, key))
                        else:
                            self.stats['duplicate'] += 1
                    
                    if not canonical_pairs:
                        continue
                    
                    # Only create Board if we have non-duplicate keys
                    self.stats['generated'] += 1
                    board = self._create_board(meta, open_indices, empty_dist, diff_dist)
                    if not board:
                        continue
                    
                    # Step 3: Yield boards - reuse first, clone rest
                    for i, (constraint, _) in enumerate(canonical_pairs):
                        if i == 0:
                            board.constraint = constraint
                            sub_r, sub_c = constraint // 3, constraint % 3
                            board.last_move = (sub_r, sub_c)
                            yield board
                        else:
                            b = board.clone()
                            b.constraint = constraint
                            sub_r, sub_c = constraint // 3, constraint % 3
                            b.last_move = (sub_r, sub_c)
                            yield b
    
    def _distribute_empty(self, num_open: int, total_empty: int) -> Generator[Tuple[int, ...], None, None]:
        """Distribute empty cells across sub-boards (1-8 each)."""
        if num_open == 1:
            if 1 <= total_empty <= 8:
                yield (total_empty,)
            return
        
        min_first = max(1, total_empty - 8 * (num_open - 1))
        max_first = min(8, total_empty - (num_open - 1))
        
        for first in range(min_first, max_first + 1):
            for rest in self._distribute_empty(num_open - 1, total_empty - first):
                yield (first,) + rest
    
    def _distribute_diff(self, num_open: int, total_diff: int, empty_dist: Tuple[int, ...]) -> Generator[Tuple[int, ...], None, None]:
        """Distribute diff across sub-boards within valid ranges."""
        if num_open == 1:
            # Check if this diff is achievable with given empty
            filled = 9 - empty_dist[0]
            if -filled <= total_diff <= filled and abs(total_diff) <= 6:
                # Check if board exists
                if self._get_local(empty_dist[0], total_diff):
                    yield (total_diff,)
            return
        
        filled = 9 - empty_dist[0]
        min_diff = max(-filled, -6)
        max_diff = min(filled, 6)
        
        for d in range(min_diff, max_diff + 1):
            # Check if board exists for this (empty, diff)
            if not self._get_local(empty_dist[0], d):
                continue
            
            for rest in self._distribute_diff(num_open - 1, total_diff - d, empty_dist[1:]):
                yield (d,) + rest
    
    def _create_board(self, meta: Tuple[int, ...], open_indices: List[int], 
                      empty_dist: Tuple[int, ...], diff_dist: Tuple[int, ...]) -> Board:
        """Create board using precomputed local boards."""
        board = Board()
        
        for i, sub_idx in enumerate(open_indices):
            boards = self._get_local(empty_dist[i], diff_dist[i])
            if not boards:
                return None
            
            cells = boards[0]  # Take first valid board
            sub_r, sub_c = sub_idx // 3, sub_idx % 3
            
            for j, val in enumerate(cells):
                r = sub_r * 3 + j // 3
                c = sub_c * 3 + j % 3
                board.boards[r][c] = val
        
        # Set meta-board state
        for sub_idx in range(9):
            sub_r, sub_c = sub_idx // 3, sub_idx % 3
            board.completed_boards[sub_r][sub_c] = meta[sub_idx]
        
        # Set current player
        total_x = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 1)
        total_o = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 2)
        board.current_player = 1 if total_x == total_o else 2
        board.winner = None
        
        return board


if __name__ == '__main__':
    for level in [1, 2]:
        enum = PositionEnumerator(empty_cells=level)
        count = sum(1 for _ in enum.enumerate(show_progress=False))
        print(f"L{level}: {count}")
