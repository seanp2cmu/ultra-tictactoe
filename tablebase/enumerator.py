"""
Position Enumerator for Tablebase Generation

Systematic enumeration:
1. Enumerate meta-board (big board) states
2. For each meta-board, fill sub-boards systematically  
3. Apply D4 symmetry reduction (no X-O swap - can't know which has more pieces)
4. Filter invalid configurations (no reachability check)
"""

from itertools import product
from typing import Generator, Tuple, List, Set
from tqdm import tqdm

from game import Board
from utils import BoardSymmetry


# Sub-board winning patterns
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


class PositionEnumerator:
    """
    Systematic enumeration of positions with exactly N empty cells.
    
    Strategy:
    1. Enumerate meta-board (big board) configurations
    2. For each meta-board, enumerate valid (X,O) count combinations
    3. For OPEN boards only: place actual pieces
    4. Apply D4 symmetry to avoid duplicates
    5. Filter: valid total X-O counts
    """
    
    # Meta-board states
    OPEN = 0    # Sub-board still in play
    X_WIN = 1   # X won sub-board
    O_WIN = 2   # O won sub-board
    DRAW = 3    # Sub-board is draw (full, no winner)
    
    
    def __init__(self, empty_cells: int = 15):
        self.empty_cells = empty_cells
        
        self.seen_hashes: Set[int] = set()  # For symmetry dedup
        self.stats = {
            'meta_boards': 0,
            'generated': 0,
            'valid_structure': 0,
            'duplicate': 0,
            'yielded': 0
        }
    
    def enumerate(self, max_positions: int = None, show_progress: bool = True) -> Generator[Board, None, None]:
        """
        Enumerate valid positions systematically (no reachability check).
        """
        count = 0
        pbar = tqdm(desc="Enumerating") if show_progress else None
        
        # Step 1: Enumerate all meta-board configurations with valid diff range
        for meta, min_diff, max_diff in self._enumerate_meta_boards():
            self.stats['meta_boards'] += 1
            
            # Step 2: For each meta-board, fill sub-boards with valid X/O counts
            for board in self._fill_subboards(meta, min_diff, max_diff):
                self.stats['generated'] += 1
                
                if board is None:
                    continue
                
                # Step 3: Dedup using normalized (meta, x_counts, o_counts) with X/O flip
                dedup_key = self._compute_normalized_key(board)
                if dedup_key in self.seen_hashes:
                    self.stats['duplicate'] += 1
                    continue
                
                self.seen_hashes.add(dedup_key)
                self.stats['valid_structure'] += 1
                self.stats['yielded'] += 1
                count += 1
                
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        'meta': self.stats['meta_boards'],
                        'valid': self.stats['valid_structure']
                    })
                
                yield board
                
                if max_positions and count >= max_positions:
                    if pbar is not None:
                        pbar.close()
                    return
        
        if pbar is not None:
            pbar.close()
    
    def _compute_normalized_key(self, board: Board) -> Tuple:
        """
        Compute normalized dedup key with X/O flip.
        
        Normalization:
        1. If o_wins > x_wins: flip
        2. If equal, first non-empty cell in OPEN sub-boards should be X
        """
        # Count X_WIN and O_WIN
        x_wins = 0
        o_wins = 0
        for r in range(3):
            for c in range(3):
                if board.completed_boards[r][c] == 1:
                    x_wins += 1
                elif board.completed_boards[r][c] == 2:
                    o_wins += 1
        
        # Determine if we need to flip
        need_flip = False
        if o_wins > x_wins:
            need_flip = True
        elif o_wins == x_wins:
            # Check first non-empty cell in OPEN sub-boards
            for sub_idx in range(9):
                sub_r, sub_c = sub_idx // 3, sub_idx % 3
                if board.completed_boards[sub_r][sub_c] == 0:  # OPEN
                    for cell_idx in range(9):
                        dr, dc = cell_idx // 3, cell_idx % 3
                        cell = board.boards[sub_r*3 + dr][sub_c*3 + dc]
                        if cell == 1:  # X first - no flip
                            break
                        elif cell == 2:  # O first - flip
                            need_flip = True
                            break
                    break
        
        # Build sub_data with optional flip
        sub_data = []
        for sub_idx in range(9):
            sub_r, sub_c = sub_idx // 3, sub_idx % 3
            state = board.completed_boards[sub_r][sub_c]
            
            if state != 0:
                # Completed - flip state if needed (1<->2, 3 stays)
                if need_flip and state in (1, 2):
                    state = 3 - state
                sub_data.append((state, 0, 0))
            else:
                # OPEN - count X and O
                x_count = sum(1 for dr in range(3) for dc in range(3) 
                             if board.boards[sub_r*3+dr][sub_c*3+dc] == 1)
                o_count = sum(1 for dr in range(3) for dc in range(3) 
                             if board.boards[sub_r*3+dr][sub_c*3+dc] == 2)
                # Flip counts if needed
                if need_flip:
                    x_count, o_count = o_count, x_count
                sub_data.append((0, x_count, o_count))
        
        return (tuple(sub_data), getattr(board, 'constraint', -1))
    
    def _enumerate_meta_boards(self) -> Generator[Tuple[Tuple[int, ...], int, int], None, None]:
        """
        Enumerate valid meta-board configurations with valid diff range.
        
        Yields: (meta, min_valid_diff, max_valid_diff)
        The diff range tells how many more X than O can be in OPEN boards.
        """
        for meta in product([0, 1, 2, 3], repeat=9):
            if self._check_meta_winner(meta) != 0:
                continue
            
            num_open = meta.count(0)
            if num_open == 0:
                continue
            
            if self.empty_cells > num_open * 8 or self.empty_cells < num_open:
                continue
            
            x_wins = meta.count(1)
            o_wins = meta.count(2)
            draws = meta.count(3)
            
            total_filled = 9 * num_open - self.empty_cells
            
            # Completed boards diff ranges
            min_completed = x_wins * (-3) + o_wins * (-7) + draws * (-1)
            max_completed = x_wins * 7 + o_wins * 3 + draws * 1
            
            # Valid open_diff: 0 <= open_diff + completed_diff <= 1
            # open_diff >= -max_completed (to get total >= 0)
            # open_diff <= 1 - min_completed (to get total <= 1)
            # Also bounded by: -total_filled <= open_diff <= total_filled
            min_valid_diff = max(-max_completed, -total_filled)
            max_valid_diff = min(1 - min_completed, total_filled)
            
            if min_valid_diff > max_valid_diff:
                continue
            
            yield (meta, min_valid_diff, max_valid_diff)
    
    def _check_meta_winner(self, meta: Tuple[int, ...]) -> int:
        """Check if meta-board has a winner."""
        for a, b, c in WIN_PATTERNS:
            if meta[a] == meta[b] == meta[c] and meta[a] in (1, 2):
                return meta[a]
        return 0
    
    def _fill_subboards(self, meta: Tuple[int, ...], min_diff: int, max_diff: int) -> Generator[Board, None, None]:
        """
        Fill sub-boards using pre-calculated valid diff range.
        
        Args:
            meta: Meta-board configuration
            min_diff, max_diff: Valid range for (X - O) in OPEN boards
        """
        open_indices = [i for i, s in enumerate(meta) if s == 0]
        num_open = len(open_indices)
        
        if num_open == 0:
            return
        
        total_filled = 9 * num_open - self.empty_cells
        
        # For each valid diff in range, calculate X and O counts
        for diff in range(min_diff, max_diff + 1):
            # X - O = diff, X + O = total_filled
            # X = (total_filled + diff) / 2
            if (total_filled + diff) % 2 != 0:
                continue  # Must be integer
            
            x_total = (total_filled + diff) // 2
            o_total = (total_filled - diff) // 2
            
            if x_total < 0 or o_total < 0:
                continue
            
            # Distribute empty cells and pieces
            for empty_dist in self._distribute_empty(num_open, self.empty_cells):
                # Enumerate ALL valid X/O distributions
                for base_board in self._create_boards_with_counts(meta, open_indices, empty_dist, x_total, o_total):
                    # Enumerate only specific constraints (OPEN boards)
                    # "any" (-1) is handled at lookup time by taking best of all OPEN constraints
                    for constraint in open_indices:
                        board = base_board.clone()
                        board.constraint = constraint
                        # Set last_move so get_legal_moves() returns moves for this constraint
                        # last_move = (r, c) where (r % 3, c % 3) = (constraint // 3, constraint % 3)
                        sub_r, sub_c = constraint // 3, constraint % 3
                        board.last_move = (sub_r, sub_c)  # Any move that sends to this sub-board
                        yield board
    
    def _distribute_empty(self, num_open: int, total_empty: int) -> Generator[Tuple[int, ...], None, None]:
        """
        Distribute total_empty cells across num_open sub-boards.
        Each sub-board can have 1-8 empty cells.
        """
        if num_open == 0:
            if total_empty == 0:
                yield ()
            return
        
        if num_open == 1:
            if 1 <= total_empty <= 8:
                yield (total_empty,)
            return
        
        # Recursive distribution
        min_per_board = max(1, total_empty - 8 * (num_open - 1))
        max_per_board = min(8, total_empty - (num_open - 1))
        
        for first in range(min_per_board, max_per_board + 1):
            for rest in self._distribute_empty(num_open - 1, total_empty - first):
                yield (first,) + rest
    
    def _create_boards_with_counts(self, meta: Tuple[int, ...], open_indices: List[int], 
                                    empty_dist: Tuple[int, ...], x_total: int, o_total: int) -> Generator[Board, None, None]:
        """Create boards with all valid X/O distributions across OPEN sub-boards."""
        num_open = len(open_indices)
        filled_per_sub = [9 - empty_dist[i] for i in range(num_open)]
        
        # Generate all valid (x_count, o_count) per sub-board
        # Constraint: sum(x_counts) = x_total, sum(o_counts) = o_total
        # For each sub: 0 <= x <= filled, 0 <= o <= filled, x + o = filled
        
        def distribute_x(idx: int, x_remaining: int) -> Generator[Tuple[int, ...], None, None]:
            """Recursively distribute x_remaining across sub-boards starting at idx."""
            if idx == num_open:
                if x_remaining == 0:
                    yield ()
                return
            
            filled = filled_per_sub[idx]
            o_remaining = o_total - (x_total - x_remaining)  # O's already placed
            
            # x_here can range from max(0, filled - o_remaining_for_rest) to min(filled, x_remaining)
            o_available_for_rest = sum(filled_per_sub[j] for j in range(idx + 1, num_open))
            min_x = max(0, filled - (o_total - (x_total - x_remaining - 0)))  # ensure o_here >= 0
            
            for x_here in range(max(0, x_remaining - sum(filled_per_sub[j] for j in range(idx + 1, num_open))),
                                min(filled, x_remaining) + 1):
                o_here = filled - x_here
                # Check o_here is valid
                if o_here < 0:
                    continue
                o_used_so_far = x_total - x_remaining  # This is wrong, let me fix
                
                for rest in distribute_x(idx + 1, x_remaining - x_here):
                    yield (x_here,) + rest
        
        # Simpler approach: enumerate x_counts for each sub-board
        def gen_x_distributions(remaining_x: int, remaining_subs: List[int]) -> Generator[Tuple[int, ...], None, None]:
            if not remaining_subs:
                if remaining_x == 0:
                    yield ()
                return
            
            filled = filled_per_sub[remaining_subs[0]]
            # x_here: min is 0 or (remaining_x - sum of max possible for rest)
            # max is min(filled, remaining_x)
            rest_max_x = sum(filled_per_sub[i] for i in remaining_subs[1:])
            min_x_here = max(0, remaining_x - rest_max_x)
            max_x_here = min(filled, remaining_x)
            
            for x_here in range(min_x_here, max_x_here + 1):
                for rest in gen_x_distributions(remaining_x - x_here, remaining_subs[1:]):
                    yield (x_here,) + rest
        
        for x_dist in gen_x_distributions(x_total, list(range(num_open))):
            # Compute o_dist from x_dist (o_here = filled - x_here)
            o_dist = tuple(filled_per_sub[i] - x_dist[i] for i in range(num_open))
            
            # Verify o_total matches
            if sum(o_dist) != o_total:
                continue
            
            # Verify no negative
            if any(o < 0 for o in o_dist):
                continue
            
            # Create board with this distribution
            board = Board()
            valid = True
            
            for i, sub_idx in enumerate(open_indices):
                sub_r, sub_c = sub_idx // 3, sub_idx % 3
                x_here, o_here = x_dist[i], o_dist[i]
                empty_here = empty_dist[i]
                
                cells = self._find_valid_cell_config(x_here, o_here, empty_here)
                if cells is None:
                    valid = False
                    break
                
                for j, val in enumerate(cells):
                    r = sub_r * 3 + j // 3
                    c = sub_c * 3 + j % 3
                    board.boards[r][c] = val
            
            if not valid:
                continue
            
            # Set completed boards state
            for sub_idx in range(9):
                sub_r, sub_c = sub_idx // 3, sub_idx % 3
                board.completed_boards[sub_r][sub_c] = meta[sub_idx]
            
            board.current_player = 1 if x_total == o_total else 2
            board.winner = None
            
            yield board
    
    def _check_cells_winner(self, cells: List[int]) -> int:
        """Check winner of 9-cell sub-board."""
        for a, b, c in WIN_PATTERNS:
            if cells[a] == cells[b] == cells[c] != 0:
                return cells[a]
        return 0
    
    # Cache for valid cell configurations: (x, o, empty) -> first valid config
    _cell_config_cache = {}
    
    def _find_valid_cell_config(self, x_count: int, o_count: int, empty_count: int) -> List[int]:
        """Find first valid 9-cell configuration with no winner (cached)."""
        key = (x_count, o_count, empty_count)
        if key in self._cell_config_cache:
            return self._cell_config_cache[key]
        
        from itertools import combinations
        
        # Use combinations instead of permutations
        # Choose positions for X, then O, rest are empty
        cells = [0] * 9
        
        for x_positions in combinations(range(9), x_count):
            remaining = [i for i in range(9) if i not in x_positions]
            for o_positions in combinations(remaining, o_count):
                test = [0] * 9
                for i in x_positions:
                    test[i] = 1
                for i in o_positions:
                    test[i] = 2
                
                if self._check_cells_winner(test) == 0:
                    self._cell_config_cache[key] = test
                    return test
        
        self._cell_config_cache[key] = None
        return None
    
    def _check_subboard_winner(self, board: Board, sub_r: int, sub_c: int) -> int:
        """Check winner of a sub-board. Returns 0/1/2/3."""
        WIN_PATTERNS = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        
        cells = []
        for dr in range(3):
            for dc in range(3):
                r, c = sub_r * 3 + dr, sub_c * 3 + dc
                cells.append(board.boards[r][c])
        
        for a, b, c in WIN_PATTERNS:
            if cells[a] == cells[b] == cells[c] != 0:
                return cells[a]
        
        if 0 not in cells:
            return 3  # Draw
        
        return 0  # Open
    
    def _check_big_board_winner(self, board: Board) -> int:
        """Check winner of big board. Returns 0/1/2/3."""
        WIN_PATTERNS = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        
        cells = []
        for r in range(3):
            for c in range(3):
                cells.append(board.completed_boards[r][c])
        
        for a, b, c in WIN_PATTERNS:
            if cells[a] == cells[b] == cells[c] != 0 and cells[a] in (1, 2):
                return cells[a]
        
        # Check if all sub-boards complete
        if all(c != 0 for c in cells):
            return 3  # Draw
        
        return 0  # Game ongoing
    
def main():
    """Test position enumeration."""
    enumerator = PositionEnumerator(empty_cells=15)
    
    count = 0
    for board in enumerator.enumerate(max_positions=100):
        count += 1
        if count <= 3:
            print(f"\nPosition {count}:")
            print(f"  Empty cells: {sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 0)}")
            print(f"  Current player: {board.current_player}")
    
    print(f"\nStats: {enumerator.stats}")


if __name__ == '__main__':
    main()
