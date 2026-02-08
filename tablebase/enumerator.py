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
        
        # Step 1: Enumerate all meta-board configurations
        for meta in self._enumerate_meta_boards():
            self.stats['meta_boards'] += 1
            
            # Step 2: For each meta-board, fill sub-boards
            for board in self._fill_subboards(meta):
                self.stats['generated'] += 1
                
                if board is None:
                    continue
                
                # Step 3: Symmetry dedup (D4 only, no X-O swap)
                canonical = BoardSymmetry.get_canonical_hash(board)
                if canonical in self.seen_hashes:
                    self.stats['duplicate'] += 1
                    continue
                
                self.seen_hashes.add(canonical)
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
    
    def _enumerate_meta_boards(self) -> Generator[Tuple[int, ...], None, None]:
        """
        Enumerate valid meta-board configurations with O(1) pre-filtering.
        
        Each sub-board can be: OPEN(0), X_WIN(1), O_WIN(2), DRAW(3)
        Filter out configurations where game is already over or impossible.
        """
        for meta in product([0, 1, 2, 3], repeat=9):
            # Check big board winner
            if self._check_meta_winner(meta) != 0:
                continue  # Game already over
            
            num_open = meta.count(0)
            if num_open == 0:
                continue
            
            # Check if empty_cells can fit in OPEN boards
            if self.empty_cells > num_open * 8 or self.empty_cells < num_open:
                continue
            
            x_wins = meta.count(1)
            o_wins = meta.count(2)
            draws = meta.count(3)
            
            # X wins - O wins should be reasonable
            if abs(x_wins - o_wins) > 2:
                continue
            
            # O(1) pre-filter: check if ANY valid piece distribution exists
            # OPEN boards: total filled = 9 * num_open - empty_cells
            total_filled = 9 * num_open - self.empty_cells
            max_open_diff = total_filled   # All X
            min_open_diff = -total_filled  # All O
            
            # Completed boards diff ranges
            min_completed = x_wins * (-3) + o_wins * (-7) + draws * (-1)
            max_completed = x_wins * 7 + o_wins * 3 + draws * 1
            
            # Check overlap with [0, 1]
            total_max = max_open_diff + max_completed
            total_min = min_open_diff + min_completed
            if total_max < 0 or total_min > 1:
                continue
            
            yield meta
    
    def _check_meta_winner(self, meta: Tuple[int, ...]) -> int:
        """Check if meta-board has a winner."""
        for a, b, c in WIN_PATTERNS:
            if meta[a] == meta[b] == meta[c] and meta[a] in (1, 2):
                return meta[a]
        return 0
    
    def _fill_subboards(self, meta: Tuple[int, ...]) -> Generator[Board, None, None]:
        """
        Fill sub-boards for a given meta-board configuration.
        
        For OPEN boards: place actual pieces
        For completed boards: just set state, no pieces (hash ignores them)
        
        Target: exactly self.empty_cells PLAYABLE empty cells
        """
        open_indices = [i for i, s in enumerate(meta) if s == 0]
        num_open = len(open_indices)
        
        if num_open == 0:
            return
        
        if self.empty_cells > num_open * 8:
            return
        if self.empty_cells < num_open:
            return
        
        # Just distribute empty cells - completed boards are ignored
        for empty_dist in self._distribute_empty(num_open, self.empty_cells):
            board = self._create_board_simple(meta, open_indices, empty_dist)
            if board is not None:
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
    
    def _create_board_simple(self, meta: Tuple[int, ...], open_indices: List[int], 
                              empty_dist: Tuple[int, ...]) -> Board:
        """Create a board - only OPEN boards have pieces, O(1) validation for completed boards."""
        board = Board()
        
        open_diff = 0  # X - O for OPEN boards
        min_completed = 0
        max_completed = 0
        
        for sub_idx in range(9):
            sub_r, sub_c = sub_idx // 3, sub_idx % 3
            state = meta[sub_idx]
            board.completed_boards[sub_r][sub_c] = state
            
            if state == self.OPEN:
                open_pos = open_indices.index(sub_idx)
                empty_count = empty_dist[open_pos]
                filled = 9 - empty_count
                
                x_here = (filled + 1) // 2
                o_here = filled // 2
                open_diff += x_here - o_here
                
                cells = [1] * x_here + [2] * o_here + [0] * empty_count
                import random
                random.seed(sub_idx * 1000 + empty_count)
                random.shuffle(cells)
                
                if self._check_cells_winner(cells) != 0:
                    return None
                
                for i, val in enumerate(cells):
                    r = sub_r * 3 + i // 3
                    c = sub_c * 3 + i % 3
                    board.boards[r][c] = val
            else:
                # Completed board: accumulate diff range (hardcoded)
                # X_WIN: [-3, 7], O_WIN: [-7, 3], DRAW: [-1, 1]
                if state == self.X_WIN:
                    min_completed += -3
                    max_completed += 7
                elif state == self.O_WIN:
                    min_completed += -7
                    max_completed += 3
                else:  # DRAW
                    min_completed += -1
                    max_completed += 1
        
        # O(1) validation: check if [open_diff + min, open_diff + max] overlaps [0, 1]
        total_min = open_diff + min_completed
        total_max = open_diff + max_completed
        if total_max < 0 or total_min > 1:
            return None
        
        # Set current player (use any valid diff, prefer 0 or 1)
        if total_min <= 0 <= total_max:
            board.current_player = 1  # X's turn (diff = 0)
        else:
            board.current_player = 2  # O's turn (diff = 1)
        
        board.last_move = None
        board.winner = None
        
        return board
    
    def _check_cells_winner(self, cells: List[int]) -> int:
        """Check winner of 9-cell sub-board."""
        for a, b, c in WIN_PATTERNS:
            if cells[a] == cells[b] == cells[c] != 0:
                return cells[a]
        return 0
    
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
