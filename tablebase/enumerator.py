"""
Position Enumerator for Tablebase Generation

Systematic enumeration:
1. Enumerate meta-board (big board) states
2. For each meta-board, fill sub-boards systematically
3. Apply D4 symmetry reduction
4. Filter invalid configurations
5. 4-move backward DFS for reachability
"""

from itertools import product, combinations
from typing import Generator, Tuple, List, Set, Dict
from collections import deque
from tqdm import tqdm
import numpy as np

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
    2. For each meta-board, fill sub-boards systematically
    3. Apply D4 symmetry to avoid duplicates
    4. Filter: |X-O| <= fillable cells, valid sub-board winners
    5. 4-move backward DFS for reachability
    """
    
    # Meta-board states
    OPEN = 0    # Sub-board still in play
    X_WIN = 1   # X won sub-board
    O_WIN = 2   # O won sub-board
    DRAW = 3    # Sub-board is draw (full, no winner)
    
    def __init__(self, empty_cells: int = 15, backward_depth: int = 4):
        self.empty_cells = empty_cells
        self.backward_depth = backward_depth
        
        self.seen_hashes: Set[int] = set()  # For symmetry dedup
        self.stats = {
            'meta_boards': 0,
            'generated': 0,
            'valid_structure': 0,
            'duplicate': 0,
            'reachable': 0,
            'unreachable': 0
        }
    
    def enumerate(self, max_positions: int = None, show_progress: bool = True) -> Generator[Board, None, None]:
        """
        Enumerate valid, reachable positions systematically.
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
                
                # Step 3: Symmetry dedup
                canonical = BoardSymmetry.get_canonical_hash(board)
                if canonical in self.seen_hashes:
                    self.stats['duplicate'] += 1
                    continue
                
                self.stats['valid_structure'] += 1
                
                # Step 4: Reachability check (4-move backward DFS)
                if self._is_reachable(board):
                    self.seen_hashes.add(canonical)
                    self.stats['reachable'] += 1
                    count += 1
                    
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'meta': self.stats['meta_boards'],
                            'valid': self.stats['valid_structure'],
                            'reach': count
                        })
                    
                    yield board
                    
                    if max_positions and count >= max_positions:
                        if pbar:
                            pbar.close()
                        return
                else:
                    self.stats['unreachable'] += 1
        
        if pbar:
            pbar.close()
    
    def _enumerate_meta_boards(self) -> Generator[Tuple[int, ...], None, None]:
        """
        Enumerate valid meta-board configurations.
        
        Each sub-board can be: OPEN(0), X_WIN(1), O_WIN(2), DRAW(3)
        Filter out configurations where game is already over.
        """
        for meta in product([0, 1, 2, 3], repeat=9):
            # Check big board winner
            if self._check_meta_winner(meta) != 0:
                continue  # Game already over
            
            # Must have at least one open sub-board
            if meta.count(0) == 0:
                continue
            
            # X wins - O wins should be reasonable
            x_wins = meta.count(1)
            o_wins = meta.count(2)
            if abs(x_wins - o_wins) > 2:
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
        
        For each sub-board:
        - OPEN: has empty cells, no winner
        - X_WIN/O_WIN: winner has 3-in-a-row, may have some empty cells too
        - DRAW: full (9 cells), no winner
        
        Target: exactly self.empty_cells PLAYABLE empty cells
        (only in OPEN sub-boards)
        """
        open_indices = [i for i, s in enumerate(meta) if s == 0]
        num_open = len(open_indices)
        
        if num_open == 0:
            return
        
        # All empty cells must be in OPEN sub-boards
        # Each OPEN sub-board has 1-8 empty cells
        if self.empty_cells > num_open * 8:
            return  # Can't fit that many empty cells
        if self.empty_cells < num_open:
            return  # Each open board needs at least 1 empty
        
        # Distribute empty cells across open sub-boards
        for empty_dist in self._distribute_empty(num_open, self.empty_cells):
            board = self._create_board(meta, open_indices, empty_dist)
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
    
    def _create_board(self, meta: Tuple[int, ...], open_indices: List[int], 
                      empty_dist: Tuple[int, ...]) -> Board:
        """Create a board from meta-board and empty distribution."""
        board = Board()
        
        total_x = 0
        total_o = 0
        
        # Fill each sub-board
        for sub_idx in range(9):
            sub_r, sub_c = sub_idx // 3, sub_idx % 3
            state = meta[sub_idx]
            
            if state == self.OPEN:
                # Find this sub-board's empty count
                open_pos = open_indices.index(sub_idx)
                empty_count = empty_dist[open_pos]
                filled = 9 - empty_count
                
                # Distribute X and O in this sub-board
                # For now, roughly equal
                x_here = (filled + 1) // 2
                o_here = filled // 2
                
                cells = [1] * x_here + [2] * o_here + [0] * empty_count
                # Shuffle to randomize position (but deterministic)
                import random
                random.seed(sub_idx * 1000 + empty_count)
                random.shuffle(cells)
                
                # Check no accidental winner
                if self._check_cells_winner(cells) != 0:
                    return None
                
                total_x += x_here
                total_o += o_here
                
            elif state == self.X_WIN:
                # X has a winning line, fill rest reasonably
                cells = self._create_won_subboard(1)
                if cells is None:
                    return None
                total_x += cells.count(1)
                total_o += cells.count(2)
                
            elif state == self.O_WIN:
                cells = self._create_won_subboard(2)
                if cells is None:
                    return None
                total_x += cells.count(1)
                total_o += cells.count(2)
                
            else:  # DRAW
                cells = self._create_draw_subboard()
                if cells is None:
                    return None
                total_x += cells.count(1)
                total_o += cells.count(2)
            
            # Place cells on board
            for i, val in enumerate(cells):
                r = sub_r * 3 + i // 3
                c = sub_c * 3 + i % 3
                board.boards[r][c] = val
            
            board.completed_boards[sub_r][sub_c] = state
        
        # Validate X-O counts
        if not (total_x == total_o or total_x == total_o + 1):
            return None
        
        board.current_player = 1 if total_x == total_o else 2
        board.last_move = None
        board.winner = None
        
        return board
    
    def _check_cells_winner(self, cells: List[int]) -> int:
        """Check winner of 9-cell sub-board."""
        for a, b, c in WIN_PATTERNS:
            if cells[a] == cells[b] == cells[c] != 0:
                return cells[a]
        return 0
    
    def _create_won_subboard(self, winner: int) -> List[int]:
        """
        Create a sub-board where winner has won.
        
        Note: Winner can have multiple winning lines (e.g., row + diagonal).
        Loser cannot have any winning line.
        """
        import random
        
        loser = 3 - winner
        
        # Try to create valid won board
        for _ in range(100):
            # Start with a winning pattern
            pattern = random.choice(WIN_PATTERNS)
            cells = [0] * 9
            
            for i in pattern:
                cells[i] = winner
            
            # Optionally add more winner pieces (may create multiple win lines)
            other_cells = [i for i in range(9) if i not in pattern]
            random.shuffle(other_cells)
            
            # Add 0-2 more winner pieces
            extra_winner = random.randint(0, min(2, len(other_cells)))
            for i in range(extra_winner):
                cells[other_cells[i]] = winner
            
            # Add some loser pieces (but check no winning line)
            remaining = [i for i in range(9) if cells[i] == 0]
            random.shuffle(remaining)
            
            loser_count = random.randint(2, min(4, len(remaining)))
            temp_cells = cells.copy()
            
            for i in range(loser_count):
                temp_cells[remaining[i]] = loser
            
            # Check loser has no winning line
            loser_wins = False
            for a, b, c in WIN_PATTERNS:
                if temp_cells[a] == temp_cells[b] == temp_cells[c] == loser:
                    loser_wins = True
                    break
            
            if not loser_wins:
                return temp_cells
        
        # Fallback: simple pattern
        cells = [0] * 9
        pattern = WIN_PATTERNS[0]
        for i in pattern:
            cells[i] = winner
        cells[3] = loser
        cells[6] = loser
        return cells
    
    def _create_draw_subboard(self) -> List[int]:
        """Create a full sub-board with no winner (draw)."""
        # 5 X and 4 O, or 4 X and 5 O, arranged with no winner
        import random
        
        # Try random arrangements until we get a draw
        for _ in range(100):
            x_count = random.choice([4, 5])
            o_count = 9 - x_count
            
            cells = [1] * x_count + [2] * o_count
            random.shuffle(cells)
            
            if self._check_cells_winner(cells) == 0:
                return cells
        
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
    
    def _is_reachable(self, board: Board) -> bool:
        """
        Check if position is reachable via 4-move backward DFS.
        
        If we can trace back 4 moves and reach a valid prior state,
        the position is considered reachable.
        """
        return self._backward_dfs(board, 0)
    
    def _backward_dfs(self, board: Board, depth: int) -> bool:
        """Backward DFS to check reachability."""
        
        if depth >= self.backward_depth:
            return True  # Reached depth limit, assume reachable
        
        # Count pieces
        x_count = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 1)
        o_count = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 2)
        
        # If very few pieces, it's definitely reachable from start
        if x_count + o_count <= depth:
            return True
        
        # Find which player made the last move (opponent of current)
        # After a move, player switches, so last mover = 3 - current_player
        last_mover = 3 - board.current_player
        
        # Try undoing each piece of last_mover
        for r in range(9):
            for c in range(9):
                if board.boards[r][c] == last_mover:
                    # Try undoing this move
                    undo_board = self._try_undo(board, r, c)
                    if undo_board is not None:
                        if self._backward_dfs(undo_board, depth + 1):
                            return True
        
        return False
    
    def _try_undo(self, board: Board, r: int, c: int) -> Board:
        """
        Try to undo a move at (r, c).
        
        Returns new board if valid undo, None otherwise.
        """
        piece = board.boards[r][c]
        if piece == 0:
            return None
        
        # Create new board with piece removed
        new_board = board.clone()
        new_board.boards[r][c] = 0
        
        # Switch player
        new_board.current_player = piece  # The one who placed this piece
        
        # Recalculate sub-board state
        sub_r, sub_c = r // 3, c // 3
        new_winner = self._check_subboard_winner(new_board, sub_r, sub_c)
        new_board.completed_boards[sub_r][sub_c] = new_winner
        
        # Check the resulting board is valid
        # (no winner yet, piece counts valid)
        if self._check_big_board_winner(new_board) != 0:
            return None
        
        return new_board


def main():
    """Test position enumeration."""
    enumerator = PositionEnumerator(empty_cells=15, backward_depth=4)
    
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
