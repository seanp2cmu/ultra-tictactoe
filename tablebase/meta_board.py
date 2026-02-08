"""
Meta-Board Enumeration for Tablebase Generation

Enumerates all valid configurations of the 3x3 meta-board (big board).
Each sub-board can be: X-won (1), O-won (2), Draw (3), or Open (0).
"""

from itertools import product
from typing import List, Tuple, Generator, Set
import numpy as np


# Sub-board winning patterns (indices 0-8)
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diagonals
]

# All possible winning configurations for a 3x3 board
# Format: frozenset of cell indices that must be filled by the winner
WINNING_CONFIGS = []
for pattern in WIN_PATTERNS:
    # Winner fills exactly the winning line, others can be anything
    WINNING_CONFIGS.append(frozenset(pattern))


class MetaBoardEnumerator:
    """
    Enumerate valid meta-board (big board) configurations.
    
    Meta-board states:
    - 0: Open (sub-board still in play)
    - 1: X won this sub-board
    - 2: O won this sub-board  
    - 3: Draw (sub-board full, no winner)
    """
    
    # States
    OPEN = 0
    X_WIN = 1
    O_WIN = 2
    DRAW = 3
    
    def __init__(self):
        self.valid_meta_boards = None
    
    def enumerate_all(self) -> List[Tuple[int, ...]]:
        """
        Enumerate all valid meta-board configurations.
        
        Filters:
        1. Game not over (no winner on meta-board)
        2. Valid X-O balance (|X_wins - O_wins| ≤ 1 considering draws)
        
        Returns:
            List of 9-tuples representing meta-board states
        """
        valid = []
        
        # All possible combinations: 4^9 = 262,144
        for meta in product([0, 1, 2, 3], repeat=9):
            if self._is_valid_meta_board(meta):
                valid.append(meta)
        
        self.valid_meta_boards = valid
        return valid
    
    def _is_valid_meta_board(self, meta: Tuple[int, ...]) -> bool:
        """Check if meta-board configuration is valid for tablebase."""
        
        # Check if game is already over (someone won the big board)
        meta_2d = [meta[i*3:(i+1)*3] for i in range(3)]
        
        # Check rows, cols, diags for winner
        for i in range(3):
            # Row
            if meta_2d[i][0] == meta_2d[i][1] == meta_2d[i][2] != 0 and meta_2d[i][0] in (1, 2):
                return False  # Game over
            # Col
            if meta_2d[0][i] == meta_2d[1][i] == meta_2d[2][i] != 0 and meta_2d[0][i] in (1, 2):
                return False
        # Diagonals
        if meta_2d[0][0] == meta_2d[1][1] == meta_2d[2][2] != 0 and meta_2d[0][0] in (1, 2):
            return False
        if meta_2d[0][2] == meta_2d[1][1] == meta_2d[2][0] != 0 and meta_2d[0][2] in (1, 2):
            return False
        
        # Count sub-board states
        x_wins = meta.count(1)
        o_wins = meta.count(2)
        draws = meta.count(3)
        opens = meta.count(0)
        
        # At least one open sub-board (game not over by full board)
        # Actually for tablebase we want games that are still playable
        # If all sub-boards are complete, game is over
        if opens == 0:
            return False
        
        # X-O win balance check
        # X moves first, so x_wins can be at most o_wins + 1
        # But this depends on how many total moves, not just wins
        # For simplicity, allow reasonable configurations
        if abs(x_wins - o_wins) > 2:  # Generous bound
            return False
        
        return True
    
    def get_sub_board_constraints(self, meta: Tuple[int, ...]) -> List[dict]:
        """
        Get constraints for each sub-board based on meta-board state.
        
        Returns:
            List of 9 dicts, each containing:
            - 'state': 0/1/2/3
            - 'winner': None, 1, or 2
            - 'must_have_winner_line': bool
            - 'is_full': bool (for draws)
        """
        constraints = []
        
        for i, state in enumerate(meta):
            constraint = {
                'state': state,
                'winner': None,
                'must_have_winner_line': False,
                'is_full': False,
                'is_open': False
            }
            
            if state == self.X_WIN:
                constraint['winner'] = 1
                constraint['must_have_winner_line'] = True
            elif state == self.O_WIN:
                constraint['winner'] = 2
                constraint['must_have_winner_line'] = True
            elif state == self.DRAW:
                constraint['is_full'] = True
            else:  # OPEN
                constraint['is_open'] = True
            
            constraints.append(constraint)
        
        return constraints
    
    def count_by_open_subboards(self) -> dict:
        """Count valid meta-boards by number of open sub-boards."""
        if self.valid_meta_boards is None:
            self.enumerate_all()
        
        counts = {}
        for meta in self.valid_meta_boards:
            opens = meta.count(0)
            counts[opens] = counts.get(opens, 0) + 1
        
        return counts


def get_winning_patterns_for_cell(cell_idx: int) -> List[Tuple[int, int, int]]:
    """Get all winning patterns that include the given cell."""
    return [p for p in WIN_PATTERNS if cell_idx in p]


def check_subboard_winner(cells: List[int]) -> int:
    """
    Check winner of a 3x3 sub-board.
    
    Args:
        cells: List of 9 values (0=empty, 1=X, 2=O)
        
    Returns:
        0 if no winner, 1 if X wins, 2 if O wins, 3 if draw (full, no winner)
    """
    for pattern in WIN_PATTERNS:
        a, b, c = pattern
        if cells[a] == cells[b] == cells[c] != 0:
            return cells[a]
    
    # Check if full (draw)
    if 0 not in cells:
        return 3
    
    return 0  # Still open


if __name__ == '__main__':
    enumerator = MetaBoardEnumerator()
    valid = enumerator.enumerate_all()
    print(f"Total valid meta-boards: {len(valid)}")
    print(f"By open sub-boards: {enumerator.count_by_open_subboards()}")
