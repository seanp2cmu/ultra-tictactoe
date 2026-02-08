"""
BFS + Memoization Tablebase Solver

Uses BFS from terminal positions with memoization.
Includes 4-move DFS reachability check to filter illegal positions.
"""

from typing import Dict, Tuple, Optional, Set, List
from collections import deque
from functools import lru_cache
import numpy as np
from game import Board


class TablebaseSolver:
    """
    Solve endgame positions using BFS + memoization.
    
    Strategy:
    1. Enumerate positions with ≤N empty cells
    2. Check reachability with 4-move backward DFS
    3. Solve via BFS from terminals + memoization
    
    Stores: (result, dtw, best_move) for each position
    
    Supports incremental building: load smaller tablebase to speed up larger ones.
    """
    
    def __init__(self, max_depth: int = 20, base_tablebase: Optional[Dict] = None):
        """
        Args:
            max_depth: Maximum search depth
            base_tablebase: Pre-computed positions dict to use as lookup (e.g., from smaller tablebase)
        """
        # Memoization cache: board_hash -> (result, dtw, best_move)
        # result: 1 = current player wins, -1 = loses, 0 = draw
        # best_move: (row, col) tuple or None
        self.cache: Dict[int, Tuple[int, int, Optional[Tuple[int, int]]]] = {}
        self.max_depth = max_depth
        self.stats = {'cache_hits': 0, 'solves': 0, 'unreachable': 0, 'depth_limit': 0, 'base_hits': 0}
        
        # Load base tablebase for faster lookups
        self.base_tablebase = base_tablebase or {}
    
    def solve(self, board: Board, max_depth: Optional[int] = None) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """
        Solve position with memoization.
        
        Returns:
            (result, dtw, best_move) from current player's perspective
        """
        board_hash = self._hash_board(board)
        
        if board_hash in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[board_hash]
        
        # Check base tablebase (from smaller threshold)
        if board_hash in self.base_tablebase:
            self.stats['base_hits'] += 1
            result = self.base_tablebase[board_hash]
            self.cache[board_hash] = result
            return result
        
        result, dtw, best_move = self._solve_recursive(board)
        self.cache[board_hash] = (result, dtw, best_move)
        self.stats['solves'] += 1
        
        return (result, dtw, best_move)
    
    def _solve_recursive(self, board: Board, depth: int = 0) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """Recursive minimax with memoization (complete search, no pruning)."""
        
        # Terminal check
        if board.winner is not None:
            if board.winner == 3:  # Draw
                return (0, depth, None)
            # Winner is set - current player lost (opponent won last turn)
            return (-1, depth, None)
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return (0, depth, None)  # No moves = draw
        
        # Depth limit for safety (shouldn't hit with ≤15 empty)
        if depth >= self.max_depth:
            self.stats['depth_limit'] += 1
            return (0, depth, None)
        
        best_value = -2
        best_dtw = 999
        best_move = None
        
        for move in legal_moves:
            child = board.clone()
            child.make_move(move[0], move[1])
            
            child_hash = self._hash_board(child)
            
            if child_hash in self.cache:
                child_result, child_dtw, _ = self.cache[child_hash]
            elif child_hash in self.base_tablebase:
                # Use base tablebase result
                child_result, child_dtw, _ = self.base_tablebase[child_hash]
                self.cache[child_hash] = self.base_tablebase[child_hash]
                self.stats['base_hits'] += 1
            else:
                child_result, child_dtw, _ = self._solve_recursive(child, depth + 1)
                self.cache[child_hash] = (child_result, child_dtw, None)
            
            # Negate for perspective switch
            value = -child_result
            
            if value > best_value or (value == best_value and child_dtw < best_dtw):
                best_value = value
                best_dtw = child_dtw
                best_move = move
        
        return (best_value, best_dtw + 1, best_move)
    
    def _hash_board(self, board: Board) -> int:
        """Create hash for board position."""
        cells = tuple(board.boards[r][c] for r in range(9) for c in range(9))
        
        # Include constraint (which sub-board must play in)
        constraint = -1
        if board.last_move:
            sub_r, sub_c = board.last_move[0] % 3, board.last_move[1] % 3
            # Check if that sub-board is still open
            if board.completed_boards[sub_r][sub_c] == 0:
                constraint = sub_r * 3 + sub_c
        
        return hash((cells, board.current_player, constraint))


class ReachabilityChecker:
    """
    Check if a board position is reachable from the start.
    
    Uses 4-move backward DFS: if we can find a valid path 4 moves back,
    the position is likely reachable.
    """
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
    
    def is_reachable(self, board: Board) -> bool:
        """
        Check if board is reachable via backward DFS.
        
        Returns True if we can trace back max_depth moves to a simpler position.
        """
        # Basic validity checks
        if not self._is_valid_board(board):
            return False
        
        # Count pieces
        x_count = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 1)
        o_count = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 2)
        
        # X moves first, so x_count == o_count or x_count == o_count + 1
        if not (x_count == o_count or x_count == o_count + 1):
            return False
        
        # Current player check
        expected_player = 1 if x_count == o_count else 2
        if board.current_player != expected_player:
            return False
        
        # Try backward DFS
        return self._backward_dfs(board, 0)
    
    def _backward_dfs(self, board: Board, depth: int) -> bool:
        """
        DFS backward to check reachability.
        
        Try to find a valid predecessor state.
        """
        if depth >= self.max_depth:
            return True  # Reached depth limit, assume reachable
        
        # Find the last move (opponent's move)
        opponent = 3 - board.current_player
        
        # Find all cells that could have been the last move
        candidates = self._find_undo_candidates(board, opponent)
        
        if not candidates:
            # No valid undo candidates
            # If board is near-empty, it's valid start
            total_pieces = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] != 0)
            return total_pieces <= depth
        
        # Try undoing each candidate
        for r, c in candidates:
            undo_board = self._try_undo_move(board, r, c)
            if undo_board is not None:
                if self._backward_dfs(undo_board, depth + 1):
                    return True
        
        return False
    
    def _is_valid_board(self, board: Board) -> bool:
        """Basic board validity check."""
        # Check each sub-board's completion status matches its contents
        for sub_r in range(3):
            for sub_c in range(3):
                sub_state = board.completed_boards[sub_r][sub_c]
                
                # Get sub-board cells
                cells = []
                for dr in range(3):
                    for dc in range(3):
                        r, c = sub_r * 3 + dr, sub_c * 3 + dc
                        cells.append(board.boards[r][c])
                
                actual_state = self._check_subboard_winner(cells)
                
                if sub_state != 0 and sub_state != actual_state:
                    return False
        
        return True
    
    def _check_subboard_winner(self, cells: List[int]) -> int:
        """Check winner of 3x3 sub-board. Returns 0/1/2/3."""
        WIN_PATTERNS = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        
        for a, b, c in WIN_PATTERNS:
            if cells[a] == cells[b] == cells[c] != 0:
                return cells[a]
        
        if 0 not in cells:
            return 3  # Draw
        
        return 0  # Open
    
    def _find_undo_candidates(self, board: Board, player: int) -> List[Tuple[int, int]]:
        """Find cells that could be undone (last move by player)."""
        candidates = []
        
        for r in range(9):
            for c in range(9):
                if board.boards[r][c] == player:
                    # Check if this could have been the last move
                    # It must satisfy the constraint from the move before it
                    candidates.append((r, c))
        
        return candidates
    
    def _try_undo_move(self, board: Board, r: int, c: int) -> Optional[Board]:
        """
        Try to undo a move at (r, c).
        
        Returns new board state if valid, None otherwise.
        """
        undo_board = board.clone()
        
        # Remove the piece
        undo_board.boards[r][c] = 0
        
        # Switch player
        undo_board.current_player = 3 - board.current_player
        
        # Update sub-board state if needed
        sub_r, sub_c = r // 3, c // 3
        
        # Recalculate sub-board state
        cells = []
        for dr in range(3):
            for dc in range(3):
                rr, cc = sub_r * 3 + dr, sub_c * 3 + dc
                cells.append(undo_board.boards[rr][cc])
        
        new_state = self._check_subboard_winner(cells)
        undo_board.completed_boards[sub_r][sub_c] = new_state
        
        # The constraint: where did the previous player have to play?
        # This move (r, c) was in sub-board (sub_r, sub_c)
        # So the move before must have sent us here
        # Meaning the move before was at cell (%3, %3) = (sub_r, sub_c) within its sub-board
        
        # Set last_move to indicate constraint
        # The previous move's cell position (within its sub-board) determined where we had to play
        # This is complex to track backward, so we'll be lenient
        undo_board.last_move = None  # Allow any sub-board
        
        return undo_board


def count_empty(board: Board) -> int:
    """Count playable empty cells."""
    count = 0
    for r in range(9):
        for c in range(9):
            if board.boards[r][c] == 0:
                sub_r, sub_c = r // 3, c // 3
                if board.completed_boards[sub_r][sub_c] == 0:
                    count += 1
    return count


if __name__ == '__main__':
    # Test solver
    solver = TablebaseSolver()
    checker = ReachabilityChecker()
    
    board = Board()
    # Play some moves
    moves = [(4, 4), (3, 3), (0, 0), (0, 4)]
    for r, c in moves:
        if (r, c) in board.get_legal_moves():
            board.make_move(r, c)
    
    print(f"Empty cells: {count_empty(board)}")
    print(f"Reachable: {checker.is_reachable(board)}")
    
    result, dtw = solver.solve(board)
    print(f"Result: {result}, DTW: {dtw}")
    print(f"Stats: {solver.stats}")
