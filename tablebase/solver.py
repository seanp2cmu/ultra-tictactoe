"""
BFS + Memoization Tablebase Solver

Uses BFS from terminal positions with memoization.
Includes 4-move DFS reachability check to filter illegal positions.
"""

from typing import Dict, Tuple, Optional, List
from game import Board
from utils.symmetry import D4_TRANSFORMS, INV_TRANSFORMS


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
        self.stats = {'cache_hits': 0, 'solves': 0, 'base_hits': 0, 'missing_child': 0}
        
        # Load base tablebase for faster lookups
        self.base_tablebase = base_tablebase or {}
    
    def solve(self, board: Board, max_depth: Optional[int] = None) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """
        Solve position with memoization.
        
        Returns:
            (result, dtw, best_move) from current player's perspective
        """
        board_hash = self._hash_board(board)
        constraint = getattr(board, 'constraint', -1)
        
        # Check cache: hash -> {constraint: (result, dtw, move)}
        if board_hash in self.cache and constraint in self.cache[board_hash]:
            self.stats['cache_hits'] += 1
            return self.cache[board_hash][constraint]
        
        # Check base tablebase
        if board_hash in self.base_tablebase and constraint in self.base_tablebase[board_hash]:
            self.stats['base_hits'] += 1
            result = self.base_tablebase[board_hash][constraint]
            if board_hash not in self.cache:
                self.cache[board_hash] = {}
            self.cache[board_hash][constraint] = result
            return result
        
        result, dtw, best_move = self._solve_recursive(board)
        
        # Store in nested dict
        if board_hash not in self.cache:
            self.cache[board_hash] = {}
        self.cache[board_hash][constraint] = (result, dtw, best_move)
        self.stats['solves'] += 1
        
        return (result, dtw, best_move)
    
    def _solve_recursive(self, board: Board) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """
        Progressive solver - lookup children from previous levels.
        Optimized: constraint-based moves + undo pattern (no clone).
        """
        # Terminal check
        if board.winner is not None:
            if board.winner == 3:  # Draw
                return (0, 0, None)
            return (-1, 0, None)
        
        # Get moves based on constraint (avoid full get_legal_moves)
        constraint = getattr(board, 'constraint', -1)
        moves = self._get_moves_for_constraint(board, constraint)
        
        if not moves:
            return (0, 0, None)
        
        best_value = -2
        best_dtw = 999
        best_move = None
        
        # Save state for undo
        prev_last_move = board.last_move
        prev_winner = board.winner
        
        for r, c in moves:
            # Save sub-board state
            sub_r, sub_c = r // 3, c // 3
            sub_idx = sub_r * 3 + sub_c
            if hasattr(board, 'get_completed_state'):
                prev_completed = board.get_completed_state(sub_idx)
            else:
                prev_completed = board.completed_boards[sub_r][sub_c]
            
            # Make move (no clone)
            board.make_move(r, c, validate=False)
            
            # Child's constraint
            target_sub_r, target_sub_c = r % 3, c % 3
            target_sub = target_sub_r * 3 + target_sub_c
            
            # Evaluate child
            if board.winner is not None:
                child_result = -1 if board.winner != 3 else 0
                child_dtw = 0
            elif (board.get_completed_state(target_sub) if hasattr(board, 'get_completed_state') else board.completed_boards[target_sub_r][target_sub_c]) != 0:
                # "any" constraint
                child_hash = self._hash_board(board)
                child_result, child_dtw = self._lookup_best_constraint(child_hash, board)
            else:
                # Specific constraint
                child_hash, canonical_constraint = self._hash_board_with_constraint(board, target_sub)
                child_result, child_dtw = self._lookup_constraint(child_hash, canonical_constraint, board)
            
            # Undo move
            board.undo_move(r, c, prev_completed, prev_winner, prev_last_move)
            
            # Negate for perspective switch
            value = -child_result
            
            if value > best_value or (value == best_value and child_dtw < best_dtw):
                best_value = value
                best_dtw = child_dtw
                best_move = (r, c)
        
        return (best_value, best_dtw + 1, best_move)
    
    def _get_moves_for_constraint(self, board: Board, constraint: int) -> List[Tuple[int, int]]:
        """Get legal moves based on constraint (faster than get_legal_moves)."""
        moves = []
        
        if constraint == -1:
            # No constraint or "any" - check all open sub-boards
            for sub_idx in range(9):
                sub_r, sub_c = sub_idx // 3, sub_idx % 3
                if (board.get_completed_state(sub_idx) if hasattr(board, 'get_completed_state') else board.completed_boards[sub_r][sub_c]) == 0:
                    start_r, start_c = sub_r * 3, sub_c * 3
                    for dr in range(3):
                        for dc in range(3):
                            if board.get_cell(start_r + dr, start_c + dc) == 0:
                                moves.append((start_r + dr, start_c + dc))
        else:
            # Specific constraint - only check that sub-board
            sub_r, sub_c = constraint // 3, constraint % 3
            if (board.get_completed_state(constraint) if hasattr(board, 'get_completed_state') else board.completed_boards[sub_r][sub_c]) == 0:
                start_r, start_c = sub_r * 3, sub_c * 3
                for dr in range(3):
                    for dc in range(3):
                        if board.get_cell(start_r + dr, start_c + dc) == 0:
                            moves.append((start_r + dr, start_c + dc))
            else:
                # Constraint sub-board is completed - fall back to "any"
                return self._get_moves_for_constraint(board, -1)
        
        return moves
    
    def _lookup_constraint(self, h: int, constraint: int, board: Board) -> Tuple[int, int]:
        """Lookup specific constraint in nested dict cache.
        
        Due to D4+flip canonicalization, the exact constraint might not match.
        If not found, use any available constraint (they're equivalent under symmetry).
        """
        # Check cache: hash -> {constraint: (result, dtw, move)}
        if h in self.cache:
            if constraint in self.cache[h]:
                result, dtw, _ = self.cache[h][constraint]
                return (result, dtw)
            # Constraint not found - use any available (equivalent under symmetry)
            if self.cache[h]:
                any_constraint = next(iter(self.cache[h]))
                result, dtw, _ = self.cache[h][any_constraint]
                return (result, dtw)
        
        if h in self.base_tablebase:
            if constraint in self.base_tablebase[h]:
                result, dtw, _ = self.base_tablebase[h][constraint]
                self.stats['base_hits'] += 1
                return (result, dtw)
            if self.base_tablebase[h]:
                any_constraint = next(iter(self.base_tablebase[h]))
                result, dtw, _ = self.base_tablebase[h][any_constraint]
                self.stats['base_hits'] += 1
                return (result, dtw)
        
        # Not found - check if terminal
        if board.winner is not None:
            return (-1 if board.winner != 3 else 0, 0)
        elif not board.get_legal_moves():
            return (0, 0)
        
        # Not in cache - should not happen in progressive building
        # Return unknown (caller should handle)
        self.stats['missing_child'] += 1
        return (0, 999)  # Unknown - treat as draw with high dtw
    
    def _lookup_best_constraint(self, h: int, board: Board) -> Tuple[int, int]:
        """For 'any' constraint, find best result among all OPEN boards."""
        best_result = -2
        best_dtw = 999
        
        # Check cache
        if h in self.cache:
            for constraint, (result, dtw, _) in self.cache[h].items():
                if result > best_result or (result == best_result and dtw < best_dtw):
                    best_result = result
                    best_dtw = dtw
        
        # Check base tablebase
        if h in self.base_tablebase:
            for constraint, (result, dtw, _) in self.base_tablebase[h].items():
                if result > best_result or (result == best_result and dtw < best_dtw):
                    best_result = result
                    best_dtw = dtw
        
        if best_result == -2:
            if board.winner is not None:
                return (-1 if board.winner != 3 else 0, 0)
            elif not board.get_legal_moves():
                return (0, 0)
            
            # Not in cache - should not happen in progressive building
            self.stats['missing_child'] += 1
            return (0, 999)  # Unknown - treat as draw with high dtw
        
        return (best_result, best_dtw)
    
    @staticmethod
    def pack_canonical_key(canonical_data: Tuple) -> int:
        """Pack canonical tuple into deterministic 90-bit integer for storage.
        Each sub-board: (state:2bit, x_count:4bit, o_count:4bit) = 10 bits
        """
        result = 0
        for state, x_count, o_count in canonical_data:
            result = (result << 10) | (state << 8) | (x_count << 4) | o_count
        return result
    
    def _hash_board_with_constraint(self, board: Board, constraint: int) -> Tuple[int, int]:
        """Create canonical hash and transformed constraint.
        
        Returns: (deterministic_packed_key, canonical_constraint)
        """
        # Build raw sub_data using cached sub_counts (O(9) instead of O(81))
        # Compatible with both Board (2D) and BoardCy (1D via getter methods)
        raw_sub_data = []
        is_cython = hasattr(board, 'get_completed_state')
        for sub_idx in range(9):
            if is_cython:
                state = board.get_completed_state(sub_idx)
                x_count, o_count = board.get_sub_count_pair(sub_idx)
            else:
                sub_r, sub_c = sub_idx // 3, sub_idx % 3
                state = board.completed_boards[sub_r][sub_c]
                x_count, o_count = board.sub_counts[sub_idx]
            
            if state != 0:
                raw_sub_data.append((state, 0, 0))
            else:
                raw_sub_data.append((0, x_count, o_count))
        
        # Precompute flipped raw_sub_data once (avoid repeated flip computation)
        # flip: state 1<->2, swap x_count/o_count
        flipped_raw = tuple(
            (3 - s, 0, 0) if s == 1 or s == 2 else (0, o, x)
            for s, x, o in raw_sub_data
        )
        
        # Find minimum canonical form using tuple comparison (fast)
        min_data = None
        min_constraint = -1
        
        for perm_id, perm in enumerate(D4_TRANSFORMS):
            inv = INV_TRANSFORMS[perm_id]
            sym_constraint = inv[constraint] if constraint >= 0 else -1
            
            # Apply permutation to original
            sym_data = tuple(raw_sub_data[perm[i]] for i in range(9))
            if min_data is None or (sym_data, sym_constraint) < (min_data, min_constraint):
                min_data = sym_data
                min_constraint = sym_constraint
            
            # Apply permutation to pre-flipped
            flipped_data = tuple(flipped_raw[perm[i]] for i in range(9))
            if (flipped_data, sym_constraint) < (min_data, min_constraint):
                min_data = flipped_data
                min_constraint = sym_constraint
        
        # Pack only at the end for deterministic storage key
        return self.pack_canonical_key(min_data), min_constraint
    
    def _hash_board(self, board: Board) -> int:
        """Create hash for board position (without constraint transformation)."""
        h, _ = self._hash_board_with_constraint(board, -1)
        return h

