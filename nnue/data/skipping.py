"""Position filtering/skipping logic for NNUE data generation."""
import random
from typing import Optional

from game import Board
from ai.endgame import DTWCalculator
from nnue.config import DataGenConfig

# 3x3 win lines as (cell_a, cell_b, cell_c) indices within a sub-board
_WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diags
]


class PositionFilter:
    """Filter positions for NNUE training data quality."""
    
    def __init__(
        self,
        config: Optional[DataGenConfig] = None,
        dtw_calculator: Optional[DTWCalculator] = None
    ):
        self.config = config or DataGenConfig()
        self.dtw = dtw_calculator
        
    def should_save(
        self,
        board: Board,
        ply: int,
        eval_value: float,
        rng: Optional[random.Random] = None
    ) -> bool:
        """
        Determine if a position should be saved for NNUE training.
        
        Args:
            board: Current board state
            ply: Current ply (move number)
            eval_value: MCTS evaluation (-1 to 1)
            rng: Random number generator for reproducibility
            
        Returns:
            True if position should be saved
        """
        rng = rng or random.Random()
        
        # 1. Ply range check
        if ply < self.config.write_minply:
            return False
        if ply > self.config.write_maxply:
            return False
        
        # 2. Eval limit check (skip decisive positions)
        if abs(eval_value) > self.config.eval_limit:
            return False
        
        # 3. Random skip for diversity
        if rng.random() < self.config.random_skip_rate:
            return False
        
        # 4. Skip endgame positions (DTW can solve them exactly)
        if self.config.skip_endgame and self.dtw is not None:
            if self.dtw.is_endgame(board):
                return False
        
        # 5. Skip noisy positions (only in early/mid game)
        if self.config.skip_noisy and ply <= self.config.skip_noisy_maxply:
            if not self._is_quiet(board):
                return False
        
        return True
    
    def get_skip_reason(
        self,
        board: Board,
        ply: int,
        eval_value: float
    ) -> Optional[str]:
        """Get reason why position would be skipped (for debugging)."""
        if ply < self.config.write_minply:
            return f"ply {ply} < min {self.config.write_minply}"
        if ply > self.config.write_maxply:
            return f"ply {ply} > max {self.config.write_maxply}"
        if abs(eval_value) > self.config.eval_limit:
            return f"|eval| {abs(eval_value):.2f} > limit {self.config.eval_limit}"
        if self.config.skip_endgame and self.dtw is not None:
            if self.dtw.is_endgame(board):
                return "endgame position"
        if self.config.skip_noisy and ply <= self.config.skip_noisy_maxply:
            if not self._is_quiet(board):
                return "noisy position (tactical threat)"
        return None
    
    def _is_quiet(self, board: Board) -> bool:
        """Check if position has no immediate tactical threats.
        
        A position is "noisy" if any active sub-board has a line
        with 2 pieces of one color + 1 empty cell (= one move from winning).
        Quiet positions are where the search would call NNUE eval directly.
        """
        legal = board.get_legal_moves()
        if not legal:
            return True
        
        player = board.current_player
        opponent = 3 - player
        
        # Find which sub-boards have legal moves
        checked_subs = set()
        for r, c in legal:
            sub_key = (r // 3, c // 3)
            if sub_key in checked_subs:
                continue
            checked_subs.add(sub_key)
            
            sub_r, sub_c = sub_key
            # Read the 9 cells of this sub-board
            cells = []
            for lr in range(3):
                for lc in range(3):
                    cells.append(board.get_cell(sub_r * 3 + lr, sub_c * 3 + lc))
            
            # Check each win line for a 2+1 threat
            for a, b, cc in _WIN_LINES:
                vals = (cells[a], cells[b], cells[cc])
                empties = vals.count(0)
                if empties != 1:
                    continue
                # 2 of same color + 1 empty = tactical threat
                if vals.count(player) == 2 or vals.count(opponent) == 2:
                    return False
        
        return True
