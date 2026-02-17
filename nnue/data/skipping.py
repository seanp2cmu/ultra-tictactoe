"""Position filtering/skipping logic for NNUE data generation."""
import random
from typing import Optional

from game import Board
from ai.endgame import DTWCalculator
from nnue.config import DataGenConfig


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
        return None
