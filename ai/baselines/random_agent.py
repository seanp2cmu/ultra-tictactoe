"""
Random Agent - selects random legal moves.
Used as performance floor baseline.
"""
import random
from game import Board


class RandomAgent:
    """Agent that plays random legal moves."""
    
    def __init__(self):
        self.name = "Random"
    
    def select_action(self, board: Board, temperature: float = None) -> int:
        """Select a random legal move.
        
        Args:
            board: Current board state
            temperature: Ignored (for API compatibility)
            
        Returns:
            action: Move as action index (row * 9 + col)
        """
        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            return 0
        
        move = random.choice(legal_moves)
        return move[0] * 9 + move[1]
