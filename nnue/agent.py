"""NNUE Agent — uses NNUE + alpha-beta search to play."""
from nnue.core.model import NNUE
from nnue.engine.search import NNUESearch


class NNUEAgent:
    """Agent using NNUE evaluation with alpha-beta search."""
    
    def __init__(self, model_path=None, depth=8, time_limit_ms=None, device='cpu'):
        """
        Args:
            model_path: Path to saved NNUE model (None = random weights)
            depth: Max search depth
            time_limit_ms: Time limit per move (overrides depth if set)
            device: torch device for NNUE inference
        """
        if model_path:
            self.model = NNUE.load(model_path, device=device)
        else:
            self.model = NNUE()
            self.model.eval()
        
        self.search = NNUESearch(self.model)
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.name = f"NNUE-d{depth}"
    
    def select_action(self, board):
        """Select best move.
        
        Args:
            board: Board object
            temperature: Ignored (API compatibility)
            
        Returns:
            action: row * 9 + col
        """
        move, score, info = self.search.search(
            board, 
            max_depth=self.depth,
            time_limit_ms=self.time_limit_ms,
        )
        
        if move is None:
            legal = board.get_legal_moves()
            if legal:
                move = legal[0]
            else:
                return 0
        
        return move[0] * 9 + move[1]
