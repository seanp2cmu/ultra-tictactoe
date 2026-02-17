"""NNUE Agent — uses C++ NNUE + alpha-beta search to play."""
import os
import tempfile

import nnue_cpp


class NNUEAgent:
    """Agent using C++ NNUE evaluation with alpha-beta search."""
    
    def __init__(self, model_path=None, depth=8, time_limit_ms=None,
                 qsearch=2, tt_size_mb=16):
        """
        Args:
            model_path: Path to saved NNUE model (.pt auto-exports to .nnue)
            depth: Max search depth
            time_limit_ms: Time limit per move in ms (overrides depth if set)
            qsearch: Quiescence search mode (0=off, 1=on, 2=auto)
            tt_size_mb: Transposition table size in MB
        """
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self._model = nnue_cpp.NNUEModel()
        
        if model_path:
            nnue_path = self._ensure_nnue_weights(model_path)
            self._model.load(nnue_path)
        else:
            from nnue.core.model import NNUE
            from nnue.cpp.export_weights import export_weights
            py_model = NNUE()
            py_model.eval()
            with tempfile.NamedTemporaryFile(suffix='.nnue', delete=False) as f:
                export_weights(py_model, f.name)
                self._model.load(f.name)
        
        self._engine = nnue_cpp.NNUESearchEngine(self._model, tt_size_mb)
        self._engine.set_qsearch(qsearch)
        self.name = f"NNUE-d{depth}"
    
    def _ensure_nnue_weights(self, model_path):
        """Convert .pt to .nnue if the .nnue file doesn't exist or is older."""
        nnue_path = model_path.replace('.pt', '.nnue')
        
        needs_export = (
            not os.path.exists(nnue_path) or
            os.path.getmtime(model_path) > os.path.getmtime(nnue_path)
        )
        
        if needs_export:
            from nnue.core.model import NNUE
            from nnue.cpp.export_weights import export_weights
            py_model = NNUE.load(model_path)
            export_weights(py_model, nnue_path)
        
        return nnue_path
    
    def select_action(self, board):
        """Select best move.
        
        Args:
            board: C++ Board object
            
        Returns:
            action: row * 9 + col
        """
        result = self._engine.search(
            board,
            max_depth=self.depth,
            time_limit_ms=self.time_limit_ms or 0,
        )
        
        if result.best_r < 0:
            legal = board.get_legal_moves()
            if legal:
                r, c = legal[0]
                return r * 9 + c
            return 0
        
        return result.best_r * 9 + result.best_c
    
    def clear(self):
        """Clear search state (TT, history, killers)."""
        self._engine.clear()
