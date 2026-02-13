"""
Distance to Win (DTW) Calculator - C++ Implementation
"""
from .transposition_table import CompressedTranspositionTable
from game import Board, BoardCpp, DTWCalculatorCpp


def _board_to_cpp(board):
    """Convert BoardCy to BoardCpp for C++ DTW (direct bitmask copy)."""
    board_cpp = BoardCpp()
    
    # Direct bitmask copy (O(9) instead of O(81))
    board_cpp.x_masks = list(board.x_masks)
    board_cpp.o_masks = list(board.o_masks)
    
    # sub_counts: compute from bitmasks (popcount)
    sub_counts = []
    for i in range(9):
        x_count = bin(board.x_masks[i]).count('1')
        o_count = bin(board.o_masks[i]).count('1')
        sub_counts.append([x_count, o_count])
    board_cpp.sub_counts = sub_counts
    
    board_cpp.set_completed_boards_2d(board.get_completed_boards_2d())
    board_cpp.completed_mask = board.completed_mask
    board_cpp.current_player = board.current_player
    board_cpp.winner = board.winner
    if board.last_move:
        board_cpp.last_move = board.last_move
    return board_cpp


class DTWCalculator:
    """DTW Calculator using C++ implementation."""
    
    def __init__(self, use_cache=True, hot_size=50000, cold_size=500000,
                 endgame_threshold=15, max_nodes=10000000):
        self.use_cache = use_cache
        self.endgame_threshold = endgame_threshold
        self.max_nodes = max_nodes
        
        # Statistics
        self._total_searches = 0
        
        # C++ DTW
        self._cpp_dtw = DTWCalculatorCpp(use_cache, endgame_threshold, max_nodes)
        
        # Python cache for BoardCy keys
        if use_cache:
            self.tt = CompressedTranspositionTable(
                hot_size=hot_size, 
                cold_size=cold_size
            )
        else:
            self.tt = None
    
    def is_endgame(self, board: Board):
        return board.count_playable_empty_cells() <= self.endgame_threshold
    
    def lookup_cache(self, board: Board):
        """Cache lookup only (no search). Returns cached result or None.
        
        NOTE: best_move is set to None because Python cache uses canonical hash
        but best_move coordinates are orientation-dependent.
        """
        if self.use_cache and self.tt:
            cached = self.tt.get(board)
            if cached is not None:
                result, dtw, _ = cached  # Ignore cached best_move
                return (result, dtw, None)
        return None
    
    def calculate_dtw(self, board: Board, _empty_count: int = None, need_best_move: bool = True):
        """
        DTW calculation using C++ Alpha-Beta Search.
        
        Args:
            board: Board to analyze
            _empty_count: Optional pre-computed empty cell count
            need_best_move: If True, always compute fresh to get correct best_move coords
        
        Returns:
            (result, dtw, best_move) or None
            - result: 1 (win), -1 (loss), 0 (draw)
            - dtw: Distance to Win/Loss
            - best_move: (row, col) or None
        """
        # Check Python cache first (but only if we don't need best_move)
        # Python cache uses canonical hash, so best_move coords may be wrong
        if self.use_cache and self.tt and not need_best_move:
            cached = self.tt.get(board)
            if cached is not None:
                result, dtw, _ = cached
                return (result, dtw, None)
        
        empty_count = _empty_count if _empty_count is not None else board.count_playable_empty_cells()
        if empty_count > self.endgame_threshold:
            return None
        
        # Convert to C++ Board and calculate
        board_cpp = _board_to_cpp(board)
        result = self._cpp_dtw.calculate_dtw(board_cpp)
        self._total_searches += 1
        
        # Store in Python cache
        if result is not None and self.use_cache and self.tt:
            self.tt.put(board, result[0], result[1], result[2])
        
        return result
    
    def get_best_winning_move(self, board: Board):
        """
        Returns the best winning move (for endgame)
        
        Returns:
            move: (row, col) or None
            dtw: DTW of the move
        """
        if not self.is_endgame(board):
            return None, float('inf')
        
        result_data = self.calculate_dtw(board)
        
        if result_data is None:
            return None, float('inf')
        
        result, dtw, best_move = result_data
        
        if result == 1 and best_move is not None:
            return best_move, dtw
        
        return None, float('inf')
    
    def get_adjusted_value(self, board: Board, network_value: float):
        """
        Adjusts value based on DTW (only for endgame)
        
        Args:
            board: current board
            network_value: value predicted by network (-1 ~ 1)
        
        Returns:
            adjusted_value: adjusted value based on DTW
        """
        if not self.is_endgame(board):
            return network_value
        
        result_data = self.calculate_dtw(board)
        
        if result_data is None:
            return network_value
        
        result, _, _ = result_data
        
        return float(result)
    
    def get_stats(self):
        stats = {}
        if self.use_cache and self.tt:
            stats = self.tt.get_stats()
        cpp_stats = self._cpp_dtw.get_stats()
        stats.update(cpp_stats)
        stats['dtw_searches'] = self._total_searches
        return stats
    
    def reset_search_stats(self):
        self._total_searches = 0
        self._cpp_dtw.reset_search_stats()
    