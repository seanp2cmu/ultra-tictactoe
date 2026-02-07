"""
Distance to Win (DTW) Calculator
"""
from .transposition_table import CompressedTranspositionTable
from game import Board


class DTWCalculator:
    _MOVE_PRIORITY = (
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 0, 1, 1, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 0, 1, 1, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 0, 1, 1, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
    )
    
    @staticmethod
    def _get_move_priority(move):
        r, c = move
        local_r, local_c = r % 3, c % 3
        if local_r == 1 and local_c == 1:
            return 0
        elif (local_r == 0 or local_r == 2) and (local_c == 0 or local_c == 2):
            return 1
        return 2
    
    def __init__(self, use_cache=True, hot_size=50000, cold_size=500000,
                 endgame_threshold=15, max_nodes=100000):
        """
        Args:
            use_cache: if True, use transposition table
            hot_size: hot cache size
            cold_size: cold cache size
            endgame_threshold: endgame threshold (complete search)
            max_nodes: maximum nodes to search before giving up
        
        Note: 
            - ≤15 cells: complete DTW search
            - >15 cells: MCTS only (no shallow search)
        """
        self.use_cache = use_cache
        self.endgame_threshold = endgame_threshold
        self.max_nodes = max_nodes
        self._node_count = 0
        
        # Statistics
        self._total_searches = 0
        self._total_nodes = 0
        self._aborted_searches = 0
        
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
        """Cache lookup only (no search). Returns cached result or None."""
        if self.use_cache and self.tt:
            return self.tt.get(board)
        return None
    
    def calculate_dtw(self, board: Board, _empty_count: int = None):
        """
        DTW calculation (Alpha-Beta Search)
        
        15 cells or less: complete search
        16 cells or more: shallow search (depth limit)
        
        Returns:
            (result, dtw, best_move) or None
            - result: 1 (win), -1 (loss), 0 (draw)
            - dtw: Distance to Win/Loss
            - best_move: (row, col) or None
        """
        if self.use_cache and self.tt:
            cached = self.tt.get(board)
            if cached is not None:
                return cached
        
        empty_count = _empty_count if _empty_count is not None else board.count_playable_empty_cells()
        if empty_count > self.endgame_threshold:
            return None
        
        self._node_count = 0  # Reset node counter
        result, dtw, best_move = self._alpha_beta_search(board)
        
        self._total_searches += 1
        self._total_nodes += self._node_count
        
        # If search was aborted due to node limit, return None
        if result == -2:
            self._aborted_searches += 1
            return None
        
        if self.use_cache and self.tt:
            self.tt.put(board, result, dtw, best_move)
        
        return (result, dtw, best_move)
    
    def _alpha_beta_search(self, board: Board, depth: int = 0, alpha: int = -2, beta: int = 2):
        """
        Alpha-Beta Pruning Search

        Complete search for 15 cells or less using alpha-beta pruning
        Shallow search for 16 cells or more using alpha-beta pruning with depth limit
        
        Args:
            board: current board
            depth: current recursive depth (DTW calculation)
            alpha: Alpha value (minimum guarantee for maximizing player)
            beta: Beta value (maximum guarantee for minimizing player)
        
        Returns:
            (result, dtw, best_move)
            - result: 1 (win), -1 (loss), 0 (draw), -2 (aborted)
            - dtw: Distance to Win/Loss
            - best_move: (row, col) or None
        """
        self._node_count += 1
        if self._node_count > self.max_nodes:
            return (-2, float('inf'), None)  # Abort: too many nodes
        
        if board.winner is not None:
            if board.winner == board.current_player:
                return (1, 0, None)
            elif board.winner == 3:
                return (0, 0, None)
            else:
                return (-1, 0, None)
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return (0, 0, None)
        
        legal_moves = sorted(legal_moves, key=DTWCalculator._get_move_priority)
        
        best_move = None
        best_result = -2
        best_dtw = float('inf')
        
        for move in legal_moves:
            next_board = board.clone()
            next_board.make_move(move[0], move[1])
            
            if self.use_cache and self.tt:
                cached = self.tt.get(next_board)
                if cached is not None:
                    opponent_result, opponent_dtw, _ = cached
                else:
                    opponent_result, opponent_dtw, _ = self._alpha_beta_search(
                        next_board, depth + 1, -beta, -alpha
                    )
                    if opponent_result == -2:  # Aborted
                        return (-2, float('inf'), None)
                    self.tt.put(next_board, opponent_result, opponent_dtw, None)
            else:
                opponent_result, opponent_dtw, _ = self._alpha_beta_search(
                    next_board, depth + 1, -beta, -alpha
                )
                if opponent_result == -2:  # Aborted
                    return (-2, float('inf'), None)
            
            my_result = -opponent_result
            my_dtw = opponent_dtw + 1 if opponent_dtw != float('inf') else float('inf')
            
            if my_result > best_result:
                best_result = my_result
                best_dtw = my_dtw
                best_move = move
                alpha = max(alpha, my_result)
            elif my_result == best_result:
                if my_result > 0:
                    if my_dtw < best_dtw:
                        best_dtw = my_dtw
                        best_move = move
                elif my_result < 0:
                    if my_dtw > best_dtw:
                        best_dtw = my_dtw
                        best_move = move
                else:
                    if my_dtw < best_dtw:
                        best_dtw = my_dtw
                        best_move = move
            
            if alpha >= beta:
                break
        
        return (best_result, best_dtw, best_move)
    
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
        stats['dtw_searches'] = self._total_searches
        stats['dtw_nodes'] = self._total_nodes
        stats['dtw_aborted'] = self._aborted_searches
        stats['dtw_avg_nodes'] = self._total_nodes / max(1, self._total_searches)
        return stats
    
    def reset_search_stats(self):
        self._total_searches = 0
        self._total_nodes = 0
        self._aborted_searches = 0
    