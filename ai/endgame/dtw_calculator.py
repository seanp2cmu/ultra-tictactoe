"""
Distance to Win (DTW) Calculator
엔드게임에서 확정 승리까지의 최단 거리 계산
"""
from .transposition_table import CompressedTranspositionTable
from game import Board


class DTWCalculator:
    def __init__(self, use_cache=True, hot_size=50000, cold_size=500000, use_symmetry=True, endgame_threshold=15):
        """
        Args:
            use_cache: Transposition Table 사용 여부
            hot_size: Hot cache 크기
            cold_size: Cold cache 크기
            use_symmetry: 보드 대칭 정규화 (8배 메모리 절약)
            endgame_threshold: 엔드게임 판단 기준 (플레이 가능한 빈칸 개수)
        
        Note: 15칸 이하는 완전 탐색 (25칸은 너무 오래 걸림)
        """
        self.use_cache = use_cache
        self.endgame_threshold = endgame_threshold
        
        if use_cache:
            self.tt = CompressedTranspositionTable(
                hot_size=hot_size, 
                cold_size=cold_size,
                use_symmetry=use_symmetry
            )
        else:
            self.tt = None
    
    def is_endgame(self, board: Board):
        """엔드게임 판단 (플레이 가능한 빈칸이 threshold 이하)"""
        return board.count_playable_empty_cells() <= self.endgame_threshold
    
    def calculate_dtw(self, board: Board):
        """
        DTW 계산 (Alpha-Beta Search)
        
        15칸 이하만 계산: Alpha-Beta 완전 탐색
        16칸 이상: None 반환 (MCTS 사용)
        
        Returns:
            (result, dtw, best_move) or None
            - result: 1 (승), -1 (패), 0 (무승부)
            - dtw: Distance to Win/Loss
            - best_move: (row, col) or None
        """
        # 캐시 확인
        if self.use_cache and self.tt:
            cached = self.tt.get(board)
            if cached is not None:
                return cached
        
        # === threshold 초과: DTW 계산 안 함, MCTS 사용 ===
        if board.count_playable_empty_cells() > self.endgame_threshold:
            return None
        
        # === threshold 이하: Alpha-Beta 완전 탐색 ===
        result, dtw, best_move = self._alpha_beta_search(board)
        
        if self.use_cache and self.tt:
            self.tt.put(board, result, dtw, best_move)
        
        return (result, dtw, best_move)
    
    def _alpha_beta_search(self, board: Board, depth: int = 0, alpha: int = -2, beta: int = 2):
        """
        Alpha-Beta Pruning 탐색
        
        15칸 이하는 완전 탐색 (depth 제한 없음)
        Alpha-Beta + 캐싱으로 효율적 탐색
        
        Args:
            board: 현재 보드
            depth: 현재 재귀 깊이 (DTW 계산용)
            alpha: Alpha 값 (최대화 플레이어의 최소 보장 값)
            beta: Beta 값 (최소화 플레이어의 최대 보장 값)
        
        Returns:
            (result, dtw, best_move)
            - result: 1 (승), -1 (패), 0 (무승부)
            - dtw: Distance to Win/Loss
            - best_move: (row, col) or None
        """
        # 터미널 체크
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
        
        best_move = None
        best_result = -2  # -1보다 작게 시작
        best_dtw = float('inf')
        
        for move in legal_moves:
            next_board = board.clone()
            next_board.make_move(move[0], move[1])
            
            # 캐시 먼저 확인
            if self.use_cache and self.tt:
                cached = self.tt.get(next_board)
                if cached is not None:
                    opponent_result, opponent_dtw, _ = cached
                else:
                    # Alpha-Beta Pruning 적용
                    opponent_result, opponent_dtw, _ = self._alpha_beta_search(
                        next_board, depth + 1, -beta, -alpha
                    )
                    self.tt.put(next_board, opponent_result, opponent_dtw, None)
            else:
                opponent_result, opponent_dtw, _ = self._alpha_beta_search(
                    next_board, depth + 1, -beta, -alpha
                )
            
            my_result = -opponent_result
            my_dtw = opponent_dtw + 1 if opponent_dtw != float('inf') else float('inf')
            
            # 최선의 수 선택
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
            
            # 🔥 Alpha-Beta Pruning
            if alpha >= beta:
                break  # 나머지 브랜치 탐색 생략
        
        return (best_result, best_dtw, best_move)
    
    def get_best_winning_move(self, board: Board):
        """
        확정 승리 수 반환 (25칸 이하만)
        
        Returns:
            move: (row, col) 또는 None
            dtw: 해당 수의 DTW
        """
        if not self.is_endgame(board):
            return None, float('inf')
        
        # Retrograde Analysis 수행
        result_data = self.calculate_dtw(board)
        
        if result_data is None:
            return None, float('inf')
        
        result, dtw, best_move = result_data
        
        # 승리 확정이면 best_move 반환
        if result == 1 and best_move is not None:
            return best_move, dtw
        
        return None, float('inf')
    
    def get_adjusted_value(self, board: Board, network_value: float):
        """
        DTW를 고려한 value 조정 (25칸 이하만)
        
        Args:
            board: 현재 보드
            network_value: 네트워크가 예측한 value (-1 ~ 1)
        
        Returns:
            adjusted_value: DTW로 조정된 value
        """
        if not self.is_endgame(board):
            return network_value
        
        result_data = self.calculate_dtw(board)
        
        if result_data is None:
            return network_value
        
        result, _, _ = result_data
        
        # result: 1 (승), -1 (패), 0 (무승부)
        return float(result)
    
    def get_stats(self):
        """캐시 통계 반환"""
        if self.use_cache and self.tt:
            return self.tt.get_stats()
        return {}
    
    def clear_cache(self):
        """캐시 초기화"""
        if self.use_cache and self.tt:
            self.tt.clear()
