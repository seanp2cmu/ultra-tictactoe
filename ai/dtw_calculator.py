"""
Distance to Win (DTW) Calculator
엔드게임에서 확정 승리까지의 최단 거리 계산
"""
import copy
from .transposition_table import CompressedTranspositionTable
from game import Board


class DTWCalculator:
    def __init__(self, max_depth=15, use_cache=True, hot_size=50000, cold_size=500000, use_symmetry=True, endgame_threshold=25):
        """
        Args:
            max_depth: 최대 탐색 깊이
            use_cache: Transposition Table 사용 여부
            hot_size: Hot cache 크기
            cold_size: Cold cache 크기
            use_symmetry: 보드 대칭 정규화 (8배 메모리 절약)
            endgame_threshold: 엔드게임 판단 기준 (빈 칸 개수)
        """
        self.max_depth = max_depth
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
        """엔드게임 판단 (빈 칸이 threshold 이하 - Tablebase 영역)"""
        empty_count = sum(1 for row in board.boards for cell in row if cell == 0)
        return empty_count <= self.endgame_threshold
    
    def calculate_dtw(self, board: Board):
        """
        DTW 계산 (Retrograde Analysis 방식)
        
        25칸 이하만 계산: 완벽한 Retrograde Analysis
        26칸 이상: None 반환 (MCTS 사용)
        
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
        
        empty_count = sum(1 for row in board.boards for cell in row if cell == 0)
        
        # === threshold 초과: DTW 계산 안 함, MCTS 사용 ===
        if empty_count > self.endgame_threshold:
            return None
        
        # === threshold 이하: Retrograde Analysis (완벽) ===
        result, dtw, best_move = self._retrograde_analysis(board)
        
        if self.use_cache and self.tt:
            self.tt.put(board, result, dtw, best_move)
        
        return (result, dtw, best_move)
    
    def _retrograde_analysis(self, board: Board):
        """
        완벽한 Retrograde Analysis (25칸 이하)
        depth 제한 없이 끝까지 계산
        
        Returns:
            (result, dtw, best_move)
            - result: 1 (승), -1 (패), 0 (무승부)
            - dtw: Distance to Win/Loss
            - best_move: (row, col) or None
        """
        # 터미널 체크
        if board.winner is not None:
            if board.winner == board.current_player:
                return (1, 0, None)  # 승리
            elif board.winner == 3:
                return (0, 0, None)  # 무승부
            else:
                return (-1, 0, None)  # 패배
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return (0, 0, None)  # 무승부
        
        # 모든 수 탐색 (depth 제한 없음)
        best_move = None
        best_result = -1  # 최악부터 시작
        best_dtw = float('inf')
        
        for move in legal_moves:
            next_board = copy.deepcopy(board)
            next_board.make_move(move[0], move[1])
            
            # 캐시 먼저 확인
            if self.use_cache and self.tt:
                cached = self.tt.get(next_board)
                if cached is not None:
                    opponent_result, opponent_dtw, _ = cached
                else:
                    # 재귀 (depth 제한 없음)
                    opponent_result, opponent_dtw, _ = self._retrograde_analysis(next_board)
                    self.tt.put(next_board, opponent_result, opponent_dtw, None)
            else:
                opponent_result, opponent_dtw, _ = self._retrograde_analysis(next_board)
            
            # 상대 관점을 내 관점으로 변환
            my_result = -opponent_result
            my_dtw = opponent_dtw + 1 if opponent_dtw != float('inf') else float('inf')
            
            # 최선의 수 선택
            if my_result > best_result:
                best_result = my_result
                best_dtw = my_dtw
                best_move = move
            elif my_result == best_result and my_dtw < best_dtw:
                best_dtw = my_dtw
                best_move = move
        
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
        
        result, dtw, _ = result_data
        
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
