"""
Distance to Win (DTW) Calculator
엔드게임에서 확정 승리까지의 최단 거리 계산
"""
from .transposition_table import CompressedTranspositionTable
from game import Board


class DTWCalculator:
    # 클래스 레벨 상수: move ordering 우선순위 (매번 함수 생성 방지)
    # 중앙 = 0, 코너 = 1, 변 = 2
    _MOVE_PRIORITY = (
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # row 0
        (1, 0, 1, 1, 0, 1, 1, 0, 1),  # row 1
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # row 2
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # row 3
        (1, 0, 1, 1, 0, 1, 1, 0, 1),  # row 4
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # row 5
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # row 6
        (1, 0, 1, 1, 0, 1, 1, 0, 1),  # row 7
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # row 8
    )
    
    @staticmethod
    def _get_move_priority(move):
        """Move ordering 우선순위 (중앙 > 코너 > 변)"""
        r, c = move
        local_r, local_c = r % 3, c % 3
        if local_r == 1 and local_c == 1:
            return 0  # 중앙
        elif (local_r == 0 or local_r == 2) and (local_c == 0 or local_c == 2):
            return 1  # 코너
        return 2  # 변
    
    def __init__(self, use_cache=True, hot_size=50000, cold_size=500000,
                 endgame_threshold=15, midgame_threshold=45, shallow_depth=8):
        """
        Args:
            use_cache: Transposition Table 사용 여부
            hot_size: Hot cache 크기
            cold_size: Cold cache 크기
            endgame_threshold: 엔드게임 판단 기준 (완전 탐색)
            midgame_threshold: 중반 판단 기준 (얕은 탐색)
            shallow_depth: 중반 얕은 탐색 depth 제한
        
        Note: 
            - ≤15칸: 완전 탐색
            - 16-45칸: 얕은 탐색 (depth 제한)
            - >45칸: MCTS만
            - 보드 대칭 정규화 항상 사용 (8배 메모리 절약)
        """
        self.use_cache = use_cache
        self.endgame_threshold = endgame_threshold
        self.midgame_threshold = midgame_threshold
        self.shallow_depth = shallow_depth
        
        if use_cache:
            self.tt = CompressedTranspositionTable(
                hot_size=hot_size, 
                cold_size=cold_size
            )
        else:
            self.tt = None
    
    def is_endgame(self, board: Board):
        """엔드게임 판단 (완전 탐색 가능)"""
        return board.count_playable_empty_cells() <= self.endgame_threshold
    
    def is_midgame(self, board: Board):
        """중반 판단 (얕은 탐색 적용 가능)"""
        cells = board.count_playable_empty_cells()
        return self.endgame_threshold < cells <= self.midgame_threshold
    
    def calculate_dtw(self, board: Board, _empty_count: int = None):
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
        # _empty_count가 전달되면 재계산 생략
        empty_count = _empty_count if _empty_count is not None else board.count_playable_empty_cells()
        if empty_count > self.endgame_threshold:
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
        
        # Move ordering: 중앙 우선 (더 좋은 pruning을 위해)
        legal_moves = sorted(legal_moves, key=DTWCalculator._get_move_priority)
        
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
    
    def check_candidate_moves(self, board: Board, candidate_moves: list):
        """
        중반: 후보수들에 대해 얕은 Alpha-Beta로 승/패 확정 체크
        
        Args:
            board: 현재 보드
            candidate_moves: MCTS 상위 후보수 리스트 [(row, col), ...]
        
        Returns:
            {
                'winning_move': (row, col) or None,  # 승리 확정 수
                'losing_moves': [(row, col), ...],   # 패배 확정 수들
                'safe_moves': [(row, col), ...]      # 안전한 수들
            }
        """
        result = {
            'winning_move': None,
            'losing_moves': [],
            'safe_moves': []
        }
        
        # 엔드게임이면 완전 탐색 사용
        if self.is_endgame(board):
            dtw_result = self.calculate_dtw(board)
            if dtw_result and dtw_result[0] == 1 and dtw_result[2]:
                result['winning_move'] = dtw_result[2]
            return result
        
        # 중반 범위가 아니면 빈 결과 반환
        if not self.is_midgame(board):
            result['safe_moves'] = candidate_moves
            return result
        
        for move in candidate_moves:
            next_board = board.clone()
            next_board.make_move(move[0], move[1])
            
            # 얕은 Alpha-Beta로 평가
            move_result, _, _ = self._shallow_alpha_beta(next_board, depth=0)
            
            # 상대 관점이므로 부호 반전
            my_result = -move_result
            
            if my_result == 1:
                # 승리 확정! 즉시 반환
                result['winning_move'] = move
                return result
            elif my_result == -1:
                # 패배 확정 - 피해야 함
                result['losing_moves'].append(move)
            else:
                # 미정 - 안전
                result['safe_moves'].append(move)
        
        return result
    
    def _shallow_alpha_beta(self, board: Board, depth: int = 0, alpha: int = -2, beta: int = 2):
        """
        얕은 Alpha-Beta 탐색 (depth 제한)
        
        Args:
            board: 현재 보드
            depth: 현재 깊이
        
        Returns:
            (result, dtw, best_move)
            - result: 1 (승리 확정), -1 (패배 확정), 0 (미정/무승부)
        """
        # 터미널 체크
        if board.winner is not None:
            if board.winner == board.current_player:
                return (1, depth, None)
            elif board.winner == 3:
                return (0, depth, None)
            else:
                return (-1, depth, None)
        
        # Depth 제한 도달 → 미정(0) 반환
        if depth >= self.shallow_depth:
            return (0, depth, None)
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return (0, depth, None)
        
        best_result = -2
        best_move = None
        
        for move in legal_moves:
            next_board = board.clone()
            next_board.make_move(move[0], move[1])
            
            opponent_result, _, _ = self._shallow_alpha_beta(
                next_board, depth + 1, -beta, -alpha
            )
            
            my_result = -opponent_result
            
            if my_result > best_result:
                best_result = my_result
                best_move = move
                alpha = max(alpha, my_result)
            
            # 승리 확정 발견 시 즉시 반환
            if my_result == 1:
                return (1, depth, move)
            
            # Alpha-Beta Pruning
            if alpha >= beta:
                break
        
        return (best_result, depth, best_move)
    
    def clear_cache(self):
        """캐시 초기화"""
        if self.use_cache and self.tt:
            self.tt.clear()
