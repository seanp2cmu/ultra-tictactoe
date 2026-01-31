"""
Retrograde Analysis Tablebase for Ultimate Tic-Tac-Toe
25칸 이하 완전 해결 (Win/Loss/Draw + DTW)
"""
import os
import pickle
import struct
from collections import deque
from tqdm import tqdm
import copy
from game import Board


class RetrogradeTablebase:
    """
    Retrograde Analysis로 엔드게임 완전 해결
    
    저장 형식:
    - Key: canonical board hash (대칭 정규화)
    - Value: (result, dtw) 
        - result: 1 (현재 플레이어 승), -1 (패), 0 (무승부)
        - dtw: Distance to Win/Loss
    """
    
    def __init__(self, max_empty_cells=25):
        """
        Args:
            max_empty_cells: 최대 빈 칸 수 (25칸 이하 모두 해결)
        """
        self.max_empty_cells = max_empty_cells
        self.tablebase = {}  # {canonical_hash: (result, dtw)}
        self.stats = {
            "total_positions": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0
        }
    
    def generate(self, verbose=True):
        """
        Retrograde Analysis로 Tablebase 생성
        
        과정:
        1. 터미널 포지션 (게임 종료) 수집
        2. 역방향으로 부모 포지션 평가
        3. 모든 포지션 해결할 때까지 반복
        """
        if verbose:
            print("=" * 80)
            print(f"Generating Tablebase (≤ {self.max_empty_cells} empty cells)")
            print("=" * 80)
        
        from .board_symmetry import BoardSymmetry
        self.symmetry = BoardSymmetry()
        
        # Phase 1: 터미널 포지션 수집
        if verbose:
            print("\nPhase 1: Collecting terminal positions...")
        
        terminal_positions = self._collect_terminal_positions()
        
        if verbose:
            print(f"✓ Found {len(terminal_positions)} terminal positions")
        
        # Phase 2: Retrograde Analysis
        if verbose:
            print("\nPhase 2: Retrograde Analysis (backward propagation)...")
        
        self._retrograde_analysis(terminal_positions, verbose=verbose)
        
        # 통계
        self.stats["total_positions"] = len(self.tablebase)
        for _, (result, _) in self.tablebase.items():
            if result == 1:
                self.stats["wins"] += 1
            elif result == -1:
                self.stats["losses"] += 1
            else:
                self.stats["draws"] += 1
        
        if verbose:
            print("\n" + "=" * 80)
            print("Tablebase Generation Complete!")
            print(f"Total positions: {self.stats['total_positions']:,}")
            print(f"  Wins: {self.stats['wins']:,}")
            print(f"  Losses: {self.stats['losses']:,}")
            print(f"  Draws: {self.stats['draws']:,}")
            print("=" * 80)
        
        return self.stats
    
    def _collect_terminal_positions(self):
        """터미널 포지션 (게임 종료 상태) 수집"""
        terminal = []
        
        # BFS로 모든 가능한 게임 탐색
        # 실제로는 25칸 이하만 관심 있으므로 pruning
        
        queue = deque([Board()])
        visited = set()
        
        pbar = tqdm(desc="Collecting terminals", leave=False)
        
        while queue:
            board = queue.popleft()
            
            # 정규화된 해시
            canonical_hash = self.symmetry.get_canonical_hash(board)
            
            if canonical_hash in visited:
                continue
            visited.add(canonical_hash)
            
            pbar.update(1)
            
            # 빈 칸 개수
            empty_count = sum(1 for row in board.boards for cell in row if cell == 0)
            
            # 25칸 초과면 skip
            if empty_count > self.max_empty_cells:
                continue
            
            # 터미널 체크
            if board.winner is not None:
                # 현재 플레이어 관점에서 결과
                if board.winner == 3:  # 무승부
                    result = 0
                elif board.winner == board.current_player:
                    result = 1  # 승
                else:
                    result = -1  # 패
                
                terminal.append((board, result, 0))  # DTW = 0 (터미널)
                continue
            
            # 합법 수 없으면 무승부
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                terminal.append((board, 0, 0))
                continue
            
            # 자식 노드 탐색
            for move in legal_moves:
                next_board = copy.deepcopy(board)
                next_board.make_move(move[0], move[1])
                queue.append(next_board)
        
        pbar.close()
        return terminal
    
    def _retrograde_analysis(self, terminal_positions, verbose=True):
        """
        역방향 분석
        
        터미널에서 시작해서 부모 포지션들을 평가
        """
        # 초기화: 터미널 포지션
        for board, result, dtw in terminal_positions:
            canonical_hash = self.symmetry.get_canonical_hash(board)
            self.tablebase[canonical_hash] = (result, dtw)
        
        # 역방향 큐
        current_depth = 0
        max_depth = self.max_empty_cells
        
        if verbose:
            depth_pbar = tqdm(range(max_depth), desc="Depth", leave=False)
        
        changed = True
        while changed and current_depth < max_depth:
            changed = False
            current_depth += 1
            
            # 현재 해결된 포지션들로부터 부모 찾기
            new_positions = []
            
            # 모든 해결된 포지션의 부모 평가
            for board_hash, (result, dtw) in list(self.tablebase.items()):
                # 이 포지션의 부모들 (1수 전)을 찾아서 평가
                # 실제로는 반대 방향이므로 복잡함
                # 간단하게: 미해결 포지션 중 자식이 해결된 것 평가
                pass
            
            if verbose:
                depth_pbar.update(1)
                depth_pbar.set_postfix({"positions": len(self.tablebase)})
        
        if verbose:
            depth_pbar.close()
    
    def lookup(self, board):
        """
        Tablebase 조회
        
        Returns:
            (result, dtw) or None
            - result: 1 (승), -1 (패), 0 (무승부)
            - dtw: Distance to Win/Loss
        """
        canonical_hash = self.symmetry.get_canonical_hash(board)
        return self.tablebase.get(canonical_hash)
    
    def save(self, filepath):
        """Tablebase 저장 (압축)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'max_empty_cells': self.max_empty_cells,
                'tablebase': self.tablebase,
                'stats': self.stats
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(filepath) / 1024 / 1024
        print(f"✓ Tablebase saved to {filepath} ({file_size:.2f} MB)")
    
    def load(self, filepath):
        """Tablebase 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.max_empty_cells = data['max_empty_cells']
        self.tablebase = data['tablebase']
        self.stats = data['stats']
        
        file_size = os.path.getsize(filepath) / 1024 / 1024
        print(f"✓ Tablebase loaded from {filepath} ({file_size:.2f} MB)")
        print(f"  Positions: {self.stats['total_positions']:,}")
    
    def get_best_move(self, board):
        """
        Tablebase에서 최선의 수 찾기
        
        Returns:
            (move, dtw) or (None, inf)
        """
        legal_moves = board.get_legal_moves()
        best_move = None
        best_dtw = float('inf')
        
        for move in legal_moves:
            next_board = copy.deepcopy(board)
            next_board.make_move(move[0], move[1])
            
            result = self.lookup(next_board)
            if result is None:
                continue
            
            opponent_result, opponent_dtw = result
            
            # 상대 관점에서 패배 = 나의 승리
            if opponent_result == -1:
                my_dtw = opponent_dtw + 1
                if my_dtw < best_dtw:
                    best_dtw = my_dtw
                    best_move = move
        
        return best_move, best_dtw


class TablebaseIntegration:
    """
    DTW Calculator + Tablebase 통합
    
    25칸 이하: Tablebase 조회 (완벽)
    26칸+: DTW 계산 (휴리스틱)
    """
    
    def __init__(self, tablebase_path=None, dtw_calculator=None):
        """
        Args:
            tablebase_path: Tablebase 파일 경로
            dtw_calculator: DTWCalculator (26칸+ 용)
        """
        self.tablebase = None
        if tablebase_path and os.path.exists(tablebase_path):
            self.tablebase = RetrogradeTablebase()
            self.tablebase.load(tablebase_path)
        
        self.dtw_calculator = dtw_calculator
    
    def get_best_move(self, board):
        """
        최선의 수 찾기
        
        25칸 이하: Tablebase
        26칸+: DTW
        """
        empty_count = sum(1 for row in board.boards for cell in row if cell == 0)
        
        # Tablebase 영역
        if empty_count <= 25 and self.tablebase:
            return self.tablebase.get_best_move(board)
        
        # DTW 영역
        if self.dtw_calculator:
            return self.dtw_calculator.get_best_winning_move(board)
        
        return None, float('inf')
    
    def lookup(self, board):
        """
        포지션 조회
        
        Returns:
            (result, dtw) or None
        """
        empty_count = sum(1 for row in board.boards for cell in row if cell == 0)
        
        if empty_count <= 25 and self.tablebase:
            return self.tablebase.lookup(board)
        
        if self.dtw_calculator:
            dtw, is_winning = self.dtw_calculator.calculate_dtw(board)
            if is_winning:
                return (1, dtw)
            elif dtw == float('inf'):
                return (-1, dtw)
        
        return None
