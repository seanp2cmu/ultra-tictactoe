import numpy as np

class Board:
    # 클래스 변수: 모든 인스턴스가 공유 (메모리 절약)
    CHECKER = (
        ((0,0), (0,1), (0,2)),
        ((1,0), (1,1), (1,2)),
        ((2,0), (2,1), (2,2)),
        ((0,0), (1,0), (2,0)),
        ((0,1), (1,1), (2,1)),
        ((0,2), (1,2), (2,2)),
        ((0,0), (1,1), (2,2)),
        ((0,2), (1,1), (2,0))
    )
    
    def __init__(self):
        self.boards = [[0 for _ in range(9)] for _ in range(9)] # empty: 0, player 1: 1, player 2: 2
        self.completed_boards = [[0 for _ in range(3)] for _ in range(3)] # empty: 0, player 1: 1, player 2: 2, draw: 3
        self.current_player = 1
        self.winner = None
        self.last_move = None
    
    def clone(self):
        """
        Fast board cloning (80x faster than copy.deepcopy)
        Only copies mutable state, shares immutable data
        """
        new_board = Board.__new__(Board)
        new_board.boards = [row[:] for row in self.boards]
        new_board.completed_boards = [row[:] for row in self.completed_boards]
        # CHECKER는 클래스 변수이므로 복사 불필요
        new_board.current_player = self.current_player
        new_board.winner = self.winner
        new_board.last_move = self.last_move
        return new_board

    def get_legal_moves(self):
        legal_moves = []
        
        if self.last_move is None:
            for r in range(9):
                for c in range(9):
                    board_r, board_c = r // 3, c // 3
                    if self.boards[r][c] == 0 and self.completed_boards[board_r][board_c] == 0:
                        legal_moves.append((r, c))
        else:
            last_r, last_c = self.last_move
            target_board_r = last_r % 3
            target_board_c = last_c % 3
            
            if self.completed_boards[target_board_r][target_board_c] == 0:
                start_r, start_c = target_board_r * 3, target_board_c * 3
                for r in range(start_r, start_r + 3):
                    for c in range(start_c, start_c + 3):
                        if self.boards[r][c] == 0:
                            legal_moves.append((r, c))
            else:
                for r in range(9):
                    for c in range(9):
                        board_r, board_c = r // 3, c // 3
                        if self.boards[r][c] == 0 and self.completed_boards[board_r][board_c] == 0:
                            legal_moves.append((r, c))
        return legal_moves
    
    def _is_valid_move(self, r, c) -> bool:
        """Fast validation without computing all legal moves. O(1) instead of O(81)."""
        # 범위 체크
        if not (0 <= r < 9 and 0 <= c < 9):
            return False
        
        # 이미 놓인 칸인지
        if self.boards[r][c] != 0:
            return False
        
        # 해당 소보드가 완료됐는지
        board_r, board_c = r // 3, c // 3
        if self.completed_boards[board_r][board_c] != 0:
            return False
        
        # 첫 수는 어디든 가능
        if self.last_move is None:
            return True
        
        # 지정된 소보드인지 확인
        last_r, last_c = self.last_move
        target_board_r = last_r % 3
        target_board_c = last_c % 3
        
        # 지정된 소보드가 완료됐으면 어디든 가능
        if self.completed_boards[target_board_r][target_board_c] != 0:
            return True
        
        # 지정된 소보드에만 둘 수 있음
        return board_r == target_board_r and board_c == target_board_c
    
    def make_move(self, r, c, validate=True):
        if validate and not self._is_valid_move(r, c):
            raise ValueError("Illegal move")
        
        self.boards[r][c] = self.current_player
        self.last_move = (r, c)
        
        self.update_completed_boards(r, c)
        self.check_winner()
        
        self.current_player = self.current_player % 2 + 1

    def update_completed_boards(self, r, c):
        board_r, board_c = r // 3, c // 3
        start_r, start_c = board_r * 3, board_c * 3
        
        for pattern in Board.CHECKER:
            if all(self.boards[start_r + pr][start_c + pc] == self.current_player for pr, pc in pattern):
                self.completed_boards[board_r][board_c] = self.current_player
                return
        
        if all(self.boards[start_r + pr][start_c + pc] != 0 for pr in range(3) for pc in range(3)):
            self.completed_boards[board_r][board_c] = 3

    def check_winner(self):
        for pattern in Board.CHECKER:
            if all(self.completed_boards[pr][pc] == self.current_player for pr, pc in pattern):
                self.winner = self.current_player
                return
        
        if all(self.completed_boards[r][c] != 0 for r in range(3) for c in range(3)):
            self.winner = 3

    def is_game_over(self): 
        return self.winner is not None
    
    def count_playable_empty_cells(self) -> int:
        """Count only playable empty cells (excluding completed small boards)."""
        empty_count = 0
        
        for br in range(3):
            for bc in range(3):
                if self.completed_boards[br][bc] == 0:
                    # Count zeros in 3x3 small board (pure Python, no numpy)
                    start_r, start_c = br * 3, bc * 3
                    for r in range(start_r, start_r + 3):
                        for c in range(start_c, start_c + 3):
                            if self.boards[r][c] == 0:
                                empty_count += 1
        
        return empty_count
    
    @staticmethod
    def get_phase_from_state(state) -> tuple[float, str]:
        """
        Get game phase from state array using playable empty cells.
        
        Args:
            state: numpy array (2, 9, 9) or (C, 9, 9) where [player1, player2, ...]
                   Can also be any object with indexable [0] and [1]
        
        Returns:
            (weight, category):
            - weight: Training weight (0.3-1.2)
            - category: Phase name
        
        Uses playable empty cells (excluding completed boards).
        """
        
        # Handle different state formats
        state = np.array(state)
        if state.ndim == 2:
            # Single plane (9, 9) - treat as all playable, count empty
            total_empty = np.sum(state == 0)
            if total_empty >= 50:
                return 1.0, "opening"
            elif total_empty >= 40:
                return 1.0, "early_mid"
            elif total_empty >= 30:
                return 1.0, "mid"
            elif total_empty >= 25:
                return 1.2, "transition"
            elif total_empty >= 20:
                return 0.8, "near_endgame"
            elif total_empty >= 10:
                return 0.5, "endgame"
            else:
                return 0.3, "deep_endgame"
        
        # Multi-channel state (C, 9, 9)
        player1 = state[0]
        player2 = state[1]
        
        # Count playable empty cells (excluding completed small boards)
        playable_empty = 0
        for br in range(3):
            for bc in range(3):
                # Extract 3x3 small board
                start_r, start_c = br * 3, bc * 3
                small_p1 = player1[start_r:start_r+3, start_c:start_c+3]
                small_p2 = player2[start_r:start_r+3, start_c:start_c+3]
                
                # Check if small board is completed
                if Board._is_small_board_completed(small_p1, small_p2):
                    continue  # Skip completed boards
                
                # Count empty cells in this small board
                small_empty = np.sum((small_p1 == 0) & (small_p2 == 0))
                playable_empty += small_empty
        
        # Phase classification based on playable empty cells
        if playable_empty >= 50:
            return 1.0, "opening"
        elif playable_empty >= 40:
            return 1.0, "early_mid"
        elif playable_empty >= 30:
            return 1.0, "mid"
        elif playable_empty >= 25:
            return 1.2, "transition"
        elif playable_empty >= 20:
            return 0.8, "near_endgame"
        elif playable_empty >= 10:
            return 0.5, "endgame"
        else:
            return 0.3, "deep_endgame"
    
    @staticmethod
    def _is_small_board_completed(p1_board, p2_board):
        """
        Check if a 3x3 small board is completed (won or full).
        
        Args:
            p1_board: (3, 3) player 1 positions
            p2_board: (3, 3) player 2 positions
        
        Returns:
            bool: True if completed
        """
        
        # Check if someone won (vectorized)
        for player_board in [p1_board, p2_board]:
            # Check rows (vectorized)
            if np.any(np.sum(player_board, axis=1) == 3):
                return True
            # Check columns (vectorized)
            if np.any(np.sum(player_board, axis=0) == 3):
                return True
            # Check diagonals
            if np.trace(player_board) == 3:  # main diagonal
                return True
            if np.trace(np.fliplr(player_board)) == 3:  # anti-diagonal
                return True
        
        # Check if full (no empty cells)
        if np.sum((p1_board == 0) & (p2_board == 0)) == 0:
            return True
        
        return False
    
    def get_phase(self) -> tuple[float, str]:
        """
        Get game phase weight and category for this board instance.
        
        Returns:
            (weight, category):
            - weight: Training weight (0.3-1.2)
            - category: Phase name
        
        Categories:
        - opening: 50+ empty cells
        - early_mid: 40-49 empty
        - mid: 30-39 empty
        - transition: 26-29 empty (most important!)
        - near_endgame: 20-25 empty
        - endgame: 10-19 empty
        - deep_endgame: 0-9 empty
        """
        # 최적화: numpy 변환 없이 직접 빈 칸 수 계산
        playable_empty = self.count_playable_empty_cells()
        
        if playable_empty >= 50:
            return 1.0, "opening"
        elif playable_empty >= 40:
            return 1.0, "early_mid"
        elif playable_empty >= 30:
            return 1.0, "mid"
        elif playable_empty >= 25:
            return 1.2, "transition"
        elif playable_empty >= 20:
            return 0.8, "near_endgame"
        elif playable_empty >= 10:
            return 0.5, "endgame"
        else:
            return 0.3, "deep_endgame"