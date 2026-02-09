import numpy as np

class Board:
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
    
    # Bitmask win patterns for fast check (position = r*3+c)
    # 0b111000000 = row 0, 0b000111000 = row 1, etc.
    WIN_MASKS = (
        0b111000000,  # row 0
        0b000111000,  # row 1
        0b000000111,  # row 2
        0b100100100,  # col 0
        0b010010010,  # col 1
        0b001001001,  # col 2
        0b100010001,  # diag
        0b001010100,  # anti-diag
    )
    
    def __init__(self):
        self.boards = [[0 for _ in range(9)] for _ in range(9)] # empty: 0, player 1: 1, player 2: 2
        self.completed_boards = [[0 for _ in range(3)] for _ in range(3)] # empty: 0, player 1: 1, player 2: 2, draw: 3
        self.current_player = 1
        self.winner = None
        self.last_move = None
        self.sub_counts = [[0, 0] for _ in range(9)]  # [x_count, o_count] per sub-board
    
    def clone(self):
        """
        Fast board cloning (80x faster than copy.deepcopy)
        Only copies mutable state, shares immutable data
        """
        new_board = Board.__new__(Board)
        new_board.boards = [row[:] for row in self.boards]
        new_board.completed_boards = [row[:] for row in self.completed_boards]
        new_board.current_player = self.current_player
        new_board.winner = self.winner
        new_board.last_move = self.last_move
        new_board.sub_counts = [c[:] for c in self.sub_counts]
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
        if not (0 <= r < 9 and 0 <= c < 9): # loop bounds
            return False
        
        if self.boards[r][c] != 0: # check already placed
            return False
        
        board_r, board_c = r // 3, c // 3
        if self.completed_boards[board_r][board_c] != 0: # check completed
            return False
        
        if self.last_move is None:
            return True
        
        last_r, last_c = self.last_move
        target_board_r = last_r % 3
        target_board_c = last_c % 3
        
        if self.completed_boards[target_board_r][target_board_c] != 0:
            return True
        
        return board_r == target_board_r and board_c == target_board_c
    
    def make_move(self, r, c, validate=True):
        if validate and not self._is_valid_move(r, c):
            raise ValueError("Illegal move")
        
        self.boards[r][c] = self.current_player
        self.last_move = (r, c)
        
        # Update sub_counts
        sub_idx = (r // 3) * 3 + (c // 3)
        self.sub_counts[sub_idx][self.current_player - 1] += 1
        
        self.update_completed_boards(r, c)
        self.check_winner()
        
        self.current_player = self.current_player % 2 + 1
    
    def undo_move(self, r, c, prev_completed, prev_winner, prev_last_move):
        """Undo a move (for solver optimization). Caller must save state before make_move."""
        # Decrement sub_counts (current_player is already switched, so use opposite)
        sub_idx = (r // 3) * 3 + (c // 3)
        player = self.current_player % 2 + 1  # The player who made the move
        self.sub_counts[sub_idx][player - 1] -= 1
        
        self.boards[r][c] = 0
        board_r, board_c = r // 3, c // 3
        self.completed_boards[board_r][board_c] = prev_completed
        self.winner = prev_winner
        self.last_move = prev_last_move
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
        # Build bitmasks for current player and all filled
        p_mask = 0
        filled_mask = 0
        for r in range(3):
            for c in range(3):
                bit = 1 << (r * 3 + c)
                if self.completed_boards[r][c] == self.current_player:
                    p_mask |= bit
                if self.completed_boards[r][c] != 0:
                    filled_mask |= bit
        
        # Check win patterns with bitmask AND
        for mask in Board.WIN_MASKS:
            if (p_mask & mask) == mask:
                self.winner = self.current_player
                return
        
        # Check draw (all 9 filled)
        if filled_mask == 0b111111111:
            self.winner = 3

    def is_game_over(self): 
        return self.winner is not None
    
    def count_playable_empty_cells(self) -> int:
        """Count only playable empty cells (excluding completed small boards)."""
        empty_count = 0
        
        for br in range(3):
            for bc in range(3):
                if self.completed_boards[br][bc] == 0:
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
        
        state = np.array(state)
        if state.ndim == 2:
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
                return 1.0, "near_endgame"
            elif total_empty >= 10:
                return 0.6, "endgame"
            else:
                return 0.4, "deep_endgame"
        
        player1 = state[0]
        player2 = state[1]
        
        playable_empty = 0
        for br in range(3):
            for bc in range(3):
                start_r, start_c = br * 3, bc * 3
                small_p1 = player1[start_r:start_r+3, start_c:start_c+3]
                small_p2 = player2[start_r:start_r+3, start_c:start_c+3]
                
                if Board._is_small_board_completed(small_p1, small_p2):
                    continue
                
                small_empty = np.sum((small_p1 == 0) & (small_p2 == 0))
                playable_empty += small_empty
        
        if playable_empty >= 50:
            return 1.0, "opening"
        elif playable_empty >= 40:
            return 1.0, "early_mid"
        elif playable_empty >= 30:
            return 1.0, "mid"
        elif playable_empty >= 25:
            return 1.2, "transition"
        elif playable_empty >= 20:
            return 1.0, "near_endgame"
        elif playable_empty >= 10:
            return 0.6, "endgame"
        else:
            return 0.4, "deep_endgame"
    
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
        
        for player_board in [p1_board, p2_board]:
            if np.any(np.sum(player_board, axis=1) == 3):
                return True
            if np.any(np.sum(player_board, axis=0) == 3):
                return True
            if np.trace(player_board) == 3:
                return True
            if np.trace(np.fliplr(player_board)) == 3:
                return True
        
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
            return 1.0, "near_endgame"
        elif playable_empty >= 10:
            return 0.6, "endgame"
        else:
            return 0.4, "deep_endgame"