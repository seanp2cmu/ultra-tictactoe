class Board: 
    def __init__(self):
        self.boards = [[0 for _ in range(9)] for _ in range(9)] # empty: 0, player 1: 1, player 2: 2
        self.completed_boards = [[0 for _ in range(3)] for _ in range(3)] # empty: 0, player 1: 1, player 2: 2, draw: 3
        
        self.checker = [
            [(0,0), (0,1), (0,2)],
            [(1,0), (1,1), (1,2)],
            [(2,0), (2,1), (2,2)],
            [(0,0), (1,0), (2,0)],
            [(0,1), (1,1), (2,1)],
            [(0,2), (1,2), (2,2)],
            [(0,0), (1,1), (2,2)],
            [(0,2), (1,1), (2,0)]
        ]

        self.current_player = 1
        self.winner = None
        self.last_move = None

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
    
    def make_move(self, r, c):
        if (r, c) not in self.get_legal_moves():
            raise ValueError("Illegal move")
        
        self.boards[r][c] = self.current_player
        self.last_move = (r, c)
        
        self.update_completed_boards(r, c)
        self.check_winner()
        
        self.current_player = self.current_player % 2 + 1

    def update_completed_boards(self, r, c):
        board_r, board_c = r // 3, c // 3
        start_r, start_c = board_r * 3, board_c * 3
        
        for pattern in self.checker:
            if all(self.boards[start_r + pr][start_c + pc] == self.current_player for pr, pc in pattern):
                self.completed_boards[board_r][board_c] = self.current_player
                return
        
        if all(self.boards[start_r + pr][start_c + pc] != 0 for pr in range(3) for pc in range(3)):
            self.completed_boards[board_r][board_c] = 3

    def check_winner(self):
        for pattern in self.checker:
            if all(self.completed_boards[pr][pc] == self.current_player for pr, pc in pattern):
                self.winner = self.current_player
                return
        
        if all(self.completed_boards[r][c] != 0 for r in range(3) for c in range(3)):
            self.winner = 3

    def is_game_over(self): 
        return self.winner is not None
    
    def count_empty_cells(self) -> int:
        """Count total empty cells (including cells in completed small boards)."""
        return sum(1 for row in self.boards for cell in row if cell == 0)
    
    def count_playable_empty_cells(self) -> int:
        """Count only playable empty cells (excluding completed small boards)."""
        empty_count = 0
        for br in range(3):
            for bc in range(3):
                if self.completed_boards[br][bc] == 0:
                    for r in range(3):
                        for c in range(3):
                            if self.boards[br * 3 + r][bc * 3 + c] == 0:
                                empty_count += 1
        return empty_count