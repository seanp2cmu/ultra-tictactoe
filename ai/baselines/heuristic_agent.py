"""
Heuristic Greedy Agent - evaluates each move and picks the best.
No search, just one-step lookahead with UTTT-specific heuristics.
"""
from typing import List, Tuple
from game import Board


class HeuristicAgent:
    """Agent that uses heuristics to evaluate and select moves."""
    
    def __init__(self):
        self.name = "Heuristic"
        
        # Weights for different heuristic components
        self.weights = {
            'win_local': 1000,      # Win a local board
            'block_local': 800,     # Block opponent from winning local board
            'win_meta': 10000,      # Win the game (meta board)
            'block_meta': 8000,     # Block opponent from winning meta
            'two_in_row_local': 50, # Create 2-in-a-row on local board
            'two_in_row_meta': 200, # Create 2-in-a-row on meta board
            'center_local': 10,     # Take center of local board
            'center_meta': 30,      # Play in center local board
            'corner_local': 5,      # Take corner of local board
            'bad_send': -100,       # Sending opponent to good board
        }
    
    def select_action(self, board: Board, temperature: float = None) -> int:
        """Select best move based on heuristic evaluation.
        
        Args:
            board: Current board state
            temperature: Ignored (for API compatibility)
            
        Returns:
            action: Move as action index (row * 9 + col)
        """
        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            return 0
        
        best_score = float('-inf')
        best_move = legal_moves[0]
        
        for move in legal_moves:
            score = self._evaluate_move(board, move)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move[0] * 9 + best_move[1]
    
    def _evaluate_move(self, board: Board, move: Tuple[int, int]) -> float:
        """Evaluate a move using heuristics."""
        row, col = move
        local_row, local_col = row % 3, col % 3
        meta_row, meta_col = row // 3, col // 3
        player = board.current_player
        opponent = 3 - player
        
        score = 0.0
        
        # Get the local board
        local_board_idx = meta_row * 3 + meta_col
        local_board = board.get_sub_board(local_board_idx)
        
        # 1. Check if this move wins the local board
        if self._would_win_local(local_board, local_row, local_col, player):
            score += self.weights['win_local']
            
            # Check if winning this local board wins the game
            if self._would_win_meta(board.completed_boards, meta_row, meta_col, player):
                score += self.weights['win_meta']
        
        # 2. Check if this move blocks opponent from winning local board
        if self._would_win_local(local_board, local_row, local_col, opponent):
            score += self.weights['block_local']
        
        # 3. Check 2-in-a-row creation on local board
        score += self._count_two_in_row(local_board, local_row, local_col, player) * self.weights['two_in_row_local']
        
        # 4. Center and corner bonuses
        if local_row == 1 and local_col == 1:
            score += self.weights['center_local']
        if meta_row == 1 and meta_col == 1:
            score += self.weights['center_meta']
        if local_row in [0, 2] and local_col in [0, 2]:
            score += self.weights['corner_local']
        
        # 5. Evaluate where we send the opponent
        send_row, send_col = local_row, local_col
        if board.completed_boards[send_row][send_col] == 0:
            # Check if we're sending opponent to a board where they can win
            send_board_idx = send_row * 3 + send_col
            send_board = board.get_sub_board(send_board_idx)
            
            # Count opponent's winning opportunities in that board
            opp_wins = self._count_winning_opportunities(send_board, opponent)
            score += opp_wins * self.weights['bad_send']
        
        # 6. Check if winning this local board creates 2-in-a-row on meta
        if self._would_win_local(local_board, local_row, local_col, player):
            score += self._count_two_in_row_meta(board.completed_boards, meta_row, meta_col, player) * self.weights['two_in_row_meta']
        
        return score
    
    def _would_win_local(self, local_board: List[int], row: int, col: int, player: int) -> bool:
        """Check if placing at (row, col) would win the local board."""
        # Simulate the move
        idx = row * 3 + col
        if local_board[idx] != 0:
            return False
        
        # Check row
        r_start = row * 3
        row_cells = [local_board[r_start], local_board[r_start + 1], local_board[r_start + 2]]
        row_cells[col] = player
        if row_cells.count(player) == 3:
            return True
        
        # Check column
        col_cells = [local_board[col], local_board[3 + col], local_board[6 + col]]
        col_cells[row] = player
        if col_cells.count(player) == 3:
            return True
        
        # Check diagonals
        if row == col:
            diag = [local_board[0], local_board[4], local_board[8]]
            diag[row] = player
            if diag.count(player) == 3:
                return True
        
        if row + col == 2:
            anti_diag = [local_board[2], local_board[4], local_board[6]]
            anti_diag[row] = player
            if anti_diag.count(player) == 3:
                return True
        
        return False
    
    def _would_win_meta(self, completed: List[List[int]], row: int, col: int, player: int) -> bool:
        """Check if winning local board at (row, col) would win the meta board."""
        # Simulate winning the local board
        test_completed = [r[:] for r in completed]
        test_completed[row][col] = player
        
        # Check row
        if test_completed[row].count(player) == 3:
            return True
        
        # Check column
        if sum(1 for r in range(3) if test_completed[r][col] == player) == 3:
            return True
        
        # Check diagonals
        if row == col:
            if sum(1 for i in range(3) if test_completed[i][i] == player) == 3:
                return True
        
        if row + col == 2:
            if sum(1 for i in range(3) if test_completed[i][2-i] == player) == 3:
                return True
        
        return False
    
    def _count_two_in_row(self, local_board: List[int], row: int, col: int, player: int) -> int:
        """Count how many 2-in-a-row this move creates."""
        count = 0
        idx = row * 3 + col
        
        # Simulate move
        test_board = local_board[:]
        test_board[idx] = player
        
        # Check all lines through this cell
        lines = []
        
        # Row
        r_start = row * 3
        lines.append([test_board[r_start], test_board[r_start + 1], test_board[r_start + 2]])
        
        # Column
        lines.append([test_board[col], test_board[3 + col], test_board[6 + col]])
        
        # Diagonals
        if row == col:
            lines.append([test_board[0], test_board[4], test_board[8]])
        if row + col == 2:
            lines.append([test_board[2], test_board[4], test_board[6]])
        
        for line in lines:
            if line.count(player) == 2 and line.count(0) == 1:
                count += 1
        
        return count
    
    def _count_two_in_row_meta(self, completed: List[List[int]], row: int, col: int, player: int) -> int:
        """Count 2-in-a-row on meta board after winning local board at (row, col)."""
        count = 0
        
        # Simulate winning
        test = [r[:] for r in completed]
        test[row][col] = player
        
        # Check row
        row_line = test[row]
        if row_line.count(player) == 2 and row_line.count(0) == 1:
            count += 1
        
        # Check column
        col_line = [test[r][col] for r in range(3)]
        if col_line.count(player) == 2 and col_line.count(0) == 1:
            count += 1
        
        # Diagonals
        if row == col:
            diag = [test[i][i] for i in range(3)]
            if diag.count(player) == 2 and diag.count(0) == 1:
                count += 1
        
        if row + col == 2:
            anti = [test[i][2-i] for i in range(3)]
            if anti.count(player) == 2 and anti.count(0) == 1:
                count += 1
        
        return count
    
    def _count_winning_opportunities(self, local_board: List[int], player: int) -> int:
        """Count winning opportunities for player in a local board."""
        count = 0
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Cols
            [0, 4, 8], [2, 4, 6]              # Diags
        ]
        
        for line in lines:
            cells = [local_board[i] for i in line]
            # Winning opportunity: 2 of player's pieces and 1 empty
            if cells.count(player) == 2 and cells.count(0) == 1:
                count += 1
        
        return count
