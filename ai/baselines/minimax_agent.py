"""
Minimax Agent with Alpha-Beta Pruning.
Depth-limited search with heuristic evaluation.
"""
from typing import List
from game import Board


def _get_completed_2d(board):
    """Get completed_boards as 2D list (BoardCy compatible)."""
    if hasattr(board, 'get_completed_boards_2d'):
        return board.get_completed_boards_2d()
    return board.completed_boards


class MinimaxAgent:
    """Agent using minimax search with alpha-beta pruning."""
    
    def __init__(self, depth: int = 3):
        """
        Args:
            depth: Maximum search depth (2-4 recommended)
        """
        self.name = f"Minimax-{depth}"
        self.depth = depth
        self.nodes_searched = 0
    
    def select_action(self, board: Board, temperature: float = None) -> int:
        """Select best move using minimax search.
        
        Args:
            board: Current board state
            temperature: Ignored (for API compatibility)
            
        Returns:
            action: Move as action index (row * 9 + col)
        """
        self.nodes_searched = 0
        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            return 0
        
        best_score = float('-inf')
        best_move = legal_moves[0]
        alpha = float('-inf')
        beta = float('inf')
        
        for move in legal_moves:
            # Make move on copy
            new_board = self._copy_board(board)
            new_board.make_move(move[0], move[1])
            
            # Minimax with alpha-beta
            score = self._minimax(new_board, self.depth - 1, alpha, beta, False, board.current_player)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_move[0] * 9 + best_move[1]
    
    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, 
                 maximizing: bool, original_player: int) -> float:
        """Minimax with alpha-beta pruning."""
        self.nodes_searched += 1
        
        # Terminal or depth limit (BoardCy uses -1 for no winner)
        if board.winner is not None and board.winner != -1:
            if board.winner == original_player:
                return 10000 + depth  # Win (prefer faster wins)
            elif board.winner == 3:
                return 0  # Draw
            else:
                return -10000 - depth  # Loss (prefer slower losses)
        
        if depth == 0:
            return self._evaluate(board, original_player)
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return 0
        
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                new_board = self._copy_board(board)
                new_board.make_move(move[0], move[1])
                eval_score = self._minimax(new_board, depth - 1, alpha, beta, False, original_player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_board = self._copy_board(board)
                new_board.make_move(move[0], move[1])
                eval_score = self._minimax(new_board, depth - 1, alpha, beta, True, original_player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def _copy_board(self, board: Board) -> Board:
        """Create a deep copy of the board."""
        return board.clone()
    
    def _evaluate(self, board: Board, player: int) -> float:
        """Evaluate board position for player using heuristics."""
        opponent = 3 - player
        score = 0.0
        
        # 1. Count won local boards
        completed = _get_completed_2d(board)
        player_boards = 0
        opponent_boards = 0
        for row in completed:
            for cell in row:
                if cell == player:
                    player_boards += 1
                elif cell == opponent:
                    opponent_boards += 1
        
        score += (player_boards - opponent_boards) * 100
        
        # 2. Meta board 2-in-a-row
        score += self._count_meta_threats(completed, player) * 50
        score -= self._count_meta_threats(completed, opponent) * 50
        
        # 3. Local board control (pieces on uncompleted boards)
        for i in range(9):
            meta_row, meta_col = i // 3, i % 3
            if completed[meta_row][meta_col] == 0:
                local_board = board.get_sub_board(i)
                
                # Count pieces
                player_pieces = local_board.count(player)
                opponent_pieces = local_board.count(opponent)
                score += (player_pieces - opponent_pieces) * 2
                
                # 2-in-a-row on local boards
                score += self._count_local_threats(local_board, player) * 10
                score -= self._count_local_threats(local_board, opponent) * 10
                
                # Center control
                if local_board[4] == player:
                    score += 5
                elif local_board[4] == opponent:
                    score -= 5
        
        # 4. Center meta board bonus
        if completed[1][1] == player:
            score += 30
        elif completed[1][1] == opponent:
            score -= 30
        
        return score
    
    def _count_meta_threats(self, completed: List[List[int]], player: int) -> int:
        """Count 2-in-a-row threats on meta board."""
        count = 0
        lines = [
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
        ]
        
        for line in lines:
            cells = [completed[r][c] for r, c in line]
            if cells.count(player) == 2 and cells.count(0) == 1:
                count += 1
        
        return count
    
    def _count_local_threats(self, local_board: List[int], player: int) -> int:
        """Count 2-in-a-row threats on a local board."""
        count = 0
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for line in lines:
            cells = [local_board[i] for i in line]
            if cells.count(player) == 2 and cells.count(0) == 1:
                count += 1
        
        return count
