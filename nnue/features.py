"""
NNUE Feature Extraction for Ultimate Tic-Tac-Toe

Features are extracted from the perspective of the current player.
This allows the network to be symmetric and reduces training complexity.
"""

import numpy as np
from game import Board


class NNUEFeatureExtractor:
    """
    Extract NNUE-style features from Ultimate Tic-Tac-Toe board.
    
    Feature layout (288 total):
    - Cells: 81 × 3 = 243 (one-hot: my_piece, opp_piece, empty)
    - Sub-boards: 9 × 4 = 36 (one-hot: my_win, opp_win, draw, open)
    - Active mask: 9 (which sub-boards are playable)
    
    All features are from current player's perspective.
    """
    
    # Feature dimensions
    CELL_FEATURES = 81 * 3      # 243
    SUBBOARD_FEATURES = 9 * 4   # 36
    ACTIVE_FEATURES = 9         # 9
    TOTAL_FEATURES = CELL_FEATURES + SUBBOARD_FEATURES + ACTIVE_FEATURES  # 288
    
    def __init__(self):
        pass
    
    def extract(self, board: Board) -> np.ndarray:
        """
        Extract feature vector from board state.
        
        Args:
            board: Ultimate Tic-Tac-Toe board
            
        Returns:
            np.ndarray of shape (288,) with float32 values (0 or 1)
        """
        features = np.zeros(self.TOTAL_FEATURES, dtype=np.float32)
        
        current_player = board.current_player
        opponent = 3 - current_player
        
        # Cell features (81 × 3 = 243)
        idx = 0
        for sub_board_idx in range(9):
            sub_row, sub_col = sub_board_idx // 3, sub_board_idx % 3
            for cell_idx in range(9):
                cell_row, cell_col = cell_idx // 3, cell_idx % 3
                
                # Get cell value
                row = sub_row * 3 + cell_row
                col = sub_col * 3 + cell_col
                cell_value = board.boards[row][col]
                
                # One-hot encoding: [my_piece, opp_piece, empty]
                if cell_value == current_player:
                    features[idx] = 1.0  # my piece
                elif cell_value == opponent:
                    features[idx + 1] = 1.0  # opponent piece
                else:
                    features[idx + 2] = 1.0  # empty
                
                idx += 3
        
        # Sub-board features (9 × 4 = 36)
        for sub_row in range(3):
            for sub_col in range(3):
                sub_board_state = board.completed_boards[sub_row][sub_col]
                
                # One-hot encoding: [my_win, opp_win, draw, open]
                if sub_board_state == current_player:
                    features[idx] = 1.0  # I won this sub-board
                elif sub_board_state == opponent:
                    features[idx + 1] = 1.0  # Opponent won
                elif sub_board_state == 3:
                    features[idx + 2] = 1.0  # Draw
                else:
                    features[idx + 3] = 1.0  # Still open
                
                idx += 4
        
        # Active sub-board mask (9)
        legal_moves = board.get_legal_moves()
        active_subboards = set()
        for row, col in legal_moves:
            sub_row, sub_col = row // 3, col // 3
            active_subboards.add(sub_row * 3 + sub_col)
        
        for i in range(9):
            if i in active_subboards:
                features[idx + i] = 1.0
        
        return features
    
    def extract_batch(self, boards: list) -> np.ndarray:
        """
        Extract features for multiple boards.
        
        Args:
            boards: List of Board objects
            
        Returns:
            np.ndarray of shape (batch_size, 288)
        """
        batch_size = len(boards)
        features = np.zeros((batch_size, self.TOTAL_FEATURES), dtype=np.float32)
        
        for i, board in enumerate(boards):
            features[i] = self.extract(board)
        
        return features
    
    def get_feature_diff(self, old_features: np.ndarray, board: Board, 
                         move: tuple, old_player: int) -> tuple:
        """
        Compute feature difference for incremental update.
        
        This is the key to NNUE efficiency - instead of recomputing all features,
        we only compute what changed after a move.
        
        Args:
            old_features: Previous feature vector (288,)
            board: Board AFTER the move was made
            move: (row, col) of the move that was made
            old_player: Player who made the move (before board.current_player switched)
            
        Returns:
            Tuple of (added_indices, removed_indices) for sparse update
        """
        row, col = move
        sub_row, sub_col = row // 3, col // 3
        cell_row, cell_col = row % 3, col % 3
        
        added = []
        removed = []
        
        # The perspective has flipped (current_player changed)
        # So we need to recompute from new player's perspective
        # For now, return full recompute signal
        # TODO: Implement true incremental update
        
        return added, removed
