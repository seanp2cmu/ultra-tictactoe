"""
board symmetry transformation and normalization
Ultimate Tic-Tac-Toe has D4 symmetry group (8 symmetries)
"""
import numpy as np
from game import Board

class BoardSymmetry:
    """board symmetry transformation and normalization"""
    
    @staticmethod
    def rotate_90(board_2d):
        return np.rot90(board_2d, k=-1)
    
    @staticmethod
    def rotate_180(board_2d):
        return np.rot90(board_2d, k=2)
    
    @staticmethod
    def rotate_270(board_2d):
        return np.rot90(board_2d, k=1)
    
    @staticmethod
    def flip_horizontal(board_2d):
        return np.fliplr(board_2d)
    
    @staticmethod
    def flip_vertical(board_2d):
        return np.flipud(board_2d)
    
    @staticmethod
    def transpose(board_2d):
        return np.transpose(board_2d)
    
    @staticmethod
    def transpose_anti(board_2d):
        return np.rot90(np.transpose(board_2d), k=2)
    
    @staticmethod
    def get_all_symmetries(board: Board):
        """
        apply all symmetries
        
        Args:
            board: Board object
            
        Returns:
            list of (boards_2d, completed_2d) tuples (8개)

        """
        boards_arr = np.array(board.boards)
        completed_arr = np.array(board.completed_boards)
        
        return [
            (boards_arr, completed_arr),
            (np.fliplr(boards_arr), np.fliplr(completed_arr)),
            (np.flipud(boards_arr), np.flipud(completed_arr)),
            (np.rot90(boards_arr, k=2), np.rot90(completed_arr, k=2)),
            (np.rot90(boards_arr, k=-1), np.rot90(completed_arr, k=-1)),
            (np.rot90(boards_arr, k=1), np.rot90(completed_arr, k=1)),
            (boards_arr.T, completed_arr.T),
            (np.rot90(boards_arr.T, k=2), np.rot90(completed_arr.T, k=2))
        ]
    
    @staticmethod
    def to_bytes(boards_arr: np.ndarray, completed_arr: np.ndarray, current_player: int) -> bytes:
        return boards_arr.tobytes() + completed_arr.tobytes() + bytes([current_player])
    
    @staticmethod
    def get_canonical_bytes(board: Board) -> bytes:
        boards_arr = np.array(board.boards, dtype=np.int8)
        completed_arr = np.array(board.completed_boards, dtype=np.int8)
        player = board.current_player
        
        min_bytes = boards_arr.tobytes() + completed_arr.tobytes() + bytes([player])
        
        for transform in [
            lambda b, c: (np.fliplr(b), np.fliplr(c)),
            lambda b, c: (np.flipud(b), np.flipud(c)),
            lambda b, c: (np.rot90(b, 2), np.rot90(c, 2)),
            lambda b, c: (np.rot90(b, -1), np.rot90(c, -1)),
            lambda b, c: (np.rot90(b, 1), np.rot90(c, 1)),
            lambda b, c: (b.T, c.T),
            lambda b, c: (np.rot90(b.T, 2), np.rot90(c.T, 2)),
        ]:
            tb, tc = transform(boards_arr, completed_arr)
            b = np.ascontiguousarray(tb).tobytes() + np.ascontiguousarray(tc).tobytes() + bytes([player])
            if b < min_bytes:
                min_bytes = b
        
        return min_bytes
    
    @staticmethod
    def get_canonical_hash(board) -> bytes:
        return BoardSymmetry.get_canonical_bytes(board)
