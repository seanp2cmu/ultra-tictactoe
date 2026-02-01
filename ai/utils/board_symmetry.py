"""
보드 대칭 변환 및 정규화
Ultimate Tic-Tac-Toe는 D4 대칭군을 가짐 (8가지 대칭)
"""
import numpy as np
from game import Board

class BoardSymmetry:
    """보드 대칭 변환 및 정규화"""
    
    @staticmethod
    def rotate_90(board_2d):
        """시계방향 90도 회전"""
        return np.rot90(board_2d, k=-1)
    
    @staticmethod
    def rotate_180(board_2d):
        """180도 회전"""
        return np.rot90(board_2d, k=2)
    
    @staticmethod
    def rotate_270(board_2d):
        """시계방향 270도 회전"""
        return np.rot90(board_2d, k=1)
    
    @staticmethod
    def flip_horizontal(board_2d):
        """좌우 반전"""
        return np.fliplr(board_2d)
    
    @staticmethod
    def flip_vertical(board_2d):
        """상하 반전"""
        return np.flipud(board_2d)
    
    @staticmethod
    def transpose(board_2d):
        """대각선 반전 (전치)"""
        return np.transpose(board_2d)
    
    @staticmethod
    def transpose_anti(board_2d):
        """역대각선 반전"""
        return np.rot90(np.transpose(board_2d), k=2)
    
    @staticmethod
    def get_all_symmetries(board: Board):
        """
        모든 대칭 변환 적용
        
        Args:
            board: Board 객체
            
        Returns:
            list of (boards_2d, completed_2d) tuples (8개)
        """
        boards_arr = np.array(board.boards)
        completed_arr = np.array(board.completed_boards)
        
        symmetries = []
        
        # 1. 원본
        symmetries.append((boards_arr.copy(), completed_arr.copy()))
        
        # 2. 좌우 반전
        symmetries.append((
            BoardSymmetry.flip_horizontal(boards_arr),
            BoardSymmetry.flip_horizontal(completed_arr)
        ))
        
        # 3. 상하 반전
        symmetries.append((
            BoardSymmetry.flip_vertical(boards_arr),
            BoardSymmetry.flip_vertical(completed_arr)
        ))
        
        # 4. 180도 회전
        symmetries.append((
            BoardSymmetry.rotate_180(boards_arr),
            BoardSymmetry.rotate_180(completed_arr)
        ))
        
        # 5. 90도 회전
        symmetries.append((
            BoardSymmetry.rotate_90(boards_arr),
            BoardSymmetry.rotate_90(completed_arr)
        ))
        
        # 6. 270도 회전
        symmetries.append((
            BoardSymmetry.rotate_270(boards_arr),
            BoardSymmetry.rotate_270(completed_arr)
        ))
        
        # 7. 대각선 반전
        symmetries.append((
            BoardSymmetry.transpose(boards_arr),
            BoardSymmetry.transpose(completed_arr)
        ))
        
        # 8. 역대각선 반전
        symmetries.append((
            BoardSymmetry.transpose_anti(boards_arr),
            BoardSymmetry.transpose_anti(completed_arr)
        ))
        
        return symmetries
    
    @staticmethod
    def to_tuple(boards_arr: np.ndarray, completed_arr: np.ndarray, current_player):
        """numpy array를 hashable tuple로 변환"""
        boards_tuple = tuple(tuple(row) for row in boards_arr.flatten().reshape(9, 9))
        completed_tuple = tuple(tuple(row) for row in completed_arr)
        return (boards_tuple, completed_tuple, current_player)
    
    @staticmethod
    def get_canonical_form(board: Board):
        """
        정규화된 형태 (canonical form) 반환
        8가지 대칭 중 사전순으로 가장 작은 것
        
        Args:
            board: Board 객체
            
        Returns:
            tuple: 정규화된 보드 상태
        """
        symmetries = BoardSymmetry.get_all_symmetries(board)
        
        # 모든 대칭을 tuple로 변환
        tuples = [
            BoardSymmetry.to_tuple(boards, completed, board.current_player)
            for boards, completed in symmetries
        ]
        
        # 사전순으로 가장 작은 것 선택
        canonical = min(tuples)
        
        return canonical
    
    @staticmethod
    def get_canonical_hash(board):
        """
        정규화된 해시 반환
        
        Args:
            board: Board 객체
            
        Returns:
            int: 정규화된 해시값
        """
        canonical = BoardSymmetry.get_canonical_form(board)
        return hash(canonical)
