"""
BoardEncoder - 단일 변환 레이어 (Public API)
모든 board/tensor/policy 변환을 담당
"""
import numpy as np
from typing import Tuple, Callable
from game import Board
from ._board_symmetry import _BoardSymmetry


class BoardEncoder:
    """
    단일 변환 레이어 - 모든 board/policy 변환을 담당
    
    사용법:
        # Training: canonical state + canonical policy 저장
        tensor, canonical_policy = BoardEncoder.to_training_tensor(board, original_policy)
        
        # Inference: canonical 예측 후 original로 역변환
        tensor, to_original = BoardEncoder.to_inference_tensor(board)
        policy, value = network(tensor)
        original_policy = to_original(policy)
    """
    
    @staticmethod
    def board_to_tensor(board: Board) -> np.ndarray:
        """
        Board를 7채널 tensor로 변환 (단일 구현)
        
        Channels:
            0: current player pieces
            1: opponent pieces  
            2: empty cells
            3: current player가 이긴 sub-boards
            4: opponent가 이긴 sub-boards
            5: legal moves
            6: last move
        """
        tensor = np.zeros((7, 9, 9), dtype=np.float32)
        boards = np.array(board.to_array(), dtype=np.float32)
        
        current_player = board.current_player
        opponent_player = 3 - current_player
        
        tensor[0] = (boards == current_player).astype(np.float32)
        tensor[1] = (boards == opponent_player).astype(np.float32)
        tensor[2] = (boards == 0).astype(np.float32)
        
        if hasattr(board, 'get_completed_boards_2d'):
            completed = board.get_completed_boards_2d()
        elif hasattr(board, 'completed_boards'):
            completed = board.completed_boards
        else:
            completed = [[0]*3 for _ in range(3)]
        
        for br in range(3):
            for bc in range(3):
                status = completed[br][bc]
                sr, sc = br * 3, bc * 3
                if status == current_player:
                    tensor[3, sr:sr+3, sc:sc+3] = 1.0
                elif status == opponent_player:
                    tensor[4, sr:sr+3, sc:sc+3] = 1.0
        
        if hasattr(board, 'get_legal_moves'):
            for r, c in board.get_legal_moves():
                tensor[5, r, c] = 1.0
        
        if hasattr(board, 'last_move') and board.last_move is not None:
            tensor[6, board.last_move[0], board.last_move[1]] = 1.0
        
        return tensor
    
    @staticmethod
    def _create_canonical_board(board: Board) -> Tuple[Board, int]:
        """Board를 canonical form으로 변환"""
        boards_arr, completed_arr, transform_idx = _BoardSymmetry.get_canonical_with_transform(board)
        
        canonical_board = Board()
        for r in range(9):
            for c in range(9):
                if boards_arr[r, c] != 0:
                    canonical_board.set_cell(r, c, int(boards_arr[r, c]))
        
        if hasattr(canonical_board, 'set_completed_boards_2d'):
            canonical_board.set_completed_boards_2d(completed_arr.tolist())
        else:
            canonical_board.completed_boards = completed_arr.tolist()
        canonical_board.current_player = board.current_player
        
        if board.last_move is not None and transform_idx != 0:
            transforms = _BoardSymmetry._build_transforms()
            old_idx = board.last_move[0] * 9 + board.last_move[1]
            new_idx = np.where(transforms[transform_idx] == old_idx)[0]
            if len(new_idx) > 0:
                canonical_board.last_move = (new_idx[0] // 9, new_idx[0] % 9)
        else:
            canonical_board.last_move = board.last_move
        
        return canonical_board, transform_idx
    
    @staticmethod
    def to_training_tensor(board: Board, original_policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Training용: canonical state와 canonical policy 반환
        
        Args:
            board: Original board
            original_policy: MCTS에서 나온 original orientation policy
            
        Returns:
            (canonical_tensor, canonical_policy)
        """
        canonical_board, transform_idx = BoardEncoder._create_canonical_board(board)
        canonical_tensor = BoardEncoder.board_to_tensor(canonical_board)
        canonical_policy = _BoardSymmetry.transform_policy(original_policy, transform_idx)
        
        return canonical_tensor, canonical_policy
    
    @staticmethod
    def to_inference_tensor(board: Board) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """
        Inference용: canonical tensor + 역변환 함수 반환
        
        Args:
            board: Original board
            
        Returns:
            (canonical_tensor, inverse_transform_fn)
        """
        canonical_board, transform_idx = BoardEncoder._create_canonical_board(board)
        canonical_tensor = BoardEncoder.board_to_tensor(canonical_board)
        
        def inverse_transform(canonical_policy: np.ndarray) -> np.ndarray:
            return _BoardSymmetry.inverse_transform_policy(canonical_policy, transform_idx)
        
        return canonical_tensor, inverse_transform
    
    @staticmethod
    def to_inference_tensor_batch(boards: list) -> Tuple[np.ndarray, list]:
        """Batch inference용"""
        tensors = []
        inverse_fns = []
        
        for board in boards:
            tensor, inverse_fn = BoardEncoder.to_inference_tensor(board)
            tensors.append(tensor)
            inverse_fns.append(inverse_fn)
        
        return np.stack(tensors), inverse_fns
    
    @staticmethod
    def get_canonical_hash(board: Board) -> bytes:
        """DTW cache용 canonical hash 반환"""
        return _BoardSymmetry.get_canonical_hash(board)
