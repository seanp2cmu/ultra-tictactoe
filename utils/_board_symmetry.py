"""
Internal board symmetry transformation (private module)
Ultimate Tic-Tac-Toe has D4 symmetry group (8 symmetries)
"""
import numpy as np
from game import Board


class _BoardSymmetry:
    """Board symmetry transformation and normalization (internal use only)"""
    
    _TRANSFORMS = None
    _C_TRANSFORMS = None  # 3x3 completed board transforms
    
    @staticmethod
    def _build_c_transforms():
        """Build transform index maps for 3x3 completed board."""
        if _BoardSymmetry._C_TRANSFORMS is not None:
            return _BoardSymmetry._C_TRANSFORMS
        cidx = np.arange(9).reshape(3, 3)
        ct = [
            cidx.flatten(),
            np.fliplr(cidx).flatten(),
            np.flipud(cidx).flatten(),
            np.rot90(cidx, 2).flatten(),
            np.rot90(cidx, -1).flatten(),
            np.rot90(cidx, 1).flatten(),
            cidx.T.flatten(),
            np.rot90(cidx.T, 2).flatten(),
        ]
        _BoardSymmetry._C_TRANSFORMS = ct
        return ct
    
    @staticmethod
    def _build_transforms():
        """Build transform index maps for 9x9 board."""
        if _BoardSymmetry._TRANSFORMS is not None:
            return _BoardSymmetry._TRANSFORMS
        
        transforms = []
        transforms.append(np.arange(81))  # Identity
        
        idx = np.arange(81).reshape(9, 9)
        transforms.append(np.fliplr(idx).flatten())  # Flip horizontal
        transforms.append(np.flipud(idx).flatten())  # Flip vertical
        transforms.append(np.rot90(idx, k=2).flatten())  # Rotate 180
        transforms.append(np.rot90(idx, k=-1).flatten())  # Rotate 90 clockwise
        transforms.append(np.rot90(idx, k=1).flatten())  # Rotate 270 clockwise
        transforms.append(idx.T.flatten())  # Transpose
        transforms.append(np.rot90(idx.T, k=2).flatten())  # Anti-transpose
        
        _BoardSymmetry._TRANSFORMS = transforms
        return transforms
    
    @staticmethod
    def get_canonical_with_transform(board: Board):
        """
        Get canonical form of board and the transform index used.
        
        Returns:
            (canonical_boards, canonical_completed, transform_idx)
        """
        boards_arr = np.array(board.to_array(), dtype=np.int8)
        if hasattr(board, 'get_completed_state'):
            completed_flat = [board.get_completed_state(i) for i in range(9)]
            completed_arr = np.array(completed_flat, dtype=np.int8).reshape(3, 3)
        else:
            completed_arr = np.array(board.completed_boards, dtype=np.int8)
        player = board.current_player
        
        min_bytes = boards_arr.tobytes() + completed_arr.tobytes() + bytes([player])
        best_idx = 0
        best_boards = boards_arr
        best_completed = completed_arr
        
        transform_ops = [
            lambda b, c: (np.fliplr(b), np.fliplr(c)),
            lambda b, c: (np.flipud(b), np.flipud(c)),
            lambda b, c: (np.rot90(b, 2), np.rot90(c, 2)),
            lambda b, c: (np.rot90(b, -1), np.rot90(c, -1)),
            lambda b, c: (np.rot90(b, 1), np.rot90(c, 1)),
            lambda b, c: (b.T, c.T),
            lambda b, c: (np.rot90(b.T, 2), np.rot90(c.T, 2)),
        ]
        
        for i, transform in enumerate(transform_ops):
            tb, tc = transform(boards_arr, completed_arr)
            b = np.ascontiguousarray(tb).tobytes() + np.ascontiguousarray(tc).tobytes() + bytes([player])
            if b < min_bytes:
                min_bytes = b
                best_idx = i + 1
                best_boards = np.ascontiguousarray(tb)
                best_completed = np.ascontiguousarray(tc)
        
        return best_boards, best_completed, best_idx
    
    @staticmethod
    def get_canonical_hash(board: Board) -> bytes:
        """Get canonical hash for board."""
        boards_arr = np.array(board.to_array(), dtype=np.int8)
        if hasattr(board, 'get_completed_state'):
            completed_flat = [board.get_completed_state(i) for i in range(9)]
            completed_arr = np.array(completed_flat, dtype=np.int8).reshape(3, 3)
        else:
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
    def transform_policy(policy: np.ndarray, transform_idx: int) -> np.ndarray:
        """Transform policy vector (81,) to canonical orientation."""
        transforms = _BoardSymmetry._build_transforms()
        transform_map = transforms[transform_idx]
        inverse_map = np.argsort(transform_map)
        return policy[inverse_map]
    
    @staticmethod
    def inverse_transform_policy(policy: np.ndarray, transform_idx: int) -> np.ndarray:
        """Inverse transform policy from canonical back to original orientation."""
        transforms = _BoardSymmetry._build_transforms()
        transform_map = transforms[transform_idx]
        return policy[transform_map]

# Use Cython version if available
try:
    from ._board_symmetry_cy import _BoardSymmetryCy as _BoardSymmetry
except ImportError:
    pass
