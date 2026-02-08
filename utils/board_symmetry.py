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
        """Get canonical bytes for board (without constraint - stored separately)."""
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
    
    @staticmethod
    def swap_xo(board: Board) -> Board:
        """
        Create a new board with X and O swapped.
        Used for X-O symmetry deduplication.
        """
        swapped = board.clone()
        
        # Swap pieces on board (1 <-> 2)
        for r in range(9):
            for c in range(9):
                if swapped.boards[r][c] == 1:
                    swapped.boards[r][c] = 2
                elif swapped.boards[r][c] == 2:
                    swapped.boards[r][c] = 1
        
        # Swap completed_boards (1 <-> 2, keep 0 and 3)
        for r in range(3):
            for c in range(3):
                if swapped.completed_boards[r][c] == 1:
                    swapped.completed_boards[r][c] = 2
                elif swapped.completed_boards[r][c] == 2:
                    swapped.completed_boards[r][c] = 1
        
        # Swap current player
        swapped.current_player = 3 - swapped.current_player
        
        return swapped
    
    @staticmethod
    def get_canonical_hash_with_swap(board: Board) -> bytes:
        """
        Get canonical hash considering both D4 symmetry AND X-O swap.
        Returns min(canonical(board), canonical(swap_xo(board))).
        
        This ensures:
        - Storage: only 1 entry per equivalence class
        - Lookup: compute same hash, 1 dict lookup
        """
        original_hash = BoardSymmetry.get_canonical_bytes(board)
        swapped = BoardSymmetry.swap_xo(board)
        swapped_hash = BoardSymmetry.get_canonical_bytes(swapped)
        
        return min(original_hash, swapped_hash)
    
    # Transform indices for policy vector (81 cells flattened)
    _TRANSFORMS = None
    
    @staticmethod
    def _build_transforms():
        """Build transform index maps for 9x9 board."""
        if BoardSymmetry._TRANSFORMS is not None:
            return BoardSymmetry._TRANSFORMS
        
        # Each transform: how to map original index -> transformed index
        # For a 9x9 board with indices 0-80
        transforms = []
        
        # Identity
        transforms.append(np.arange(81))
        
        # Flip horizontal
        idx = np.arange(81).reshape(9, 9)
        transforms.append(np.fliplr(idx).flatten())
        
        # Flip vertical
        transforms.append(np.flipud(idx).flatten())
        
        # Rotate 180
        transforms.append(np.rot90(idx, k=2).flatten())
        
        # Rotate 90 clockwise (k=-1)
        transforms.append(np.rot90(idx, k=-1).flatten())
        
        # Rotate 270 clockwise (k=1)
        transforms.append(np.rot90(idx, k=1).flatten())
        
        # Transpose
        transforms.append(idx.T.flatten())
        
        # Anti-transpose (rot180 of transpose)
        transforms.append(np.rot90(idx.T, k=2).flatten())
        
        BoardSymmetry._TRANSFORMS = transforms
        return transforms
    
    @staticmethod
    def get_canonical_with_transform(board: Board):
        """
        Get canonical form of board and the transform index used.
        
        Returns:
            (canonical_boards, canonical_completed, transform_idx)
        """
        boards_arr = np.array(board.boards, dtype=np.int8)
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
                best_idx = i + 1  # +1 because identity is 0
                best_boards = np.ascontiguousarray(tb)
                best_completed = np.ascontiguousarray(tc)
        
        return best_boards, best_completed, best_idx
    
    @staticmethod
    def transform_policy(policy: np.ndarray, transform_idx: int) -> np.ndarray:
        """
        Transform policy vector (81,) using the same transform.
        
        Args:
            policy: Original policy probabilities (81,)
            transform_idx: Transform index from get_canonical_with_transform
            
        Returns:
            Transformed policy (81,)
        """
        transforms = BoardSymmetry._build_transforms()
        transform_map = transforms[transform_idx]
        
        # Map: new_policy[transform_map[i]] = policy[i]
        # We need inverse: new_policy[i] = policy[inverse_map[i]]
        inverse_map = np.argsort(transform_map)
        return policy[inverse_map]
    
    @staticmethod 
    def canonicalize_training_sample(board: Board, policy: np.ndarray):
        """
        Convert board and policy to canonical form for training.
        
        Args:
            board: Board object
            policy: Policy probabilities (81,)
            
        Returns:
            (canonical_boards_arr, canonical_completed_arr, transformed_policy, current_player)
        """
        boards, completed, transform_idx = BoardSymmetry.get_canonical_with_transform(board)
        transformed_policy = BoardSymmetry.transform_policy(policy, transform_idx)
        return boards, completed, transformed_policy, board.current_player
    
    @staticmethod
    def inverse_transform_policy(policy: np.ndarray, transform_idx: int) -> np.ndarray:
        """
        Inverse transform policy from canonical back to original orientation.
        
        Args:
            policy: Canonical policy probabilities (81,)
            transform_idx: Transform index from get_canonical_with_transform
            
        Returns:
            Policy in original orientation (81,)
        """
        transforms = BoardSymmetry._build_transforms()
        transform_map = transforms[transform_idx]
        # Forward: new[transform_map[i]] = old[i]
        # So: original[i] = canonical[transform_map[i]]
        return policy[transform_map]
    
    @staticmethod
    def predict_with_canonical(board: Board, predict_fn) -> tuple:
        """
        Make prediction using canonical form, then transform output back.
        
        Args:
            board: Original board
            predict_fn: Function that takes Board and returns (policy, value)
            
        Returns:
            (policy, value) in original board orientation
        """
        boards_arr, completed_arr, transform_idx = BoardSymmetry.get_canonical_with_transform(board)
        
        # Create canonical board for prediction
        canonical_board = Board()
        canonical_board.boards = boards_arr.tolist()
        canonical_board.completed_boards = completed_arr.tolist()
        canonical_board.current_player = board.current_player
        
        # Transform last_move too
        if board.last_move is not None and transform_idx != 0:
            transforms = BoardSymmetry._build_transforms()
            old_idx = board.last_move[0] * 9 + board.last_move[1]
            new_idx = np.where(transforms[transform_idx] == old_idx)[0]
            if len(new_idx) > 0:
                canonical_board.last_move = (new_idx[0] // 9, new_idx[0] % 9)
        else:
            canonical_board.last_move = board.last_move
        
        # Get prediction in canonical space
        canonical_policy, value = predict_fn(canonical_board)
        
        # Transform policy back to original orientation
        original_policy = BoardSymmetry.inverse_transform_policy(canonical_policy, transform_idx)
        
        return original_policy, value
