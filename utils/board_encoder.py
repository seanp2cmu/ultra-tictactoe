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
    
    # Pre-computed comparison key bases (class-level, computed once)
    _KEY_BASES_27 = None

    @staticmethod
    def _get_key_bases():
        if BoardEncoder._KEY_BASES_27 is None:
            BoardEncoder._KEY_BASES_27 = np.array(
                [3 ** (26 - i) for i in range(27)], dtype=np.uint64
            )
        return BoardEncoder._KEY_BASES_27

    @staticmethod
    def to_inference_tensor_batch(boards: list) -> Tuple[np.ndarray, list]:
        """Vectorized batch inference — all canonical transforms computed at once."""
        n = len(boards)
        if n == 0:
            return np.zeros((0, 7, 9, 9), dtype=np.float32), []

        # ── 1. Extract raw data from all boards ──
        transforms_81 = _BoardSymmetry._build_transforms()       # 8 × (81,)
        transforms_9  = _BoardSymmetry._build_c_transforms()     # 8 × (9,)
        P27 = BoardEncoder._get_key_bases()
        P9  = np.array([3 ** (8 - i) for i in range(9)], dtype=np.uint64)

        all_flat = np.empty((n, 81), dtype=np.uint8)
        all_comp = np.empty((n, 9), dtype=np.uint8)
        all_player = np.empty(n, dtype=np.uint8)
        all_legal = []       # list of arrays of flat indices
        all_last  = np.full(n, -1, dtype=np.int16)

        has_completed_state = hasattr(boards[0], 'get_completed_state')
        for i, b in enumerate(boards):
            all_flat[i] = np.array(b.to_array(), dtype=np.uint8).ravel()
            if has_completed_state:
                all_comp[i] = np.array(
                    [b.get_completed_state(j) for j in range(9)], dtype=np.uint8
                )
            else:
                all_comp[i] = np.array(b.completed_boards, dtype=np.uint8).ravel()
            all_player[i] = b.current_player
            legal = b.get_legal_moves()
            all_legal.append(np.array([r * 9 + c for r, c in legal], dtype=np.int16))
            if b.last_move is not None:
                all_last[i] = b.last_move[0] * 9 + b.last_move[1]

        # ── 2. Vectorized canonical transform selection ──
        # Apply all 8 transforms via fancy indexing: (8, N, 81) and (8, N, 9)
        t_boards = np.stack([all_flat[:, t] for t in transforms_81])   # (8, N, 81)
        t_comps  = np.stack([all_comp[:, t] for t in transforms_9])    # (8, N, 9)

        # Compute comparison keys (3 chunks of 27 for board + 1 for completed+player)
        kb_a = t_boards[:, :, :27].astype(np.uint64) @ P27             # (8, N)
        kb_b = t_boards[:, :, 27:54].astype(np.uint64) @ P27           # (8, N)
        kb_c = t_boards[:, :, 54:].astype(np.uint64) @ P27             # (8, N)
        kc   = t_comps.astype(np.uint64) @ P9                          # (8, N)
        kc   = kc * 3 + all_player[None, :].astype(np.uint64)

        # Cascaded min-finding: board chunk a → b → c → completed
        MAXU = np.iinfo(np.uint64).max
        cand = np.ones((8, n), dtype=bool)

        min_a = kb_a.min(axis=0)
        cand &= (kb_a == min_a[None, :])

        masked_b = np.where(cand, kb_b, MAXU)
        min_b = masked_b.min(axis=0)
        cand &= (kb_b == min_b[None, :])

        masked_c = np.where(cand, kb_c, MAXU)
        min_c = masked_c.min(axis=0)
        cand &= (kb_c == min_c[None, :])

        masked_d = np.where(cand, kc, MAXU)
        min_d = masked_d.min(axis=0)
        cand &= (kc == min_d[None, :])

        best_idx = np.argmax(cand, axis=0).astype(np.int32)  # (N,) first surviving

        # ── 3. Gather best-transform boards & completed ──
        arange_n = np.arange(n)
        best_flat = t_boards[best_idx, arange_n]          # (N, 81)
        best_comp = t_comps[best_idx, arange_n]           # (N, 9)

        # ── 4. Build 7-channel tensor (vectorized) ──
        batch = np.zeros((n, 7, 9, 9), dtype=np.float32)
        best_r = best_flat.reshape(n, 9, 9).astype(np.float32)
        p = all_player[:, None, None].astype(np.float32)
        o = (3 - all_player)[:, None, None].astype(np.float32)

        batch[:, 0] = (best_r == p)
        batch[:, 1] = (best_r == o)
        batch[:, 2] = (best_r == 0)

        # Channels 3,4: completed sub-boards
        best_c33 = best_comp.reshape(n, 3, 3)
        p1 = all_player
        o1 = 3 - all_player
        for br in range(3):
            for bc in range(3):
                sr, sc = br * 3, bc * 3
                mask_p = (best_c33[:, br, bc] == p1)
                mask_o = (best_c33[:, br, bc] == o1)
                if mask_p.any():
                    batch[mask_p, 3, sr:sr+3, sc:sc+3] = 1.0
                if mask_o.any():
                    batch[mask_o, 4, sr:sr+3, sc:sc+3] = 1.0

        # Channel 5: legal moves (transform to canonical)
        # Channel 6: last move
        inv_maps = [np.argsort(transforms_81[t]) for t in range(8)]
        for i in range(n):
            inv = inv_maps[best_idx[i]]
            for idx in all_legal[i]:
                ni = inv[idx]
                batch[i, 5, ni // 9, ni % 9] = 1.0
            if all_last[i] >= 0:
                ni = inv[all_last[i]]
                batch[i, 6, ni // 9, ni % 9] = 1.0

        # ── 5. Build inverse transform index array for batch policy output ──
        inv_idx = np.stack([transforms_81[idx] for idx in best_idx])  # (N, 81)

        return batch, inv_idx
    
    @staticmethod
    def get_canonical_hash(board: Board) -> bytes:
        """DTW cache용 canonical hash 반환"""
        return _BoardSymmetry.get_canonical_hash(board)

# Use Cython version if available
try:
    from ._board_encoder_cy import BoardEncoderCy as BoardEncoder
except ImportError:
    pass
