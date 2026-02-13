# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython implementation of BoardEncoder.
Drop-in replacement for board_encoder.py with identical API.
Accesses BoardCy cdef-public bitmask fields for fast extraction.
"""
cimport cython
import numpy as np
cimport numpy as np
from libc.string cimport memset

np.import_array()

from ._board_symmetry_cy import _BoardSymmetryCy as _BoardSymmetry

# ── Pre-computed key bases ──
cdef unsigned long long _KEY_BASES_27[27]
cdef unsigned long long _KEY_BASES_9[9]
cdef bint _key_bases_built = False

cdef void _build_key_bases() noexcept nogil:
    global _key_bases_built
    cdef int i
    cdef unsigned long long v
    v = 1
    for i in range(27):
        _KEY_BASES_27[26 - i] = v
        v *= 3
    v = 1
    for i in range(9):
        _KEY_BASES_9[8 - i] = v
        v *= 3
    _key_bases_built = True

cdef inline void _ensure_key_bases() noexcept nogil:
    if not _key_bases_built:
        _build_key_bases()


# ── Fast C-level board data extraction using bitmask arrays ──
cdef void _extract_flat_from_masks(
    unsigned short *x_masks, unsigned short *o_masks,
    unsigned char *out
) noexcept nogil:
    """Extract 81-element flat board from bitmask arrays. Matches to_array layout."""
    cdef int sub_idx, cell_idx, sub_r, sub_c, r, c
    cdef unsigned short xm, om
    for sub_idx in range(9):
        sub_r = sub_idx / 3
        sub_c = sub_idx % 3
        xm = x_masks[sub_idx]
        om = o_masks[sub_idx]
        for cell_idx in range(9):
            r = sub_r * 3 + cell_idx / 3
            c = sub_c * 3 + cell_idx % 3
            if xm & (1 << cell_idx):
                out[r * 9 + c] = 1
            elif om & (1 << cell_idx):
                out[r * 9 + c] = 2
            else:
                out[r * 9 + c] = 0

cdef int _extract_legal_from_masks(
    unsigned short *x_masks, unsigned short *o_masks,
    unsigned short completed_mask,
    signed char last_move_r, signed char last_move_c,
    short *out
) noexcept nogil:
    """Extract legal move flat indices from bitmasks. Returns count."""
    cdef int count = 0
    cdef int sub_idx, cell_idx, sub_r, sub_c, r, c
    cdef int target_sub_idx
    cdef unsigned short occupied

    if last_move_r < 0:
        for sub_idx in range(9):
            if completed_mask & (1 << sub_idx):
                continue
            occupied = x_masks[sub_idx] | o_masks[sub_idx]
            sub_r = sub_idx / 3
            sub_c = sub_idx % 3
            for cell_idx in range(9):
                if not (occupied & (1 << cell_idx)):
                    r = sub_r * 3 + cell_idx / 3
                    c = sub_c * 3 + cell_idx % 3
                    out[count] = r * 9 + c
                    count += 1
    else:
        target_sub_idx = (last_move_r % 3) * 3 + (last_move_c % 3)
        if not (completed_mask & (1 << target_sub_idx)):
            occupied = x_masks[target_sub_idx] | o_masks[target_sub_idx]
            sub_r = target_sub_idx / 3
            sub_c = target_sub_idx % 3
            for cell_idx in range(9):
                if not (occupied & (1 << cell_idx)):
                    r = sub_r * 3 + cell_idx / 3
                    c = sub_c * 3 + cell_idx % 3
                    out[count] = r * 9 + c
                    count += 1
        else:
            for sub_idx in range(9):
                if completed_mask & (1 << sub_idx):
                    continue
                occupied = x_masks[sub_idx] | o_masks[sub_idx]
                sub_r = sub_idx / 3
                sub_c = sub_idx % 3
                for cell_idx in range(9):
                    if not (occupied & (1 << cell_idx)):
                        r = sub_r * 3 + cell_idx / 3
                        c = sub_c * 3 + cell_idx % 3
                        out[count] = r * 9 + c
                        count += 1
    return count


class BoardEncoderCy:
    """Cython-accelerated BoardEncoder — identical API to Python version."""

    @staticmethod
    def board_to_tensor(board):
        """Board -> 7-channel tensor. Same as Python version."""
        cdef np.ndarray[float, ndim=3] tensor = np.zeros((7, 9, 9), dtype=np.float32)
        boards = np.array(board.to_array(), dtype=np.float32)

        cdef int current_player = board.current_player
        cdef int opponent_player = 3 - current_player

        tensor[0] = (boards == current_player).astype(np.float32)
        tensor[1] = (boards == opponent_player).astype(np.float32)
        tensor[2] = (boards == 0).astype(np.float32)

        if hasattr(board, 'get_completed_boards_2d'):
            completed = board.get_completed_boards_2d()
        elif hasattr(board, 'completed_boards'):
            completed = board.completed_boards
        else:
            completed = [[0]*3 for _ in range(3)]

        cdef int br, bc, sr, sc, status
        for br in range(3):
            for bc in range(3):
                status = completed[br][bc]
                sr = br * 3
                sc = bc * 3
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
    def _create_canonical_board(board):
        """Board -> canonical form. Same as Python version."""
        from game import Board as GameBoard
        boards_arr, completed_arr, transform_idx = _BoardSymmetry.get_canonical_with_transform(board)

        canonical_board = GameBoard()
        cdef int r, c
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
    def to_training_tensor(board, original_policy):
        """Training: canonical state + canonical policy."""
        canonical_board, transform_idx = BoardEncoderCy._create_canonical_board(board)
        canonical_tensor = BoardEncoderCy.board_to_tensor(canonical_board)
        canonical_policy = _BoardSymmetry.transform_policy(original_policy, transform_idx)
        return canonical_tensor, canonical_policy

    @staticmethod
    def to_inference_tensor(board):
        """Inference: canonical tensor + inverse function."""
        canonical_board, transform_idx = BoardEncoderCy._create_canonical_board(board)
        canonical_tensor = BoardEncoderCy.board_to_tensor(canonical_board)

        def inverse_transform(canonical_policy):
            return _BoardSymmetry.inverse_transform_policy(canonical_policy, transform_idx)

        return canonical_tensor, inverse_transform

    @staticmethod
    def _get_key_bases():
        _ensure_key_bases()
        return np.array([_KEY_BASES_27[i] for i in range(27)], dtype=np.uint64)

    @staticmethod
    def to_inference_tensor_batch(list boards):
        """Vectorized batch inference — direct bitmask access, no Python method calls."""
        cdef int n = len(boards)
        if n == 0:
            return np.zeros((0, 7, 9, 9), dtype=np.float32), []

        _ensure_key_bases()

        transforms_81_list = _BoardSymmetry._build_transforms()
        transforms_9_list = _BoardSymmetry._build_c_transforms()

        # ── 1. Extract raw data from BoardCy bitmasks (fast C-level) ──
        all_flat = np.empty((n, 81), dtype=np.uint8)
        all_comp = np.empty((n, 9), dtype=np.uint8)
        all_player_arr = np.empty(n, dtype=np.uint8)
        all_last = np.full(n, -1, dtype=np.int16)

        cdef unsigned char *flat_ptr = <unsigned char *>np.PyArray_DATA(all_flat)
        cdef unsigned char *comp_ptr = <unsigned char *>np.PyArray_DATA(all_comp)
        cdef short *last_ptr = <short *>np.PyArray_DATA(all_last)

        # Temp C arrays to hold one board's bitmasks
        cdef unsigned short xm[9]
        cdef unsigned short om[9]
        cdef unsigned char cb[9]
        cdef unsigned short cmask
        cdef signed char lmr, lmc
        cdef unsigned char cp

        cdef short legal_buf[81]
        cdef int legal_count
        cdef int i, j
        all_legal = []

        for i in range(n):
            b = boards[i]
            # Read cdef-public fields (fast Python attr access, no method calls)
            xm_py = b.x_masks    # list of 9 ints
            om_py = b.o_masks
            for j in range(9):
                xm[j] = xm_py[j]
                om[j] = om_py[j]

            # Extract flat board in C (nogil)
            _extract_flat_from_masks(xm, om, flat_ptr + i * 81)

            # Completed boards (use get_completed_state for C array access)
            for j in range(9):
                cb[j] = b.get_completed_state(j)
                comp_ptr[i * 9 + j] = cb[j]

            cp = b.current_player
            all_player_arr[i] = cp
            cmask = b.completed_mask
            lmr = b.last_move_r
            lmc = b.last_move_c

            # Legal moves in C (nogil)
            legal_count = _extract_legal_from_masks(xm, om, cmask, lmr, lmc, legal_buf)
            legal_arr = np.empty(legal_count, dtype=np.int16)
            for j in range(legal_count):
                legal_arr[j] = legal_buf[j]
            all_legal.append(legal_arr)

            # Last move
            if lmr >= 0:
                last_ptr[i] = lmr * 9 + lmc

        # ── 2. Vectorized canonical transform selection (numpy) ──
        idx_81 = np.array(transforms_81_list)    # (8, 81)
        idx_9  = np.array(transforms_9_list)     # (8, 9)
        t_boards = all_flat[:, idx_81].transpose(1, 0, 2)   # (8, N, 81)
        t_comps  = all_comp[:, idx_9].transpose(1, 0, 2)    # (8, N, 9)

        P27 = np.array([_KEY_BASES_27[k] for k in range(27)], dtype=np.uint64)
        P9  = np.array([_KEY_BASES_9[k] for k in range(9)], dtype=np.uint64)

        kb_a = t_boards[:, :, :27].astype(np.uint64) @ P27
        kb_b = t_boards[:, :, 27:54].astype(np.uint64) @ P27
        kb_c = t_boards[:, :, 54:].astype(np.uint64) @ P27
        kc   = t_comps.astype(np.uint64) @ P9
        kc   = kc * 3 + all_player_arr[None, :].astype(np.uint64)

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

        cdef np.ndarray[int, ndim=1] best_idx = np.argmax(cand, axis=0).astype(np.int32)

        # ── 3. Gather best-transform boards & completed ──
        arange_n = np.arange(n)
        best_flat = t_boards[best_idx, arange_n]
        best_comp = t_comps[best_idx, arange_n]

        # ── 4. Build 7-channel tensor ──
        cdef np.ndarray[float, ndim=4] batch = np.zeros((n, 7, 9, 9), dtype=np.float32)
        best_r = best_flat.reshape(n, 9, 9).astype(np.float32)
        p = all_player_arr[:, None, None].astype(np.float32)
        o = (3 - all_player_arr)[:, None, None].astype(np.float32)

        batch[:, 0] = (best_r == p)
        batch[:, 1] = (best_r == o)
        batch[:, 2] = (best_r == 0)

        # Channels 3,4: completed sub-boards
        best_c33 = best_comp.reshape(n, 3, 3)
        p1 = all_player_arr
        o1 = 3 - all_player_arr
        cdef int br, bc, sr, sc
        for br in range(3):
            for bc in range(3):
                sr = br * 3
                sc = bc * 3
                s_col = best_c33[:, br, bc]
                mask_p = (s_col == p1)
                mask_o = (s_col == o1)
                if mask_p.any():
                    batch[mask_p, 3, sr:sr+3, sc:sc+3] = 1.0
                if mask_o.any():
                    batch[mask_o, 4, sr:sr+3, sc:sc+3] = 1.0

        # ── Channel 5 & 6: legal moves + last move (typed C loop) ──
        cdef int inv_maps[8][81]
        cdef int t, pos
        for t in range(8):
            for pos in range(81):
                inv_maps[t][<int>transforms_81_list[t][pos]] = pos

        cdef int bi, ni, idx_val
        cdef short last_val
        cdef np.ndarray[short, ndim=1] legal_arr_np
        for i in range(n):
            bi = best_idx[i]
            legal_arr_np = all_legal[i]
            for j in range(len(legal_arr_np)):
                idx_val = legal_arr_np[j]
                ni = inv_maps[bi][idx_val]
                batch[i, 5, ni / 9, ni % 9] = 1.0
            last_val = all_last[i]
            if last_val >= 0:
                ni = inv_maps[bi][last_val]
                batch[i, 6, ni / 9, ni % 9] = 1.0

        # ── 5. Build inverse transform index array ──
        cdef np.ndarray[long, ndim=2] inv_idx = np.empty((n, 81), dtype=np.intp)
        for i in range(n):
            bi = best_idx[i]
            for j in range(81):
                inv_idx[i, j] = transforms_81_list[bi][j]

        return batch, inv_idx

    @staticmethod
    def get_canonical_hash(board):
        """DTW cache canonical hash."""
        return _BoardSymmetry.get_canonical_hash(board)
