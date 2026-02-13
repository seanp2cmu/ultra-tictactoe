# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython implementation of _BoardSymmetry.
Drop-in replacement for _board_symmetry.py with identical API.
"""
cimport cython
import numpy as np
cimport numpy as np
from libc.string cimport memcpy

np.import_array()

# ── Pre-computed transform tables (module-level, built once) ──
cdef int _transforms_81_arr[8][81]
cdef int _c_transforms_9_arr[8][9]
cdef bint _transforms_built = False
cdef bint _c_transforms_built = False

cdef void _build_transforms_c() noexcept nogil:
    """Build 9x9 transform index tables in C."""
    global _transforms_built
    cdef int i, r, c, idx
    cdef int grid[9][9]
    cdef int flipped[9][9]
    cdef int result[81]

    # Build identity grid
    for r in range(9):
        for c in range(9):
            grid[r][c] = r * 9 + c

    # Transform 0: Identity
    for i in range(81):
        _transforms_81_arr[0][i] = i

    # Transform 1: Flip horizontal (fliplr)
    for r in range(9):
        for c in range(9):
            _transforms_81_arr[1][r * 9 + c] = grid[r][8 - c]

    # Transform 2: Flip vertical (flipud)
    for r in range(9):
        for c in range(9):
            _transforms_81_arr[2][r * 9 + c] = grid[8 - r][c]

    # Transform 3: Rotate 180
    for r in range(9):
        for c in range(9):
            _transforms_81_arr[3][r * 9 + c] = grid[8 - r][8 - c]

    # Transform 4: Rotate 90 CW (rot90 k=-1)
    for r in range(9):
        for c in range(9):
            _transforms_81_arr[4][r * 9 + c] = grid[8 - c][r]

    # Transform 5: Rotate 270 CW (rot90 k=1)
    for r in range(9):
        for c in range(9):
            _transforms_81_arr[5][r * 9 + c] = grid[c][8 - r]

    # Transform 6: Transpose
    for r in range(9):
        for c in range(9):
            _transforms_81_arr[6][r * 9 + c] = grid[c][r]

    # Transform 7: Anti-transpose (rot90(T, 2))
    for r in range(9):
        for c in range(9):
            _transforms_81_arr[7][r * 9 + c] = grid[8 - c][8 - r]

    _transforms_built = True


cdef void _build_c_transforms_c() noexcept nogil:
    """Build 3x3 transform index tables in C."""
    global _c_transforms_built
    cdef int r, c
    cdef int grid[3][3]

    for r in range(3):
        for c in range(3):
            grid[r][c] = r * 3 + c

    # 0: Identity
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[0][r * 3 + c] = grid[r][c]
    # 1: fliplr
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[1][r * 3 + c] = grid[r][2 - c]
    # 2: flipud
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[2][r * 3 + c] = grid[2 - r][c]
    # 3: rot180
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[3][r * 3 + c] = grid[2 - r][2 - c]
    # 4: rot90 CW
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[4][r * 3 + c] = grid[2 - c][r]
    # 5: rot270 CW
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[5][r * 3 + c] = grid[c][2 - r]
    # 6: transpose
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[6][r * 3 + c] = grid[c][r]
    # 7: anti-transpose
    for r in range(3):
        for c in range(3):
            _c_transforms_9_arr[7][r * 3 + c] = grid[2 - c][2 - r]

    _c_transforms_built = True


cdef inline void _ensure_transforms() noexcept nogil:
    if not _transforms_built:
        _build_transforms_c()

cdef inline void _ensure_c_transforms() noexcept nogil:
    if not _c_transforms_built:
        _build_c_transforms_c()


class _BoardSymmetryCy:
    """Cython-accelerated board symmetry transformation."""

    @staticmethod
    def _build_transforms():
        """Return list of 8 numpy int arrays (81,) — same API as Python version."""
        _ensure_transforms()
        result = []
        cdef int t, i
        for t in range(8):
            arr = np.empty(81, dtype=np.intp)
            for i in range(81):
                arr[i] = _transforms_81_arr[t][i]
            result.append(arr)
        return result

    @staticmethod
    def _build_c_transforms():
        """Return list of 8 numpy int arrays (9,) — same API as Python version."""
        _ensure_c_transforms()
        result = []
        cdef int t, i
        for t in range(8):
            arr = np.empty(9, dtype=np.intp)
            for i in range(9):
                arr[i] = _c_transforms_9_arr[t][i]
            result.append(arr)
        return result

    @staticmethod
    def get_canonical_with_transform(board):
        """
        Get canonical form and transform index. Matches Python version exactly.
        Uses tobytes() comparison for identical ordering.
        """
        _ensure_transforms()
        _ensure_c_transforms()

        cdef int t, i, best_idx
        cdef unsigned char flat[81]
        cdef unsigned char comp[9]
        cdef unsigned char player
        cdef unsigned char t_flat[81]
        cdef unsigned char t_comp[9]

        # Extract board data
        boards_arr = np.array(board.to_array(), dtype=np.int8)
        if hasattr(board, 'get_completed_state'):
            completed_flat = [board.get_completed_state(j) for j in range(9)]
            completed_arr = np.array(completed_flat, dtype=np.int8).reshape(3, 3)
        else:
            completed_arr = np.array(board.completed_boards, dtype=np.int8)
        player_val = board.current_player

        min_bytes = boards_arr.tobytes() + completed_arr.tobytes() + bytes([player_val])
        best_idx = 0
        best_boards = boards_arr
        best_completed = completed_arr

        # Use the same lambda-based transforms as Python for exact compatibility
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
            b = np.ascontiguousarray(tb).tobytes() + np.ascontiguousarray(tc).tobytes() + bytes([player_val])
            if b < min_bytes:
                min_bytes = b
                best_idx = i + 1
                best_boards = np.ascontiguousarray(tb)
                best_completed = np.ascontiguousarray(tc)

        return best_boards, best_completed, best_idx

    @staticmethod
    def get_canonical_hash(board):
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
    def transform_policy(np.ndarray policy, int transform_idx):
        """Transform policy to canonical orientation."""
        _ensure_transforms()
        cdef np.ndarray[long, ndim=1] inverse_map = np.empty(81, dtype=np.intp)
        cdef int i, pos
        # argsort of transform
        for i in range(81):
            pos = _transforms_81_arr[transform_idx][i]
            inverse_map[pos] = i
        return policy[inverse_map]

    @staticmethod
    def inverse_transform_policy(np.ndarray policy, int transform_idx):
        """Inverse transform policy from canonical to original."""
        _ensure_transforms()
        cdef np.ndarray[long, ndim=1] transform_map = np.empty(81, dtype=np.intp)
        cdef int i
        for i in range(81):
            transform_map[i] = _transforms_81_arr[transform_idx][i]
        return policy[transform_map]
