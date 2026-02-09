"""
Precomputed D4 symmetry transformations for Ultimate Tic-Tac-Toe.
All symmetry-related constants and lookup tables in one place.
"""

# D4 symmetry group: 8 index permutations for 3x3 grid
# Each permutation maps original position -> new position
D4_TRANSFORMS = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),  # identity
    (6, 3, 0, 7, 4, 1, 8, 5, 2),  # rotate 90° CW
    (8, 7, 6, 5, 4, 3, 2, 1, 0),  # rotate 180°
    (2, 5, 8, 1, 4, 7, 0, 3, 6),  # rotate 270° CW
    (2, 1, 0, 5, 4, 3, 8, 7, 6),  # flip horizontal
    (6, 7, 8, 3, 4, 5, 0, 1, 2),  # flip vertical
    (0, 3, 6, 1, 4, 7, 2, 5, 8),  # flip main diagonal
    (8, 5, 2, 7, 4, 1, 6, 3, 0),  # flip anti-diagonal
)

# Inverse transforms: INV_TRANSFORMS[perm_id][new_pos] = original_pos
INV_TRANSFORMS = tuple(
    tuple(perm.index(i) for i in range(9)) for perm in D4_TRANSFORMS
)

# Precomputed rotated 9-bit masks for all 512 possible masks × 8 transforms
# ROTATED_MASKS[perm_id][mask] = rotated_mask
# Usage: rotated = ROTATED_MASKS[perm_id][original_mask]
def _build_rotated_masks():
    table = []
    for perm in D4_TRANSFORMS:
        perm_table = []
        for mask in range(512):
            rotated = 0
            for i in range(9):
                if mask & (1 << i):
                    rotated |= (1 << perm[i])
            perm_table.append(rotated)
        table.append(tuple(perm_table))
    return tuple(table)

ROTATED_MASKS = _build_rotated_masks()

# Win patterns for 3x3 board (bitmask form)
WIN_MASKS = (
    0b111000000,  # row 0
    0b000111000,  # row 1
    0b000000111,  # row 2
    0b100100100,  # col 0
    0b010010010,  # col 1
    0b001001001,  # col 2
    0b100010001,  # diagonal
    0b001010100,  # anti-diagonal
)

# Win check lines for iteration (row, col, diag patterns)
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
)
