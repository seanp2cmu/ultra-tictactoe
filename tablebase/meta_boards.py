"""
Precomputed Meta-Board Configurations

4^9 = 262,144 전수 열거 후 필터링 + D4 + X/O flip 정규화.
키: "numopen_empty" -> 값: [(meta, min_diff, max_diff), ...]

Generate:
    python -m tablebase.meta_boards
"""

import os
import pickle
from typing import Dict, List, Tuple
from collections import defaultdict
from itertools import product


DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'meta_boards.pkl')


WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]

D4_TRANSFORMS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity
    [6, 3, 0, 7, 4, 1, 8, 5, 2],  # rotate 90°
    [8, 7, 6, 5, 4, 3, 2, 1, 0],  # rotate 180°
    [2, 5, 8, 1, 4, 7, 0, 3, 6],  # rotate 270°
    [2, 1, 0, 5, 4, 3, 8, 7, 6],  # flip horizontal
    [6, 7, 8, 3, 4, 5, 0, 1, 2],  # flip vertical
    [0, 3, 6, 1, 4, 7, 2, 5, 8],  # flip main diagonal
    [8, 5, 2, 7, 4, 1, 6, 3, 0],  # flip anti-diagonal
]

# Precompute inverse maps for O(1) constraint transformation
INV_TRANSFORMS = [[perm.index(i) for i in range(9)] for perm in D4_TRANSFORMS]


def pack_sub_data(sub_data) -> int:
    """Pack 9 sub-boards into deterministic 90-bit integer.
    Each sub-board: (state:2bit, x_count:4bit, o_count:4bit) = 10 bits
    """
    result = 0
    for state, x_count, o_count in sub_data:
        result = (result << 10) | (state << 8) | (x_count << 4) | o_count
    return result


def _make_key(num_open: int, empty_cells: int) -> str:
    return f"{num_open}_{empty_cells}"


def _check_winner(meta: Tuple[int, ...]) -> int:
    for a, b, c in WIN_PATTERNS:
        if meta[a] == meta[b] == meta[c] and meta[a] in (1, 2):
            return meta[a]
    return 0


def _canonical(meta: Tuple[int, ...]) -> Tuple[int, ...]:
    candidates = []
    for p in D4_TRANSFORMS:
        sym = tuple(meta[p[i]] for i in range(9))
        candidates.append(sym)
        flipped = tuple(3 - v if v in (1, 2) else v for v in sym)
        candidates.append(flipped)
    return min(candidates)


def _compute_diff_range(meta: Tuple[int, ...], empty_cells: int) -> Tuple[int, int]:
    """Compute (min_diff, max_diff) for OPEN boards."""
    num_open = meta.count(0)
    x_wins = meta.count(1)
    o_wins = meta.count(2)
    draws = meta.count(3)
    
    total_filled = 9 * num_open - empty_cells
    
    min_completed = x_wins * (-3) + o_wins * (-7) + draws * (-1)
    max_completed = x_wins * 7 + o_wins * 3 + draws * 1
    
    min_diff = max(-max_completed, -total_filled)
    max_diff = min(1 - min_completed, total_filled)
    
    max_possible = num_open * 6
    min_diff = max(min_diff, -max_possible)
    max_diff = min(max_diff, max_possible)
    
    return (min_diff, max_diff)


def _generate() -> Dict[str, List[Tuple[Tuple[int, ...], int, int]]]:
    """Generate all valid meta-boards."""
    seen = set()
    canonical_metas = []
    
    for meta in product([0, 1, 2, 3], repeat=9):
        if _check_winner(meta) != 0:
            continue
        canonical = _canonical(meta)
        if canonical in seen:
            continue
        seen.add(canonical)
        num_open = canonical.count(0)
        if num_open == 0:
            continue
        canonical_metas.append(canonical)
    
    buckets = defaultdict(list)
    for meta in canonical_metas:
        num_open = meta.count(0)
        for empty_cells in range(num_open, num_open * 8 + 1):
            min_diff, max_diff = _compute_diff_range(meta, empty_cells)
            if min_diff <= max_diff:
                key = _make_key(num_open, empty_cells)
                buckets[key].append((meta, min_diff, max_diff))
    
    return dict(buckets)


def _load() -> Dict[str, List[Tuple[Tuple[int, ...], int, int]]]:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Run 'python -m tablebase.meta_boards' first")
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)


def _save(data) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {DATA_FILE}")


# Load precomputed data
META_BOARDS: Dict[str, List[Tuple[Tuple[int, ...], int, int]]] = _load() if os.path.exists(DATA_FILE) else {}


def get_meta_boards(empty_cells: int) -> List[Tuple[Tuple[int, ...], int, int]]:
    """
    Get all valid meta-boards for given empty_cells.
    
    Returns:
        List[(meta, min_diff, max_diff)]
    """
    result = []
    for num_open in range(1, 10):
        key = _make_key(num_open, empty_cells)
        result.extend(META_BOARDS.get(key, []))
    return result


def get_meta_boards_by_open(num_open: int, empty_cells: int) -> List[Tuple[Tuple[int, ...], int, int]]:
    """Get meta-boards for specific num_open and empty_cells."""
    key = _make_key(num_open, empty_cells)
    return META_BOARDS.get(key, [])


def stats() -> Dict:
    total = sum(len(v) for v in META_BOARDS.values())
    return {
        'total_entries': total,
        'buckets': len(META_BOARDS)
    }


if __name__ == '__main__':
    print("Generating meta boards...")
    data = _generate()
    _save(data)
    
    META_BOARDS.update(data)
    s = stats()
    print(f"\nTotal entries: {s['total_entries']}")
    print(f"Buckets: {s['buckets']}")
    
    for empty in [1, 2, 3]:
        metas = get_meta_boards(empty)
        print(f"empty={empty}: {len(metas)} metas")
