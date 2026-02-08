"""
Precomputed Local 3x3 Board States

3^9 = 19,683 전수 열거 후 도달 가능한 OPEN 상태만 저장.
"diff_empty" 키로 인덱싱. diff = x_count - o_count.

Usage:
    boards = get_open_boards(empty=3, diff=1)  # key: "1_3"
    
Generate:
    python -m tablebase.local_boards
"""

import os
import pickle
from typing import Dict, List, Tuple
from collections import defaultdict
from itertools import product

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'local_boards.pkl')

WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diagonals
]


def _make_key(diff: int, empty: int) -> str:
    """Create bucket key: 'diff_empty'."""
    return f"{diff}_{empty}"


def _check_winner(cells: Tuple[int, ...]) -> int:
    """Check winner. Returns 0=none, 1=X, 2=O."""
    for a, b, c in WIN_PATTERNS:
        if cells[a] == cells[b] == cells[c] != 0:
            return cells[a]
    return 0


def _is_open_and_reachable(cells: Tuple[int, ...]) -> bool:
    """Check if board is OPEN (no winner) and reachable."""
    if cells.count(0) == 0:
        return False
    if _check_winner(cells) != 0:
        return False
    x_count = cells.count(1)
    o_count = cells.count(2)
    diff = x_count - o_count
    if not (-6 <= diff <= 6):
        return False
    return True


def _generate() -> Dict[str, List[Tuple[int, ...]]]:
    """Generate all OPEN 3x3 boards."""
    buckets = defaultdict(list)
    for cells in product([0, 1, 2], repeat=9):
        if not _is_open_and_reachable(cells):
            continue
        empty = cells.count(0)
        diff = cells.count(1) - cells.count(2)
        key = _make_key(diff, empty)
        buckets[key].append(cells)
    return dict(buckets)


def _load() -> Dict[str, List[Tuple[int, ...]]]:
    """Load from pickle file."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Run 'python -m tablebase.local_boards' first to generate {DATA_FILE}")
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)


def _save(data: Dict[str, List[Tuple[int, ...]]]) -> None:
    """Save to pickle file."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {DATA_FILE}")


# Load precomputed data
OPEN_BOARDS: Dict[str, List[Tuple[int, ...]]] = _load() if os.path.exists(DATA_FILE) else {}


def get_open_boards(empty: int, diff: int) -> List[Tuple[int, ...]]:
    """
    Get all OPEN boards with given empty count and diff.
    
    Args:
        empty: Number of empty cells (1-8)
        diff: x_count - o_count
    
    Returns:
        List of 9-tuples (0=empty, 1=X, 2=O)
    """
    key = _make_key(diff, empty)
    return OPEN_BOARDS.get(key, [])


def stats() -> Dict[str, int]:
    """Return statistics about precomputed boards."""
    total = sum(len(v) for v in OPEN_BOARDS.values())
    
    by_empty = defaultdict(int)
    by_diff = defaultdict(int)
    for key, boards in OPEN_BOARDS.items():
        diff, empty = key.split('_')
        by_empty[int(empty)] += len(boards)
        by_diff[int(diff)] += len(boards)
    
    return {
        'total_open': total,
        'buckets': len(OPEN_BOARDS),
        'by_empty': dict(by_empty),
        'by_diff': dict(by_diff)
    }


if __name__ == '__main__':
    print("Generating local boards...")
    data = _generate()
    _save(data)
    
    # Reload and show stats
    OPEN_BOARDS.update(data)
    s = stats()
    print(f"\nTotal OPEN boards: {s['total_open']}")
    print(f"Buckets: {s['buckets']}")
    print(f"\nBy empty count:")
    for e in sorted(s['by_empty'].keys()):
        print(f"  {e} empty: {s['by_empty'][e]}")
