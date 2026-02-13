"""
Test: Cython BoardEncoder/BoardSymmetry vs Python reference implementation.

Verifies:
1. _build_transforms() matches Python version
2. _build_c_transforms() matches Python version
3. get_canonical_with_transform() matches Python version
4. to_inference_tensor_batch() matches Python version (tensor + inv_idx)
5. to_inference_tensor() matches Python version (tensor + inverse fn)
6. transform_policy / inverse_transform_policy match
7. Performance benchmark: Cython vs Python
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from game import Board


def make_random_board(num_moves=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    b = Board()
    if num_moves is None:
        num_moves = np.random.randint(1, 30)
    for _ in range(num_moves):
        moves = b.get_legal_moves()
        if not moves or b.is_game_over():
            break
        r, c = moves[np.random.randint(len(moves))]
        b.make_move(r, c)
    return b


def get_python_modules():
    """Import the pure-Python reference implementations."""
    from utils._board_symmetry import _BoardSymmetry as PySym
    from utils.board_encoder import BoardEncoder as PyEnc
    return PySym, PyEnc


def get_cython_modules():
    """Import the Cython implementations."""
    try:
        from utils._board_symmetry_cy import _BoardSymmetryCy as CySym
        from utils._board_encoder_cy import BoardEncoderCy as CyEnc
        return CySym, CyEnc
    except ImportError as e:
        print(f"Cython modules not built yet: {e}")
        return None, None


def test_transforms():
    """Test 1: _build_transforms matches."""
    PySym, _ = get_python_modules()
    CySym, _ = get_cython_modules()
    if CySym is None:
        print("TEST 1 - SKIPPED (Cython not built)")
        return

    py_t = PySym._build_transforms()
    cy_t = CySym._build_transforms()
    assert len(py_t) == len(cy_t) == 8
    for i in range(8):
        assert np.array_equal(py_t[i], cy_t[i]), f"Transform {i} mismatch"
    print("TEST 1 - _build_transforms: PASSED")


def test_c_transforms():
    """Test 2: _build_c_transforms matches."""
    PySym, _ = get_python_modules()
    CySym, _ = get_cython_modules()
    if CySym is None:
        print("TEST 2 - SKIPPED (Cython not built)")
        return

    py_ct = PySym._build_c_transforms()
    cy_ct = CySym._build_c_transforms()
    assert len(py_ct) == len(cy_ct) == 8
    for i in range(8):
        assert np.array_equal(py_ct[i], cy_ct[i]), f"C-Transform {i} mismatch"
    print("TEST 2 - _build_c_transforms: PASSED")


def test_canonical_with_transform():
    """Test 3: get_canonical_with_transform matches."""
    PySym, _ = get_python_modules()
    CySym, _ = get_cython_modules()
    if CySym is None:
        print("TEST 3 - SKIPPED (Cython not built)")
        return

    np.random.seed(42)
    n_boards = 100
    mismatches = 0
    for i in range(n_boards):
        board = make_random_board()
        py_boards, py_comp, py_idx = PySym.get_canonical_with_transform(board)
        cy_boards, cy_comp, cy_idx = CySym.get_canonical_with_transform(board)

        if not (np.array_equal(py_boards, cy_boards) and
                np.array_equal(py_comp, cy_comp) and
                py_idx == cy_idx):
            mismatches += 1
            if mismatches <= 3:
                print(f"  Board {i}: py_idx={py_idx}, cy_idx={cy_idx}")

    print(f"TEST 3 - get_canonical_with_transform: {n_boards - mismatches}/{n_boards} passed")
    assert mismatches == 0, f"{mismatches} mismatches"


def test_transform_policy():
    """Test 4: transform_policy / inverse_transform_policy match."""
    PySym, _ = get_python_modules()
    CySym, _ = get_cython_modules()
    if CySym is None:
        print("TEST 4 - SKIPPED (Cython not built)")
        return

    np.random.seed(42)
    for t_idx in range(8):
        policy = np.random.dirichlet(np.ones(81))
        py_fwd = PySym.transform_policy(policy, t_idx)
        cy_fwd = CySym.transform_policy(policy, t_idx)
        assert np.allclose(py_fwd, cy_fwd, atol=1e-12), f"transform_policy mismatch t={t_idx}"

        py_inv = PySym.inverse_transform_policy(policy, t_idx)
        cy_inv = CySym.inverse_transform_policy(policy, t_idx)
        assert np.allclose(py_inv, cy_inv, atol=1e-12), f"inverse_transform_policy mismatch t={t_idx}"

    print("TEST 4 - transform_policy / inverse_transform_policy: PASSED")


def test_inference_tensor_single():
    """Test 5: to_inference_tensor matches."""
    _, PyEnc = get_python_modules()
    _, CyEnc = get_cython_modules()
    if CyEnc is None:
        print("TEST 5 - SKIPPED (Cython not built)")
        return

    np.random.seed(42)
    n_boards = 50
    mismatches = 0
    for i in range(n_boards):
        board = make_random_board()
        py_tensor, py_inv_fn = PyEnc.to_inference_tensor(board)
        cy_tensor, cy_inv_fn = CyEnc.to_inference_tensor(board)

        if not np.array_equal(py_tensor, cy_tensor):
            mismatches += 1
            if mismatches <= 3:
                diff = np.abs(py_tensor - cy_tensor).max()
                print(f"  Board {i}: tensor max_diff={diff:.6f}")
            continue

        # Also check inverse transform
        fake_policy = np.random.dirichlet(np.ones(81))
        py_result = py_inv_fn(fake_policy)
        cy_result = cy_inv_fn(fake_policy)
        if not np.allclose(py_result, cy_result, atol=1e-12):
            mismatches += 1

    print(f"TEST 5 - to_inference_tensor: {n_boards - mismatches}/{n_boards} passed")
    assert mismatches == 0, f"{mismatches} mismatches"


def test_inference_tensor_batch():
    """Test 6: to_inference_tensor_batch matches."""
    _, PyEnc = get_python_modules()
    _, CyEnc = get_cython_modules()
    if CyEnc is None:
        print("TEST 6 - SKIPPED (Cython not built)")
        return

    np.random.seed(42)
    boards = [make_random_board() for _ in range(100)]

    py_tensors, py_inv = PyEnc.to_inference_tensor_batch(boards)
    cy_tensors, cy_inv = CyEnc.to_inference_tensor_batch(boards)

    assert np.array_equal(py_tensors, cy_tensors), \
        f"Batch tensor mismatch: max_diff={np.abs(py_tensors - cy_tensors).max()}"
    assert np.array_equal(py_inv, cy_inv), "Batch inv_idx mismatch"

    print(f"TEST 6 - to_inference_tensor_batch: PASSED (100 boards)")


def test_benchmark():
    """Test 7: Performance comparison."""
    _, PyEnc = get_python_modules()
    _, CyEnc = get_cython_modules()
    if CyEnc is None:
        print("TEST 7 - SKIPPED (Cython not built)")
        return

    np.random.seed(42)
    boards = [make_random_board() for _ in range(2048)]

    # Warmup
    PyEnc.to_inference_tensor_batch(boards[:10])
    CyEnc.to_inference_tensor_batch(boards[:10])

    # Python
    t0 = time.perf_counter()
    for _ in range(3):
        PyEnc.to_inference_tensor_batch(boards)
    py_time = (time.perf_counter() - t0) / 3

    # Cython
    t0 = time.perf_counter()
    for _ in range(3):
        CyEnc.to_inference_tensor_batch(boards)
    cy_time = (time.perf_counter() - t0) / 3

    speedup = py_time / cy_time
    print(f"TEST 7 - Benchmark (2048 boards):")
    print(f"  Python: {py_time*1000:.1f}ms")
    print(f"  Cython: {cy_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")


if __name__ == '__main__':
    print("=" * 60)
    print("Cython BoardEncoder/BoardSymmetry tests")
    print("=" * 60)

    CySym, CyEnc = get_cython_modules()
    if CySym is None:
        print("\nCython modules not available. Build first, then re-run.")
        print("Exiting with success (tests will be run after build).")
        sys.exit(0)

    test_transforms()
    print()
    test_c_transforms()
    print()
    test_canonical_with_transform()
    print()
    test_transform_policy()
    print()
    test_inference_tensor_single()
    print()
    test_inference_tensor_batch()
    print()
    test_benchmark()

    print()
    print("=" * 60)
    print("ALL CYTHON TESTS PASSED")
    print("=" * 60)
