"""
Test: Cython BoardEncoder/BoardSymmetry standalone correctness.

Verifies:
1. _build_transforms() produces valid 8 D4 symmetry transforms
2. _build_c_transforms() produces valid 8 D4 3x3 transforms
3. transform_policy round-trip: transform -> inverse = identity
4. to_inference_tensor single vs batch consistency
5. to_inference_tensor_batch inverse transform correctness
6. Benchmark
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from game import Board
from utils._board_symmetry_cy import _BoardSymmetryCy as _BoardSymmetry
from utils._board_encoder_cy import BoardEncoderCy as BoardEncoder


def make_random_board(num_moves=None):
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


def test_transforms():
    """Test 1: 8 transforms are valid permutations of 81 elements."""
    transforms = _BoardSymmetry._build_transforms()
    assert len(transforms) == 8
    for i, t in enumerate(transforms):
        assert t.shape == (81,), f"Transform {i} shape: {t.shape}"
        assert len(set(t)) == 81, f"Transform {i} not a permutation"
    # Identity must be first
    assert np.array_equal(transforms[0], np.arange(81))
    print("TEST 1 - _build_transforms: PASSED")


def test_c_transforms():
    """Test 2: 8 c-transforms are valid permutations of 9 elements."""
    transforms = _BoardSymmetry._build_c_transforms()
    assert len(transforms) == 8
    for i, t in enumerate(transforms):
        assert t.shape == (9,), f"C-Transform {i} shape: {t.shape}"
        assert len(set(t)) == 9, f"C-Transform {i} not a permutation"
    assert np.array_equal(transforms[0], np.arange(9))
    print("TEST 2 - _build_c_transforms: PASSED")


def test_policy_roundtrip():
    """Test 3: transform_policy -> inverse = identity for all 8 transforms."""
    np.random.seed(42)
    for t_idx in range(8):
        policy = np.random.dirichlet(np.ones(81))
        fwd = _BoardSymmetry.transform_policy(policy, t_idx)
        recovered = _BoardSymmetry.inverse_transform_policy(fwd, t_idx)
        assert np.allclose(policy, recovered, atol=1e-12), \
            f"Round-trip failed for transform {t_idx}, max_diff={np.abs(policy - recovered).max()}"
    print("TEST 3 - Policy round-trip all 8 transforms: PASSED")


def test_single_vs_batch():
    """Test 4: to_inference_tensor single vs batch consistency."""
    np.random.seed(42)
    boards = [make_random_board() for _ in range(50)]
    batch_tensors, batch_inv = BoardEncoder.to_inference_tensor_batch(boards)

    mismatches = 0
    for i, board in enumerate(boards):
        single_tensor, inv_fn = BoardEncoder.to_inference_tensor(board)
        if not np.array_equal(single_tensor, batch_tensors[i]):
            mismatches += 1

        # Check inverse transform
        fake_policy = np.random.dirichlet(np.ones(81))
        single_result = inv_fn(fake_policy)
        batch_result = fake_policy[batch_inv[i]]
        if not np.allclose(single_result, batch_result, atol=1e-10):
            mismatches += 1

    print(f"TEST 4 - Single vs batch: {50 - mismatches}/50 passed")
    assert mismatches == 0


def test_batch_inverse():
    """Test 5: batch inverse transform produces valid policies."""
    np.random.seed(42)
    boards = [make_random_board() for _ in range(100)]
    batch_tensors, batch_inv = BoardEncoder.to_inference_tensor_batch(boards)

    assert batch_tensors.shape == (100, 7, 9, 9)
    assert batch_inv.shape == (100, 81)

    # Each inv row must be a permutation of 0..80
    for i in range(100):
        assert len(set(batch_inv[i])) == 81, f"Board {i}: inv not a permutation"

    print("TEST 5 - Batch inverse transform: PASSED")


def test_benchmark():
    """Test 6: Benchmark."""
    np.random.seed(42)
    boards = [make_random_board() for _ in range(2048)]

    # Warmup
    BoardEncoder.to_inference_tensor_batch(boards[:10])

    t0 = time.perf_counter()
    for _ in range(3):
        BoardEncoder.to_inference_tensor_batch(boards)
    elapsed = (time.perf_counter() - t0) / 3

    print(f"TEST 6 - Benchmark (2048 boards): {elapsed*1000:.1f}ms")


if __name__ == '__main__':
    print("=" * 60)
    print("Cython BoardEncoder/BoardSymmetry tests")
    print("=" * 60)

    test_transforms()
    print()
    test_c_transforms()
    print()
    test_policy_roundtrip()
    print()
    test_single_vs_batch()
    print()
    test_batch_inverse()
    print()
    test_benchmark()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
