"""
Test: BoardEncoder.to_inference_tensor vs to_inference_tensor_batch consistency.

Verifies:
1. Canonical tensors match (single vs batch)
2. Inverse transform indices match
3. Round-trip: transform -> inverse gives original policy
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from game import Board
from utils._board_encoder_cy import BoardEncoderCy as BoardEncoder
from utils._board_symmetry_cy import _BoardSymmetryCy as _BoardSymmetry


def make_random_board(num_moves=None):
    """Create a board with random legal moves."""
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


def test_tensor_match():
    """Test 1: Canonical tensors from single and batch must be identical."""
    np.random.seed(42)
    n_boards = 50
    boards = [make_random_board() for _ in range(n_boards)]

    batch_tensors, batch_inv_idx = BoardEncoder.to_inference_tensor_batch(boards)

    mismatches = 0
    for i, board in enumerate(boards):
        single_tensor, inv_fn = BoardEncoder.to_inference_tensor(board)
        
        if not np.array_equal(single_tensor, batch_tensors[i]):
            diff = np.abs(single_tensor - batch_tensors[i])
            max_diff = diff.max()
            # Find which channels differ
            for ch in range(7):
                ch_diff = np.abs(single_tensor[ch] - batch_tensors[i][ch]).max()
                if ch_diff > 0:
                    print(f"  Board {i}: channel {ch} max_diff={ch_diff:.6f}")
            mismatches += 1
            if mismatches <= 3:
                print(f"  Board {i}: max_diff={max_diff:.6f}")

    print(f"TEST 1 - Tensor match: {n_boards - mismatches}/{n_boards} passed")
    assert mismatches == 0, f"{mismatches} tensor mismatches found"


def test_inverse_transform_match():
    """Test 2: Inverse transform from single and batch must produce same result."""
    np.random.seed(42)
    n_boards = 50
    boards = [make_random_board() for _ in range(n_boards)]

    batch_tensors, batch_inv_idx = BoardEncoder.to_inference_tensor_batch(boards)
    transforms_81 = _BoardSymmetry._build_transforms()

    mismatches = 0
    for i, board in enumerate(boards):
        single_tensor, inv_fn = BoardEncoder.to_inference_tensor(board)
        
        # Create a fake canonical policy (random)
        fake_canonical_policy = np.random.dirichlet(np.ones(81))
        
        # Single inverse
        single_result = inv_fn(fake_canonical_policy)
        
        # Batch inverse (same logic as predict_batch)
        batch_result = fake_canonical_policy[batch_inv_idx[i]]
        
        if not np.allclose(single_result, batch_result, atol=1e-10):
            max_diff = np.abs(single_result - batch_result).max()
            mismatches += 1
            if mismatches <= 3:
                print(f"  Board {i}: inverse max_diff={max_diff:.10f}")

    print(f"TEST 2 - Inverse transform match: {n_boards - mismatches}/{n_boards} passed")
    assert mismatches == 0, f"{mismatches} inverse transform mismatches found"


def test_roundtrip():
    """Test 3: apply transform then inverse = identity."""
    np.random.seed(123)
    transforms = _BoardSymmetry._build_transforms()
    
    for t_idx in range(8):
        original = np.random.dirichlet(np.ones(81))
        
        # Forward: original -> canonical
        inv_map = np.argsort(transforms[t_idx])
        canonical = original[inv_map]
        
        # Inverse: canonical -> original (using transforms[t_idx] as index)
        recovered = canonical[transforms[t_idx]]
        
        assert np.allclose(original, recovered, atol=1e-15), \
            f"Transform {t_idx}: roundtrip failed, max_diff={np.abs(original - recovered).max()}"
    
    print("TEST 3 - Round-trip all 8 transforms: PASSED")


def test_canonical_selection_consistency():
    """Test 4: Single and batch select the same canonical transform."""
    np.random.seed(42)
    n_boards = 100
    boards = [make_random_board() for _ in range(n_boards)]
    
    batch_tensors, batch_inv_idx = BoardEncoder.to_inference_tensor_batch(boards)
    transforms_81 = _BoardSymmetry._build_transforms()
    
    mismatches = 0
    for i, board in enumerate(boards):
        # Single: get transform_idx
        _, _, single_t_idx = _BoardSymmetry.get_canonical_with_transform(board)
        
        # Batch: reconstruct transform_idx from inv_idx
        # inv_idx[i] == transforms_81[best_idx[i]]
        batch_t_idx = None
        for t in range(8):
            if np.array_equal(batch_inv_idx[i], transforms_81[t]):
                batch_t_idx = t
                break
        
        if single_t_idx != batch_t_idx:
            mismatches += 1
            if mismatches <= 3:
                print(f"  Board {i}: single_t_idx={single_t_idx}, batch_t_idx={batch_t_idx}")
    
    print(f"TEST 4 - Canonical selection: {n_boards - mismatches}/{n_boards} passed")
    assert mismatches == 0, f"{mismatches} canonical selection mismatches"


def test_predict_consistency_no_trt():
    """Test 5: predict vs predict_batch with same PyTorch backend (no TRT)."""
    try:
        import torch
        from ai.core.alpha_zero_net import AlphaZeroNet
    except ImportError:
        print("TEST 5 - SKIPPED (torch not available)")
        return
    
    net = AlphaZeroNet(device='cpu')
    # Ensure TRT is disabled
    net.trt_engine = None
    
    np.random.seed(42)
    boards = [make_random_board() for _ in range(20)]
    
    batch_policies, batch_values = net.predict_batch(boards)
    
    max_policy_diff = 0.0
    max_value_diff = 0.0
    for i, board in enumerate(boards):
        single_policy, single_value = net.predict(board)
        
        p_diff = np.abs(single_policy - batch_policies[i]).max()
        v_diff = abs(single_value - batch_values[i][0])
        max_policy_diff = max(max_policy_diff, p_diff)
        max_value_diff = max(max_value_diff, v_diff)
    
    print(f"TEST 5 - predict consistency (CPU, no TRT):")
    print(f"  max policy diff: {max_policy_diff:.2e}")
    print(f"  max value diff:  {max_value_diff:.2e}")
    assert max_policy_diff < 1e-5, f"Policy mismatch: {max_policy_diff}"
    assert max_value_diff < 1e-5, f"Value mismatch: {max_value_diff}"
    print("  PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("BoardEncoder consistency tests")
    print("=" * 60)
    
    test_roundtrip()
    print()
    test_canonical_selection_consistency()
    print()
    test_tensor_match()
    print()
    test_inverse_transform_match()
    print()
    test_predict_consistency_no_trt()
    
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
