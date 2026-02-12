#!/usr/bin/env python3
"""Test parallel self-play implementation."""
import sys
import time
sys.path.insert(0, '.')

import numpy as np
from game import Board
from ai.core import AlphaZeroNet
from ai.training.parallel_self_play import ParallelMCTS, ParallelSelfPlayWorker
from ai.endgame import DTWCalculator


def test_parallel_mcts():
    """Test ParallelMCTS with multiple games."""
    print("=" * 60)
    print("Test 1: ParallelMCTS")
    print("=" * 60)
    
    network = AlphaZeroNet(device='mps')
    dtw = DTWCalculator(use_cache=True)
    
    mcts = ParallelMCTS(
        network=network,
        num_simulations=50,  # Low for testing
        dtw_calculator=dtw
    )
    
    # Create 4 games at different states
    games = []
    for i in range(4):
        board = Board()
        # Make some random moves
        for _ in range(i * 2):
            legal = board.get_legal_moves()
            if legal and board.winner in (None, -1):
                board.make_move(*legal[np.random.randint(len(legal))])
        games.append({'board': board})
    
    print(f"Testing with {len(games)} games...")
    
    t0 = time.perf_counter()
    results = mcts.search_parallel(games, temperature=1.0, add_noise=True)
    elapsed = time.perf_counter() - t0
    
    print(f"Time: {elapsed*1000:.1f}ms")
    print(f"Results: {len(results)} (policy, action) tuples")
    
    for i, (policy, action) in enumerate(results):
        assert policy.shape == (81,), f"Policy shape mismatch: {policy.shape}"
        assert 0 <= action < 81, f"Invalid action: {action}"
        assert abs(policy.sum() - 1.0) < 0.01, f"Policy not normalized: {policy.sum()}"
        print(f"  Game {i}: action={action}, policy_max={policy.max():.3f}")
    
    print("✓ ParallelMCTS test passed\n")
    return True


def test_parallel_worker():
    """Test ParallelSelfPlayWorker."""
    print("=" * 60)
    print("Test 2: ParallelSelfPlayWorker")
    print("=" * 60)
    
    network = AlphaZeroNet(device='mps')
    dtw = DTWCalculator(use_cache=True)
    
    worker = ParallelSelfPlayWorker(
        network=network,
        dtw_calculator=dtw,
        num_simulations=30,  # Low for testing
        temperature=1.0,
        parallel_games=4
    )
    
    print("Playing 4 games in parallel (2 batches)...")
    
    t0 = time.perf_counter()
    data = worker.play_games(num_games=4)
    elapsed = time.perf_counter() - t0
    
    print(f"Time: {elapsed:.1f}s")
    print(f"Samples generated: {len(data)}")
    
    # Validate data format
    for i, (state, policy, value, dtw_val) in enumerate(data[:5]):
        assert state.shape == (7, 9, 9), f"State shape mismatch: {state.shape}"
        assert policy.shape == (81,), f"Policy shape mismatch: {policy.shape}"
        assert 0 <= value <= 1, f"Value out of range: {value}"
    
    print(f"  First 5 samples validated")
    print(f"  Avg samples per game: {len(data) / 4:.1f}")
    
    print("✓ ParallelSelfPlayWorker test passed\n")
    return True


def test_speed_comparison():
    """Compare sequential vs parallel speed."""
    print("=" * 60)
    print("Test 3: Speed Comparison (Sequential vs Parallel)")
    print("=" * 60)
    
    network = AlphaZeroNet(device='mps')
    dtw = DTWCalculator(use_cache=True)
    
    num_sims = 50
    num_games = 2
    
    # Sequential (1 game at a time)
    from ai.training.self_play import SelfPlayWorker
    
    print(f"Sequential: {num_games} games, {num_sims} sims each...")
    t0 = time.perf_counter()
    seq_samples = 0
    for _ in range(num_games):
        worker = SelfPlayWorker(
            network=network,
            dtw_calculator=dtw,
            num_simulations=num_sims,
            temperature=1.0
        )
        data = worker.play_game()
        seq_samples += len(data)
    seq_time = time.perf_counter() - t0
    print(f"  Time: {seq_time:.1f}s, Samples: {seq_samples}")
    
    # Parallel
    print(f"Parallel: {num_games} games simultaneously, {num_sims} sims each...")
    parallel_worker = ParallelSelfPlayWorker(
        network=network,
        dtw_calculator=dtw,
        num_simulations=num_sims,
        temperature=1.0,
        parallel_games=num_games
    )
    
    t0 = time.perf_counter()
    par_data = parallel_worker.play_games(num_games)
    par_time = time.perf_counter() - t0
    print(f"  Time: {par_time:.1f}s, Samples: {len(par_data)}")
    
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup > 1:
        print("✓ Parallel is faster")
    else:
        print("⚠ Parallel is slower (may be due to overhead with small test)")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("PARALLEL SELF-PLAY TESTS (MPS)")
    print("=" * 60 + "\n")
    
    try:
        test_parallel_mcts()
        test_parallel_worker()
        test_speed_comparison()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
