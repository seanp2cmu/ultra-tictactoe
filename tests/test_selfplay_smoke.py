"""Smoke test: run a few self-play games to verify all optimizations work end-to-end."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

def test_selfplay_smoke():
    from ai.core.alpha_zero_net import AlphaZeroNet
    from ai.training.self_play_worker import SelfPlayWorker
    from ai.training.parallel_mcts import reset_parallel_timing, get_parallel_timing

    net = AlphaZeroNet(device='cpu')
    net.trt_engine = None

    worker = SelfPlayWorker(
        network=net,
        dtw_calculator=None,
        num_simulations=5,
        temperature=1.0,
        parallel_games=4
    )

    reset_parallel_timing()
    t0 = time.perf_counter()
    data = worker.play_games(num_games=4, disable_tqdm=True)
    elapsed = time.perf_counter() - t0

    timing = get_parallel_timing()
    print(f"Self-play smoke test:")
    print(f"  Games: 4, Samples: {len(data)}")
    print(f"  Total: {elapsed:.2f}s")
    print(f"  Network: {timing['network_time']:.2f}s ({timing['network_time']/elapsed*100:.0f}%)")
    print(f"  MCTS overhead: {timing['mcts_overhead']:.2f}s ({timing['mcts_overhead']/elapsed*100:.0f}%)")

    assert len(data) > 0, "No training data generated"
    for state, policy, value, game_id in data[:3]:
        assert state.shape == (7, 9, 9), f"Bad state shape: {state.shape}"
        assert policy.shape == (81,), f"Bad policy shape: {policy.shape}"
        assert 0.0 <= value <= 1.0, f"Bad value: {value}"

    print("  PASSED")

if __name__ == '__main__':
    test_selfplay_smoke()
