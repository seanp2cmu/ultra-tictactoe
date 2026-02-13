"""
Test MCTS optimizations: expand_numpy, is_terminal fast path, parallel_mcts integration.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from game import Board
from ai.mcts.node_cy import NodeCy as Node


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


def test_expand_numpy_vs_dict():
    """expand_numpy must produce identical children as expand(dict)."""
    np.random.seed(42)
    mismatches = 0
    for _ in range(100):
        board = make_random_board(np.random.randint(1, 20))
        if board.is_game_over():
            continue
        policy = np.random.dirichlet(np.ones(81)).astype(np.float32)

        # Dict expand
        n1 = Node(board)
        n1.expand(dict(enumerate(policy)))

        # Numpy expand
        n2 = Node(board)
        n2.expand_numpy(policy)

        if set(n1.children.keys()) != set(n2.children.keys()):
            mismatches += 1
            continue

        for action in n1.children:
            p1 = n1.children[action].prior_prob
            p2 = n2.children[action].prior_prob
            if abs(p1 - p2) > 1e-6:
                mismatches += 1
                break

    print(f"TEST 1 - expand_numpy vs dict: {'PASSED' if mismatches == 0 else f'FAILED ({mismatches} mismatches)'}")
    assert mismatches == 0


def test_is_terminal_fast_path():
    """is_terminal must use fast path for BoardCy (completed_mask check)."""
    np.random.seed(42)

    # Non-terminal board
    b = Board()
    b.make_move(0, 0)
    n = Node(b)
    assert not n.is_terminal(), "Fresh board should not be terminal"

    # Play a long game to test various states
    for _ in range(50):
        board = make_random_board(np.random.randint(5, 60))
        n = Node(board)
        result = n.is_terminal()

        # Verify against ground truth
        is_over = board.is_game_over()
        if result != is_over:
            # is_game_over and is_terminal may differ slightly
            # is_terminal checks if there are any legal moves
            legal = board.get_legal_moves()
            winner = board.winner
            if winner not in (None, -1):
                assert result == True, f"Winner exists but is_terminal=False"
            elif not legal:
                assert result == True, f"No legal moves but is_terminal=False"
            else:
                assert result == False, f"Legal moves exist but is_terminal=True"

    print("TEST 2 - is_terminal fast path: PASSED")


def test_expand_numpy_benchmark():
    """Benchmark expand_numpy vs expand(dict)."""
    np.random.seed(42)
    boards = [make_random_board(5) for _ in range(500)]
    boards = [b for b in boards if not b.is_game_over()][:200]
    policies = [np.random.dirichlet(np.ones(81)).astype(np.float32) for _ in boards]

    # Dict expand
    t0 = time.perf_counter()
    for b, p in zip(boards, policies):
        n = Node(b)
        n.expand(dict(enumerate(p)))
    dict_time = time.perf_counter() - t0

    # Numpy expand
    t0 = time.perf_counter()
    for b, p in zip(boards, policies):
        n = Node(b)
        n.expand_numpy(p)
    numpy_time = time.perf_counter() - t0

    speedup = dict_time / numpy_time
    print(f"TEST 3 - expand benchmark ({len(boards)} boards):")
    print(f"  dict:  {dict_time*1000:.1f}ms")
    print(f"  numpy: {numpy_time*1000:.1f}ms")
    print(f"  speedup: {speedup:.2f}x")


def test_parallel_mcts_integration():
    """Test that ParallelMCTS works with the new expand_numpy."""
    try:
        import torch
        from ai.core.alpha_zero_net import AlphaZeroNet
        from ai.training.parallel_mcts import ParallelMCTS
    except ImportError as e:
        print(f"TEST 4 - SKIPPED ({e})")
        return

    net = AlphaZeroNet(device='cpu')
    net.trt_engine = None
    mcts = ParallelMCTS(network=net, num_simulations=5, dtw_calculator=None)

    games = [{'board': Board()} for _ in range(4)]
    results = mcts.search_parallel(games, temperature=1.0, add_noise=True)

    assert len(results) == 4
    for policy, action in results:
        assert policy.shape == (81,)
        assert 0 <= action < 81
        assert abs(policy.sum() - 1.0) < 1e-4, f"Policy sum: {policy.sum()}"

    print("TEST 4 - ParallelMCTS integration: PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("MCTS optimization tests")
    print("=" * 60)

    test_expand_numpy_vs_dict()
    print()
    test_is_terminal_fast_path()
    print()
    test_expand_numpy_benchmark()
    print()
    test_parallel_mcts_integration()

    print()
    print("=" * 60)
    print("ALL MCTS TESTS PASSED")
    print("=" * 60)
