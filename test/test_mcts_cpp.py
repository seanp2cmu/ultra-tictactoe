"""Test C++ MCTS implementation."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'cpp')

import pytest
import time

from board_cpp import Board, Node


class TestNodeBasics:
    def test_init(self):
        board = Board()
        node = Node(board)
        assert node.visits == 0
        assert node.value_sum == 0
        assert node.prior_prob == 0
        assert node.parent is None
        assert len(node.children) == 0

    def test_init_with_params(self):
        board = Board()
        node = Node(board, action=40, prior_prob=0.5)
        assert node.action == 40
        assert node.prior_prob == 0.5


class TestNodeState:
    def test_is_expanded_false(self):
        board = Board()
        node = Node(board)
        assert not node.is_expanded()

    def test_is_expanded_true(self):
        board = Board()
        node = Node(board)
        action_probs = {i: 1.0/81 for i in range(81)}
        node.expand(action_probs)
        assert node.is_expanded()

    def test_is_terminal_false(self):
        board = Board()
        node = Node(board)
        assert not node.is_terminal()

    def test_is_terminal_true(self):
        board = Board()
        board.winner = 1
        node = Node(board)
        assert node.is_terminal()

    def test_value_zero_visits(self):
        board = Board()
        node = Node(board)
        assert node.value() == 0

    def test_value_with_visits(self):
        board = Board()
        node = Node(board)
        node.visits = 10
        node.value_sum = 5
        assert node.value() == 0.5


class TestNodeExpand:
    def test_expand_creates_children(self):
        board = Board()
        node = Node(board)
        action_probs = {i: 1.0/81 for i in range(81)}
        node.expand(action_probs)
        assert len(node.children) == 81

    def test_expand_children_have_priors(self):
        board = Board()
        node = Node(board)
        action_probs = {40: 0.5, 41: 0.3}
        node.expand(action_probs)
        assert abs(node.children[40].prior_prob - 0.5) < 0.01
        assert abs(node.children[41].prior_prob - 0.3) < 0.01


class TestNodeSelect:
    def test_select_child_ucb(self):
        board = Board()
        node = Node(board)
        action_probs = {40: 0.5, 41: 0.3, 42: 0.2}
        node.expand(action_probs)
        node.visits = 100
        
        node.children[40].visits = 50
        node.children[40].value_sum = 25
        node.children[41].visits = 1
        node.children[41].value_sum = 0.9
        
        action, child = node.select_child(1.0)
        assert action is not None
        assert child is not None


class TestNodeUpdate:
    def test_update(self):
        board = Board()
        node = Node(board)
        node.update(0.5)
        assert node.visits == 1
        assert node.value_sum == 0.5

    def test_update_multiple(self):
        board = Board()
        node = Node(board)
        node.update(0.5)
        node.update(0.3)
        node.update(-0.2)
        assert node.visits == 3
        assert abs(node.value_sum - 0.6) < 1e-5


class TestMCTSSimulation:
    def test_mcts_simulation(self):
        board = Board()
        root = Node(board)
        num_simulations = 100
        
        for _ in range(num_simulations):
            node = root
            path = [node]
            
            while node.is_expanded() and not node.is_terminal():
                _, node = node.select_child(1.0)
                path.append(node)
            
            if not node.is_terminal() and not node.is_expanded():
                action_probs = {i: 1.0/81 for i in range(81)}
                node.expand(action_probs)
            
            value = 0.5
            for n in reversed(path):
                n.update(value)
                value = -value
        
        assert root.visits == num_simulations
        assert len(root.children) == 81


class TestBenchmark:
    def test_expand_speed(self):
        board = Board()
        action_probs = {i: 1.0/81 for i in range(81)}
        
        n = 1000
        t0 = time.perf_counter()
        for _ in range(n):
            node = Node(board)
            node.expand(action_probs)
        elapsed = time.perf_counter() - t0
        print(f"\nC++ Expand {n}x: {elapsed*1000:.2f}ms ({elapsed/n*1e6:.2f}µs/op)")

    def test_select_speed(self):
        board = Board()
        node = Node(board)
        action_probs = {i: 1.0/81 for i in range(81)}
        node.expand(action_probs)
        node.visits = 1000
        for child in node.children.values():
            child.visits = 10
            child.value_sum = 5
        
        n = 10000
        t0 = time.perf_counter()
        for _ in range(n):
            node.select_child(1.0)
        elapsed = time.perf_counter() - t0
        print(f"\nC++ Select {n}x: {elapsed*1000:.2f}ms ({elapsed/n*1e6:.2f}µs/op)")

    def test_mcts_speed(self):
        board = Board()
        n_sims = 500
        t0 = time.perf_counter()
        
        root = Node(board)
        for _ in range(n_sims):
            node = root
            path = [node]
            
            while node.is_expanded() and not node.is_terminal():
                _, node = node.select_child(1.0)
                path.append(node)
            
            if not node.is_terminal() and not node.is_expanded():
                action_probs = {i: 0.01 for i in range(81)}
                node.expand(action_probs)
            
            value = 0.5
            for n in reversed(path):
                n.update(value)
                value = -value
        
        elapsed = time.perf_counter() - t0
        print(f"\nC++ MCTS {n_sims} sims: {elapsed*1000:.2f}ms ({elapsed/n_sims*1000:.3f}ms/sim)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
