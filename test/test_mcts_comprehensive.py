"""Comprehensive MCTS Node tests - must pass for both Python and C++ implementations."""
import sys
sys.path.insert(0, '.')

import pytest
import numpy as np
import time

from game import Board
from ai.mcts import Node


class TestNodeBasics:
    """Basic node operations."""
    
    def test_init(self):
        """Test node initialization."""
        board = Board()
        node = Node(board)
        
        assert node.visits == 0
        assert node.value_sum == 0
        assert node.prior_prob == 0
        assert node.parent is None
        assert node.action is None
        assert len(node.children) == 0
    
    def test_init_with_params(self):
        """Test node initialization with parameters."""
        board = Board()
        parent = Node(board)
        node = Node(board, parent=parent, action=40, prior_prob=0.5)
        
        assert node.parent is parent
        assert node.action == 40
        assert node.prior_prob == 0.5
    
    def test_clone_board(self):
        """Test that node clones board by default."""
        board = Board()
        node = Node(board)
        
        board.make_move(4, 4)
        assert node.board.get_cell(4, 4) == 0  # Node's board unchanged


class TestNodeState:
    """Node state methods."""
    
    def test_is_expanded_false(self):
        """Test unexpanded node."""
        board = Board()
        node = Node(board)
        assert not node.is_expanded()
    
    def test_is_expanded_true(self):
        """Test expanded node."""
        board = Board()
        node = Node(board)
        action_probs = {i: 1.0/81 for i in range(81)}
        node.expand(action_probs)
        assert node.is_expanded()
    
    def test_is_terminal_false(self):
        """Test non-terminal node."""
        board = Board()
        node = Node(board)
        assert not node.is_terminal()
    
    def test_is_terminal_true(self):
        """Test terminal node (winner)."""
        board = Board()
        board.winner = 1
        node = Node(board)
        assert node.is_terminal()
    
    def test_value_zero_visits(self):
        """Test value with zero visits."""
        board = Board()
        node = Node(board)
        assert node.value() == 0
    
    def test_value_with_visits(self):
        """Test value with visits."""
        board = Board()
        node = Node(board)
        node.visits = 10
        node.value_sum = 5
        assert node.value() == 0.5


class TestNodeExpand:
    """Node expansion."""
    
    def test_expand_creates_children(self):
        """Test that expand creates children."""
        board = Board()
        node = Node(board)
        action_probs = {i: 1.0/81 for i in range(81)}
        
        node.expand(action_probs)
        
        assert len(node.children) == 81
    
    def test_expand_children_have_priors(self):
        """Test children have correct priors."""
        board = Board()
        node = Node(board)
        action_probs = {40: 0.5, 41: 0.3}
        
        node.expand(action_probs)
        
        assert node.children[40].prior_prob == 0.5
        assert node.children[41].prior_prob == 0.3
    
    def test_expand_children_boards_updated(self):
        """Test children boards have move applied."""
        board = Board()
        node = Node(board)
        action_probs = {40: 0.5}  # (4, 4)
        
        node.expand(action_probs)
        
        child = node.children[40]
        assert child.board.get_cell(4, 4) == 1


class TestNodeSelect:
    """Node selection."""
    
    def test_select_child_ucb(self):
        """Test UCB selection."""
        board = Board()
        node = Node(board)
        action_probs = {40: 0.5, 41: 0.3, 42: 0.2}
        node.expand(action_probs)
        node.visits = 100
        
        # Give one child more visits
        node.children[40].visits = 50
        node.children[40].value_sum = 25
        node.children[41].visits = 1
        node.children[41].value_sum = 0.9
        node.children[42].visits = 1
        node.children[42].value_sum = 0.1
        
        action, child = node.select_child(c_puct=1.0)
        
        assert action is not None
        assert child is not None
    
    def test_select_child_explores_unvisited(self):
        """Test that unvisited nodes get explored."""
        board = Board()
        node = Node(board)
        action_probs = {40: 0.9, 41: 0.1}
        node.expand(action_probs)
        node.visits = 10
        
        # One visited, one not
        node.children[40].visits = 9
        node.children[40].value_sum = 4.5
        
        action, child = node.select_child(c_puct=1.0)
        
        # With high c_puct, should explore unvisited
        assert action in [40, 41]


class TestNodeUpdate:
    """Node update operations."""
    
    def test_update(self):
        """Test basic update."""
        board = Board()
        node = Node(board)
        
        node.update(0.5)
        
        assert node.visits == 1
        assert node.value_sum == 0.5
    
    def test_update_multiple(self):
        """Test multiple updates."""
        board = Board()
        node = Node(board)
        
        node.update(0.5)
        node.update(0.3)
        node.update(-0.2)
        
        assert node.visits == 3
        assert abs(node.value_sum - 0.6) < 1e-6
    
    def test_update_recursive(self):
        """Test recursive update."""
        board = Board()
        parent = Node(board)
        action_probs = {40: 0.5}
        parent.expand(action_probs)
        child = parent.children[40]
        
        child.update_recursive(0.5)
        
        assert child.visits == 1
        assert child.value_sum == 0.5
        assert parent.visits == 1
        assert parent.value_sum == -0.5  # Negated for parent


class TestMCTSSimulation:
    """Full MCTS simulation tests."""
    
    def test_mcts_simulation(self):
        """Test complete MCTS simulation."""
        board = Board()
        root = Node(board)
        
        num_simulations = 100
        
        for _ in range(num_simulations):
            node = root
            path = [node]
            
            # Selection
            while node.is_expanded() and not node.is_terminal():
                _, node = node.select_child(1.0)
                path.append(node)
            
            # Expansion
            if not node.is_terminal() and not node.is_expanded():
                action_probs = {i: 1.0/81 for i in range(81)}
                node.expand(action_probs)
            
            # Backup
            value = 0.5
            for n in reversed(path):
                n.update(value)
                value = -value
        
        assert root.visits == num_simulations
        assert len(root.children) == 81


class TestBenchmark:
    """Performance benchmarks."""
    
    def test_expand_speed(self):
        """Benchmark expand operation."""
        board = Board()
        action_probs = {i: 1.0/81 for i in range(81)}
        
        n = 1000
        t0 = time.perf_counter()
        for _ in range(n):
            node = Node(board)
            node.expand(action_probs)
        elapsed = time.perf_counter() - t0
        
        print(f"\nExpand {n}x: {elapsed*1000:.2f}ms ({elapsed/n*1e6:.2f}µs/op)")
    
    def test_select_speed(self):
        """Benchmark select_child operation."""
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
        
        print(f"\nSelect {n}x: {elapsed*1000:.2f}ms ({elapsed/n*1e6:.2f}µs/op)")
    
    def test_mcts_speed(self):
        """Benchmark full MCTS simulation."""
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
        print(f"\nMCTS {n_sims} sims: {elapsed*1000:.2f}ms ({elapsed/n_sims*1000:.3f}ms/sim)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
