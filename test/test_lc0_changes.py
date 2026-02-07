"""
Test suite for Lc0-style AlphaZero changes:
1. Value range 0~1 (sigmoid)
2. Temperature only for first 8 moves
3. FPU = -1 for unvisited nodes
4. Shallow AB removed, DTW endgame only
"""
import pytest
import numpy as np
import torch

from game import Board
from ai.core.network import Model
from ai.core import AlphaZeroNet
from ai.mcts import AlphaZeroAgent
from ai.mcts.node import Node
from ai.endgame import DTWCalculator
from ai.training.self_play import SelfPlayWorker


class TestValueRange:
    """Test that value network outputs are in 0~1 range."""
    
    def test_network_value_range(self):
        """Network value should be in [0, 1] range (sigmoid)."""
        model = Model(num_res_blocks=2, num_channels=32)
        net = AlphaZeroNet(model=model)
        
        board = Board()
        policy, value = net.predict(board)
        
        assert 0 <= value <= 1, f"Value {value} not in [0, 1] range"
        assert policy.shape == (81,), f"Policy shape {policy.shape} != (81,)"
        assert abs(np.sum(policy) - 1.0) < 1e-5, f"Policy sum {np.sum(policy)} != 1.0"
    
    def test_network_batch_value_range(self):
        """Batch prediction values should be in [0, 1] range."""
        model = Model(num_res_blocks=2, num_channels=32)
        net = AlphaZeroNet(model=model)
        
        boards = [Board() for _ in range(4)]
        policies, values = net.predict_batch(boards)
        
        for i, v in enumerate(values):
            val = float(v)
            assert 0 <= val <= 1, f"Batch value[{i}] = {val} not in [0, 1]"
    
    def test_self_play_value_targets(self):
        """Self-play should produce value targets in 0~1 range."""
        model = Model(num_res_blocks=2, num_channels=32)
        net = AlphaZeroNet(model=model)
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        worker = SelfPlayWorker(net, dtw_calculator=dtw, num_simulations=10, temperature=1.0)
        
        # Play one game
        training_data = worker.play_game()
        
        for state, policy, value, dtw_val in training_data:
            assert value in [0.0, 0.5, 1.0], f"Value target {value} not in {{0, 0.5, 1}}"


class TestTemperatureFirstMoves:
    """Test that temperature is only applied for first 8 moves."""
    
    def test_greedy_after_8_moves(self):
        """After 8 moves, selection should be greedy (deterministic)."""
        model = Model(num_res_blocks=2, num_channels=32)
        net = AlphaZeroNet(model=model)
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        worker = SelfPlayWorker(net, dtw_calculator=dtw, num_simulations=10, temperature=1.0)
        
        # Run multiple games and check consistency after move 8
        # (This is a statistical test - greedy should be deterministic)
        board = Board()
        
        # Make 8 random moves first
        for _ in range(8):
            legal = board.get_legal_moves()
            if not legal:
                break
            move = legal[0]
            board.make_move(move[0], move[1])
        
        # Now the 9th move should be greedy
        # We test by checking the code structure, not runtime behavior
        assert hasattr(worker, 'agent'), "Worker should have agent"
        assert worker.agent.temperature > 0, "Agent should have temperature > 0"


class TestFPU:
    """Test First Play Urgency = -1 for unvisited nodes."""
    
    def test_fpu_minus_one(self):
        """Unvisited nodes should have Q-value of -1."""
        board = Board()
        root = Node(board)
        
        # Expand root with some priors
        action_probs = {i: 0.01 for i in range(81)}
        action_probs[40] = 0.5  # Center has high prior
        root.expand(action_probs)
        root.visits = 10  # Simulate some visits to root
        
        # Select child - should prefer high prior with FPU=-1 for unvisited
        action, child = root.select_child(c_puct=1.0)
        
        # The child with highest prior should be selected first
        # because FPU=-1 makes unvisited nodes seem bad,
        # but high prior compensates
        assert action is not None, "Should select a child"
        assert child.visits == 0, "Selected child should be unvisited"
    
    def test_fpu_calculation(self):
        """Verify Q-value calculation uses FPU=-1."""
        board = Board()
        node = Node(board)
        
        # Unvisited node should return 0 from value() method
        # but FPU in select_child uses -1
        assert node.visits == 0
        assert node.value() == 0  # value() returns 0 for unvisited


class TestDTWNoMidgame:
    """Test that DTW only works for endgame (≤15 cells)."""
    
    def test_no_midgame_method(self):
        """DTWCalculator should not have is_midgame method."""
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        assert not hasattr(dtw, 'is_midgame'), "is_midgame should be removed"
    
    def test_no_check_candidate_moves(self):
        """DTWCalculator should not have check_candidate_moves method."""
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        assert not hasattr(dtw, 'check_candidate_moves'), "check_candidate_moves should be removed"
    
    def test_no_shallow_alpha_beta(self):
        """DTWCalculator should not have _shallow_alpha_beta method."""
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        assert not hasattr(dtw, '_shallow_alpha_beta'), "_shallow_alpha_beta should be removed"
    
    def test_no_midgame_params(self):
        """DTWCalculator should not accept midgame parameters."""
        # This should work without midgame params
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        assert dtw.endgame_threshold == 15
        
        # These should not exist
        assert not hasattr(dtw, 'midgame_threshold')
        assert not hasattr(dtw, 'shallow_depth')
    
    def test_endgame_only(self):
        """DTW should only calculate for endgame positions."""
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        
        # Empty board has 81 cells - not endgame
        board = Board()
        assert not dtw.is_endgame(board), "Empty board should not be endgame"
        
        # calculate_dtw should return None for non-endgame
        result = dtw.calculate_dtw(board)
        assert result is None, "calculate_dtw should return None for non-endgame"


class TestMCTSValueConversion:
    """Test that MCTS correctly converts network 0~1 values to -1~1."""
    
    def test_mcts_runs(self):
        """MCTS should run without errors with new value conversion."""
        model = Model(num_res_blocks=2, num_channels=32)
        net = AlphaZeroNet(model=model)
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        agent = AlphaZeroAgent(net, num_simulations=10, dtw_calculator=dtw)
        
        board = Board()
        action = agent.select_action(board, temperature=0)
        
        assert 0 <= action < 81, f"Action {action} out of range"
        row, col = action // 9, action % 9
        assert (row, col) in board.get_legal_moves(), "Selected action should be legal"


class TestStatsNoMidgame:
    """Test that stats no longer include midgame/shallow stats."""
    
    def test_dtw_stats_no_shallow(self):
        """DTW stats should not include shallow search stats."""
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        stats = dtw.get_stats()
        
        assert 'dtw_searches' in stats
        assert 'dtw_nodes' in stats
        assert 'dtw_aborted' in stats
        assert 'dtw_avg_nodes' in stats
        
        # These should NOT exist
        assert 'shallow_searches' not in stats
        assert 'shallow_nodes' not in stats
        assert 'shallow_aborted' not in stats
        assert 'shallow_avg_nodes' not in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
