"""AlphaZero MCTS agent with DTW endgame support."""
from typing import List, Tuple, Dict, Optional
import random
import time
import numpy as np
from game import Board

from .node import Node
from ai.core import AlphaZeroNet

# 디버깅용 타이밍
_mcts_timing = {
    'network_predict': 0.0,
    'network_predict_batch': 0.0,
    'dtw_in_mcts': 0.0,
    'dtw_in_mcts_count': 0,
    'expand': 0.0,
    'select': 0.0,
    'backprop': 0.0,
}

def reset_mcts_timing():
    global _mcts_timing
    _mcts_timing = {k: 0.0 if isinstance(v, float) else 0 for k, v in _mcts_timing.items()}

def get_mcts_timing():
    return _mcts_timing.copy()


class AlphaZeroAgent:
    """MCTS agent with neural network and DTW endgame."""
    
    def __init__(
        self,
        network: AlphaZeroNet,
        num_simulations: int = 100,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        batch_size: int = 8,
        dtw_calculator=None
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.batch_size = batch_size
        
        if dtw_calculator is not None:
            self.dtw_calculator = dtw_calculator
        else:
            from ..endgame import DTWCalculator
            self.dtw_calculator = DTWCalculator(
                use_cache=True,
                hot_size=50000,
                cold_size=500000
            )
    
    def search(self, board: Board) -> Node:
        """Run MCTS from root position."""
        global _mcts_timing
        root = Node(board)
        
        t0 = time.time()
        policy_probs, _ = self.network.predict(board)
        _mcts_timing['network_predict'] += time.time() - t0
        
        action_probs = dict(enumerate(policy_probs))
        root.expand(action_probs)
        
        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_size = min(self.batch_size, self.num_simulations - batch_idx * self.batch_size)
            
            search_paths: List[List[Node]] = []
            leaf_nodes: List[Node] = []
            leaf_boards: List[Board] = []
            
            for _ in range(batch_size):
                node = root
                search_path = [node]
                
                t0 = time.time()
                while node.is_expanded() and not node.is_terminal():
                    _, node = node.select_child(self.c_puct)
                    search_path.append(node)
                _mcts_timing['select'] += time.time() - t0
                
                search_paths.append(search_path)
                leaf_nodes.append(node)
                
                if not node.is_terminal():
                    leaf_boards.append(node.board)
            
            if leaf_boards:
                t0 = time.time()
                policy_probs_batch, values_batch = self.network.predict_batch(leaf_boards)
                _mcts_timing['network_predict_batch'] += time.time() - t0
            
            leaf_idx = 0
            for i, node in enumerate(leaf_nodes):
                if node.is_terminal():
                    # Terminal value in MCTS range (-1~1)
                    if node.board.winner is None or node.board.winner == 3:
                        value = 0.0  # draw
                    else:
                        if node.board.winner == node.board.current_player:
                            value = 1.0  # win
                        else:
                            value = -1.0  # loss
                else:
                    policy_probs = policy_probs_batch[leaf_idx]
                    # Network outputs 0~1, convert to MCTS range -1~1
                    net_value = values_batch[leaf_idx].item() if hasattr(values_batch[leaf_idx], 'item') else float(values_batch[leaf_idx].squeeze())
                    value = 2.0 * net_value - 1.0  # 0~1 -> -1~1
                    leaf_idx += 1
                    
                    if self.dtw_calculator.is_endgame(node.board):
                        cached = self.dtw_calculator.lookup_cache(node.board)
                        if cached is not None:
                            result, _, _ = cached
                            value = float(result)  # DTW result is already -1/0/1
                    
                    t0 = time.time()
                    action_probs = dict(enumerate(policy_probs))
                    node.expand(action_probs)
                    _mcts_timing['expand'] += time.time() - t0
                
                t0 = time.time()
                for path_node in reversed(search_paths[i]):
                    path_node.update(value)
                    value = -value
                _mcts_timing['backprop'] += time.time() - t0
        
        return root
    
    def select_action(self, board: Board, temperature: Optional[float] = None) -> int:
        """Select action using MCTS."""
        if temperature is None:
            temperature = self.temperature
        
        if self.dtw_calculator.is_endgame(board):
                best_move, dtw = self.dtw_calculator.get_best_winning_move(board)
                if best_move and dtw < float('inf'):
                    return best_move[0] * 9 + best_move[1]
        
        root = self.search(board)
        
        action_visits = [(action, child.visits) for action, child in root.children.items()]
        
        if not action_visits:
            legal_moves = board.get_legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
                return move[0] * 9 + move[1]
            return 0
        
        actions, visits = zip(*action_visits)
        visits = np.array(visits, dtype=np.float32)
        
        if temperature == 0:
            action_idx = np.argmax(visits)
            return actions[action_idx]
        
        visits = visits ** (1.0 / temperature)
        probs = visits / np.sum(visits)
        action_idx = np.random.choice(len(actions), p=probs)
        
        return actions[action_idx]
    
    def get_action_probs(self, board: Board, temperature: Optional[float] = None) -> Dict[int, float]:
        """Get action probability distribution."""
        if temperature is None:
            temperature = self.temperature
        
        root = self.search(board)
        
        action_visits = [(action, child.visits) for action, child in root.children.items()]
        
        if not action_visits:
            return {}
        
        actions, visits = zip(*action_visits)
        visits = np.array(visits, dtype=np.float32)
        
        if temperature == 0:
            action_idx = np.argmax(visits)
            probs = np.zeros(len(actions))
            probs[action_idx] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)
        
        return {action: prob for action, prob in zip(actions, probs)}
