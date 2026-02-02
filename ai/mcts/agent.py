"""AlphaZero MCTS agent with DTW endgame support."""
from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from game import Board

from .node import Node
from ai.core import AlphaZeroNet


class AlphaZeroAgent:
    """MCTS agent with neural network and DTW tablebase."""
    
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
        
        # DTW: 외부 주입 또는 기본 생성
        if dtw_calculator is not None:
            self.dtw_calculator = dtw_calculator
        else:
            from ..endgame import DTWCalculator
            self.dtw_calculator = DTWCalculator(
                use_cache=True,
                hot_size=50000,   # 5만 (테스트용 작은 캐시)
                cold_size=500000  # 50만
            )
    
    def search(self, board: Board) -> Node:
        """Run MCTS from root position."""
        root = Node(board)
        
        policy_probs, _ = self.network.predict(board)
        action_probs = dict(enumerate(policy_probs))
        root.expand(action_probs)
        
        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_size = min(self.batch_size, self.num_simulations - batch_idx * self.batch_size)
            
            search_paths: List[List[Node]] = []
            leaf_nodes: List[Node] = []
            leaf_boards: List[Board] = []
            tablebase_results: List[Tuple[int, float, Dict[int, float]]] = []
            
            for _ in range(batch_size):
                node = root
                search_path = [node]
                
                while node.is_expanded() and not node.is_terminal():
                    _, node = node.select_child(self.c_puct)
                    search_path.append(node)
                
                search_paths.append(search_path)
                node_idx = len(leaf_nodes)
                leaf_nodes.append(node)
                
                tablebase_hit = False
                if not node.is_terminal() and self.dtw_calculator.is_endgame(node.board):
                        result_data = self.dtw_calculator.calculate_dtw(node.board)
                        
                        if result_data is not None:
                            result, dtw, _ = result_data
                            value = float(result)
                            
                            legal_moves = node.board.get_legal_moves()
                            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                            expand_probs = {move[0] * 9 + move[1]: uniform_prob for move in legal_moves}
                            
                            tablebase_results.append((node_idx, value, expand_probs))
                            tablebase_hit = True
                
                if not node.is_terminal() and not tablebase_hit:
                    leaf_boards.append(node.board)
            
            # Convert to dict for O(1) lookup instead of O(n) search
            tablebase_dict = {idx: (value, expand_probs) for idx, value, expand_probs in tablebase_results}
            
            if leaf_boards:
                policy_probs_batch, values_batch = self.network.predict_batch(leaf_boards)
            
            leaf_idx = 0
            for i, node in enumerate(leaf_nodes):
                if node.is_terminal():
                    if node.board.winner is None or node.board.winner == 3:
                        value = 0.0
                    else:
                        if node.board.winner == node.board.current_player:
                            value = 1.0
                        else:
                            value = -1.0
                elif i in tablebase_dict:
                    value, expand_probs = tablebase_dict[i]
                    if expand_probs:
                        node.expand(expand_probs)
                else:
                    policy_probs = policy_probs_batch[leaf_idx]
                    value = values_batch[leaf_idx].item() if hasattr(values_batch[leaf_idx], 'item') else float(values_batch[leaf_idx].squeeze())
                    leaf_idx += 1
                    
                    action_probs = dict(enumerate(policy_probs))
                    node.expand(action_probs)
                
                for path_node in reversed(search_paths[i]):
                    path_node.update(value)
                    value = -value
        
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
