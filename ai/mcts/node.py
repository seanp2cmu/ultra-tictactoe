"""MCTS node for AlphaZero."""
from typing import Dict, Optional, Tuple
import math

from game import Board


class Node:
    """Monte Carlo Tree Search node."""
    
    def __init__(
        self,
        board: Board,
        parent: Optional['Node'] = None,
        action: Optional[int] = None,
        prior_prob: float = 0
    ) -> None:
        self.board = board.clone()
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.children: Dict[int, 'Node'] = {}
        self.visits = 0
        self.value_sum = 0
        
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def is_terminal(self) -> bool:
        return self.board.winner is not None or len(self.board.get_legal_moves()) == 0
    
    def value(self) -> float:
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[int, 'Node']:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            q_value = child.value()
            u_value = c_puct * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs: Dict[int, float]) -> None:
        """Expand node by creating children for legal moves."""
        legal_moves = self.board.get_legal_moves()
        
        for move in legal_moves:
            action = move[0] * 9 + move[1]
            if action not in self.children:
                next_board = self.board.clone()
                next_board.make_move(move[0], move[1])
                prior = action_probs.get(action, 1e-8)
                self.children[action] = Node(next_board, parent=self, action=action, prior_prob=prior)
    
    def update(self, value: float) -> None:
        """Update visit count and value sum."""
        self.visits += 1
        self.value_sum += value
    
    def update_recursive(self, value: float) -> None:
        """Recursively update ancestors."""
        if self.parent:
            self.parent.update_recursive(-value)
        self.update(value)
