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
        prior_prob: float = 0,
        _clone: bool = True
    ) -> None:
        self.board = board.clone() if _clone else board
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.children: Dict[int, 'Node'] = {}
        self.visits = 0
        self.value_sum = 0
        self._is_terminal: Optional[bool] = None
        
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def is_terminal(self) -> bool:
        if self._is_terminal is not None:
            return self._is_terminal
        
        if self.board.winner is not None:
            self._is_terminal = True
            return True
        
        for br in range(3):
            for bc in range(3):
                if self.board.completed_boards[br][bc] == 0:
                    for r in range(br*3, br*3+3):
                        for c in range(bc*3, bc*3+3):
                            if self.board.boards[r][c] == 0:
                                self._is_terminal = False
                                return False
        
        self._is_terminal = True
        return True
    
    def value(self) -> float:
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[int, 'Node']:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sqrt_visits = math.sqrt(self.visits)
        exploration_factor = c_puct * sqrt_visits
        
        for action, child in self.children.items():
            # FPU (First Play Urgency) = -1 for unvisited nodes (AlphaZero style)
            q_value = child.value_sum / child.visits if child.visits > 0 else -1.0
            u_value = exploration_factor * child.prior_prob / (1 + child.visits)
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
                self.children[action] = Node(next_board, parent=self, action=action, prior_prob=prior, _clone=False)
    
    def update(self, value: float) -> None:
        """Update visit count and value sum."""
        self.visits += 1
        self.value_sum += value
    
    def update_recursive(self, value: float) -> None:
        """Recursively update ancestors."""
        if self.parent:
            self.parent.update_recursive(-value)
        self.update(value)
