# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython-optimized MCTS Node."""

from libc.math cimport sqrt
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem
import numpy as np
cimport numpy as np

from game import Board


cdef class NodeCy:
    """Cython-optimized MCTS Node."""
    
    cdef public object board
    cdef public object parent
    cdef public int action
    cdef public float prior_prob
    cdef public dict children
    cdef public int visits
    cdef public float value_sum
    cdef public int _is_terminal  # -1: unknown, 0: false, 1: true
    cdef public int virtual_loss
    
    def __init__(self, board, parent=None, int action=-1, float prior_prob=0.0, bint _clone=True):
        self.board = board.clone() if _clone else board
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self._is_terminal = -1
        self.virtual_loss = 0
    
    cpdef bint is_expanded(self):
        return len(self.children) > 0
    
    cpdef bint is_terminal(self):
        cdef int br, bc, r, c
        cdef object completed
        
        if self._is_terminal != -1:
            return self._is_terminal == 1
        
        winner = self.board.winner
        if winner is not None and winner != -1:
            self._is_terminal = 1
            return True
        
        # Fast path for BoardCy: check completed_mask bitmask
        if hasattr(self.board, 'completed_mask'):
            if self.board.completed_mask == 0x1FF:  # all 9 sub-boards done
                self._is_terminal = 1
                return True
            # If any sub-board is open, there are legal moves
            self._is_terminal = 0
            return False
        
        if hasattr(self.board, 'get_completed_boards_2d'):
            completed = self.board.get_completed_boards_2d()
        else:
            completed = self.board.completed_boards
        
        for br in range(3):
            for bc in range(3):
                if completed[br][bc] == 0:
                    for r in range(br*3, br*3+3):
                        for c in range(bc*3, bc*3+3):
                            if self.board.get_cell(r, c) == 0:
                                self._is_terminal = 0
                                return False
        
        self._is_terminal = 1
        return True
    
    cpdef float value(self):
        cdef int total = self.visits + self.virtual_loss
        if total == 0:
            return 0.0
        return (self.value_sum - <float>self.virtual_loss) / <float>total
    
    cpdef tuple select_child(self, float c_puct=1.0):
        """Select child with highest UCB score."""
        cdef float best_score = -1e9
        cdef int best_action = -1
        cdef object best_child = None
        cdef float sqrt_visits, exploration_factor, parent_value, fpu
        cdef float q_value, u_value, score
        cdef int action
        cdef NodeCy child
        
        sqrt_visits = sqrt(<float>(self.visits + self.virtual_loss))
        exploration_factor = c_puct * sqrt_visits
        
        parent_value = self.value() if self.visits > 0 else 0.0
        fpu = -parent_value
        
        cdef int child_total
        for action, child in self.children.items():
            child_total = child.visits + child.virtual_loss
            if child_total > 0:
                q_value = (child.value_sum - <float>child.virtual_loss) / <float>child_total
            else:
                q_value = fpu
            u_value = exploration_factor * child.prior_prob / (1 + child_total)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    cpdef void expand(self, dict action_probs):
        """Expand node by creating children for legal moves (dict API)."""
        cdef list legal_moves = self.board.get_legal_moves()
        cdef int action
        cdef float prior
        cdef object next_board
        cdef tuple move
        
        for move in legal_moves:
            action = move[0] * 9 + move[1]
            if action not in self.children:
                next_board = self.board.clone()
                next_board.make_move(move[0], move[1])
                prior = action_probs.get(action, 1e-8)
                self.children[action] = NodeCy(next_board, parent=self, action=action, prior_prob=prior, _clone=False)
    
    cpdef void expand_numpy(self, float[:] policy):
        """Expand node using a flat numpy policy array (avoids dict creation)."""
        cdef list legal_moves = self.board.get_legal_moves()
        cdef int action
        cdef float prior
        cdef object next_board
        cdef tuple move
        
        for move in legal_moves:
            action = move[0] * 9 + move[1]
            if action not in self.children:
                next_board = self.board.clone()
                next_board.make_move(move[0], move[1])
                prior = policy[action]
                self.children[action] = NodeCy(next_board, parent=self, action=action, prior_prob=prior, _clone=False)
    
    cpdef void add_virtual_loss(self, int n=1):
        """Add virtual loss to discourage re-selection."""
        self.virtual_loss += n
    
    cpdef void revert_virtual_loss(self, int n=1):
        """Remove virtual loss after real backup."""
        self.virtual_loss -= n
    
    cpdef void update(self, float value):
        """Update visit count and value sum."""
        self.visits += 1
        self.value_sum += value
    
    cpdef void update_recursive(self, float value):
        """Recursively update ancestors."""
        if self.parent is not None:
            (<NodeCy>self.parent).update_recursive(-value)
        self.update(value)
