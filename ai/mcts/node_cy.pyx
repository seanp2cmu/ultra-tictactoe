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
    """Cython-optimized MCTS Node with lazy board creation."""
    
    cdef public object board
    cdef public object parent
    cdef public int action
    cdef public float prior_prob
    cdef public dict children
    cdef public int visits
    cdef public float value_sum
    cdef public int _is_terminal  # -1: unknown, 0: false, 1: true
    cdef public int virtual_loss
    cdef public bint _board_ready  # True if board is materialized
    
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
        self._board_ready = True
    
    cdef inline void _ensure_board(self):
        """Lazily materialize board from parent if needed."""
        if self._board_ready:
            return
        # board field holds parent's board; clone and apply action
        cdef object parent_board = self.board
        self.board = parent_board.clone()
        self.board.make_move(self.action // 9, self.action % 9)
        self._board_ready = True
    
    cpdef bint is_expanded(self):
        return len(self.children) > 0
    
    cpdef bint is_terminal(self):
        cdef int br, bc, r, c
        cdef object completed
        
        if self._is_terminal != -1:
            return self._is_terminal == 1
        
        self._ensure_board()
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
        return self.value_sum / <float>total
    
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
                # Negate: child stores value from child's (opponent's) perspective
                # Parent needs value from its own perspective
                # Virtual loss: only inflate denominator, don't modify value_sum
                q_value = -child.value_sum / <float>child_total
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
        """Expand node by creating children for legal moves (dict API) — lazy board creation."""
        self._ensure_board()
        cdef list legal_moves = self.board.get_legal_moves()
        cdef int action
        cdef float prior
        cdef tuple move
        cdef NodeCy child
        
        for move in legal_moves:
            action = move[0] * 9 + move[1]
            if action not in self.children:
                prior = action_probs.get(action, 1e-8)
                child = NodeCy.__new__(NodeCy)
                child.board = self.board
                child.parent = self
                child.action = action
                child.prior_prob = prior
                child.children = {}
                child.visits = 0
                child.value_sum = 0.0
                child._is_terminal = -1
                child.virtual_loss = 0
                child._board_ready = False
                self.children[action] = child
    
    cpdef void expand_numpy(self, float[:] policy):
        """Expand node using a flat numpy policy array — lazy board creation.
        
        Children are created with a reference to parent's board instead of
        cloning. The actual clone+make_move happens lazily when the child
        is first visited (via _ensure_board).
        """
        self._ensure_board()
        cdef list legal_moves = self.board.get_legal_moves()
        cdef int action
        cdef float prior
        cdef tuple move
        cdef NodeCy child
        
        for move in legal_moves:
            action = move[0] * 9 + move[1]
            if action not in self.children:
                prior = policy[action]
                # Lazy child: store parent's board ref, defer clone+make_move
                child = NodeCy.__new__(NodeCy)
                child.board = self.board  # NOT cloned yet
                child.parent = self
                child.action = action
                child.prior_prob = prior
                child.children = {}
                child.visits = 0
                child.value_sum = 0.0
                child._is_terminal = -1
                child.virtual_loss = 0
                child._board_ready = False  # lazy
                self.children[action] = child
    
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


# ── Batch MCTS operations (called from ParallelMCTS) ──

cpdef tuple select_leaf_vl(NodeCy root, float c_puct):
    """Select one leaf from root, applying virtual loss along the path.
    
    Returns (leaf_node, search_path) where search_path is a list of NodeCy.
    Virtual loss is applied to all nodes on the path.
    """
    cdef NodeCy node = root
    cdef list search_path = [root]
    cdef int action
    
    while node.is_expanded() and not node.is_terminal():
        action, node = node.select_child(c_puct)
        node._ensure_board()  # materialize board on visit
        search_path.append(node)
    
    # Apply virtual loss to entire path (direct field access, no method call)
    cdef NodeCy n
    for n in search_path:
        n.virtual_loss += 1
    
    return node, search_path


def select_multi_leaves_cy(list roots, list indices, int leaves_per_game, float c_puct):
    """Select multiple leaves per game using virtual loss for diversity.
    
    Returns (leaves, boards) where:
      leaves = list of (game_idx, node, search_path)
      boards = list of board objects for inference
    """
    cdef list leaves = []
    cdef list boards = []
    cdef int idx, j
    cdef NodeCy root, node, n
    cdef list search_path
    
    cdef int k, path_len
    cdef float value
    cdef NodeCy path_node
    
    for idx in indices:
        root = <NodeCy>roots[idx]
        for j in range(leaves_per_game):
            node, search_path = select_leaf_vl(root, c_puct)
            if not node.is_terminal() and not node.is_expanded():
                leaves.append((idx, node, search_path))
                boards.append(node.board)
            elif node.is_terminal():
                # Terminal node: backpropagate actual game result
                winner = node.board.winner
                if winner is not None and winner != -1 and winner != 3:
                    # winner != current_player always at terminal
                    value = -1.0  # current player lost
                else:
                    value = 0.0  # draw
                # Revert VL + backprop (leaf to root)
                path_len = len(search_path)
                for k in range(path_len - 1, -1, -1):
                    path_node = <NodeCy>search_path[k]
                    path_node.virtual_loss -= 1
                    path_node.visits += 1
                    path_node.value_sum += value
                    value = -value
            else:
                # Already expanded — just revert virtual loss
                for n in search_path:
                    n.virtual_loss -= 1
    
    return leaves, boards


def expand_backprop_batch_cy(list leaves, float[:,:] policies, float[:] values_scaled):
    """Expand leaf nodes, backpropagate values, and revert virtual losses.
    
    Args:
        leaves: list of (game_idx, node, search_path) tuples
        policies: (N, 81) float32 array of policy outputs
        values_scaled: (N,) float32 array of pre-scaled values (2*v - 1)
    """
    cdef int i, j, path_len
    cdef int game_idx
    cdef NodeCy node, path_node
    cdef float value
    cdef list search_path
    
    for i in range(len(leaves)):
        game_idx, node, search_path = leaves[i]
        
        # Expand with policy
        node.expand_numpy(policies[i])
        
        # Revert virtual loss + backprop (reversed path)
        value = values_scaled[i]
        path_len = len(search_path)
        for j in range(path_len - 1, -1, -1):
            path_node = <NodeCy>search_path[j]
            path_node.virtual_loss -= 1
            path_node.visits += 1
            path_node.value_sum += value
            value = -value


def revert_vl_batch_cy(list leaves):
    """Revert virtual losses for all leaves without expand/backprop."""
    cdef NodeCy n
    cdef list search_path
    
    for _, _, search_path in leaves:
        for n in search_path:
            n.virtual_loss -= 1
