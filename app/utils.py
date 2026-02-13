"""Utility functions for the app"""
import numpy as np
from game import Board


def board_from_dict(board_dict):
    """Create a Board object from dictionary representation"""
    board = Board()
    boards_data = board_dict.get('boards', [[0]*9 for _ in range(9)])
    for r in range(9):
        for c in range(9):
            if boards_data[r][c] != 0:
                board.set_cell(r, c, boards_data[r][c])
    # Set completed_boards (BoardCy compatible)
    cb_data = board_dict.get('completed_boards', [[0]*3 for _ in range(3)])
    if hasattr(board, 'set_completed_boards_2d'):
        board.set_completed_boards_2d(cb_data)
    else:
        board.completed_boards = cb_data
    # Sync completed_mask
    completed = board.get_completed_boards_2d() if hasattr(board, 'get_completed_boards_2d') else board.completed_boards
    for sub_idx in range(9):
        sub_r, sub_c = sub_idx // 3, sub_idx % 3
        if completed[sub_r][sub_c] != 0:
            board.completed_mask |= (1 << sub_idx)
    board.current_player = board_dict.get('current_player', 1)
    # BoardCy uses -1 for no winner, Python Board uses None
    raw_winner = board_dict.get('winner', None)
    board.winner = -1 if raw_winner is None else raw_winner
    board.last_move = tuple(board_dict['last_move']) if board_dict.get('last_move') else None
    return board


def get_principal_variation(root, depth=10):
    """Extract principal variation (best move sequence) from MCTS tree"""
    pv = []
    node = root
    
    for _ in range(depth):
        if not node.children or node.is_terminal():
            break
        
        best_action = max(node.children.items(), key=lambda x: x[1].visits)[0]
        best_child = node.children[best_action]
        
        move_r = best_action // 9
        move_c = best_action % 9
        
        pv.append({
            'action': int(best_action),
            'move': [int(move_r), int(move_c)],
            'visits': int(best_child.visits),
            'value': float(best_child.value()),
            'eval': float(best_child.value() * 100)
        })
        
        node = best_child
    
    return pv
