"""
Comprehensive Board tests for bitmask refactoring.
These tests must pass before and after the refactoring.
"""

import pytest
from game import Board


class TestBoardBasics:
    """Basic board operations."""
    
    def test_init(self):
        board = Board()
        assert board.current_player == 1
        assert board.winner is None
        assert board.last_move is None
    
    def test_make_move_first(self):
        board = Board()
        board.make_move(4, 4)
        assert board.boards[4][4] == 1
        assert board.current_player == 2
        assert board.last_move == (4, 4)
    
    def test_make_move_sequence(self):
        board = Board()
        board.make_move(4, 4)  # X at center
        board.make_move(3, 3)  # O at (3,3) - constrained to sub-board (1,1)
        assert board.boards[4][4] == 1
        assert board.boards[3][3] == 2
        assert board.current_player == 1
    
    def test_clone(self):
        board = Board()
        board.make_move(4, 4)
        board.make_move(3, 3)
        
        clone = board.clone()
        assert clone.boards[4][4] == 1
        assert clone.boards[3][3] == 2
        assert clone.current_player == board.current_player
        
        # Modify original, clone should not change
        board.make_move(0, 0)
        assert clone.boards[0][0] == 0


class TestLegalMoves:
    """Legal move generation."""
    
    def test_first_move_all_cells(self):
        board = Board()
        moves = board.get_legal_moves()
        assert len(moves) == 81
    
    def test_constrained_moves(self):
        board = Board()
        board.make_move(4, 4)  # X at center of center sub-board
        moves = board.get_legal_moves()
        # Next player constrained to sub-board (1,1) = center
        # 8 cells available (center taken)
        assert len(moves) == 8
        for r, c in moves:
            assert 3 <= r <= 5 and 3 <= c <= 5
            assert (r, c) != (4, 4)
    
    def test_any_constraint_when_target_full(self):
        board = Board()
        # When target sub-board is completed, player can move anywhere
        # last_move (0,0) constrains to sub-board (0%3, 0%3) = (0,0)
        board.completed_boards[0][0] = 1  # Mark sub-board (0,0) as completed
        board.last_move = (0, 0)  # Constrains to (0,0) which is completed
        moves = board.get_legal_moves()
        # Should have moves in all non-completed sub-boards (72 cells)
        assert len(moves) == 72
    
    def test_is_valid_move(self):
        board = Board()
        assert board._is_valid_move(4, 4) == True
        board.make_move(4, 4)
        assert board._is_valid_move(4, 4) == False  # Already taken
        assert board._is_valid_move(0, 0) == False  # Wrong sub-board


class TestSubBoardCompletion:
    """Sub-board win/draw detection."""
    
    def test_sub_board_win_row(self):
        board = Board()
        # Fill row 0 of sub-board (0,0)
        board.set_cell(0, 0, 1)
        board.set_cell(0, 1, 1)
        board.current_player = 1
        board.make_move(0, 2, validate=False)
        assert board.completed_boards[0][0] == 1
    
    def test_sub_board_win_col(self):
        board = Board()
        board.set_cell(0, 0, 1)
        board.set_cell(1, 0, 1)
        board.current_player = 1
        board.make_move(2, 0, validate=False)
        assert board.completed_boards[0][0] == 1
    
    def test_sub_board_win_diag(self):
        board = Board()
        board.set_cell(0, 0, 1)
        board.set_cell(1, 1, 1)
        board.current_player = 1
        board.make_move(2, 2, validate=False)
        assert board.completed_boards[0][0] == 1
    
    def test_sub_board_draw(self):
        board = Board()
        # Fill sub-board (0,0) with no winner
        pattern = [
            (0, 0, 1), (0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 1, 1), (1, 2, 2),
            (2, 0, 2), (2, 1, 1)
        ]
        for r, c, p in pattern:
            board.set_cell(r, c, p)
        board.current_player = 2
        board.make_move(2, 2, validate=False)
        assert board.completed_boards[0][0] == 3  # Draw


class TestGameWinner:
    """Game-level win detection."""
    
    def test_game_win_row(self):
        board = Board()
        board.completed_boards[0][0] = 1
        board.completed_boards[0][1] = 1
        board.completed_boards[0][2] = 1
        board.current_player = 1
        board.check_winner()
        assert board.winner == 1
    
    def test_game_win_diag(self):
        board = Board()
        board.completed_boards[0][0] = 2
        board.completed_boards[1][1] = 2
        board.completed_boards[2][2] = 2
        board.current_player = 2
        board.check_winner()
        assert board.winner == 2
    
    def test_game_draw(self):
        board = Board()
        board.completed_boards = [
            [1, 2, 1],
            [1, 2, 1],
            [2, 1, 2]
        ]
        board.current_player = 1
        board.check_winner()
        assert board.winner == 3  # Draw


class TestUndoMove:
    """Undo move functionality."""
    
    def test_undo_basic(self):
        board = Board()
        board.make_move(4, 4)
        
        prev_last_move = None
        prev_winner = None
        prev_completed = board.completed_boards[1][1]
        
        board.undo_move(4, 4, prev_completed, prev_winner, prev_last_move)
        
        assert board.boards[4][4] == 0
        assert board.current_player == 1
        assert board.last_move is None
    
    def test_undo_sequence(self):
        board = Board()
        
        # Make moves
        board.make_move(4, 4)
        state1 = (board.last_move, board.winner, board.completed_boards[1][1])
        
        board.make_move(3, 3)
        state2 = (board.last_move, board.winner, board.completed_boards[0][0])
        
        # Undo second move
        board.undo_move(3, 3, 0, None, (4, 4))
        assert board.boards[3][3] == 0
        assert board.current_player == 2
        
        # Undo first move
        board.undo_move(4, 4, 0, None, None)
        assert board.boards[4][4] == 0
        assert board.current_player == 1


class TestSubCounts:
    """Sub-board piece counting."""
    
    def test_sub_counts_init(self):
        board = Board()
        for counts in board.sub_counts:
            assert counts == [0, 0]
    
    def test_sub_counts_after_move(self):
        board = Board()
        board.make_move(4, 4)  # X in sub-board 4
        assert board.sub_counts[4] == [1, 0]
        
        board.make_move(3, 3)  # O in sub-board 4
        assert board.sub_counts[4] == [1, 1]
    
    def test_sub_counts_after_undo(self):
        board = Board()
        board.make_move(4, 4)
        board.undo_move(4, 4, 0, None, None)
        assert board.sub_counts[4] == [0, 0]


class TestBoardAccess:
    """Direct board cell access (for compatibility)."""
    
    def test_read_cell(self):
        board = Board()
        board.make_move(4, 4)
        assert board.boards[4][4] == 1
        assert board.boards[0][0] == 0
    
    def test_write_cell(self):
        board = Board()
        board.set_cell(0, 0, 1)
        assert board.boards[0][0] == 1
        # Bitmask should also be updated
        assert board.x_masks[0] & 1 == 1


class TestCountEmpty:
    """Empty cell counting."""
    
    def test_count_initial(self):
        board = Board()
        assert board.count_playable_empty_cells() == 81
    
    def test_count_after_moves(self):
        board = Board()
        board.make_move(4, 4)
        board.make_move(3, 3)
        assert board.count_playable_empty_cells() == 79
    
    def test_count_with_completed_sub(self):
        board = Board()
        # Complete sub-board (0,0)
        board.completed_boards[0][0] = 1
        # Should not count cells in completed sub-board
        count = board.count_playable_empty_cells()
        assert count == 72  # 81 - 9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
