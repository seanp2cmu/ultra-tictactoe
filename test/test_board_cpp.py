"""Test C++ Board implementation against comprehensive tests."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'cpp')

import pytest
import time

# Import C++ Board
from board_cpp import Board


class TestBoardBasics:
    """Basic board operations."""
    
    def test_init(self):
        """Test board initialization."""
        board = Board()
        assert board.current_player == 1
        assert board.winner == -1  # C++ uses -1 for no winner
        assert board.last_move is None
        assert board.completed_mask == 0
        assert len(board.x_masks) == 9
        assert len(board.o_masks) == 9
        assert all(m == 0 for m in board.x_masks)
        assert all(m == 0 for m in board.o_masks)
    
    def test_clone(self):
        """Test board cloning."""
        board = Board()
        board.make_move(4, 4)
        board.make_move(3, 3)
        
        clone = board.clone()
        
        assert clone.current_player == board.current_player
        assert clone.last_move == board.last_move
        assert clone.get_cell(4, 4) == board.get_cell(4, 4)
        assert clone.get_cell(3, 3) == board.get_cell(3, 3)
        
        # Last move (3,3) sends to sub-board (0,0)
        clone.make_move(0, 0)
        assert clone.get_cell(0, 0) != board.get_cell(0, 0)
    
    def test_get_cell(self):
        """Test cell access."""
        board = Board()
        assert board.get_cell(0, 0) == 0
        
        board.make_move(0, 0)
        assert board.get_cell(0, 0) == 1
        
        board.make_move(0, 1)
        assert board.get_cell(0, 1) == 2
    
    def test_set_cell(self):
        """Test direct cell setting."""
        board = Board()
        board.set_cell(4, 4, 1)
        assert board.get_cell(4, 4) == 1
        
        board.set_cell(4, 4, 2)
        assert board.get_cell(4, 4) == 2
        
        board.set_cell(4, 4, 0)
        assert board.get_cell(4, 4) == 0


class TestBoardMoves:
    """Move making and validation."""
    
    def test_make_move_basic(self):
        """Test basic move making."""
        board = Board()
        board.make_move(4, 4)
        
        assert board.get_cell(4, 4) == 1
        assert board.current_player == 2
        assert board.last_move == (4, 4)
    
    def test_make_move_alternates_players(self):
        """Test player alternation."""
        board = Board()
        assert board.current_player == 1
        
        board.make_move(4, 4)
        assert board.current_player == 2
        
        board.make_move(3, 3)
        assert board.current_player == 1
    
    def test_invalid_move_occupied(self):
        """Test that occupied cell raises error."""
        board = Board()
        board.make_move(4, 4)
        
        with pytest.raises(Exception):
            board.make_move(4, 4)
    
    def test_invalid_move_wrong_subboard(self):
        """Test constraint validation."""
        board = Board()
        board.make_move(4, 4)
        
        with pytest.raises(Exception):
            board.make_move(0, 0)


class TestLegalMoves:
    """Legal move generation."""
    
    def test_initial_legal_moves(self):
        """Test all cells legal at start."""
        board = Board()
        legal = board.get_legal_moves()
        assert len(legal) == 81
    
    def test_constrained_legal_moves(self):
        """Test moves constrained to sub-board."""
        board = Board()
        board.make_move(4, 4)
        
        legal = board.get_legal_moves()
        assert len(legal) == 8
        for r, c in legal:
            assert 3 <= r < 6 and 3 <= c < 6


class TestSubBoard:
    """Sub-board operations."""
    
    def test_get_sub_board(self):
        """Test sub-board extraction."""
        board = Board()
        board.make_move(0, 0)
        board.make_move(0, 1)
        
        sub = board.get_sub_board(0)
        assert sub[0] == 1
        assert sub[1] == 2
        assert sub[2] == 0
    
    def test_to_array(self):
        """Test full board array conversion."""
        board = Board()
        board.set_cell(0, 0, 1)
        board.set_cell(8, 8, 2)
        
        arr = board.to_array()
        assert arr[0][0] == 1
        assert arr[8][8] == 2
        assert arr[4][4] == 0


class TestWinConditions:
    """Win and draw detection."""
    
    def test_sub_board_win_row(self):
        """Test sub-board row win."""
        board = Board()
        board.set_cell(0, 0, 1)
        board.set_cell(0, 1, 1)
        board.current_player = 1
        board.make_move(0, 2, False)
        
        assert board.completed_boards[0][0] == 1
    
    def test_game_win(self):
        """Test overall game win."""
        board = Board()
        cb = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for c in range(3):
            cb[0][c] = 1
        board.completed_boards = cb
        board.completed_mask = 0b111
        
        board.current_player = 1
        board.check_winner()
        
        assert board.winner == 1


class TestMiscOperations:
    """Miscellaneous operations."""
    
    def test_count_playable_empty_cells(self):
        """Test empty cell counting."""
        board = Board()
        assert board.count_playable_empty_cells() == 81
        
        board.make_move(4, 4)
        assert board.count_playable_empty_cells() == 80
    
    def test_is_game_over(self):
        """Test game over detection."""
        board = Board()
        assert not board.is_game_over()
        
        board.winner = 1
        assert board.is_game_over()
    
    def test_swap_xo(self):
        """Test X/O swap."""
        board = Board()
        board.make_move(4, 4)
        board.make_move(3, 3)
        
        board.swap_xo()
        
        assert board.get_cell(4, 4) == 2
        assert board.get_cell(3, 3) == 1


class TestBenchmark:
    """Performance benchmarks."""
    
    def test_clone_speed(self):
        """Benchmark clone operation."""
        board = Board()
        board.make_move(4, 4)
        board.make_move(3, 3)
        
        n = 100000
        t0 = time.perf_counter()
        for _ in range(n):
            board.clone()
        elapsed = time.perf_counter() - t0
        
        print(f"\nC++ Clone {n}x: {elapsed*1000:.2f}ms ({elapsed/n*1e6:.2f}µs/op)")
    
    def test_make_move_speed(self):
        """Benchmark make_move operation."""
        n = 10000
        t0 = time.perf_counter()
        for _ in range(n):
            board = Board()
            for move in [(4, 4), (3, 3), (0, 0), (0, 1)]:
                board.make_move(*move)
        elapsed = time.perf_counter() - t0
        
        print(f"\nC++ Make 4 moves {n}x: {elapsed*1000:.2f}ms ({elapsed/n*1e6:.2f}µs/game)")
    
    def test_get_legal_moves_speed(self):
        """Benchmark get_legal_moves operation."""
        board = Board()
        board.make_move(4, 4)
        
        n = 100000
        t0 = time.perf_counter()
        for _ in range(n):
            board.get_legal_moves()
        elapsed = time.perf_counter() - t0
        
        print(f"\nC++ get_legal_moves {n}x: {elapsed*1000:.2f}ms ({elapsed/n*1e6:.2f}µs/op)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
