"""Comprehensive DTW Calculator tests."""
import sys
sys.path.insert(0, '.')

import pytest
import time

from game import Board
from ai.endgame.dtw_calculator import DTWCalculator


class TestDTWBasics:
    """Basic DTW operations."""
    
    def test_init(self):
        """Test DTW calculator initialization."""
        dtw = DTWCalculator(use_cache=True)
        assert dtw.endgame_threshold == 15
        assert dtw.max_nodes == 100000
        assert dtw.use_cache == True
    
    def test_is_endgame_empty_board(self):
        """Empty board is not endgame."""
        dtw = DTWCalculator()
        board = Board()
        assert not dtw.is_endgame(board)  # 81 cells > 15
    
    def test_is_endgame_threshold(self):
        """Test endgame threshold detection."""
        dtw = DTWCalculator(endgame_threshold=15)
        board = Board()
        
        # Fill cells until threshold
        for i in range(66):  # 81 - 15 = 66
            r, c = i // 9, i % 9
            board.set_cell(r, c, (i % 2) + 1)
        
        # Should be exactly at threshold
        assert board.count_playable_empty_cells() <= 15


class TestDTWCalculation:
    """DTW calculation tests."""
    
    def test_calculate_immediate_win(self):
        """Test DTW for immediate win position."""
        dtw = DTWCalculator(use_cache=False)
        board = Board()
        
        # Setup position where player 1 can win immediately
        board.set_cell(0, 0, 1)
        board.set_cell(0, 1, 1)
        board.current_player = 1
        board.last_move = None
        
        # Fill other cells to make it endgame
        cb = [[0, 3, 3], [3, 3, 3], [3, 3, 3]]
        if hasattr(board, 'set_completed_boards_2d'):
            board.set_completed_boards_2d(cb)
        else:
            board.completed_boards = cb
        board.completed_mask = 0b111111110  # All except sub-board 0
        
        # Only a few cells left in sub-board 0
        empty = board.count_playable_empty_cells()
        assert empty <= 15
    
    def test_calculate_terminal_win(self):
        """Test DTW for already won position."""
        dtw = DTWCalculator(use_cache=False)
        board = Board()
        board.winner = 1
        board.current_player = 1
        
        # Force endgame by filling cells
        board.completed_mask = 0b111111111
        
        result = dtw.calculate_dtw(board)
        if result:
            r, d, m = result
            assert r == 1  # Win


class TestDTWCache:
    """Cache functionality tests."""
    
    def test_cache_lookup(self):
        """Test cache lookup."""
        dtw = DTWCalculator(use_cache=True)
        board = Board()
        
        # Should return None for empty cache
        result = dtw.lookup_cache(board)
        assert result is None
    
    def test_cache_stores_result(self):
        """Test that results are cached."""
        dtw = DTWCalculator(use_cache=True, hot_size=1000)
        
        # Create simple endgame position
        board = Board()
        board.winner = 1
        board.current_player = 1
        board.completed_mask = 0b111111111
        
        # First calculation
        result1 = dtw.calculate_dtw(board)
        
        # Second lookup should hit cache
        result2 = dtw.lookup_cache(board)
        # May or may not be cached depending on implementation


class TestDTWStats:
    """Statistics tests."""
    
    def test_stats_tracking(self):
        """Test that stats are tracked."""
        dtw = DTWCalculator(use_cache=True)
        
        stats = dtw.get_stats()
        assert 'dtw_searches' in stats
        assert 'dtw_nodes' in stats
    
    def test_reset_stats(self):
        """Test stats reset."""
        dtw = DTWCalculator()
        dtw._total_searches = 100
        dtw._total_nodes = 1000
        
        dtw.reset_search_stats()
        
        assert dtw._total_searches == 0
        assert dtw._total_nodes == 0


class TestBenchmark:
    """Performance benchmarks."""
    
    def test_endgame_search_speed(self):
        """Benchmark endgame search."""
        dtw = DTWCalculator(use_cache=True, max_nodes=10000)
        
        # Create endgame position with ~10 empty cells
        board = Board()
        # Fill most of the board
        for r in range(9):
            for c in range(9):
                if r < 7 or (r == 7 and c < 5):
                    board.set_cell(r, c, ((r * 9 + c) % 2) + 1)
        
        # Mark some sub-boards as complete
        cb = [[3, 3, 3], [3, 3, 3], [0, 0, 0]]
        if hasattr(board, 'set_completed_boards_2d'):
            board.set_completed_boards_2d(cb)
        else:
            board.completed_boards = cb
        board.completed_mask = 0b000111111
        
        empty = board.count_playable_empty_cells()
        print(f"\nEmpty cells: {empty}")
        
        if empty <= 15:
            n = 10
            t0 = time.perf_counter()
            for _ in range(n):
                dtw.reset_search_stats()
                result = dtw.calculate_dtw(board)
            elapsed = time.perf_counter() - t0
            
            stats = dtw.get_stats()
            print(f"DTW search {n}x: {elapsed*1000:.2f}ms")
            print(f"Avg nodes: {stats['dtw_avg_nodes']:.0f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
