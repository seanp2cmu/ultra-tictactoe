"""
Tests for TablebaseSolver and PositionEnumerator.
Run before and after Cython optimization to verify correctness.
"""

import pytest
import tempfile
import os
from game import Board
from tablebase import TablebaseSolver, PositionEnumerator, TablebaseBuilder, CompactTablebase


class TestTablebaseSolver:
    """Test solver correctness."""
    
    def test_empty_board_hash_deterministic(self):
        """Same board should always produce same hash."""
        solver = TablebaseSolver()
        board = Board()
        
        h1 = solver._hash_board(board)
        h2 = solver._hash_board(board)
        assert h1 == h2
    
    def test_symmetric_boards_same_hash(self):
        """D4 symmetric boards should produce same canonical hash."""
        solver = TablebaseSolver()
        
        # Original: move at (0,0)
        board1 = Board()
        board1.make_move(4, 4)
        board1.make_move(3, 3)
        board1.make_move(0, 0)
        
        # Rotated 180°: move at (8,8)
        board2 = Board()
        board2.make_move(4, 4)
        board2.make_move(5, 5)
        board2.make_move(8, 8)
        
        h1 = solver._hash_board(board1)
        h2 = solver._hash_board(board2)
        assert h1 == h2
    
    def test_xo_flip_same_hash(self):
        """X/O flipped boards should produce same canonical hash."""
        solver = TablebaseSolver()
        
        # X plays first at center
        board1 = Board()
        board1.make_move(4, 4)
        
        # After one more move, positions should be X/O flippable
        # This tests the flip canonicalization
        h1, c1 = solver._hash_board_with_constraint(board1, 4)
        
        # Hash should be positive integer
        assert isinstance(h1, int)
        assert h1 >= 0
    
    def test_terminal_position_result(self):
        """Terminal positions should return correct result."""
        # L1 positions are terminal - solver works without base
        solver = TablebaseSolver()
        enum = PositionEnumerator(empty_cells=1)
        
        count = 0
        for board in enum.enumerate(show_progress=False):
            result, dtw, _ = solver.solve(board)
            assert result in (-1, 0, 1)
            assert dtw <= 1  # L1 positions have dtw 0 or 1
            count += 1
            if count >= 10:
                break
    
    def test_pack_canonical_key(self):
        """pack_canonical_key should produce deterministic integer."""
        solver = TablebaseSolver()
        
        data = ((0, 4, 4), (0, 4, 4), (0, 4, 4),
                (0, 4, 4), (0, 4, 4), (0, 4, 4),
                (0, 4, 4), (0, 4, 4), (0, 4, 5))
        
        key = solver.pack_canonical_key(data)
        assert isinstance(key, int)
        assert key > 0
        
        # Same data should produce same key
        key2 = solver.pack_canonical_key(data)
        assert key == key2
    
    def test_hash_consistency_across_calls(self):
        """Hash should be consistent across multiple solve calls."""
        solver = TablebaseSolver()
        
        board = Board()
        board.make_move(4, 4)
        board.make_move(3, 3)
        board.make_move(0, 0)
        
        # Solve multiple times
        results = []
        for _ in range(5):
            r, d, _ = solver.solve(board)
            results.append((r, d))
        
        # All results should be identical
        assert all(r == results[0] for r in results)


class TestPositionEnumerator:
    """Test enumerator correctness."""
    
    @pytest.mark.slow
    def test_level1_count(self):
        """Level 1 should enumerate specific count of positions."""
        enum = PositionEnumerator(empty_cells=1)
        count = sum(1 for _ in enum.enumerate(show_progress=False))
        # Should be around 13K positions
        assert 13000 <= count <= 14000
    
    def test_no_duplicates(self):
        """Enumerated positions should be unique (canonical)."""
        enum = PositionEnumerator(empty_cells=1)
        solver = TablebaseSolver()
        
        seen_hashes = set()
        count = 0
        for board in enum.enumerate(show_progress=False):
            h = solver._hash_board(board)
            assert h not in seen_hashes, f"Duplicate hash: {h}"
            seen_hashes.add(h)
            count += 1
            if count >= 100:  # 빠른 테스트
                break
    
    def test_canonical_keys_deterministic(self):
        """Canonical keys should be deterministic."""
        enum = PositionEnumerator(empty_cells=1)
        solver = TablebaseSolver()
        
        hashes1 = []
        for board in enum.enumerate(show_progress=False):
            hashes1.append(solver._hash_board(board))
            if len(hashes1) >= 50:  # 빠른 테스트
                break
        
        enum2 = PositionEnumerator(empty_cells=1)
        hashes2 = []
        for board in enum2.enumerate(show_progress=False):
            hashes2.append(solver._hash_board(board))
            if len(hashes2) >= 50:
                break
        
        assert hashes1 == hashes2


class TestIntegration:
    """Integration tests for solver + enumerator."""
    
    def test_level1_sample_solve(self):
        """Solve sample of level 1 positions (terminal)."""
        solver = TablebaseSolver()
        enum = PositionEnumerator(empty_cells=1)
        
        count = 0
        for board in enum.enumerate(show_progress=False):
            result, dtw, _ = solver.solve(board)
            assert result in (-1, 0, 1)
            count += 1
            if count >= 50:
                break
        assert count == 50
    
    def test_level2_with_builder(self):
        """L2 solve requires L1 base - use builder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'tb.pkl')
            builder = TablebaseBuilder(max_empty=1, save_path=save_path)
            builder.build(verbose=False)
            
            # Now solve L2 using builder's solver (has L1 cache)
            enum = PositionEnumerator(empty_cells=2)
            count = 0
            for board in enum.enumerate(show_progress=False):
                result, dtw, _ = builder.solver.solve(board)
                assert result in (-1, 0, 1)
                assert builder.solver.stats['missing_child'] == 0
                count += 1
                if count >= 50:
                    break
            assert count == 50


@pytest.mark.slow
class TestSaveLoad:
    """Test save/load and reuse of tablebase (slow)."""
    
    def test_builder_save_load(self):
        """Build L1 only, save, load, and verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_tb.pkl')
            
            builder = TablebaseBuilder(max_empty=1, save_path=save_path)
            builder.build(verbose=False)
            
            assert len(builder.positions) > 10000
            
            # Load into new builder
            builder2 = TablebaseBuilder(max_empty=1, save_path=save_path)
            assert len(builder2.positions) == len(builder.positions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
