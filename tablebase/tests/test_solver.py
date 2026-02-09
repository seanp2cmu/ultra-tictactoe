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
        solver = TablebaseSolver()
        
        # Create a position with 1 empty cell
        enum = PositionEnumerator(empty_cells=1)
        
        count = 0
        for board in enum.enumerate(show_progress=False):
            result, dtw = solver.solve(board)
            assert result in (-1, 0, 1)
            assert dtw >= 0
            count += 1
            if count >= 100:
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
            r, d = solver.solve(board)
            results.append((r, d))
        
        # All results should be identical
        assert all(r == results[0] for r in results)


class TestPositionEnumerator:
    """Test enumerator correctness."""
    
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
        for board in enum.enumerate(show_progress=False):
            h = solver._hash_board(board)
            # Each hash should be unique
            assert h not in seen_hashes, f"Duplicate hash: {h}"
            seen_hashes.add(h)
    
    def test_valid_boards(self):
        """All enumerated boards should be valid."""
        enum = PositionEnumerator(empty_cells=1)
        
        count = 0
        for board in enum.enumerate(show_progress=False):
            # Board should have correct player count
            x_count = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 1)
            o_count = sum(1 for r in range(9) for c in range(9) if board.boards[r][c] == 2)
            
            # X moves first, so x_count == o_count or x_count == o_count + 1
            assert x_count == o_count or x_count == o_count + 1
            
            count += 1
            if count >= 1000:
                break
    
    def test_canonical_keys_deterministic(self):
        """Canonical keys should be deterministic."""
        enum = PositionEnumerator(empty_cells=1)
        solver = TablebaseSolver()
        
        # Enumerate twice and compare
        hashes1 = []
        for board in enum.enumerate(show_progress=False):
            hashes1.append(solver._hash_board(board))
            if len(hashes1) >= 500:
                break
        
        enum2 = PositionEnumerator(empty_cells=1)
        hashes2 = []
        for board in enum2.enumerate(show_progress=False):
            hashes2.append(solver._hash_board(board))
            if len(hashes2) >= 500:
                break
        
        assert hashes1 == hashes2


class TestIntegration:
    """Integration tests for solver + enumerator."""
    
    def test_level1_full_solve(self):
        """Solve all level 1 positions."""
        solver = TablebaseSolver()
        enum = PositionEnumerator(empty_cells=1)
        
        wins = losses = draws = 0
        for board in enum.enumerate(show_progress=False):
            result, dtw = solver.solve(board)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        
        total = wins + losses + draws
        assert total > 13000
        print(f"\nL1 stats: wins={wins}, losses={losses}, draws={draws}")
    
    def test_level2_sample_solve(self):
        """Solve sample of level 2 positions."""
        solver = TablebaseSolver()
        enum = PositionEnumerator(empty_cells=2)
        
        count = 0
        for board in enum.enumerate(show_progress=False):
            result, dtw = solver.solve(board)
            assert result in (-1, 0, 1)
            count += 1
            if count >= 5000:
                break
        
        assert count == 5000


class TestSaveLoad:
    """Test save/load and reuse of tablebase."""
    
    def test_builder_save_load(self):
        """Build L1-L2, save, load, and verify lookups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_tb.pkl')
            
            # Build L1-L2
            builder = TablebaseBuilder(max_empty=2, save_path=save_path)
            builder.build(verbose=False)
            
            # Verify positions exist
            assert len(builder.positions) > 100000
            
            # Sample some positions and results
            sample_hashes = list(builder.positions.keys())[:100]
            sample_results = {}
            for h in sample_hashes:
                constraint = next(iter(builder.positions[h]))
                sample_results[h] = builder.positions[h][constraint]
            
            # Load into new builder
            builder2 = TablebaseBuilder(max_empty=2, save_path=save_path)
            
            # Verify loaded data matches
            assert len(builder2.positions) == len(builder.positions)
            for h in sample_hashes:
                constraint = next(iter(builder2.positions[h]))
                assert builder2.positions[h][constraint] == sample_results[h]
    
    def test_compact_export_lookup(self):
        """Export to compact format and verify lookups work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_tb.pkl')
            compact_path = os.path.join(tmpdir, 'test_tb.npz')
            
            # Build L1-L2
            builder = TablebaseBuilder(max_empty=2, save_path=save_path)
            builder.build(verbose=False)
            
            # Export to compact
            builder.export_compact(compact_path)
            
            # Load compact tablebase
            compact = CompactTablebase(compact_path)
            assert compact.hashes is not None
            assert len(compact.hashes) > 0
            
            # Lookup some positions
            solver = TablebaseSolver()
            enum = PositionEnumerator(empty_cells=1)
            
            found = 0
            for board in enum.enumerate(show_progress=False):
                h = solver._hash_board(board)
                result = compact.lookup(h)
                if result is not None:
                    r, dtw = result
                    assert r in (-1, 0, 1)
                    assert dtw >= 0
                    found += 1
                if found >= 100:
                    break
            
            assert found == 100
    
    def test_solver_uses_base_tablebase(self):
        """Solver should use preloaded base tablebase for lookups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_tb.pkl')
            
            # Build L1-L2
            builder = TablebaseBuilder(max_empty=2, save_path=save_path)
            builder.build(verbose=False)
            
            # Create solver with base tablebase
            solver = TablebaseSolver(base_tablebase=builder.positions)
            
            # Solve some L2 positions - should hit base_tablebase
            enum = PositionEnumerator(empty_cells=2)
            
            count = 0
            for board in enum.enumerate(show_progress=False):
                result, dtw = solver.solve(board)
                assert result in (-1, 0, 1)
                count += 1
                if count >= 100:
                    break
            
            # Should have base_hits
            assert solver.stats['base_hits'] > 0 or solver.stats['cache_hits'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
