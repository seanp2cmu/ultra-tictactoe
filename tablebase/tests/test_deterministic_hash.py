"""Test deterministic hash and save/reload stability."""
import pytest
import pickle
import tempfile
import os

from game import Board
from tablebase.solver import TablebaseSolver
from tablebase.enumerator import PositionEnumerator
from tablebase.tablebase import CompactTablebase


class TestDeterministicHash:
    """Test that hashes are deterministic across runs."""
    
    def test_same_board_same_hash(self):
        """Same board should always produce same hash."""
        solver = TablebaseSolver()
        
        board1 = Board()
        board1.make_move(4, 4)
        board1.make_move(3, 3)
        
        board2 = Board()
        board2.make_move(4, 4)
        board2.make_move(3, 3)
        
        h1 = solver._hash_board(board1)
        h2 = solver._hash_board(board2)
        
        assert h1 == h2
        assert isinstance(h1, int)
        # Verify it's a packed integer, not Python hash
        assert h1 > 0  # pack_canonical_key returns positive int
    
    def test_hash_is_integer_not_python_hash(self):
        """Hash should be a deterministic packed integer."""
        solver = TablebaseSolver()
        
        board = Board()
        board.make_move(4, 4)
        
        h = solver._hash_board(board)
        
        # Should be same across multiple calls
        for _ in range(10):
            assert solver._hash_board(board) == h
    
    def test_packed_key_format(self):
        """Verify packed key is deterministic bit-packing."""
        # Each sub-board: 10 bits (2 state + 4 x_count + 4 o_count)
        # Total: 90 bits
        solver = TablebaseSolver()
        
        board = Board()
        h = solver._hash_board(board)
        
        # Empty board should have consistent hash
        assert h >= 0
        assert h < (1 << 90)  # 90-bit max


class TestSaveReload:
    """Test that tablebase can be saved and reloaded correctly."""
    
    def test_solver_cache_pickle(self):
        """Solver cache should survive pickle round-trip."""
        solver = TablebaseSolver()
        
        # Solve some positions
        enum = PositionEnumerator(1)
        positions = list(enum.enumerate(show_progress=False))[:100]
        
        for board in positions:
            solver.solve(board)
        
        original_cache = dict(solver.cache)
        
        # Pickle and unpickle
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(solver.cache, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                loaded_cache = pickle.load(f)
            
            # Verify all keys match
            assert set(original_cache.keys()) == set(loaded_cache.keys())
            
            # Verify values match
            for h in original_cache:
                assert original_cache[h] == loaded_cache[h]
        finally:
            os.unlink(temp_path)
    
    def test_compact_tablebase_save_reload(self):
        """CompactTablebase should save and reload correctly."""
        solver = TablebaseSolver()
        
        # Solve positions
        enum = PositionEnumerator(1)
        positions = list(enum.enumerate(show_progress=False))[:100]
        
        for board in positions:
            solver.solve(board)
        
        # Build compact tablebase
        tb = CompactTablebase()
        tb.build_from_dict(solver.cache)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            temp_path = f.name
        
        try:
            tb.save(temp_path)
            
            # Reload in new instance
            tb2 = CompactTablebase()
            tb2.load(temp_path)
            
            # Verify lookups match
            for h in solver.cache:
                result1 = tb.lookup(h)
                result2 = tb2.lookup(h)
                assert result1 == result2, f"Mismatch for hash {h}"
        finally:
            os.unlink(temp_path)
    
    def test_hash_consistent_after_reload(self):
        """Hash computation should be consistent after process restart simulation."""
        solver1 = TablebaseSolver()
        solver2 = TablebaseSolver()
        
        board = Board()
        board.make_move(4, 4)
        board.make_move(3, 3)
        
        h1 = solver1._hash_board(board)
        h2 = solver2._hash_board(board)
        
        # Both solvers should compute same hash
        assert h1 == h2
    
    def test_full_level1_save_reload(self):
        """Full level 1 tablebase should survive save/reload."""
        solver = TablebaseSolver()
        
        # Solve all level 1 positions
        enum = PositionEnumerator(1)
        positions = list(enum.enumerate(show_progress=False))
        
        for board in positions:
            solver.solve(board)
        
        # Build and save
        tb = CompactTablebase()
        tb.build_from_dict(solver.cache)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            temp_path = f.name
        
        try:
            tb.save(temp_path)
            
            # Reload
            tb2 = CompactTablebase()
            tb2.load(temp_path)
            
            # Verify all lookups
            success = 0
            for h in solver.cache:
                if tb2.lookup(h) is not None:
                    success += 1
            
            assert success == len(solver.cache), f"Only {success}/{len(solver.cache)} lookups succeeded"
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
