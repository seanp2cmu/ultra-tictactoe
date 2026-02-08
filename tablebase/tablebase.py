"""
Tablebase Storage and Lookup for Ultimate Tic-Tac-Toe

Efficient storage and retrieval of precomputed endgame positions.
"""

import os
import pickle
import struct
import mmap
from typing import Optional, Tuple, Dict
import numpy as np
from game import Board


class Tablebase:
    """
    Endgame tablebase for Ultimate Tic-Tac-Toe.
    
    Provides O(1) lookup for precomputed positions.
    Stores (result, dtw) for each position.
    """
    
    def __init__(self, path: Optional[str] = None):
        """
        Args:
            path: Path to tablebase file (optional, can load later)
        """
        self.positions: Dict[int, Tuple[int, int]] = {}
        self.max_empty = 15
        self.stats = {}
        
        if path and os.path.exists(path):
            self.load(path)
    
    def load(self, path: str):
        """Load tablebase from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.positions = data['positions']
        self.stats = data.get('stats', {})
        self.max_empty = data.get('max_empty', 15)
        
        print(f"✓ Loaded tablebase: {len(self.positions)} positions")
    
    def save(self, path: str):
        """Save tablebase to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'stats': self.stats,
                'max_empty': self.max_empty
            }, f)
        
        print(f"✓ Saved tablebase: {len(self.positions)} positions")
    
    def lookup(self, board: Board) -> Optional[Tuple[int, int]]:
        """
        Look up position in tablebase.
        
        Args:
            board: Board position to look up
            
        Returns:
            (result, dtw) if found, None otherwise
            - result: 1 (current player wins), -1 (loses), 0 (draw)
            - dtw: distance to win (moves until game ends)
        """
        # Check if position qualifies for tablebase
        empty_count = self._count_empty(board)
        if empty_count > self.max_empty:
            return None
        
        # Get canonical hash
        board_hash = self._hash_board(board)
        
        # Try direct lookup
        if board_hash in self.positions:
            return self.positions[board_hash]
        
        # Try symmetric positions
        for sym_hash in self._get_symmetric_hashes(board):
            if sym_hash in self.positions:
                return self.positions[sym_hash]
        
        return None
    
    def probe(self, board: Board) -> Optional[dict]:
        """
        Probe tablebase with detailed result.
        
        Returns:
            Dict with 'result', 'dtw', 'best_move' if found
        """
        result = self.lookup(board)
        if result is None:
            return None
        
        game_result, dtw = result
        
        # Find best move (one that maintains the result)
        best_move = self._find_best_move(board, game_result, dtw)
        
        return {
            'result': game_result,
            'dtw': dtw,
            'best_move': best_move,
            'result_str': self._result_to_str(game_result)
        }
    
    def _count_empty(self, board: Board) -> int:
        """Count empty cells on the board."""
        count = 0
        for row in range(9):
            for col in range(9):
                if board.boards[row][col] == 0:
                    # Check if in completed sub-board
                    sub_row, sub_col = row // 3, col // 3
                    if board.completed_boards[sub_row][sub_col] == 0:
                        count += 1
        return count
    
    def _hash_board(self, board: Board) -> int:
        """Create hash for board position."""
        # Flatten board to tuple
        cells = []
        for row in range(9):
            for col in range(9):
                cells.append(board.boards[row][col])
        
        # Include current player in hash
        cells.append(board.current_player)
        
        # Include last move constraint
        if board.last_move:
            cells.append(board.last_move[0] * 9 + board.last_move[1])
        else:
            cells.append(-1)
        
        return hash(tuple(cells))
    
    def _get_symmetric_hashes(self, board: Board) -> list:
        """Get hashes for all symmetric positions (D4 symmetry)."""
        # TODO: Implement D4 symmetry on the big board
        # For now, return empty list
        return []
    
    def _find_best_move(self, board: Board, target_result: int, target_dtw: int) -> Optional[Tuple[int, int]]:
        """Find a move that achieves the target result."""
        legal_moves = board.get_legal_moves()
        
        for move in legal_moves:
            # Make move and check tablebase
            board.make_move(move[0], move[1])
            
            result = self.lookup(board)
            if result:
                child_result, child_dtw = result
                # Result is from opponent's perspective after our move
                if -child_result == target_result:
                    board.undo_move()
                    return move
            
            board.undo_move()
        
        return None
    
    def _result_to_str(self, result: int) -> str:
        """Convert result code to string."""
        if result == 1:
            return "Win"
        elif result == -1:
            return "Loss"
        else:
            return "Draw"
    
    def get_stats(self) -> dict:
        """Get tablebase statistics."""
        return {
            'total_positions': len(self.positions),
            'max_empty': self.max_empty,
            'size_mb': self._estimate_size_mb(),
            **self.stats
        }
    
    def _estimate_size_mb(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimate: each position ~100 bytes (hash + values + overhead)
        return len(self.positions) * 100 / (1024 * 1024)


class CompactTablebase:
    """
    Memory-efficient tablebase using numpy arrays.
    
    For production use with millions of positions.
    Uses memory-mapped file for large tablebases.
    """
    
    def __init__(self, path: Optional[str] = None):
        self.hashes: Optional[np.ndarray] = None  # uint64 array
        self.results: Optional[np.ndarray] = None  # int8 array
        self.dtws: Optional[np.ndarray] = None  # uint8 array
        self.path = path
        
        if path and os.path.exists(path):
            self.load(path)
    
    def build_from_dict(self, positions: Dict[int, Tuple[int, int]]):
        """Build compact representation from position dict."""
        n = len(positions)
        
        self.hashes = np.zeros(n, dtype=np.uint64)
        self.results = np.zeros(n, dtype=np.int8)
        self.dtws = np.zeros(n, dtype=np.uint8)
        
        for i, (h, (r, d)) in enumerate(positions.items()):
            self.hashes[i] = h & 0xFFFFFFFFFFFFFFFF  # Ensure positive
            self.results[i] = r
            self.dtws[i] = min(d, 255)
        
        # Sort by hash for binary search
        sort_idx = np.argsort(self.hashes)
        self.hashes = self.hashes[sort_idx]
        self.results = self.results[sort_idx]
        self.dtws = self.dtws[sort_idx]
    
    def lookup(self, board_hash: int) -> Optional[Tuple[int, int]]:
        """O(log n) lookup using binary search."""
        if self.hashes is None:
            return None
        
        h = board_hash & 0xFFFFFFFFFFFFFFFF
        idx = np.searchsorted(self.hashes, h)
        
        if idx < len(self.hashes) and self.hashes[idx] == h:
            return (int(self.results[idx]), int(self.dtws[idx]))
        
        return None
    
    def save(self, path: str):
        """Save compact tablebase."""
        np.savez_compressed(
            path,
            hashes=self.hashes,
            results=self.results,
            dtws=self.dtws
        )
        print(f"✓ Saved compact tablebase: {len(self.hashes)} positions")
    
    def load(self, path: str):
        """Load compact tablebase."""
        data = np.load(path)
        self.hashes = data['hashes']
        self.results = data['results']
        self.dtws = data['dtws']
        print(f"✓ Loaded compact tablebase: {len(self.hashes)} positions")
    
    def get_size_mb(self) -> float:
        """Get actual memory usage in MB."""
        if self.hashes is None:
            return 0
        total = (self.hashes.nbytes + self.results.nbytes + self.dtws.nbytes)
        return total / (1024 * 1024)
