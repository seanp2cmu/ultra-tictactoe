"""
Compressed Transposition Table for DTW caching
"""
from collections import OrderedDict
from game import Board

from utils import BoardSymmetry

class CompressedTranspositionTable:
    def __init__(self, hot_size=50000, cold_size=500000):
        """
        Args:
            hot_size: Hot cache size (faster access, no compression)
            cold_size: Cold cache size (slower access, compression)
        """
        self.hot = OrderedDict()
        self.hot_size = hot_size
        
        self.cold = {}
        self.cold_size = cold_size
        
        self.symmetry = BoardSymmetry()
        
        self.stats = {
            "hot_hits": 0,
            "cold_hits": 0,
            "misses": 0,
            "evictions": 0,
            "symmetry_saves": 0 
        }
    
    def get_hash(self, board: Board):
        return self.symmetry.get_canonical_hash(board)
    
    def compress_entry(self, result, dtw, best_move):
        """
        entry compression
        - result: -1/0/1 → 1byte (128=lose, 129=draw, 130=win)
        - dtw: int (max 255) → 1byte (255=inf)
        - best_move: (row, col) → 1byte (row*9+col, 255=None)
        total: 3bytes (original 16bytes compressed 5x)
        """
        result_byte = 128 + result + 1
        
        dtw_byte = min(dtw if dtw != float('inf') else 255, 255)
        
        if best_move is None:
            move_byte = 255
        else:
            row, col = best_move
            move_byte = row * 9 + col
        
        return bytes([result_byte, dtw_byte, move_byte])
    
    def decompress_entry(self, compressed):
        """decompress entry"""
        result = compressed[0] - 128 - 1
        
        dtw = compressed[1]
        if dtw == 255:
            dtw = float('inf')
        
        move_byte = compressed[2]
        if move_byte == 255:
            best_move = None
        else:
            best_move = (move_byte // 9, move_byte % 9)
        
        return result, dtw, best_move
    
    def get(self, board):
        """
        get entry from cache
        Hot → Cold order
        
        Returns:
            (result, dtw, best_move) or None
        """
        key = self.get_hash(board)
        
        if key in self.hot:
            self.stats["hot_hits"] += 1
            self.hot.move_to_end(key)
            return self.hot[key]
        
        if key in self.cold:
            self.stats["cold_hits"] += 1
            compressed = self.cold[key]
            result, dtw, best_move = self.decompress_entry(compressed)
            
            self._promote_to_hot(key, result, dtw, best_move)
            return (result, dtw, best_move)
        
        self.stats["misses"] += 1
        return None
    
    def put(self, board, result, dtw, best_move=None):
        """
        store entry in cache
        
        Args:
            board: Board object
            result: 1 (win), -1 (lose), 0 (draw)
            dtw: Distance to Win/Loss
            best_move: (row, col) or None
        """
        key = self.get_hash(board)
        
        if key in self.hot:
            self.stats["symmetry_saves"] += 1
            self.hot[key] = (result, dtw, best_move)
            self.hot.move_to_end(key)
            return
        
        if key in self.cold:
            self.stats["symmetry_saves"] += 1
            self._promote_to_hot(key, result, dtw, best_move)
            return
        
        if len(self.hot) >= self.hot_size:
            self._evict_from_hot()
        
        self.hot[key] = (result, dtw, best_move)
    
    def _promote_to_hot(self, key, result, dtw, best_move):
        """promote entry from cold to hot"""
        if key in self.cold:
            del self.cold[key]
        
        if len(self.hot) >= self.hot_size:
            self._evict_from_hot()
        
        self.hot[key] = (result, dtw, best_move)
    
    def _evict_from_hot(self):
        """evict entry from hot"""
        old_key, (old_result, old_dtw, old_best_move) = self.hot.popitem(last=False)
        self.stats["evictions"] += 1
        
        if len(self.cold) >= self.cold_size:
            self._evict_from_cold()
        
        compressed = self.compress_entry(old_result, old_dtw, old_best_move)
        self.cold[old_key] = compressed
    
    def _evict_from_cold(self):
        """evict entry from cold"""
        if not self.cold:
            return
        
        first_key = next(iter(self.cold))
        del self.cold[first_key]
    
    def get_memory_usage(self):
        """estimate memory usage (MB)"""
        hot_memory = len(self.hot) * 24
        cold_memory = len(self.cold) * 3
        overhead = (len(self.hot) + len(self.cold)) * 8
        
        total_mb = (hot_memory + cold_memory + overhead) / 1024 / 1024
        
        return {
            "hot_mb": hot_memory / 1024 / 1024,
            "cold_mb": cold_memory / 1024 / 1024,
            "total_mb": total_mb,
            "hot_entries": len(self.hot),
            "cold_entries": len(self.cold)
        }
    
    def get_stats(self):
        """return statistics"""
        total = sum([self.stats["hot_hits"], self.stats["cold_hits"], self.stats["misses"]])
        hit_rate = (self.stats["hot_hits"] + self.stats["cold_hits"]) / total if total > 0 else 0
        
        memory = self.get_memory_usage()
        
        return {
            **self.stats,
            "total_queries": total,
            "hit_rate": f"{hit_rate:.2%}",
            **memory
        }
    
    def save_to_file(self, filepath):
        """
        save DTW cache to disk
        
        Args:
            filepath: path to save file (e.g. './model/dtw_cache.pkl')
        
        size: ~1 GB (20M positions example)
        - Hot: 20M x 56 bytes ≈ 112 MB
        - Cold: 200M x 35 bytes ≈ 700 MB
        - Overhead: ~200 MB
        """
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        data = {
            'hot': dict(self.hot),
            'cold': self.cold,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Silent save - log to file if needed
        pass
    
    def load_from_file(self, filepath):
        """
        load DTW cache from disk
        
        Args:
            filepath: path to load file
        
        Returns:
            bool: success or failure
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            print(f"⚠ DTW cache not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # OrderedDict로 복원
            self.hot = OrderedDict(data['hot'])
            self.cold = data['cold']
            self.stats = data['stats']
            
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✓ DTW cache loaded: {filepath} ({size_mb:.1f} MB)")
            print(f"  Entries: {len(self.hot)} hot + {len(self.cold)} cold")
            
            stats = self.get_stats()
            if 'hit_rate' in stats:
                print(f"  Cache stats available")
            
            return True
        
        except Exception as e:
            print(f"✗ Failed to load DTW cache: {e}")
            return False
