"""
Compressed Transposition Table for DTW caching
Hot/Cold 2-tier 구조 + 보드 대칭 정규화로 메모리 효율 극대화
"""
from collections import OrderedDict
from game import Board

from ai.utils import BoardSymmetry

class CompressedTranspositionTable:
    def __init__(self, hot_size=50000, cold_size=500000, use_symmetry=True):
        """
        Args:
            hot_size: Hot cache 크기 (빠른 접근, 압축 안 함)
            cold_size: Cold cache 크기 (느린 접근, 압축)
            use_symmetry: 보드 대칭 정규화 사용 (8배 메모리 절약)
        """
        self.hot = OrderedDict()
        self.hot_size = hot_size
        
        self.cold = {}
        self.cold_size = cold_size
        
        self.use_symmetry = use_symmetry
        if use_symmetry:
            self.symmetry = BoardSymmetry()
        
        self.stats = {
            "hot_hits": 0,
            "cold_hits": 0,
            "misses": 0,
            "evictions": 0,
            "symmetry_saves": 0  # 대칭으로 인한 중복 방지 횟수
        }
    
    def get_hash(self, board: Board):
        """
        보드를 해시로 변환
        use_symmetry=True면 정규화된 형태로 변환 (8배 절약)
        """
        if self.use_symmetry:
            return self.symmetry.get_canonical_hash(board)
        else:
            boards_tuple = tuple(tuple(row) for row in board.boards)
            completed_tuple = tuple(tuple(row) for row in board.completed_boards)
            return hash((boards_tuple, completed_tuple, board.current_player))
    
    def compress_entry(self, result, dtw, best_move):
        """
        엔트리 압축
        - result: -1/0/1 → 1바이트 (128=패, 129=무, 130=승)
        - dtw: int (최대 254) → 1바이트 (255=inf)
        - best_move: (row, col) → 1바이트 (row*9+col, 255=None)
        총: 3바이트 (원본 16바이트 대비 5배 압축)
        """
        # Result 인코딩: -1→128, 0→129, 1→130
        result_byte = 128 + result + 1
        
        # DTW 인코딩
        dtw_byte = min(dtw if dtw != float('inf') else 255, 255)
        
        # Best move 인코딩
        if best_move is None:
            move_byte = 255
        else:
            row, col = best_move
            move_byte = row * 9 + col
        
        return bytes([result_byte, dtw_byte, move_byte])
    
    def decompress_entry(self, compressed):
        """압축된 엔트리 복원"""
        # Result 디코딩
        result = compressed[0] - 128 - 1
        
        # DTW 디코딩
        dtw = compressed[1]
        if dtw == 255:
            dtw = float('inf')
        
        # Best move 디코딩
        move_byte = compressed[2]
        if move_byte == 255:
            best_move = None
        else:
            best_move = (move_byte // 9, move_byte % 9)
        
        return result, dtw, best_move
    
    def get(self, board):
        """
        캐시에서 조회
        Hot → Cold 순서
        
        Returns:
            (result, dtw, best_move) or None
        """
        key = self.get_hash(board)
        
        # Hot 확인
        if key in self.hot:
            self.stats["hot_hits"] += 1
            self.hot.move_to_end(key)
            return self.hot[key]
        
        # Cold 확인
        if key in self.cold:
            self.stats["cold_hits"] += 1
            compressed = self.cold[key]
            result, dtw, best_move = self.decompress_entry(compressed)
            
            # Hot으로 승격
            self._promote_to_hot(key, result, dtw, best_move)
            return (result, dtw, best_move)
        
        # Miss
        self.stats["misses"] += 1
        return None
    
    def put(self, board, result, dtw, best_move=None):
        """
        Hot에 저장
        
        Args:
            board: Board 객체
            result: 1 (승), -1 (패), 0 (무승부)
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
            # Cold에서 Hot으로 승격하며 업데이트
            self._promote_to_hot(key, result, dtw, best_move)
            return
        
        if len(self.hot) >= self.hot_size:
            self._evict_from_hot()
        
        self.hot[key] = (result, dtw, best_move)
    
    def _promote_to_hot(self, key, result, dtw, best_move):
        """Cold → Hot 승격"""
        if key in self.cold:
            del self.cold[key]
        
        if len(self.hot) >= self.hot_size:
            self._evict_from_hot()
        
        self.hot[key] = (result, dtw, best_move)
    
    def _evict_from_hot(self):
        """Hot에서 가장 오래된 것을 Cold로"""
        old_key, (old_result, old_dtw, old_best_move) = self.hot.popitem(last=False)
        self.stats["evictions"] += 1
        
        if len(self.cold) >= self.cold_size:
            self._evict_from_cold()
        
        compressed = self.compress_entry(old_result, old_dtw, old_best_move)
        self.cold[old_key] = compressed
    
    def _evict_from_cold(self):
        """Cold에서 임의의 엔트리 삭제 (FIFO)"""
        if not self.cold:
            return
        
        # dict의 첫 번째 엔트리 제거
        first_key = next(iter(self.cold))
        del self.cold[first_key]
    
    def get_memory_usage(self):
        """메모리 사용량 추정 (MB)"""
        # Hot: (result, dtw, best_move) = 튜플 약 24바이트
        hot_memory = len(self.hot) * 24
        # Cold: 3바이트 압축
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
        """통계 반환"""
        total = sum([self.stats["hot_hits"], self.stats["cold_hits"], self.stats["misses"]])
        hit_rate = (self.stats["hot_hits"] + self.stats["cold_hits"]) / total if total > 0 else 0
        
        memory = self.get_memory_usage()
        
        return {
            **self.stats,
            "total_queries": total,
            "hit_rate": f"{hit_rate:.2%}",
            **memory
        }
    
    def clear(self):
        """캐시 초기화"""
        self.hot.clear()
        self.cold.clear()
        self.stats = {
            "hot_hits": 0,
            "cold_hits": 0,
            "misses": 0,
            "evictions": 0,
            "symmetry_saves": 0
        }
    
    def save_to_file(self, filepath):
        """
        Tablebase를 디스크에 저장
        
        Args:
            filepath: 저장할 파일 경로 (예: './model/tablebase.pkl')
        
        크기: ~1 GB (2000만 포지션 기준)
        - Hot: 200만 × 56 bytes ≈ 112 MB
        - Cold: 2000만 × 35 bytes ≈ 700 MB
        - Overhead: ~200 MB
        """
        import pickle
        import os
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Hot + Cold 모두 저장
        data = {
            'hot': dict(self.hot),
            'cold': self.cold,
            'stats': self.stats,
            'use_symmetry': self.use_symmetry
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"✓ Tablebase saved: {filepath} ({size_mb:.1f} MB)")
        print(f"  Entries: {len(self.hot)} hot + {len(self.cold)} cold")
    
    def load_from_file(self, filepath):
        """
        Tablebase를 디스크에서 로드
        
        Args:
            filepath: 로드할 파일 경로
        
        Returns:
            bool: 성공 여부
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            print(f"⚠ Tablebase not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # OrderedDict로 복원
            self.hot = OrderedDict(data['hot'])
            self.cold = data['cold']
            self.stats = data['stats']
            
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✓ Tablebase loaded: {filepath} ({size_mb:.1f} MB)")
            print(f"  Entries: {len(self.hot)} hot + {len(self.cold)} cold")
            
            stats = self.get_stats()
            if 'hit_rate' in stats:
                print(f"  Cache stats available")
            
            return True
        
        except Exception as e:
            print(f"✗ Failed to load tablebase: {e}")
            return False
