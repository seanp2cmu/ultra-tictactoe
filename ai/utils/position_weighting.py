"""
포지션별 학습 가중치
전환 구간(26-29칸)에 집중, 엔드게임 영역은 낮게
"""
from game.board import Board


class WeightedSampleBuffer:
    """
    가중치 기반 샘플링 버퍼
    
    전환 구간 데이터를 더 자주 샘플링
    deque 사용으로 O(1) FIFO 성능
    """
    
    def __init__(self, max_size=50000):
        from collections import deque
        self.data = deque(maxlen=max_size)
        self.weights = deque(maxlen=max_size)
        self.categories = deque(maxlen=max_size)
        self.max_size = max_size
        
        self.stats = {
            "opening": 0,
            "early_mid": 0,
            "mid": 0,
            "transition": 0,
            "near_endgame": 0,
            "endgame": 0,
            "deep_endgame": 0
        }
        
        self.total_weight = 0.0
    
    def add(self, state, policy, value, board: Board, dtw=None):
        """
        데이터 추가 (가중치 포함)
        
        Args:
            state: board state
            policy: policy probability
            value: value target
            board: Board object (for weight calculation)
            dtw: DTW value (optional)
        """
        weight, category = board.get_phase()
        
        if len(self.data) >= self.max_size:
            old_category = self.categories[0]
            old_weight = self.weights[0]
            
            self.stats[old_category] -= 1
            self.total_weight -= old_weight
        
        self.data.append((state, policy, value, dtw, weight))
        self.weights.append(weight)
        self.categories.append(category)
        
        self.stats[category] += 1
        self.total_weight += weight
    
    def sample(self, batch_size):
        """
        weight based sampling
        
        transition part more often selected
        """
        import numpy as np
        
        if len(self.data) < batch_size:
            batch = list(self.data)
        else:
            probs = [w / self.total_weight for w in self.weights]
            
            indices = np.random.choice(
                len(self.data), 
                size=batch_size, 
                replace=False,
                p=probs
            )
            batch = [self.data[i] for i in indices]
        
        states, policies, values, dtws, weights = zip(*batch)
        return (
            np.array(states), 
            np.array(policies), 
            np.array(values), 
            list(dtws),
            np.array(weights)
        )
    
    def get_stats(self):
        """return statistics"""
        total = len(self.data)
        if total == 0:
            return {}
        
        return {
            "total": total,
            "distribution": {
                cat: f"{count}/{total} ({100*count/total:.1f}%)"
                for cat, count in self.stats.items()
            },
            "avg_weight": sum(self.weights) / len(self.weights) if self.weights else 0
        }
    
    def __len__(self):
        return len(self.data)