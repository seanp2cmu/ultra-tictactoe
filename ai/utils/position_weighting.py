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
        self.categories = deque(maxlen=max_size)  # 카테고리 추적
        self.max_size = max_size
        
        # 통계
        self.stats = {
            "opening": 0,
            "early_mid": 0,
            "mid": 0,
            "transition": 0,
            "near_endgame": 0,
            "endgame": 0,
            "deep_endgame": 0
        }
        
        # 누적 가중치 (샘플링 최적화)
        self.total_weight = 0.0
    
    def add(self, state, policy, value, board: Board, dtw=None):
        """
        데이터 추가 (가중치 포함)
        
        Args:
            state: 보드 state
            policy: policy 확률
            value: value 타겟
            board: Board 객체 (가중치 계산용)
            dtw: DTW 값 (선택)
        """
        # 가중치 계산 (Board 메서드 사용)
        weight, category = board.get_phase()
        
        # deque가 가득 차면 자동으로 가장 오래된 것 제거
        if len(self.data) >= self.max_size:
            # 제거될 항목의 카테고리와 가중치
            old_category = self.categories[0]
            old_weight = self.weights[0]
            
            # 통계 업데이트
            self.stats[old_category] -= 1
            self.total_weight -= old_weight
        
        # 추가 (deque가 자동으로 maxlen 관리)
        self.data.append((state, policy, value, dtw, weight))
        self.weights.append(weight)
        self.categories.append(category)
        
        # 통계 업데이트
        self.stats[category] += 1
        self.total_weight += weight
    
    def sample(self, batch_size):
        """
        가중치 기반 샘플링 (누적 가중치 사용으로 O(n) → O(1))
        
        전환 구간(weight=1.2)이 더 자주 선택됨
        """
        import numpy as np
        
        if len(self.data) < batch_size:
            batch = list(self.data)
        else:
            # 가중치 기반 샘플링 (누적 가중치 재사용)
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
        """통계 반환"""
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


def print_weight_schedule():
    """가중치 스케줄 출력 (디버깅용)"""
    print("=" * 60)
    print("Position Weighting Schedule")
    print("=" * 60)
    print("Empty Cells | Weight | Category")
    print("-" * 60)
    print("50+         | 1.0    | Opening/Mid")
    print("40-49       | 1.0    | Early Mid")
    print("30-39       | 1.0    | Mid")
    print("26-29       | 1.2    | ★ Transition (Most Important!)")
    print("20-25       | 0.8    | Near Endgame")
    print("10-19       | 0.5    | Endgame")
    print("0-9         | 0.3    | Deep Endgame")
    print("=" * 60)
    print("\nRationale:")
    print("- Transition (26-29): Critical decision point")
    print("- Endgame (≤25): Perfect solution exists, less learning needed")
    print("- Focus on positions where neural net can add value")
    print("=" * 60)


if __name__ == "__main__":
    # 테스트
    print_weight_schedule()
