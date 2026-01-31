"""
포지션별 학습 가중치
전환 구간(26-29칸)에 집중, Tablebase 영역은 낮게
"""


def get_position_weight(board):
    """
    보드 상태에 따른 학습 가중치 반환
    
    Args:
        board: Board 객체
        
    Returns:
        float: 학습 가중치 (0.3 ~ 1.2)
    
    가중치 전략:
    - 50칸+:   1.0  (미드게임, 중요)
    - 40-49칸: 1.0  (중반, 중요)
    - 30-39칸: 1.0  (중요)
    - 26-29칸: 1.2  (전환 구간, 가장 중요!)
    - 20-25칸: 0.8  (Tablebase 근처, 덜 중요)
    - 10-19칸: 0.5  (Tablebase 확실)
    - 0-9칸:   0.3  (Tablebase 완전 확실)
    """
    empty_count = sum(1 for row in board.boards for cell in row if cell == 0)
    
    if empty_count >= 50:
        return 1.0  # 미드게임
    elif empty_count >= 40:
        return 1.0  # 중반
    elif empty_count >= 30:
        return 1.0  # 중요 구간
    elif empty_count >= 26:
        return 1.2  # 전환 구간 - 가장 중요!
    elif empty_count >= 20:
        return 0.8  # Tablebase 근처
    elif empty_count >= 10:
        return 0.5  # Tablebase 영역
    else:
        return 0.3  # Tablebase 완전 확실


def get_position_category(board):
    """
    포지션 카테고리 반환 (통계용)
    
    Returns:
        str: 카테고리 이름
    """
    empty_count = sum(1 for row in board.boards for cell in row if cell == 0)
    
    if empty_count >= 50:
        return "opening"
    elif empty_count >= 40:
        return "early_mid"
    elif empty_count >= 30:
        return "mid"
    elif empty_count >= 26:
        return "transition"  # 전환 구간
    elif empty_count >= 20:
        return "near_tablebase"
    elif empty_count >= 10:
        return "tablebase"
    else:
        return "deep_tablebase"


class WeightedSampleBuffer:
    """
    가중치 기반 샘플링 버퍼
    
    전환 구간 데이터를 더 자주 샘플링
    """
    
    def __init__(self, max_size=50000):
        self.data = []
        self.weights = []
        self.max_size = max_size
        
        # 통계
        self.stats = {
            "opening": 0,
            "early_mid": 0,
            "mid": 0,
            "transition": 0,
            "near_tablebase": 0,
            "tablebase": 0,
            "deep_tablebase": 0
        }
    
    def add(self, state, policy, value, board, dtw=None):
        """
        데이터 추가 (가중치 포함)
        
        Args:
            state: 보드 state
            policy: policy 확률
            value: value 타겟
            board: Board 객체 (가중치 계산용)
            dtw: DTW 값 (선택)
        """
        # 가중치 계산
        weight = get_position_weight(board)
        category = get_position_category(board)
        
        # FIFO with max size
        if len(self.data) >= self.max_size:
            # 가장 오래된 것 제거
            self.data.pop(0)
            self.weights.pop(0)
            
            # 통계도 업데이트 (정확하지 않지만 근사)
            for cat in self.stats:
                if self.stats[cat] > 0:
                    self.stats[cat] -= 1
                    break
        
        self.data.append((state, policy, value, dtw, weight))
        self.weights.append(weight)
        self.stats[category] += 1
    
    def sample(self, batch_size):
        """
        가중치 기반 샘플링
        
        전환 구간(weight=1.2)이 더 자주 선택됨
        """
        import random
        import numpy as np
        
        if len(self.data) < batch_size:
            batch = self.data
        else:
            # 가중치 기반 샘플링
            total_weight = sum(self.weights)
            probs = [w / total_weight for w in self.weights]
            
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
    print("20-25       | 0.8    | Near Tablebase")
    print("10-19       | 0.5    | Tablebase")
    print("0-9         | 0.3    | Deep Tablebase")
    print("=" * 60)
    print("\nRationale:")
    print("- Transition (26-29): Critical decision point")
    print("- Tablebase (≤25): Perfect solution exists, less learning needed")
    print("- Focus on positions where neural net can add value")
    print("=" * 60)


if __name__ == "__main__":
    # 테스트
    print_weight_schedule()
