# Ultra Tic-Tac-Toe AlphaZero

AlphaZero 알고리즘을 사용한 Ultimate Tic-Tac-Toe AI 학습 프로젝트

## 프로젝트 구조

```
ultra-tictacto/
├── ai/                    # AI 관련 모듈
│   ├── agent.py          # MCTS Agent 및 AlphaZero Agent
│   ├── network.py        # Neural Network (ResNet)
│   ├── trainer.py        # Self-play 및 학습 로직
│   └── env.py            # 환경 인터페이스
├── game/                  # 게임 로직
│   └── board.py          # Ultimate Tic-Tac-Toe 보드
├── ui/                    # UI 관련
│   └── game_ui.py        # Pygame UI
├── config.py             # 설정 파일 (GPU 최적화 포함)
├── train.py              # 학습 실행 스크립트
└── main.py               # 게임 실행
```

## 설치 방법

1. **가상환경 생성 및 활성화**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

## 사용법

### 학습 실행

**자동 설정 (GPU 감지)**
```bash
python train.py
```

**커스텀 설정**
```python
from config import Config, get_gpu_optimized_config, get_cpu_config
from train import train_alphazero

# GPU 최적화 설정 (RTX 5090 등)
config = get_gpu_optimized_config()
train_alphazero(config)

# CPU 설정
config = get_cpu_config()
train_alphazero(config)
```

### 게임 플레이

```bash
python main.py
```

## 설정 (config.py)

### NetworkConfig
- `num_res_blocks`: ResNet 블록 개수 (기본: 10)
- `num_channels`: 채널 수 (기본: 256)

### TrainingConfig
- `num_iterations`: 학습 반복 횟수
- `batch_size`: 배치 사이즈 (GPU: 1024, CPU: 32)
- `num_simulations`: MCTS 시뮬레이션 횟수
- `use_amp`: Mixed Precision Training 사용 여부

### 주요 특징

1. **6채널 입력**
   - Player 1 positions
   - Player 2 positions
   - Current player indicator
   - Completed boards (Player 1)
   - Completed boards (Player 2)
   - Draw boards

2. **GPU 최적화**
   - Mixed Precision Training (AMP)
   - Large batch size support (1024)
   - CUDA/MPS/CPU 자동 감지
   - **배치 MCTS**: GPU 활용을 극대화하는 배치 평가
   - **Virtual Loss**: 병렬 시뮬레이션 지원

3. **AlphaZero 구현**
   - MCTS with neural network guidance
   - Self-play data generation
   - Policy + Value dual-head network

### 🚀 성능 최적화 (v2.0)

#### 배치 MCTS
기존 순차 MCTS의 GPU 활용도가 낮은 문제를 해결하기 위해 배치 평가를 도입했습니다.

**개선 사항:**
- **배치 크기**: 8개의 MCTS 시뮬레이션을 동시에 평가
- **Virtual Loss**: 병렬 시뮬레이션 중 트리 탐색 충돌 방지
- **GPU 처리량**: 네트워크 호출 횟수 감소로 GPU 활용도 향상

**사용법:**
```python
from ai.agent import AlphaZeroAgent

# 배치 크기 설정 (기본값: 8)
agent = AlphaZeroAgent(
    network=network,
    num_simulations=100,
    batch_size=8  # GPU 메모리에 따라 조절
)
```

**예상 성능 향상:**
- Self-play 속도: ~2-3x 향상 (GPU 사용 시)
- GPU 활용률: 20-30% → 60-80%

## 네트워크 아키텍처

- **Input**: 6 channels × 9×9 board
- **Backbone**: ResNet (10 blocks, 256 channels)
- **Policy Head**: 81 output (모든 가능한 위치)
- **Value Head**: 1 output (승률 예측, -1 ~ 1)
- **Parameters**: ~11.8M (256ch, 10 blocks)

## 프로젝트 패키징

```bash
./package.sh
```

이 스크립트는:
- `requirements.txt` 자동 생성
- `.venv`, `__pycache__`, `.git` 등 제외
- 타임스탬프가 포함된 tar.gz 파일 생성

## 라이센스

MIT License
# ultra-tictactoe
# ultra-tictactoe
