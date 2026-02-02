# Ultimate Tic-Tac-Toe AlphaZero + DTW

AlphaZero + Alpha-Beta Search (DTW)를 결합한 Ultimate Tic-Tac-Toe AI

## 핵심 기술

- **AlphaZero**: 신경망 (ResNet + SE Block) + MCTS로 초중반 강력한 전략 학습
- **Distance to Win (DTW)**: 엔드게임(≤15칸)에서 Alpha-Beta 완전 탐색
- **Tablebase**: Hot/Cold 2-tier 캐싱으로 즉시 완벽한 수 반환
- **배치 MCTS**: GPU 최대 활용 + Virtual Loss 병렬화
- **최적화**: Board.clone(), 공유 DTW Calculator

## 프로젝트 구조

```
ultra-tictacto/
├── ai/
│   ├── core/
│   │   ├── network.py           # ResNet + SE Block 신경망
│   │   └── alpha_zero_net.py    # 학습/예측 래퍼
│   ├── mcts/
│   │   ├── agent.py             # AlphaZero Agent (배치 MCTS)
│   │   └── node.py              # MCTS 노드 (Virtual Loss)
│   ├── endgame/
│   │   ├── dtw_calculator.py    # Alpha-Beta Search (DTW)
│   │   └── transposition_table.py # 2-tier Tablebase (Hot/Cold)
│   ├── training/
│   │   ├── trainer.py           # DTW 통합 학습 루프
│   │   ├── self_play.py         # Self-play 워커
│   │   └── replay_buffer.py     # Position-weighted 버퍼
│   ├── prediction/
│   │   └── prediction_agent.py  # 실전/API용 Agent
│   └── utils/
│       └── batch_predictor.py   # 배치 예측 최적화
├── game/
│   └── board.py                 # Ultimate Tic-Tac-Toe (clone() 최적화)
├── test/                        # 66개 테스트 (포괄적 검증)
├── config.py                    # 설정 (기본/GPU/RTX 5090)
└── train.py                     # 학습 실행
```

## 빠른 시작

### 1. 설치

```bash
# 패키지 압축 해제
tar -xzf ultra-tictacto_YYYYMMDD_HHMMSS.tar.gz
cd ultra-tictacto

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 학습 실행

```bash
# RTX 5090 최적화 설정으로 학습
python train.py

# 또는 기본 GPU 설정
# config.py에서 get_gpu_optimized_config() 사용
```

### 3. 예측/API

```python
from ai.prediction_agent import create_prediction_agent

# 학습된 모델 + Tablebase 로드
agent = create_prediction_agent(
    model_path="./model/model_dtw_final.pth"
    # tablebase_path 자동 탐색
)

# 최선의 수 선택
move = agent.select_action(board)
```

## 주요 기능

### 1. DTW (Distance to Win) - 완벽한 엔드게임

```
엔드게임 (≤15 playable cells):
├─ Alpha-Beta Pruning 완전 탐색
├─ 승리까지 최단 거리 (DTW) 측정
├─ Transposition Table 캐싱 (Hot/Cold 2-tier)
├─ 8방향 대칭 정규화 (메모리 8배 절약)
└─ 완벽한 수 반환 ✓

중반전 (16-50 cells):
├─ MCTS + 신경망 평가
├─ DTW 캐시 참조 (높은 히트율)
└─ 강력한 플레이 ✓

초반 (51-81 cells):
└─ 순수 MCTS + 신경망 학습 ✓

※ endgame_threshold=15 (25칸은 계산 시간 초과)
```

### 2. Tablebase 저장/로드

```python
# 학습 중 자동 저장
- Iteration 20, 40, 60... 마다 체크포인트
- 최종: ./model/tablebase.pkl (~1 GB)

# 예측 시 자동 로드
agent = create_prediction_agent(model_path)
# → tablebase.pkl 자동 발견 및 로드

# 효과
- 25칸 이하: 즉시 완벽한 수 (캐시 조회)
- API 응답: <10ms
```

### 3. 배치 MCTS (GPU 최적화)

```python
기존 순차 MCTS:
└─ GPU 활용률 20-30% (네트워크 호출 낭비)

배치 MCTS:
├─ 8-16개 시뮬레이션 동시 평가
├─ Virtual Loss로 충돌 방지
├─ GPU 활용률 60-80% ✓
└─ Self-play 속도 2-3배 향상 ✓
```

## 설정 (config.py)

### RTX 5090 최적화 (`get_rtx5090_config()`)

```
네트워크:
├─ num_res_blocks: 20
├─ num_channels: 384
├─ SE Block: 채널 어텐션
└─ 파라미터: ~15M

학습:
├─ batch_size: 2048 (32GB VRAM)
├─ num_simulations: 400
├─ num_self_play_games: 200
├─ num_parallel_games: 32
├─ replay_buffer_size: 500k
└─ num_iterations: 300

DTW/Tablebase:
├─ endgame_threshold: 15 cells (완전 탐색)
├─ hot_cache: 500만 포지션
├─ cold_cache: 2000만 포지션
└─ use_symmetry: True (8배 절약)
```

### 기본 GPU 설정 (`get_gpu_optimized_config()`)

```python
네트워크:
├─ num_res_blocks: 10
├─ num_channels: 256
└─ 파라미터: ~11.8M

학습:
├─ batch_size: 1024
├─ num_simulations: 150
├─ num_self_play_games: 100
└─ num_parallel_games: 16
```

### CPU 설정 (`get_cpu_config()`)

```python
├─ batch_size: 32
├─ num_simulations: 50
├─ num_parallel_games: 4
└─ use_amp: False
```

## 기술 스택

### 신경망 아키텍처

```
Input: 7 channels × 9×9
├─ Channel 0: 현재 플레이어 돌 위치
├─ Channel 1: 상대 플레이어 돌 위치
├─ Channel 2: 현재 플레이어가 이긴 소보드
├─ Channel 3: 상대가 이긴 소보드
├─ Channel 4: 무승부 소보드
├─ Channel 5: 유효한 소보드 마스크
└─ Channel 6: 마지막 수 위치

Backbone: ResNet + SE Block
├─ 10-20 Residual Blocks
├─ 256-384 channels
├─ Squeeze-and-Excitation (채널 어텐션)
├─ Batch Normalization
└─ ReLU activation

Heads:
├─ Policy Head: Conv 1×1 → FC → 81 outputs (softmax)
└─ Value Head: Conv 1×1 → FC → 1 output (tanh, -1~1)

Parameters:
├─ 기본: ~172K (64ch, 2 blocks) - 테스트용
├─ 중간: ~11.8M (256ch, 10 blocks) - 학습용
└─ RTX 5090: ~15M (384ch, 20 blocks) - 고성능
```

### AlphaZero + DTW 통합

```
MCTS Search (Opening/Midgame):
├─ Selection: UCB + Prior (Neural Net)
├─ Expansion: Legal moves only + Neural Net prior
├─ Evaluation: Neural Net value prediction
├─ Backpropagation: 부호 교대 (-value)
└─ Action Selection: Visit count 기반

DTW Search (Endgame ≤15 cells):
├─ Alpha-Beta Pruning (효율적 탐색)
├─ Transposition Table (캐시 조회 우선)
├─ 8방향 대칭 정규화 (8배 메모리 절약)
├─ 완전 탐색 (depth 무제한)
└─ 최적의 수 + DTW 값 반환

Self-Play 통합:
├─ >15 cells: MCTS로 action_probs 생성
├─ ≤15 cells + 승리 확정 (DTW≤5): DTW 최적수 사용
└─ 학습 데이터: (state, policy, value, dtw)
```

### 최적화 기법

```
성능 최적화:
├─ Board.clone(): copy.deepcopy 대비 80배 빠름
├─ 공유 DTW Calculator: 캐시 재사용
├─ MCTS 단일 실행: search() 결과 재사용
└─ endgame_threshold=15: 계산 시간 최적화

GPU 최적화:
├─ Mixed Precision (AMP): FP16 연산으로 2배 속도
├─ Batch Predictor: 요청 모아서 배치 처리
├─ Virtual Loss: MCTS 병렬화
└─ Gradient Clipping: 학습 안정성

메모리 최적화:
├─ Symmetry Normalization: 8배 메모리 절약
├─ Hot/Cold 2-tier Cache: 접근 패턴 최적화
├─ Position Weighting: 중요 포지션 우선 샘플링
└─ Replay Buffer Deque: 효율적 FIFO
```

## 학습 루프

```
Train Iteration:
│
├─ 1. Self-Play (num_games개)
│   ├─ MCTS로 수 선택 (>15 cells)
│   ├─ DTW로 최적수 (≤15 cells, 승리 확정 시)
│   └─ 데이터 수집: (state, policy, value, dtw)
│
├─ 2. Replay Buffer 추가
│   ├─ Position weighting (게임 단계별)
│   └─ 승패 결과 역전파
│
├─ 3. Network Training (num_epochs)
│   ├─ Weighted sampling
│   ├─ Policy loss: Cross Entropy
│   ├─ Value loss: MSE
│   └─ Gradient clipping (max_norm=1.0)
│
└─ 4. Scheduler Step (Cosine Annealing)
```

## 테스트

```bash
# 전체 테스트 실행 (66개)
python -m pytest test/ -v

# 최종 검증 테스트만
python -m pytest test/test_final_verification.py -v

# 테스트 카테고리:
# - MCTS Backpropagation (4)
# - Network Training (4)
# - Train Loop (4)
# - Alpha-Beta/DTW (3)
# - Edge Cases (6)
# - Performance (3)
# - 기존 테스트 (42)
```

## 학습 결과

학습 준비 완료 (v1.0)

## 문서
- **[config.py](config.py)**: 설정 파일 (주석 포함)

## 패키징

```bash
./package.sh

# 생성물:
# ultra-tictacto_YYYYMMDD_HHMMSS.tar.gz (~44 KB)

# 포함:
# - 모든 소스 코드
# - requirements.txt
# - 문서
# - 테스트 코드

# 제외:
# - .venv (가상환경)
# - __pycache__ (컴파일 파일)
# - .git (버전 관리)
# - model/ (학습된 모델)
```

## 기여

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 라이센스

MIT License

## 참고 문헌

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [MCTS Survey](https://www.ru.is/faculty/yngvi/pdf/BrowneP12TCIAIG.pdf)
- [Ultimate Tic-Tac-Toe Rules](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
