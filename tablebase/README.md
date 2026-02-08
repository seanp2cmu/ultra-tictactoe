# Ultra Tic-Tac-Toe Tablebase

## 개요

Endgame tablebase는 게임 종료 직전 상태들의 최적해를 미리 계산해 저장합니다.
**Progressive Endgame Construction** 방식으로 1개 빈칸부터 N개 빈칸까지 순차적으로 구축합니다.

## 핵심 개념

### 1. Progressive Endgame Construction (점진적 엔드게임 구축)

```
Level 1 (1 empty) → Level 2 (2 empty) → ... → Level N
     ↓                    ↓
  Solve directly      Lookup children (Level N-1)
```

- **Level 1**: 1개 빈칸 = 한 수만 가능하므로 직접 해결 (승/패/무)
- **Level N**: N개 빈칸 상태에서 가능한 모든 수를 두면 N-1개 빈칸 상태가 됨
- 이미 계산된 하위 레벨(N-1) 결과를 참조하여 현재 레벨(N) 해결
- Minimax 원리: 상대방도 최선을 둔다고 가정

**참고**: Retrograde analysis(역행 분석)와 다름
- Retrograde: Terminal 상태에서 역으로 추적
- 우리 방식: 간단한 endgame부터 순차적으로 해결

### 2. Canonical Hashing (정규화 해싱)

동일한 게임 상태를 하나의 해시로 표현하여 저장 공간 절약:

#### D4 Symmetry (8가지 변환)
```
Original  Rot90   Rot180  Rot270  FlipH   FlipV   Diag    AntiDiag
1 2 3    7 4 1   9 8 7   3 6 9   3 2 1   7 8 9   1 4 7   9 6 3
4 5 6 →  8 5 2   6 5 4   2 5 8   6 5 4   4 5 6   2 5 8   8 5 2
7 8 9    9 6 3   3 2 1   1 4 7   9 8 7   1 2 3   3 6 9   7 4 1
```

#### X/O Flip (2가지)
- X와 O를 서로 바꿈
- X_WIN ↔ O_WIN, 1 ↔ 2

#### 총 감소율
- 8 (D4) × 2 (X/O flip) = **최대 16배 감소**

### 3. Board State Representation

```python
# 각 서브보드 상태
sub_data = (state, x_count, o_count)

# state 값:
# 0 = OPEN (진행 중)
# 1 = X_WIN
# 2 = O_WIN  
# 3 = DRAW

# 예시: 9개 서브보드
board_data = (
    (0, 3, 4),  # 서브보드 0: OPEN, X 3개, O 4개
    (1, 0, 0),  # 서브보드 1: X_WIN
    (2, 0, 0),  # 서브보드 2: O_WIN
    ...
)
```

### 4. Constraint (제약)

다음 수를 둬야 할 서브보드 인덱스:
- **-1 (any)**: 어디든 가능 (상대방이 완료된 보드로 보냈을 때)
- **0-8**: 특정 서브보드에만 착수 가능

## 저장 구조

```python
# positions: Dict[hash, Dict[constraint, Tuple[result, dtw, best_move]]]
{
    12345678: {
        0: (1, 3, (2, 5)),   # constraint=0: 승리, DTW=3, 최선수=(2,5)
        2: (-1, 2, (0, 1)),  # constraint=2: 패배, DTW=2
    },
    ...
}
```

- **result**: 1=현재 플레이어 승, -1=패, 0=무승부
- **dtw**: Depth To Win (승리까지 남은 수)
- **best_move**: 최적 착수 좌표

## 빌드 과정

### 1. Meta-Board 열거

```python
# 가능한 메타보드 조합 (4^9 = 262,144개)
# D4 + X/O flip으로 ~2,600개로 축소
for meta in product([0, 1, 2, 3], repeat=9):
    canonical = _canonical_meta(meta)  # D4 + flip 적용
```

### 2. X/O 분배 열거

```python
# 각 OPEN 서브보드에 X/O 개수 분배
# 예: x_total=7, o_total=5, 2개 OPEN
# → (4,3), (3,4), (5,2), (2,5), ...
```

### 3. 셀 구성 생성

```python
# 각 분배에 대해 승자 없는 셀 배치 찾기
cells = _find_valid_cell_config(x_count, o_count, empty_count)
# [1, 2, 0, 1, 0, 2, 1, 2, 0] → X, O, empty, X, empty, ...
```

### 4. 해결 및 저장

```python
result, dtw, best_move = solver.solve(board)
# Child lookup: 이미 계산된 하위 레벨 결과 참조
```

## 사용 방법

### 빌드

```bash
# 로컬에서 빌드 (권장)
python -m tablebase.builder --max-empty 10 --output tablebase/endgame.pkl

# 옵션:
#   --max-empty N    : N개 빈칸까지 계산 (기본: 10)
#   --output PATH    : 저장 경로
#   --max-per-level M: 레벨당 최대 M개 (테스트용)
```

### 로드 및 사용

```python
from tablebase.solver import TablebaseSolver

# 기존 tablebase 로드
solver = TablebaseSolver(base_tablebase_path="tablebase/endgame.pkl")

# 포지션 해결
result, dtw, best_move = solver.solve(board)
```

## 통계 (예상)

| Level | Positions | Cumulative |
|-------|-----------|------------|
| 1     | ~13,000   | ~13,000    |
| 2     | ~114,000  | ~127,000   |
| 3     | ~400,000  | ~527,000   |
| ...   | ...       | ...        |
| 10    | ~N M      | ~N M       |

## 파일 구조

```
tablebase/
├── __init__.py
├── builder.py      # 빌드 로직
├── solver.py       # 해결 & 조회
├── enumerator.py   # 포지션 열거
├── endgame.pkl     # 생성된 tablebase
└── README.md       # 이 문서
```

## 주의사항

1. **HuggingFace Spaces**: 재시작 시 파일 삭제됨 → 로컬에서 빌드 후 업로드
2. **메모리**: Level 10+ 시 수 GB 필요
3. **시간**: Level 10까지 수 시간 소요
4. **Resume**: 중단 후 재시작 시 `completed_empty` 레벨 스킵
