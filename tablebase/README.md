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

## 수학적 최적화 기법

### 1. 상태 공간 축소 (State Space Reduction)

#### 1.1 D4 Symmetry Group (정사각형 대칭군)
정사각형의 대칭 변환 8가지를 활용하여 동치인 상태를 하나로 통합:

```
|G| = 8 (D4 group order)
- 항등원 (e)
- 회전: r₉₀, r₁₈₀, r₂₇₀
- 반사: σₕ (수평), σᵥ (수직), σ_d (대각), σ_d' (반대각)

이론적 감소율: 최대 8배
실제 감소율: ~7-8배 (일부 상태는 자기 자신과 대칭)
```

인덱스 변환 테이블:
```python
D4_TRANSFORMS = [
    [0,1,2,3,4,5,6,7,8],  # e (항등)
    [6,3,0,7,4,1,8,5,2],  # r₉₀
    [8,7,6,5,4,3,2,1,0],  # r₁₈₀
    [2,5,8,1,4,7,0,3,6],  # r₂₇₀
    [2,1,0,5,4,3,8,7,6],  # σₕ
    [6,7,8,3,4,5,0,1,2],  # σᵥ
    [0,3,6,1,4,7,2,5,8],  # σ_d
    [8,5,2,7,4,1,6,3,0],  # σ_d'
]
```

#### 1.2 X/O Flip Symmetry (색상 교환 대칭)
X와 O는 게임 규칙상 동등하므로 교환해도 동치:

```
변환: X ↔ O, X_WIN ↔ O_WIN
이론적 감소율: 2배
결합 감소율: D4 × Flip = 최대 16배
```

#### 1.3 Canonical Form (정규형)
모든 대칭 변환 중 사전순 최소값 선택:

```python
def canonical(board):
    candidates = []
    for transform in D4_TRANSFORMS:
        sym = apply_transform(board, transform)
        candidates.append(sym)
        candidates.append(flip_xo(sym))  # X/O 교환
    return min(candidates)  # 사전순 최소
```

### 2. 유효성 필터링 (Validity Filtering)

#### 2.1 Meta-Board 승자 필터
이미 승부가 결정된 메타보드 제외:

```python
# 3x3 승리 패턴 (8가지)
WIN_PATTERNS = [(0,1,2), (3,4,5), (6,7,8),  # 가로
                (0,3,6), (1,4,7), (2,5,8),  # 세로
                (0,4,8), (2,4,6)]           # 대각

for a, b, c in WIN_PATTERNS:
    if meta[a] == meta[b] == meta[c] in (1, 2):
        skip()  # 이미 승자 있음
```

#### 2.2 X-O 개수 제약 (Count Constraint)
유효한 게임 상태에서 X와 O의 개수 차이는 0 또는 1:

```
|X_count - O_count| ∈ {0, 1}

완료된 서브보드별 X-O 차이 범위:
- X_WIN: [-3, +7] (X가 3개로 승리 ~ X가 9개, O가 2개)
- O_WIN: [-7, +3]
- DRAW:  [-1, +1] (9칸 모두 채워짐, 차이 1 이하)

전체 diff = Σ(서브보드별 diff) 
유효 조건: diff ∈ {0, 1}
```

#### 2.3 서브보드 승자 없음 필터
OPEN 서브보드에서 이미 3줄이 완성되면 무효:

```python
def has_winner(cells):
    for a, b, c in WIN_PATTERNS:
        if cells[a] == cells[b] == cells[c] != 0:
            return True
    return False
```

### 3. 열거 최적화 (Enumeration Optimization)

#### 3.1 빈칸 분배 (Empty Cell Distribution)
N개 빈칸을 k개 OPEN 서브보드에 분배:

```
각 서브보드: 1 ≤ empty ≤ 8 (0이면 완료, 9면 빈 보드)
제약: Σ(empty_i) = N

분배 수: 조합 문제 (정수 분할)
예: N=5, k=2 → (1,4), (2,3), (3,2), (4,1)
```

#### 3.2 X/O 분배 (Piece Distribution)
총 X, O 개수를 OPEN 서브보드들에 분배:

```
filled_i = 9 - empty_i
x_i + o_i = filled_i

모든 유효한 (x₀, x₁, ..., x_{k-1}) 열거
where Σ(x_i) = x_total and 0 ≤ x_i ≤ filled_i
```

#### 3.3 셀 구성 캐싱 (Cell Configuration Caching)
동일한 (x_count, o_count, empty_count)에 대해 결과 재사용:

```python
_cell_config_cache = {}

def find_valid_config(x, o, empty):
    key = (x, o, empty)
    if key in cache:
        return cache[key]
    
    # 조합으로 탐색 (순열보다 효율적)
    # C(9, x) × C(9-x, o) vs 9!
    for x_positions in combinations(range(9), x):
        for o_positions in combinations(remaining, o):
            if not has_winner(config):
                cache[key] = config
                return config
```

### 4. 해시 충돌 처리

#### 4.1 Constraint 변환
D4 변환 시 constraint도 함께 변환:

```python
# 원본 constraint가 위치 c에 있을 때
# 변환 T 적용 후 새 위치: T.index(c)

def hash_with_constraint(board, constraint):
    for perm in D4_TRANSFORMS:
        sym_data = transform(board, perm)
        sym_constraint = perm.index(constraint)
        candidates.append((sym_data, sym_constraint))
    return min(candidates)
```

#### 4.2 동치 Constraint 처리
같은 hash의 다른 constraint는 대칭으로 동치:

```python
def lookup(hash, constraint):
    if constraint in cache[hash]:
        return cache[hash][constraint]
    # 동치인 아무 constraint 사용
    return cache[hash][any_available]
```

## 빌드 과정

### 1. Meta-Board 열거

```python
# 가능한 메타보드 조합: 4^9 = 262,144개
# D4 symmetry 적용: ~33,000개
# D4 + X/O flip 적용: ~2,600개 (canonical만)

for meta in product([0, 1, 2, 3], repeat=9):
    canonical = _canonical_meta(meta)
    if canonical != meta:
        continue  # 이미 처리됨
```

### 2. X/O 분배 열거

```python
# diff 범위 계산 후 유효한 (x_total, o_total) 생성
for diff in range(min_valid_diff, max_valid_diff + 1):
    x_total = (total_filled + diff) // 2
    o_total = (total_filled - diff) // 2
    
    # 각 서브보드에 분배
    for x_dist in gen_x_distributions(x_total, open_boards):
        yield (x_dist, o_dist)
```

### 3. 셀 구성 생성

```python
# 캐시된 조합 탐색
cells = _find_valid_cell_config(x_count, o_count, empty_count)
# 승자 없는 첫 번째 유효 구성 반환
```

### 4. 해결 및 저장

```python
result, dtw, best_move = solver.solve(board)
# Child lookup: O(1) 해시 테이블 조회
# Minimax: max(child) for current player
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
1. **메모리**: Level 10+ 시 수 GB 필요
2. **시간**: Level 10까지 수 시간 소요
3. **Resume**: 중단 후 재시작 시 `completed_empty` 레벨 스킵
