---
title: Ultra Tic-Tac-Toe AI
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Ultra Tic-Tac-Toe AI

**Pure AlphaZero Implementation for Ultimate Tic-Tac-Toe with DTW Endgame Solver**

A high-performance AI engine that combines deep reinforcement learning (AlphaZero-style) with perfect endgame play using Distance-To-Win (DTW) alpha-beta search.

## Features

- **AlphaZero-style Training**: Self-play reinforcement learning with MCTS
- **DTW Endgame Solver**: Perfect play in endgame positions (≤15 empty cells)
- **High Performance**: Cython + C++ optimizations for critical paths
- **Lc0-style Training**: Large-scale self-play with optimized replay buffer

---

## Project Structure

```
ultra-tictactoe/
├── app.py                    # Gradio web interface
├── setup.py                  # Cython build configuration
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies (HF Spaces)
│
├── ai/                       # AI module
│   ├── config.py             # Training configuration
│   ├── train.py              # Training loop
│   │
│   ├── core/                 # Neural network
│   │   ├── network.py        # Model architecture (ResNet + SE)
│   │   └── alpha_zero_net.py # Training wrapper (optimizer, AMP, torch.compile)
│   │
│   ├── mcts/                 # Monte Carlo Tree Search
│   │   ├── node.py           # Python MCTS node
│   │   ├── node_cy.pyx       # Cython MCTS node (optimized)
│   │   ├── mcts.py           # MCTS algorithm
│   │   └── agent.py          # AlphaZero agent
│   │
│   ├── endgame/              # DTW Endgame Solver
│   │   ├── dtw_calculator.py # DTW interface
│   │   └── transposition_table.py # Cache (hot/cold storage)
│   │
│   ├── training/             # Training components
│   │   ├── self_play.py      # Self-play game generation
│   │   ├── replay_buffer.py  # Lc0-style replay buffer
│   │   └── trainer.py        # Training orchestrator
│   │
│   ├── baselines/            # Baseline agents
│   │   ├── random_agent.py   # Random moves
│   │   ├── heuristic_agent.py # Rule-based
│   │   └── minimax_agent.py  # Minimax search
│   │
│   └── prediction/           # Inference utilities
│       └── prediction_agent.py
│
├── game/                     # Game logic
│   ├── board.py              # Python board implementation
│   ├── board_cy.pyx          # Cython board (training)
│   └── __init__.py           # Board import selector
│
├── cpp/                      # C++ extensions (pybind11)
│   ├── board.cpp/.hpp        # C++ board implementation
│   ├── dtw.cpp/.hpp          # C++ DTW alpha-beta search
│   ├── bindings.cpp          # pybind11 bindings
│   └── setup.py              # C++ build configuration
│
├── utils/                    # Utilities
│   ├── hf_upload.py          # HuggingFace upload
│   └── __init__.py
│
└── model/                    # Saved models & cache
    ├── best.pt               # Best model checkpoint
    ├── dtw_cache.pkl         # DTW transposition table
    └── training.log          # Training logs
```

---

## Model Architecture

### Neural Network (AlphaZero-style ResNet)

```
Input: 7 channels × 9 × 9
├── 7 input planes:
│   ├── [0-1] Current player pieces (per sub-board)
│   ├── [2-3] Opponent pieces (per sub-board)
│   ├── [4]   Valid moves mask
│   ├── [5]   Sub-board completion status
│   └── [6]   Current player indicator

Backbone: 30 Residual Blocks × 256 channels
├── Each block:
│   ├── Conv2d 3×3 → BatchNorm → ReLU
│   ├── Conv2d 3×3 → BatchNorm
│   ├── SE Block (Squeeze-and-Excitation, reduction=16)
│   └── Skip connection → ReLU

Policy Head:
├── Conv2d 1×1 → BatchNorm → ReLU
├── Flatten → Linear(162, 81)
└── Output: 81 logits (one per cell)

Value Head:
├── Conv2d 1×1 → BatchNorm → ReLU
├── Flatten → Linear(81, 64) → ReLU
├── Linear(64, 1) → Sigmoid
└── Output: Win probability [0, 1]
```

**Model Size**: ~107M parameters (~409 MB)

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Residual Blocks | 30 | Deep enough for strategic understanding |
| Channels | 256 | Balance between capacity and speed |
| SE Blocks | Yes | Channel attention improves feature selection |
| Value Output | Sigmoid [0,1] | 0=loss, 0.5=draw, 1=win |
| Activation | ReLU | Standard, fast |

---

## Optimization Strategy

### 1. Cython Extensions (`*.pyx`)

**`game/board_cy.pyx`** - Game board logic

- Bitboard representation for sub-board states
- Fast move validation and application
- ~10x faster than pure Python

**`ai/mcts/node_cy.pyx`** - MCTS node operations

- Efficient child node management
- Optimized UCB calculation
- ~5x faster tree operations

**Build**: `python setup.py build_ext --inplace`

### 2. C++ Extensions (pybind11)

**`cpp/board.cpp`** - C++ board implementation

- Used for DTW search (fastest path)
- Minimal Python overhead

**`cpp/dtw.cpp`** - Alpha-beta search

- Perfect endgame evaluation
- Transposition table integration
- ~20x faster than Python minimax

**Build**: `cd cpp && python setup.py build_ext --inplace`

### 3. PyTorch Optimizations

| Optimization | Description |
|--------------|-------------|
| `torch.compile` | Graph compilation for faster inference |
| AMP (FP16) | Mixed precision training |
| CUDAGraphs | Reduced kernel launch overhead |
| Batch Inference | Parallel position evaluation |

### 4. Performance Summary

| Component | Implementation | Speedup |
|-----------|---------------|---------|
| Board Logic | Cython (bitboard) | ~10x |
| MCTS Node | Cython | ~5x |
| DTW Search | C++ | ~20x |
| Network | torch.compile + AMP | ~2x |

---

## Training Methodology

### AlphaZero Algorithm

For each iteration:
    1. Self-Play: Generate 8,192 games using MCTS + current network
    2. Training: Update network on replay buffer samples
    3. Evaluation: (implicit via self-play improvement)

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Iterations | 400 | Total training cycles |
| Games/Iteration | 8,192 | Self-play games per cycle |
| MCTS Simulations | 800 | Simulations per move |
| Batch Size | 1,024 | Training batch size |
| Training Epochs | 40 | Epochs per iteration |
| Learning Rate | 0.002 | Initial LR (cosine decay) |
| Weight Decay | 1e-4 | L2 regularization |
| Replay Buffer | 2M | Maximum samples |

### Temperature Schedule

| Move Number | Temperature | Behavior |
|-------------|-------------|----------|
| 1-8 | 1.0 | Exploratory (proportional to visits) |
| 9+ | 0.0 | Greedy (best move only) |

### Loss Function

```
L = L_policy + L_value

L_policy = CrossEntropy(π_predicted, π_mcts)
L_value  = MSE(v_predicted, z_game_result)
```

---

## Data Collection (Self-Play)

### Parallel Self-Play

```python
# 256 games run in parallel
for batch in parallel_games:
    # 1. Get valid positions needing evaluation
    positions = [game.board for game in active_games]
    
    # 2. Batch neural network inference
    policies, values = network.predict_batch(positions)
    
    # 3. Run MCTS with network guidance
    for game, policy, value in zip(games, policies, values):
        mcts.search(game.board, prior_policy=policy)
        
    # 4. Select moves and record training data
    for game in games:
        move = select_move(mcts_policy, temperature)
        record_sample(board, mcts_policy, player)
```

### DTW Integration (Endgame)

When ≤15 empty cells remain:

1. **DTW Calculator** computes exact game-theoretic value
2. If **decisive** (win/loss for either player):
   - Skip remaining MCTS
   - Record position with perfect value
   - Early terminate game
3. Otherwise: Continue with MCTS

### Lc0-style Replay Buffer

| Feature | Description |
|---------|-------------|
| Game ID Tracking | Each sample tagged with game ID |
| Age-based Weighting | Recent games weighted higher |
| One Position Per Game | Batch sampling picks max 1 position per game |
| Deduplication | Reduces correlation in training batches |

```python
# Sampling strategy
weight = 1.0 / (1 + age_in_iterations * 0.1)
batch = sample_one_per_game(buffer, batch_size, weights)
```

---

## DTW Endgame Solver

### Distance-To-Win (DTW)

DTW measures how many moves until a forced win/loss:

- **DTW = +N**: Current player wins in N moves
- **DTW = -N**: Current player loses in N moves  
- **DTW = 0**: Draw with perfect play

### Alpha-Beta Search (C++)

```cpp
int alpha_beta(Board& board, int alpha, int beta, int depth) {
    // Transposition table lookup
    if (auto entry = tt.lookup(board.hash())) {
        return entry->value;
    }
    
    // Terminal check
    if (board.is_terminal()) {
        return evaluate_terminal(board);
    }
    
    // Search all moves
    int best = -INF;
    for (int move : board.valid_moves()) {
        board.make_move(move);
        int score = -alpha_beta(board, -beta, -alpha, depth + 1);
        board.undo_move(move);
        
        best = max(best, score);
        alpha = max(alpha, score);
        if (alpha >= beta) break;  // Pruning
    }
    
    tt.store(board.hash(), best);
    return best;
}
```

### Transposition Table

| Tier | Size | Purpose |
|------|------|---------|
| Hot Cache | 60M entries | Frequently accessed positions |
| Cold Cache | 240M entries | Archive for less common positions |

Cache is persisted to `dtw_cache.pkl` and uploaded to HuggingFace.

---

## Running the Project

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Build Cython extensions
python setup.py build_ext --inplace

# Build C++ extensions
cd cpp && python setup.py build_ext --inplace && cd ..

# Run Gradio app
python app.py
```

### Training

```bash
# Start training (interactive checkpoint selection)
python -m ai.train
```

### HuggingFace Spaces

The app automatically builds Cython/C++ extensions on startup via `packages.txt` (g++) and build logic in `app.py`.

---

## Evaluation

### Baseline Tests

| Opponent | Expected Win Rate | Notes |
|----------|-------------------|-------|
| Random | >99% | Sanity check |
| Heuristic | >95% | Rule-based agent |
| Minimax-2 | >90% | 2-ply search |
| Minimax-3 | >80% | 3-ply search |

### Self-Play Metrics

- **Loss convergence**: ~1.8 after 30 iterations
- **DTW cache hit rate**: Increases over training
- **Average game length**: ~50 moves

---

## Configuration

Edit `ai/config.py` to adjust:

```python
@dataclass
class TrainingConfig:
    num_iterations: int = 400
    num_self_play_games: int = 8192
    num_simulations: int = 800
    batch_size: int = 1024
    lr: float = 0.002
    
@dataclass
class DTWConfig:
    endgame_threshold: int = 15  # Use DTW when ≤15 cells empty
    hot_cache_size: int = 60_000_000
```

---

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Mastering Chess and Shogi
- [Lc0](https://lczero.org/) - Leela Chess Zero
- [Ultimate Tic-Tac-Toe Rules](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)

---

## License

MIT License
