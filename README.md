# Ultimate Tic-Tac-Toe AlphaZero + DTW

A hybrid AI for Ultimate Tic-Tac-Toe combining AlphaZero (ResNet + MCTS) with Distance to Win (DTW) endgame solver.

## Key Features

- **AlphaZero**: Deep ResNet with SE attention + Monte Carlo Tree Search
- **DTW (Distance to Win)**: Alpha-Beta complete search for endgame positions
- **Hybrid Search**: MCTS for opening, shallow Alpha-Beta for midgame, complete search for endgame

## Project Structure

```
ultra-tictacto/
├── ai/
│   ├── core/                       # Neural network (ResNet + SE blocks)
│   │   └── network.py              # Model, SEBlock, ResidualBlock, AlphaZeroNet
│   ├── mcts/                       # Monte Carlo Tree Search
│   │   ├── agent.py                # AlphaZeroAgent with DTW integration
│   │   └── node.py                 # MCTS Node with UCB selection
│   ├── endgame/                    # DTW endgame solver
│   │   ├── dtw_calculator.py       # Alpha-Beta search with move ordering
│   │   └── transposition_table.py  # Hot/Cold 2-tier cache with compression
│   ├── training/                   # Training pipeline
│   │   ├── trainer.py              # Main training loop
│   │   ├── self_play.py            # Self-play game generation
│   │   └── replay_buffer.py        # Experience replay
│   ├── prediction/                 # Inference helpers
│   │   └── prediction_agent.py     # create_prediction_agent, create_strong_agent
│   └── utils/                      # Utilities
│       ├── board_symmetry.py       # D4 symmetry normalization (8x memory savings)
│       └── position_weighting.py   # Weighted sampling for transition positions
├── game/
│   └── board.py                    # Ultimate Tic-Tac-Toe game logic
├── test/                           # Unit tests
├── config.py                       # Configuration dataclasses
└── train.py                        # Training script
```

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Inference

```python
from ai.prediction import create_prediction_agent, create_strong_agent

# Standard agent (400 simulations)
agent = create_prediction_agent(model_path="./model/model_final.pth")

# Strong agent (800 simulations, lower temperature)
agent = create_strong_agent(model_path="./model/model_final.pth")

# Play a move
from game import Board
board = Board()
action = agent.select_action(board, temperature=0)
row, col = action // 9, action % 9
```

## Search Strategy

```
Opening (46-81 empty cells):
└─ Pure MCTS + Neural Network policy/value

Midgame (16-45 empty cells):
├─ MCTS selects top 5 candidate moves
├─ Shallow Alpha-Beta (depth=8) checks for forced wins/losses
├─ Winning move found → select immediately
└─ Losing moves → exclude from candidates

Endgame (≤15 empty cells):
├─ Complete Alpha-Beta search
├─ Transposition table caching
├─ 8-fold symmetry normalization (D4 group)
└─ Winning position + DTW≤5 → use optimal move directly
```

## Neural Network Architecture

```
Input: 7 channels × 9×9
├─ Ch 0: Current player stones
├─ Ch 1: Opponent player stones
├─ Ch 2: Won sub-boards (current player perspective)
├─ Ch 3: Lost sub-boards (current player perspective)
├─ Ch 4: Drawn sub-boards
├─ Ch 5: Last move position
└─ Ch 6: Valid sub-board mask (where next move is allowed)

Backbone: ResNet with SE attention
├─ 30 residual blocks
├─ 512 channels
└─ Squeeze-and-Excitation (reduction=16)

Dual Heads:
├─ Policy Head: Conv 1×1 → FC → 81 outputs (softmax)
└─ Value Head: Conv 1×1 → FC → FC → 1 output (tanh)
```

## DTW Endgame Solver

The Distance to Win (DTW) calculator provides perfect play in endgame positions:

- **Alpha-Beta Pruning**: Minimax search with alpha-beta cutoffs
- **Move Ordering**: Center → corners → edges priority for better pruning
- **Transposition Table**: Hot/Cold 2-tier LRU cache
  - Hot cache: Fast access, no compression (5M entries)
  - Cold cache: Compressed to 3 bytes per entry (20M entries)
- **Symmetry Normalization**: Canonical board representation using D4 group (8x memory savings)

## Position Weighting

The training pipeline uses weighted sampling to focus learning on critical game phases:

```
Empty Cells | Weight | Category
------------|--------|------------------
50+         | 1.0    | Opening
40-49       | 1.0    | Early Mid
30-39       | 1.0    | Mid
26-29       | 1.2    | ★ Transition (Most Important)
20-25       | 1.0    | Near Endgame
10-19       | 0.6    | Endgame
0-9         | 0.4    | Deep Endgame
```

**Rationale**:
- **Transition (26-29 cells)**: Critical decision point where games are often decided. Higher weight ensures more training focus.
- **Near Endgame (20-25 cells)**: Shallow Alpha-Beta only (DTW threshold=15). NN guidance still important, so standard weight.
- **Endgame (10-19 cells)**: DTW provides perfect solutions, reduced weight but still maintains pattern learning.
- **Deep Endgame (0-9 cells)**: DTW solves completely, but some weight retained for NN to learn endgame patterns for MCTS initial evaluation.

The `WeightedSampleBuffer` implements O(1) weighted sampling using cumulative weights and numpy random choice.

## Training Strategy

### Adaptive Scheduling

Training uses progressive scheduling to optimize learning efficiency:

```
Progress   | Simulations | Games | Temperature
-----------|-------------|-------|------------
0-20%      | 200         | 83    | 1.0 (exploration)
20-50%     | 380         | 133   | 1.0
50-80%     | 560         | 183   | 0.65
80-100%    | 800         | 250   | 0.3 (exploitation)
```

**Rationale**:
- **Early training**: Network is near-random, so fewer games/sims suffice. High temperature encourages diverse exploration.
- **Mid training**: Gradual increase as network improves and data quality matters more.
- **Late training**: Maximum resources for fine-tuning with low temperature for precise play.

### Training Loop

Each iteration:
1. **Self-play**: Generate games using current network + MCTS + DTW
2. **Replay buffer**: Add samples with phase-based weights
3. **Training**: Sample weighted batch, train for 40 epochs
4. **Checkpoint**: Save model and DTW cache every 10 iterations

### Replay Buffer

- Size: 1,000,000 samples (sliding window)
- Sampling: Weighted by game phase (transition positions prioritized)
- Accumulation: Data persists across iterations for diversity

## Configuration

```python
# config.py defaults
NetworkConfig:
    num_res_blocks: 30
    num_channels: 512

TrainingConfig:
    num_iterations: 300
    num_self_play_games: 250
    num_simulations: 800 (adaptive: 200→800)
    batch_size: 4096
    lr: 0.002
    use_amp: True  # Mixed precision

DTWConfig:
    endgame_threshold: 15   # Complete search threshold
    midgame_threshold: 45   # Shallow search threshold
    shallow_depth: 8        # Midgame search depth limit
    hot_cache_size: 5M
    cold_cache_size: 20M
```

## Optimizations

- **Fast Board.clone()**: Manual copy vs deepcopy (80x faster)
- **Node clone elimination**: Avoid double cloning in MCTS expansion
- **is_terminal caching**: Memoized terminal state detection
- **Symmetry normalization**: 8x cache memory savings via D4 group
- **Mixed Precision (AMP)**: FP16 training for 2x memory efficiency
- **Adaptive simulations**: Start with 200 sims, increase to 800 as training progresses

## Testing

```bash
python -m pytest test/ -v
```

## API Reference

### AlphaZeroAgent

```python
agent = AlphaZeroAgent(
    network,                    # AlphaZeroNet instance
    num_simulations=100,        # MCTS simulations per move
    c_puct=1.0,                 # Exploration constant
    temperature=1.0,            # Action selection temperature
    dtw_calculator=None         # Optional DTWCalculator
)

# Select best action
action = agent.select_action(board, temperature=0)

# Get action probability distribution
probs = agent.get_action_probs(board, temperature=1.0)
```

### DTWCalculator

```python
dtw = DTWCalculator(
    use_cache=True,
    hot_size=5000000,
    cold_size=20000000,
    endgame_threshold=15,
    midgame_threshold=45,
    shallow_depth=8
)

# Check if endgame position
if dtw.is_endgame(board):
    result, distance, best_move = dtw.calculate_dtw(board)
    # result: 1 (win), -1 (loss), 0 (draw)
    # distance: moves to outcome
    # best_move: (row, col) tuple

# Get winning move if available
move, distance = dtw.get_best_winning_move(board)
```

### Trainer

```python
trainer = Trainer(
    lr=0.002,
    batch_size=4096,
    num_simulations=800,
    replay_buffer_size=1000000,
    device="cuda",
    use_amp=True
)

# Single training iteration
result = trainer.train_iteration(
    num_self_play_games=250,
    num_train_epochs=40,
    temperature=1.0,
    num_simulations=800
)

# Save/Load
trainer.save("./model/model.pth")
trainer.load("./model/model.pth")
```

## License

MIT License

## References

- [Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815)
- [Ultimate Tic-Tac-Toe Rules](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
