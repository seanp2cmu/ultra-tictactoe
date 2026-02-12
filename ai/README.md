---
title: Ultra Tic-Tac-Toe AI
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Ultimate Tic-Tac-Toe AlphaZero

A pure AlphaZero implementation for Ultimate Tic-Tac-Toe, following Leela Chess Zero (Lc0) best practices.

## Key Features

- **Pure AlphaZero**: ResNet with SE attention + Monte Carlo Tree Search
- **Lc0-style Training**: Temperature only for first 8 moves, FPU=-1, value range 0~1
- **DTW Endgame Solver**: Alpha-Beta complete search for endgame positions (≤15 cells)

## Architecture

### Neural Network

```
Input: 7 channels × 9×9
├─ Ch 0-1: Current/Opponent stones
├─ Ch 2-4: Won/Lost/Drawn sub-boards
├─ Ch 5: Last move position
└─ Ch 6: Valid move mask

Backbone: 30 ResBlocks × 512 channels + SE attention

Dual Heads:
├─ Policy: Conv 1×1 → FC → 81 (softmax)
└─ Value: Conv 1×1 → FC → 1 (sigmoid, 0~1)
```

### Value Range (Lc0-style)

| Value | Meaning |
|-------|---------|
| 0.0 | Loss |
| 0.5 | Draw |
| 1.0 | Win |

### MCTS Settings

- **FPU (First Play Urgency)**: -1 for unvisited nodes
- **Temperature**: Applied only for first 8 moves (exploration), greedy after
- **c_puct**: 1.0

## Search Strategy

```
Opening/Midgame (>15 empty cells):
└─ Pure MCTS + Neural Network

Endgame (≤15 empty cells):
├─ Complete Alpha-Beta DTW search
├─ Transposition table with D4 symmetry (8x savings)
└─ Winning move with DTW≤5 → use directly
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Play

```python
from ai.prediction import create_prediction_agent
from game import Board

agent = create_prediction_agent(model_path="./model/best.pt")
board = Board()
action = agent.select_action(board, temperature=0)
row, col = action // 9, action % 9
```

## Configuration

```python
NetworkConfig:
    num_res_blocks: 30
    num_channels: 512

TrainingConfig:
    num_iterations: 300
    num_self_play_games: 250
    num_simulations: 800
    batch_size: 1024
    lr: 0.002

DTWConfig:
    endgame_threshold: 15
```

## Training Notes

Based on Leela Chess Zero findings:

1. **Temperature only for first N moves** - AlphaZero applies temperature only for opening moves (30 ply in chess, 8 moves here). This prevents random blunders in mid/endgame.

2. **Value range 0~1** - Loss=0, Draw=0.5, Win=1 (not -1~1). Simpler for sigmoid output.

3. **FPU = -1** - Unvisited nodes are assumed to be losses, encouraging exploitation of known-good moves.

4. **No shallow search** - Pure MCTS + NN for all non-endgame positions. DTW only for complete endgame solving.

## Testing

```bash
python -m pytest test/ -v
```

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [Leela Chess Zero](https://lczero.org/)
- [Lc0 AlphaZero Findings](https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/)
