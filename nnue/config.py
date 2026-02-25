"""NNUE configuration."""
from dataclasses import dataclass


@dataclass
class NNUEConfig:
    """NNUE network architecture (SFNNv5+ style)."""
    
    # Network architecture
    accumulator_size: int = 256
    hidden1_size: int = 32
    hidden2_size: int = 32
    num_buckets: int = 4              # Layer stack buckets (game phase)
    bucket_divisor: int = 20          # bucket = piece_count // divisor (0-80 cells → 0-3)
    
    # Eval scaling (raw output → WDL)
    scaling_factor: float = 2.5       # sigmoid(raw / scaling) = win probability


@dataclass
class DataGenConfig:
    """Data generation settings."""
    
    model_path: str = "model/01ge7yt9/latest.pt"
    num_games: int = 10000
    num_simulations: int = 800
    output_path: str = "nnue/data/training_data.npz"
    device: str = "cuda"
    seed: int = None
    
    # Target blending: target = λ × search_score + (1-λ) × game_result
    lambda_search: float = 0.75          # 1.0 = search only, 0.0 = result only
    
    # Position filtering
    write_minply: int = 4                # Skip first N moves
    write_maxply: int = 60               # Skip after N moves
    eval_limit: float = 5.0              # Skip if |eval| > this (raw eval scale)
    random_skip_rate: float = 0.3        # Random skip probability
    skip_endgame: bool = True            # Skip DTW-solvable positions
    skip_noisy: bool = True              # Skip positions with immediate tactical threats
    skip_noisy_maxply: int = 30          # Only apply noisy filter before this ply


@dataclass
class PipelineConfig:
    """Full NNUE training pipeline settings."""
    
    # AlphaZero teacher (GPU producer) — bootstrap only
    alphazero_model: str = "model/01ge7yt9/latest.pt"
    alphazero_sims: int = 800
    az_batch_games: int = 500       # Games per AZ batch
    az_total_games: int = 500       # Total AZ games (seed data)
    az_lambda: float = 0.75         # target = λ×search + (1-λ)×result

    # C++ NNUE selfplay loop (CPU) — main data source
    selfplay_loops: int = 10        # Number of selfplay→retrain cycles
    selfplay_games: int = 5000      # Games per selfplay loop
    selfplay_depth: int = 8         # NNUE search depth for selfplay
    selfplay_threads: int = 16      # Threads for C++ NNUE selfplay
    selfplay_lambda: float = 0.75   # target = λ×search + (1-λ)×result

    # Training
    train_epochs: int = 10
    train_batch_size: int = 4096
    learning_rate: float = 0.001
    max_positions: int = 2_000_000

    # Evaluation
    eval_games: int = 100
    eval_games_minimax: int = 100
    eval_depth: int = 6             # NNUE search depth for eval games

    # General
    device: str = "cuda"
    model_dir: str = "nnue/model"


@dataclass
class TrainingConfig:
    """NNUE training settings."""
    
    batch_size: int = 4096
    learning_rate: float = 0.001
    weight_decay: float = 1e-6
    num_epochs: int = 10
    num_workers: int = 16
    grad_clip: float = 1.0
    use_amp: bool = True              # Automatic mixed precision
    lr_schedule: str = "cosine"       # "cosine" or "none"
    warmup_epochs: int = 1


