"""NNUE configuration."""
from dataclasses import dataclass


@dataclass
class NNUEConfig:
    """NNUE network architecture."""
    
    # Feature dimensions
    num_cell_features: int = 81 * 3      # 243: each cell can be empty/P1/P2
    num_meta_features: int = 9 * 4       # 36: each mini-board status
    num_active_features: int = 9         # 9: which mini-board is active
    
    # Network architecture
    accumulator_size: int = 256
    hidden1_size: int = 32
    hidden2_size: int = 32
    
    @property
    def total_features(self) -> int:
        return self.num_cell_features + self.num_meta_features + self.num_active_features


@dataclass
class DataGenConfig:
    """Data generation settings."""
    
    model_path: str = "model/sclpjdv6/latest.pt"
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
    eval_limit: float = 0.9              # Skip if |eval| > this
    random_skip_rate: float = 0.3        # Random skip probability
    skip_endgame: bool = True            # Skip DTW-solvable positions
    skip_noisy: bool = True              # Skip positions with immediate tactical threats
    skip_noisy_maxply: int = 30          # Only apply noisy filter before this ply


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


@dataclass
class PipelineConfig:
    """Full NNUE training pipeline settings."""
    
    # Phase 1: AlphaZero teacher
    skip_phase1: bool = False
    phase1_only: bool = False
    phase1_games: int = 10000
    alphazero_model: str = "model/sclpjdv6/latest.pt"
    alphazero_sims: int = 800
    
    # Phase 2+: C++ self-play
    selfplay_loops: int = 50
    selfplay_games: int = 50000
    selfplay_threads: int = 64
    selfplay_depth: int = 8
    lambda_search: float = 0.75          # target = λ × search + (1-λ) × game_result
    
    # Evaluation
    eval_games: int = 200             # Games per baseline (random, heuristic)
    eval_games_minimax: int = 100     # Games for minimax (slower)
    eval_depth: int = 5               # NNUE search depth for eval games
    
    # General
    device: str = "cuda"
    max_positions: int = 2000000      # 2M positions → ~488 iters/epoch @ bs=4096
    early_stop_patience: int = 5     # Stop if val_loss doesn't improve for N loops (0=off)
