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
    output_path: str = "nnue_data/training_data.npz"
    device: str = "cuda"
    seed: int = None
    
    # Position filtering
    write_minply: int = 4                # Skip first N moves
    write_maxply: int = 60               # Skip after N moves
    eval_limit: float = 0.9              # Skip if |eval| > this
    random_skip_rate: float = 0.3        # Random skip probability
    skip_endgame: bool = True            # Skip DTW-solvable positions


@dataclass
class TrainingConfig:
    """NNUE training settings."""
    
    batch_size: int = 4096
    learning_rate: float = 0.001
    num_epochs: int = 10
