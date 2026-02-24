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
    
    model_path: str = "auto"                  # "auto" = find latest from runs.json
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
    alphazero_model: str = "auto"      # "auto" = find latest, or explicit path
    alphazero_sims: int = 800
    
    # Phase 2+: C++ self-play
    selfplay_loops: int = 50
    selfplay_games: int = 50000
    selfplay_threads: int = 64
    selfplay_depth: int = 8
    lambda_search: float = 0.75          # target = λ × search + (1-λ) × game_result
    
    # Rescoring: re-evaluate self-play positions with AlphaZero
    rescore_enabled: bool = True          # Enable GPU rescoring
    rescore_sims: int = 50                # AlphaZero sims per position (lower = faster)
    rescore_lambda: float = 0.75          # λ × alphazero + (1-λ) × original
    rescore_max_positions: int = 200000   # Max positions to rescore per loop
    
    # Evaluation
    eval_games: int = 200             # Games per baseline (random, heuristic)
    eval_games_minimax: int = 100     # Games for minimax (slower)
    eval_depth: int = 5               # NNUE search depth for eval games
    
    # Continuous mode: loop until NNUE beats AlphaZero
    continuous: bool = False          # Enable continuous training
    target_winrate: float = 55.0      # Stop when NNUE achieves this winrate vs AlphaZero
    vs_alphazero_sims: int = 50       # AlphaZero sims for strength comparison
    vs_alphazero_games: int = 200     # Games for NNUE vs AlphaZero match
    
    # General
    device: str = "cuda"
    max_positions: int = 2000000      # 2M positions → ~488 iters/epoch @ bs=4096
    early_stop_patience: int = 5     # Stop if val_loss doesn't improve for N loops (0=off)
    
    def __post_init__(self):
        if self.alphazero_model == "auto":
            self.alphazero_model = self._find_latest_alphazero()
    
    @staticmethod
    def _find_latest_alphazero(base_dir: str = "model") -> str:
        """Find the latest AlphaZero checkpoint from runs.json."""
        import os, json
        runs_path = os.path.join(base_dir, "runs.json")
        if not os.path.exists(runs_path):
            return os.path.join(base_dir, "latest.pt")
        
        with open(runs_path) as f:
            runs = json.load(f)
        
        # Find run with highest last_iteration
        best_run, best_iter = None, -1
        for run_id, info in runs.items():
            it = info.get('last_iteration', 0)
            latest = os.path.join(base_dir, run_id, 'latest.pt')
            if os.path.exists(latest) and it > best_iter:
                best_run, best_iter = run_id, it
        
        if best_run:
            path = os.path.join(base_dir, best_run, 'latest.pt')
            print(f"[NNUE] Auto-detected AlphaZero model: {path} (iter {best_iter})")
            return path
        
        return os.path.join(base_dir, "latest.pt")
