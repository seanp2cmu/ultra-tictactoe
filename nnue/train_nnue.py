#!/usr/bin/env python3
"""
NNUE Training Pipeline — Phase 1 + Phase 2 unified.

Phase 1: Generate initial data from AlphaZero teacher
Phase 2+: Train NNUE, then self-play loop with C++ engine

All settings controlled via PipelineConfig in nnue/config.py.
Run: python -m nnue.train_nnue
"""
import time
from pathlib import Path

import numpy as np

import nnue_cpp
from nnue.core.model import NNUE
from nnue.cpp.export_weights import export_weights
from nnue.training.trainer import train_nnue
from nnue.config import PipelineConfig, TrainingConfig


# ─── Paths ────────────────────────────────────────────────────────

MODEL_DIR = Path("nnue/model")
DATA_DIR = Path("nnue/data")

PT_PATH = MODEL_DIR / "nnue_model.pt"
NNUE_PATH = MODEL_DIR / "nnue_model.nnue"


# ─── Phase 1: AlphaZero teacher datagen ──────────────────────────

def phase1_generate(cfg: PipelineConfig):
    """Generate initial NNUE training data using AlphaZero teacher."""
    from ai.core import AlphaZeroNet
    from nnue.data import NNUEDataGenerator
    from nnue.config import DataGenConfig

    output_path = DATA_DIR / "phase1_data.npz"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[Phase 1] Data already exists: {output_path}")
        data = np.load(output_path)
        print(f"  Positions: {len(data['values']):,}")
        return str(output_path)

    print(f"\n{'='*60}")
    print(f"Phase 1: AlphaZero Teacher Data Generation")
    print(f"{'='*60}")

    dg_config = DataGenConfig()
    dg_config.num_games = cfg.phase1_games
    dg_config.num_simulations = cfg.alphazero_sims
    dg_config.device = cfg.device

    print(f"Loading AlphaZero from {cfg.alphazero_model}...")
    network = AlphaZeroNet(device=cfg.device)
    if Path(cfg.alphazero_model).exists():
        network.load(cfg.alphazero_model)
        print("  Model loaded")
    else:
        print(f"  WARNING: {cfg.alphazero_model} not found, using random weights")

    generator = NNUEDataGenerator(
        network=network,
        config=dg_config,
        num_simulations=cfg.alphazero_sims,
        seed=42,
    )

    print(f"Generating {cfg.phase1_games} games (sims={cfg.alphazero_sims})...")
    t0 = time.time()
    dataset = generator.generate_dataset(
        num_games=cfg.phase1_games,
        output_path=str(output_path),
        verbose=True,
    )
    elapsed = time.time() - t0

    generator.print_stats()
    print(f"\nPhase 1 complete: {len(dataset['values']):,} positions in {elapsed:.0f}s")
    return str(output_path)


# ─── Train NNUE ──────────────────────────────────────────────────

def train_model(data_path, cfg: PipelineConfig, train_cfg: TrainingConfig):
    """Train NNUE model from .npz data. Returns (model, best_val_loss)."""
    print(f"\n{'='*60}")
    print(f"Training NNUE")
    print(f"{'='*60}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model, best_val_loss = train_nnue(
        data_path=data_path,
        output_path=str(PT_PATH),
        train_config=train_cfg,
        device=cfg.device,
    )

    print(f"\nExporting to {NNUE_PATH}...")
    model.eval()
    export_weights(model, str(NNUE_PATH))
    return model, best_val_loss


# ─── C++ self-play datagen ────────────────────────────────────────

def selfplay_generate(cfg: PipelineConfig, loop_idx: int):
    """Generate training data via C++ NNUE self-play."""
    print(f"\n{'='*60}")
    print(f"Self-Play Data Generation (loop {loop_idx})")
    print(f"{'='*60}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"selfplay_{loop_idx:03d}.npz"

    model = nnue_cpp.NNUEModel()
    if NNUE_PATH.exists():
        model.load(str(NNUE_PATH))
        print(f"  Loaded NNUE weights from {NNUE_PATH}")
    else:
        print(f"  WARNING: {NNUE_PATH} not found, using uninitialized weights")

    config = nnue_cpp.DataGenConfig()
    config.search_depth = cfg.selfplay_depth
    config.qsearch_mode = 2
    config.lambda_search = 0.75
    config.random_move_count = 8
    config.write_minply = 4
    config.write_maxply = 60
    config.eval_limit = 0.9
    config.random_skip_rate = 0.3
    config.skip_noisy = True
    config.skip_noisy_maxply = 30

    seed = 42 + loop_idx * 100000

    print(f"  Games: {cfg.selfplay_games}, Threads: {cfg.selfplay_threads}, "
          f"Depth: {cfg.selfplay_depth}")

    t0 = time.time()
    boards, values = nnue_cpp.generate_data(
        model, config,
        num_games=cfg.selfplay_games,
        num_threads=cfg.selfplay_threads,
        seed=seed,
    )
    elapsed = time.time() - t0

    rate = cfg.selfplay_games / elapsed if elapsed > 0 else 0
    print(f"  Generated {len(values):,} positions from {cfg.selfplay_games} games")
    print(f"  Time: {elapsed:.1f}s ({rate:.1f} games/s)")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")

    np.savez_compressed(str(output_path), boards=boards, values=values)
    print(f"  Saved to {output_path}")
    return str(output_path)


# ─── Merge datasets ──────────────────────────────────────────────

def merge_datasets(paths, max_positions=500000):
    """Merge multiple .npz datasets, keeping most recent if over limit."""
    all_boards, all_values = [], []

    for p in paths:
        if not Path(p).exists():
            continue
        data = np.load(p)
        all_boards.append(data['boards'])
        all_values.append(data['values'])

    if not all_boards:
        raise ValueError("No valid datasets to merge")

    boards = np.concatenate(all_boards)
    values = np.concatenate(all_values)

    if len(values) > max_positions:
        boards = boards[-max_positions:]
        values = values[-max_positions:]
        print(f"  Trimmed to {max_positions:,} most recent positions")

    merged_path = DATA_DIR / "merged.npz"
    np.savez_compressed(str(merged_path), boards=boards, values=values)
    print(f"  Merged {len(values):,} positions → {merged_path}")
    return str(merged_path)


# ─── Main pipeline ───────────────────────────────────────────────

def run(cfg: PipelineConfig = None, train_cfg: TrainingConfig = None):
    """Run the full NNUE training pipeline."""
    cfg = cfg or PipelineConfig()
    train_cfg = train_cfg or TrainingConfig()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data_paths = []

    # ── Phase 1: AlphaZero teacher ──
    if not cfg.skip_phase1:
        p1_path = phase1_generate(cfg)
        data_paths.append(p1_path)

        if cfg.phase1_only:
            print("\nPhase 1 complete.")
            return

        train_model(p1_path, cfg, train_cfg)
    else:
        if not PT_PATH.exists():
            print(f"ERROR: No model at {PT_PATH}. Run Phase 1 first.")
            return
        if not NNUE_PATH.exists():
            print(f"Exporting existing model to .nnue...")
            model = NNUE.load(str(PT_PATH))
            export_weights(model, str(NNUE_PATH))

    # ── Phase 2+: Self-play loop ──
    print(f"\n{'='*60}")
    print(f"Starting Self-Play Training Loop (max {cfg.selfplay_loops} iterations)")
    print(f"{'='*60}")

    best_loop_loss = float('inf')
    no_improve_count = 0

    for loop in range(cfg.selfplay_loops):
        print(f"\n{'─'*40}")
        print(f"Loop {loop + 1}/{cfg.selfplay_loops}")
        print(f"{'─'*40}")

        sp_path = selfplay_generate(cfg, loop_idx=loop)
        data_paths.append(sp_path)

        print(f"\nMerging {len(data_paths)} datasets...")
        merged_path = merge_datasets(data_paths, max_positions=cfg.max_positions)

        _, val_loss = train_model(merged_path, cfg, train_cfg)

        # Early stopping check
        if val_loss < best_loop_loss:
            best_loop_loss = val_loss
            no_improve_count = 0
            print(f"  ★ New best loop val_loss: {val_loss:.6f}")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{cfg.early_stop_patience})")

        if cfg.early_stop_patience > 0 and no_improve_count >= cfg.early_stop_patience:
            print(f"\n  Early stopping: no improvement for {cfg.early_stop_patience} loops.")
            break

        print(f"\nLoop {loop + 1} complete.")

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Model: {PT_PATH}")
    print(f"  C++ weights: {NNUE_PATH}")
    print(f"  Loops run: {loop + 1}, Best val_loss: {best_loop_loss:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
