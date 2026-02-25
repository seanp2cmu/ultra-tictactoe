#!/usr/bin/env python3
"""
Distributed NNUE Trainer — runs on GPU server.

Polls shared data directory for new .npz batches from workers,
merges data, trains NNUE, exports new model, bumps version.

Usage:
    python -m nnue.distributed.trainer \
        --shared-dir /mnt/shared/nnue \
        --max-loops 50 \
        --min-new-positions 50000 \
        --max-positions 2000000

Directory structure (shared storage):
    shared_dir/
    ├── model/
    │   ├── nnue_model.nnue    ← trainer exports here
    │   ├── nnue_model.pt      ← PyTorch weights
    │   └── version.txt        ← monotonic version counter
    ├── data/
    │   ├── v003_worker0_001.npz  ← consumed and moved to consumed/
    │   └── ...
    ├── consumed/              ← processed data files
    └── runs/                  ← wandb/logs
"""
import argparse
import glob
import os
import shutil
import time
from pathlib import Path

import numpy as np
import wandb

from nnue.core.model import NNUE
from nnue.config import NNUEConfig, TrainingConfig
from nnue.cpp.export_weights import export_weights
from nnue.training.trainer import train_nnue
from evaluation.evaluator import nnue_run_evaluation_suite as run_evaluation_suite
from nnue.agent import NNUEAgent


def get_model_version(shared_dir: Path) -> int:
    vfile = shared_dir / "model" / "version.txt"
    if vfile.exists():
        return int(vfile.read_text().strip())
    return 0


def set_model_version(shared_dir: Path, version: int):
    vfile = shared_dir / "model" / "version.txt"
    vfile.parent.mkdir(parents=True, exist_ok=True)
    vfile.write_text(str(version))


def collect_new_data(shared_dir: Path) -> tuple:
    """Collect all pending .npz files from data directory.

    Returns (boards, values, file_list) or (None, None, []) if no data.
    """
    data_dir = shared_dir / "data"
    if not data_dir.exists():
        return None, None, []

    # Only pick up completed files (not .tmp_*)
    files = sorted(data_dir.glob("v*.npz"))
    if not files:
        return None, None, []

    board_chunks = []
    value_chunks = []
    for f in files:
        try:
            d = np.load(str(f))
            board_chunks.append(d['boards'])
            value_chunks.append(d['values'])
        except Exception as e:
            print(f"[Trainer] Skipping corrupt file {f.name}: {e}")
            continue

    if not board_chunks:
        return None, None, []

    boards = np.concatenate(board_chunks)
    values = np.concatenate(value_chunks)
    return boards, values, files


def consume_files(shared_dir: Path, files: list):
    """Move processed files to consumed/ directory."""
    consumed_dir = shared_dir / "consumed"
    consumed_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = consumed_dir / f.name
        f.rename(dest)


def bootstrap_model(shared_dir: Path, seed_data_path: str = None):
    """Create initial NNUE model if none exists."""
    model_dir = shared_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    nnue_path = model_dir / "nnue_model.nnue"
    pt_path = model_dir / "nnue_model.pt"

    if nnue_path.exists():
        print(f"[Trainer] Existing model found: {nnue_path}")
        return

    if seed_data_path and Path(seed_data_path).exists():
        print(f"[Trainer] Bootstrapping from seed data: {seed_data_path}")
        d = np.load(seed_data_path)
        boards, values = d['boards'], d['values']

        train_cfg = TrainingConfig()
        train_cfg.num_epochs = 10
        model, val_loss, _ = train_nnue(
            data=(boards, values),
            output_path=str(pt_path),
            train_config=train_cfg,
            device="cuda",
        )
        model.eval()
        export_weights(model, str(nnue_path))
    else:
        print("[Trainer] No seed data — creating random NNUE model")
        model = NNUE()
        import torch
        torch.save(model.state_dict(), str(pt_path))
        model.eval()
        export_weights(model, str(nnue_path))

    set_model_version(shared_dir, 1)
    print(f"[Trainer] Initial model exported (v1)")


def run_trainer(args):
    shared_dir = Path(args.shared_dir)
    model_dir = shared_dir / "model"
    pt_path = model_dir / "nnue_model.pt"
    nnue_path = model_dir / "nnue_model.nnue"

    print(f"[Trainer] Starting — shared_dir={shared_dir}")
    print(f"[Trainer] min_new_positions={args.min_new_positions:,}, "
          f"max_positions={args.max_positions:,}")

    # Bootstrap if needed
    bootstrap_model(shared_dir, args.seed_data)

    # Init wandb
    wandb.init(project="uttt-nnue-distributed", name=args.run_name or None)

    # Accumulated training data
    all_boards = None
    all_values = None

    # Load existing checkpoint if available
    ckpt_path = shared_dir / "checkpoint.npz"
    if ckpt_path.exists():
        d = np.load(str(ckpt_path))
        all_boards, all_values = d['boards'], d['values']
        print(f"[Trainer] Checkpoint loaded: {len(all_values):,} positions")

    version = get_model_version(shared_dir)
    train_cfg = TrainingConfig()
    train_cfg.num_epochs = args.train_epochs
    train_cfg.batch_size = 4096

    for loop in range(args.max_loops):
        print(f"\n{'='*60}")
        print(f"[Trainer] Loop {loop+1}/{args.max_loops} — Model v{version}")
        print(f"{'='*60}")

        # Poll for new data
        print("[Trainer] Waiting for worker data...")
        total_new = 0
        new_boards_list = []
        new_values_list = []

        while total_new < args.min_new_positions:
            boards, values, files = collect_new_data(shared_dir)
            if boards is not None and len(values) > 0:
                new_boards_list.append(boards)
                new_values_list.append(values)
                total_new += len(values)
                consume_files(shared_dir, files)
                print(f"  Collected {len(values):,} new positions "
                      f"({len(files)} files) | Total new: {total_new:,}")
            else:
                time.sleep(10)  # Wait for workers

        # Merge new data
        new_boards = np.concatenate(new_boards_list)
        new_values = np.concatenate(new_values_list)

        if all_boards is not None:
            all_boards = np.concatenate([all_boards, new_boards])
            all_values = np.concatenate([all_values, new_values])
        else:
            all_boards = new_boards
            all_values = new_values

        # Trim to max_positions (keep newest)
        if len(all_values) > args.max_positions:
            all_boards = all_boards[-args.max_positions:]
            all_values = all_values[-args.max_positions:]

        print(f"[Trainer] Training on {len(all_values):,} positions "
              f"({total_new:,} new)")

        # Train (warm-start from previous)
        warm_start = str(pt_path) if pt_path.exists() else None
        model, val_loss, ep_metrics = train_nnue(
            data=(all_boards, all_values),
            output_path=str(pt_path),
            train_config=train_cfg,
            device="cuda",
            warm_start_path=warm_start,
        )

        # Export new NNUE
        model.eval()
        export_weights(model, str(nnue_path))

        # Bump version (workers will pick up new model)
        version += 1
        set_model_version(shared_dir, version)
        print(f"[Trainer] Model v{version} exported")

        # Save checkpoint
        np.savez_compressed(str(ckpt_path),
                            boards=all_boards, values=all_values)

        # Evaluate
        agent = NNUEAgent(model_path=str(pt_path), depth=6)
        eval_metrics = run_evaluation_suite(
            agent, num_games=args.eval_games, num_games_minimax=args.eval_games)

        wr = eval_metrics.get('eval/vs_random_winrate', 0)
        mm2 = eval_metrics.get('eval/vs_minimax2_winrate', 0)

        wandb.log({
            **eval_metrics,
            'step/version': version,
            'step/total_positions': len(all_values),
            'step/new_positions': total_new,
            'step/val_loss': val_loss,
        })

        print(f"[Trainer] v{version}: val={val_loss:.6f} "
              f"vs_mm2={mm2:.0f}% vs_rnd={wr:.0f}% "
              f"total_pos={len(all_values):,}")

    # Signal workers to stop
    stop_file = shared_dir / "STOP"
    stop_file.write_text("done")
    print(f"\n[Trainer] Done. {args.max_loops} loops complete.")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="NNUE Distributed Trainer")
    parser.add_argument("--shared-dir", type=str, required=True,
                        help="Path to shared storage directory")
    parser.add_argument("--seed-data", type=str, default=None,
                        help="Path to seed .npz for bootstrapping")
    parser.add_argument("--run-name", type=str, default=None,
                        help="wandb run name")
    parser.add_argument("--max-loops", type=int, default=50,
                        help="Maximum training loops")
    parser.add_argument("--min-new-positions", type=int, default=50000,
                        help="Min new positions before retraining")
    parser.add_argument("--max-positions", type=int, default=2_000_000,
                        help="Max total positions to keep")
    parser.add_argument("--train-epochs", type=int, default=10,
                        help="Training epochs per loop")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Eval games per loop")
    args = parser.parse_args()

    run_trainer(args)


if __name__ == "__main__":
    main()
