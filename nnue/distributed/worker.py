#!/usr/bin/env python3
"""
Distributed NNUE Worker — runs on CPU server(s).

Polls for new NNUE model versions, generates selfplay data via C++ engine,
and uploads .npz batches to shared storage.

Usage:
    python -m nnue.distributed.worker \
        --shared-dir /mnt/shared/nnue \
        --worker-id worker0 \
        --threads 16 \
        --games-per-batch 1000 \
        --depth 8

Directory structure (shared storage):
    shared_dir/
    ├── model/
    │   ├── nnue_model.nnue    ← trainer exports here
    │   └── version.txt        ← monotonic version counter
    ├── data/
    │   ├── v003_worker0_001.npz
    │   ├── v003_worker1_001.npz
    │   └── ...
    └── config.json            ← optional shared config
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def get_model_version(shared_dir: Path) -> int:
    """Read current model version from shared storage."""
    vfile = shared_dir / "model" / "version.txt"
    if vfile.exists():
        return int(vfile.read_text().strip())
    return 0


def wait_for_model(shared_dir: Path, timeout: float = 600) -> int:
    """Wait until a model is available. Returns version."""
    model_path = shared_dir / "model" / "nnue_model.nnue"
    t0 = time.time()
    while not model_path.exists():
        if time.time() - t0 > timeout:
            print(f"[Worker] Timeout waiting for model at {model_path}")
            sys.exit(1)
        time.sleep(5)
    return get_model_version(shared_dir)


def generate_batch(
    model_path: str,
    num_games: int,
    depth: int,
    threads: int,
    lambda_search: float,
    seed: int,
) -> tuple:
    """Generate one batch of selfplay data. Returns (boards, values)."""
    import nnue_cpp

    model = nnue_cpp.NNUEModel()
    model.load(model_path)

    config = nnue_cpp.DataGenConfig()
    config.search_depth = depth
    config.lambda_search = lambda_search
    config.early_stop_empty = 15

    boards, values = nnue_cpp.generate_data(
        model, config,
        num_games=num_games,
        num_threads=threads,
        seed=seed,
    )
    return boards, values


def save_batch(
    shared_dir: Path,
    worker_id: str,
    version: int,
    batch_num: int,
    boards: np.ndarray,
    values: np.ndarray,
):
    """Save batch to shared data directory."""
    data_dir = shared_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write to temp file then rename (atomic on same filesystem)
    filename = f"v{version:04d}_{worker_id}_{batch_num:04d}.npz"
    tmp_path = data_dir / f".tmp_{filename}"
    final_path = data_dir / filename

    np.savez_compressed(str(tmp_path), boards=boards, values=values)
    tmp_path.rename(final_path)

    print(f"[Worker] Saved {final_path.name}: {len(values):,} positions")


def run_worker(args):
    shared_dir = Path(args.shared_dir)
    worker_id = args.worker_id

    print(f"[Worker:{worker_id}] Starting — shared_dir={shared_dir}")
    print(f"[Worker:{worker_id}] Games/batch={args.games_per_batch}, "
          f"depth={args.depth}, threads={args.threads}")

    # Wait for initial model
    current_version = wait_for_model(shared_dir)
    model_path = str(shared_dir / "model" / "nnue_model.nnue")
    print(f"[Worker:{worker_id}] Model v{current_version} found")

    batch_num = 0
    total_positions = 0

    while True:
        # Check for new model version
        new_version = get_model_version(shared_dir)
        if new_version > current_version:
            print(f"[Worker:{worker_id}] Model updated: v{current_version} → v{new_version}")
            current_version = new_version
            time.sleep(1)  # Let file write finish

        # Generate a batch
        seed = hash((worker_id, current_version, batch_num)) % (2**32)
        t0 = time.time()

        boards, values = generate_batch(
            model_path=model_path,
            num_games=args.games_per_batch,
            depth=args.depth,
            threads=args.threads,
            lambda_search=args.lambda_search,
            seed=seed,
        )
        elapsed = time.time() - t0

        total_positions += len(values)
        rate = args.games_per_batch / max(1, elapsed)
        print(f"[Worker:{worker_id}] Batch {batch_num}: {len(values):,} pos, "
              f"{elapsed:.0f}s ({rate:.1f} g/s) | Total: {total_positions:,}")

        # Save to shared storage
        save_batch(shared_dir, worker_id, current_version, batch_num, boards, values)
        batch_num += 1

        # Check if trainer signaled stop
        stop_file = shared_dir / "STOP"
        if stop_file.exists():
            print(f"[Worker:{worker_id}] Stop signal received. Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="NNUE Distributed Worker")
    parser.add_argument("--shared-dir", type=str, required=True,
                        help="Path to shared storage directory")
    parser.add_argument("--worker-id", type=str, default="worker0",
                        help="Unique worker identifier")
    parser.add_argument("--threads", type=int, default=16,
                        help="Number of CPU threads for selfplay")
    parser.add_argument("--games-per-batch", type=int, default=1000,
                        help="Games per data batch")
    parser.add_argument("--depth", type=int, default=8,
                        help="NNUE search depth for selfplay")
    parser.add_argument("--lambda-search", type=float, default=0.75,
                        help="Target blending: λ×search + (1-λ)×result")
    args = parser.parse_args()

    run_worker(args)


if __name__ == "__main__":
    main()
