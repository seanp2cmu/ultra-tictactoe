#!/usr/bin/env python3
"""
Depth-8 NNUE selfplay prototyping (no wandb, local only).

1. Load existing AZ data (91K positions) as seed
2. Train initial NNUE on seed data
3. Export quantized weights → C++ selfplay → new data → merge → retrain
4. Repeat for N loops, print metrics each loop
"""
import os, sys, time, shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nnue_cpp
from nnue.core.model import NNUE
from nnue.cpp.export_weights import export_weights
from nnue.training.trainer import train_nnue
from nnue.config import NNUEConfig, TrainingConfig

# ─── Config ───
SEED_DATA = "nnue/model/cfrvhghx/checkpoint.npz"
RUN_DIR = Path("nnue/model/proto_d8")

SELFPLAY_DEPTH = 8
SELFPLAY_GAMES = 1000       # games per loop
SELFPLAY_THREADS = 16
SELFPLAY_LOOPS = 5
MAX_POSITIONS = 500_000

TRAIN_EPOCHS = 10
TRAIN_BATCH = 4096
TRAIN_LR = 0.001

EVAL_GAMES = 50  # quick eval vs random


def quick_eval(pt_path, depth=6, n_games=50):
    """Quick eval: NNUE vs random, return winrate %."""
    try:
        from nnue.agent import NNUEAgent
        from evaluation.nnue_evaluator import run_evaluation_suite
        agent = NNUEAgent(model_path=str(pt_path), depth=depth)
        metrics = run_evaluation_suite(agent, num_games=n_games, num_games_minimax=20)
        return metrics
    except Exception as e:
        print(f"  Eval failed: {e}")
        return {}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = RUN_DIR / "model.pt"
    nnue_path = RUN_DIR / "model.nnue"

    # ─── Step 1: Load seed data ───
    print(f"{'='*60}")
    print(f"Loading seed data: {SEED_DATA}")
    d = np.load(SEED_DATA)
    boards, values = d['boards'], d['values']
    print(f"  Seed: {len(values):,} positions")

    # ─── Step 2: Initial training ───
    print(f"\n{'='*60}")
    print(f"Initial training on {len(values):,} positions")
    print(f"{'='*60}")

    tcfg = TrainingConfig(
        batch_size=TRAIN_BATCH, learning_rate=TRAIN_LR,
        num_epochs=TRAIN_EPOCHS, num_workers=8,
    )

    model, val_loss, metrics = train_nnue(
        data=(boards, values), output_path=str(pt_path),
        train_config=tcfg, device='cuda',
    )
    model.eval()
    export_weights(model, str(nnue_path))
    print(f"  Initial val_loss: {val_loss:.6f}")

    # Quick eval
    ev = quick_eval(pt_path, n_games=EVAL_GAMES)
    rnd_wr = ev.get('eval/vs_random_winrate', 0)
    mm2_wr = ev.get('eval/vs_minimax2_winrate', 0)
    print(f"  vs_random: {rnd_wr:.0f}%  vs_minimax2: {mm2_wr:.0f}%")

    best_wr = rnd_wr

    # ─── Step 3: Selfplay loops ───
    print(f"\n{'='*60}")
    print(f"Selfplay loops: {SELFPLAY_LOOPS}x, {SELFPLAY_GAMES} games/loop, depth {SELFPLAY_DEPTH}")
    print(f"{'='*60}")

    for loop in range(SELFPLAY_LOOPS):
        loop_t0 = time.time()
        print(f"\n--- Loop {loop+1}/{SELFPLAY_LOOPS} ---")

        # Selfplay
        sp_model = nnue_cpp.NNUEModel()
        sp_model.load(str(nnue_path))

        sp_cfg = nnue_cpp.DataGenConfig()
        sp_cfg.search_depth = SELFPLAY_DEPTH
        sp_cfg.qsearch_mode = 0
        sp_cfg.tt_size_mb = 8
        sp_cfg.early_stop_empty = 15

        t0 = time.time()
        new_boards, new_values = nnue_cpp.generate_data(
            sp_model, sp_cfg,
            num_games=SELFPLAY_GAMES,
            num_threads=SELFPLAY_THREADS,
            seed=42 + loop * 10000,
        )
        sp_time = time.time() - t0
        sp_rate = SELFPLAY_GAMES / max(1, sp_time)
        print(f"  Selfplay: {len(new_values):,} pos from {SELFPLAY_GAMES} games "
              f"({sp_time:.1f}s, {sp_rate:.1f} g/s)")

        # Merge
        boards = np.concatenate([boards, new_boards])
        values = np.concatenate([values, new_values])
        if len(values) > MAX_POSITIONS:
            boards = boards[-MAX_POSITIONS:]
            values = values[-MAX_POSITIONS:]
        print(f"  Total data: {len(values):,} positions")

        # Retrain (warm start)
        model, val_loss, _ = train_nnue(
            data=(boards, values), output_path=str(pt_path),
            train_config=tcfg, device='cuda',
            warm_start_path=str(pt_path),
        )
        model.eval()
        export_weights(model, str(nnue_path))

        # Eval
        ev = quick_eval(pt_path, n_games=EVAL_GAMES)
        rnd_wr = ev.get('eval/vs_random_winrate', 0)
        mm2_wr = ev.get('eval/vs_minimax2_winrate', 0)
        loop_time = time.time() - loop_t0

        is_best = rnd_wr > best_wr
        if is_best:
            best_wr = rnd_wr
            shutil.copy2(str(pt_path), str(RUN_DIR / "best.pt"))
            shutil.copy2(str(nnue_path), str(RUN_DIR / "best.nnue"))

        print(f"  val_loss: {val_loss:.6f}  vs_random: {rnd_wr:.0f}%  "
              f"vs_mm2: {mm2_wr:.0f}%  time: {loop_time:.0f}s"
              f"{'  ★ BEST' if is_best else ''}")

    print(f"\n{'='*60}")
    print(f"Done! Best vs_random: {best_wr:.0f}%")
    print(f"Models: {RUN_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
