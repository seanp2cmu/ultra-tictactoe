#!/usr/bin/env python3
"""
NNUE Training Pipeline (in-memory).

All data lives in numpy arrays — no intermediate .npz files.
A single checkpoint.npz is saved only on exit/crash for resumability.

Step 1: AZ teacher datagen (GPU, multi-worker MCTS) → (boards, values)
Step 2: Initial NNUE training on AZ data
Step 3: C++ NNUE selfplay loop — generate new data → merge → retrain

Run: python -m nnue.train
"""
import datetime
import math
import os
import shutil
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import wandb
from tqdm import tqdm

import nnue_cpp
from nnue.core.model import NNUE
from nnue.cpp.export_weights import export_weights
from nnue.training.trainer import train_nnue
from evaluation.nnue_evaluator import run_evaluation_suite
from nnue.agent import NNUEAgent
from nnue.config import NNUEConfig, TrainingConfig, PipelineConfig


# ─── Logging ──────────────────────────────────────────────────────

def _log(log_path, msg):
    now = datetime.datetime.now().strftime('%H:%M:%S')
    with open(log_path, 'a') as f:
        f.write(f"[{now}] {msg}\n")


# ─── GPU Producer: AZ datagen ────────────────────────────────────

def _az_states_to_nnue_boards(states: np.ndarray) -> np.ndarray:
    """Convert AZ canonical 7×9×9 tensors to NNUE 92-dim int8 boards (vectorized).

    AZ tensor channels:
      0: current player pieces, 1: opponent pieces, 2: empty,
      3: current player won sub-boards, 4: opponent won sub-boards,
      5: legal moves, 6: last move

    NNUE board (92,):
      [0:81]  = cell values (0=empty, 1=current, 2=opponent)
      [81:90] = meta-board (sub-board winners, same convention)
      [90]    = active sub-board (-1 if any)
      [91]    = current player (always 1 in canonical form)
    """
    n = len(states)
    boards = np.zeros((n, 92), dtype=np.int8)

    # Cells: ch0 = current player (→1), ch1 = opponent (→2)
    flat0 = states[:, 0].reshape(n, 81)  # (N, 81) float32
    flat1 = states[:, 1].reshape(n, 81)
    boards[:, :81] = (flat0 > 0.5).astype(np.int8) + 2 * (flat1 > 0.5).astype(np.int8)

    # Meta-board: ch3 = current won (→1), ch4 = opponent won (→2)
    # Each won sub-board is a 3×3 block of 1.0 in the 9×9 grid
    # Sample one cell per sub-board (top-left corner)
    for br in range(3):
        for bc in range(3):
            idx = br * 3 + bc
            sr, sc = br * 3, bc * 3
            boards[:, 81 + idx] = (
                (states[:, 3, sr, sc] > 0.5).astype(np.int8) +
                2 * (states[:, 4, sr, sc] > 0.5).astype(np.int8)
            )

    # Active sub-board from last move (ch6)
    last_move_flat = states[:, 6].reshape(n, 81)  # (N, 81)
    last_move_idx = np.argmax(last_move_flat, axis=1)  # (N,)
    has_last_move = last_move_flat.max(axis=1) > 0.5  # (N,)

    last_r = last_move_idx // 9
    last_c = last_move_idx % 9
    target_sub = (last_r % 3) * 3 + (last_c % 3)  # (N,)

    # If target sub-board is completed → any board (-1)
    target_meta = boards[np.arange(n), 81 + target_sub]
    active = np.where(has_last_move & (target_meta == 0), target_sub, -1).astype(np.int8)
    boards[:, 90] = active

    # Current player: always 1 in canonical form
    boards[:, 91] = 1

    return boards


def az_produce(cfg: PipelineConfig, log_path: str, data_state: dict) -> tuple:
    """Generate AZ self-play data in batches using multi-worker parallel MCTS.

    Processes az_batch_games at a time. After each batch, results are
    appended to data_state['boards'] / data_state['values'] so that
    atexit can save partial progress if interrupted.

    Returns:
        (boards, values) numpy arrays.
    """
    from ai.core import AlphaZeroNet
    from ai.training.self_play import run_multiprocess_self_play

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"GPU Producer: AZ Datagen ({cfg.az_total_games} games, {cfg.alphazero_sims} sims)")
    tqdm.write(f"{'='*60}")

    network = AlphaZeroNet(device=cfg.device)
    if Path(cfg.alphazero_model).exists():
        network.load(cfg.alphazero_model)
        tqdm.write(f"  AZ model loaded: {cfg.alphazero_model}")
    else:
        tqdm.write(f"  WARNING: {cfg.alphazero_model} not found")

    board_chunks = []
    value_chunks = []
    games_done = 0
    gid = 0
    t_total = time.time()

    while games_done < cfg.az_total_games:
        batch_games = min(cfg.az_batch_games, cfg.az_total_games - games_done)

        t0 = time.time()
        (states, policies, values, game_ids), timing = run_multiprocess_self_play(
            network=network,
            num_games=batch_games,
            num_simulations=cfg.alphazero_sims,
            parallel_games=2048,
            temperature=1.0,
            game_id_start=gid,
            num_workers=4,
        )
        elapsed = time.time() - t0

        # Convert AZ tensors → NNUE boards
        batch_boards = _az_states_to_nnue_boards(states)
        board_chunks.append(batch_boards)
        value_chunks.append(values)

        games_done += batch_games
        gid += batch_games

        # Update shared state so atexit can save partial progress
        all_boards = np.concatenate(board_chunks)
        all_values = np.concatenate(value_chunks)
        data_state['boards'] = all_boards
        data_state['values'] = all_values

        msg = (f"AZ batch: {len(values):,} pos from {batch_games} games "
               f"({elapsed:.0f}s, {batch_games/max(1,elapsed):.1f} g/s) | "
               f"Total: {games_done}/{cfg.az_total_games} ({len(all_values):,} pos)")
        tqdm.write(f"  {msg}")
        _log(log_path, msg)

    # Cleanup TRT
    try:
        network.trt_engine.shutdown()
    except Exception:
        pass

    total_elapsed = time.time() - t_total
    tqdm.write(f"  AZ done: {len(all_values):,} pos, {total_elapsed:.0f}s total")
    return all_boards, all_values


# ─── Checkpoint save ─────────────────────────────────────────────

def _save_checkpoint(boards, values, path):
    """Save data checkpoint (called on exit or between phases)."""
    np.savez_compressed(str(path), boards=boards, values=values)
    tqdm.write(f"  Checkpoint saved: {path} ({len(values):,} positions)")


def _load_checkpoint(path):
    """Load data checkpoint if it exists. Returns (boards, values) or None."""
    p = Path(path)
    if p.exists():
        data = np.load(str(p))
        boards, values = data['boards'], data['values']
        tqdm.write(f"  Checkpoint loaded: {p} ({len(values):,} positions)")
        return boards, values
    return None


# ─── Main pipeline ───────────────────────────────────────────────

def run(cfg: PipelineConfig = None, train_cfg: TrainingConfig = None):
    """Run concurrent NNUE training pipeline (fully in-memory).

    Data lives in numpy arrays throughout. A checkpoint .npz is saved
    only on normal exit or crash (via atexit) for resumability.
    """
    import atexit, signal

    cfg = cfg or PipelineConfig()
    train_cfg = train_cfg or TrainingConfig()
    train_cfg.num_epochs = cfg.train_epochs
    train_cfg.batch_size = cfg.train_batch_size
    train_cfg.learning_rate = cfg.learning_rate

    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pt_path = model_dir / "nnue_model.pt"
    nnue_path = model_dir / "nnue_model.nnue"

    # ── Run name ──
    run_name = input("Run name (enter to skip): ").strip() or None

    wb_run = wandb.init(
        project="uttt-nnue",
        name=run_name,
        config={
            **{f"concurrent/{k}": v for k, v in cfg.__dict__.items()},
            **{f"training/{k}": v for k, v in train_cfg.__dict__.items()},
        },
    )
    run_id = wb_run.id

    run_dir = model_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.npz"
    log_path = str(run_dir / "training.log")

    _log(log_path, f"Concurrent Pipeline — run {run_id}")
    _log(log_path, f"Config: {cfg}")
    tqdm.write(f"\nRun: {run_id} | Log: {log_path}")

    # ── In-memory data state (shared dict so atexit can always access) ──
    data_state = {'boards': None, 'values': None}

    def _save_on_exit():
        if data_state['boards'] is not None:
            _save_checkpoint(data_state['boards'], data_state['values'], ckpt_path)

    atexit.register(_save_on_exit)
    signal.signal(signal.SIGTERM, lambda *_: (_save_on_exit(), exit(1)))

    global_step = 0
    best_winrate = 0.0

    # ════════════════════════════════════════════════════════════════
    # Step 1: AZ Datagen (GPU) — generate positions in batches
    # ════════════════════════════════════════════════════════════════
    loaded = _load_checkpoint(ckpt_path)
    if loaded is not None:
        data_state['boards'], data_state['values'] = loaded
        tqdm.write(f"  Resumed from run checkpoint: {ckpt_path} ({len(loaded[1]):,} pos)")
    elif cfg.seed_checkpoint and Path(cfg.seed_checkpoint).exists():
        loaded = _load_checkpoint(Path(cfg.seed_checkpoint))
        if loaded is not None:
            data_state['boards'], data_state['values'] = loaded
            tqdm.write(f"  Loaded seed checkpoint: {cfg.seed_checkpoint} ({len(loaded[1]):,} pos)")
        else:
            az_produce(cfg, log_path, data_state)
    else:
        az_produce(cfg, log_path, data_state)

    boards, values = data_state['boards'], data_state['values']

    # Trim to max_positions
    if len(values) > cfg.max_positions:
        boards = boards[-cfg.max_positions:]
        values = values[-cfg.max_positions:]
        data_state['boards'], data_state['values'] = boards, values

    # ════════════════════════════════════════════════════════════════
    # Step 2: Initial NNUE training on AZ data
    # ════════════════════════════════════════════════════════════════
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Step 2: Initial NNUE Training ({len(values):,} positions)")
    tqdm.write(f"{'='*60}")

    model, best_val, ep_metrics = train_nnue(
        data=(boards, values),
        output_path=str(pt_path),
        train_config=train_cfg,
        device=cfg.device,
    )
    global_step += len(ep_metrics)

    model.eval()
    export_weights(model, str(nnue_path))

    for m in ep_metrics:
        wandb.log(m, step=global_step - len(ep_metrics) + int(m['train/epoch']))

    # Initial eval
    eval_metrics = _run_eval(cfg, pt_path)
    eval_metrics['step/phase'] = 0
    wandb.log(eval_metrics, step=global_step)
    _log(log_path, f"Initial: val={best_val:.6f} "
         f"vs_mm2={eval_metrics.get('eval/vs_minimax2_winrate', 0):.0f}%")

    wr = eval_metrics.get('eval/vs_random_winrate', 0)
    if wr > best_winrate:
        best_winrate = wr
        shutil.copy2(str(pt_path), str(run_dir / "best.pt"))
        shutil.copy2(str(nnue_path), str(run_dir / "best.nnue"))

    # ════════════════════════════════════════════════════════════════
    # Step 3: C++ NNUE selfplay loops — generate new data → retrain
    # ════════════════════════════════════════════════════════════════
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Step 3: NNUE SelfPlay Loops ({cfg.selfplay_loops} iterations, "
               f"{cfg.selfplay_games} games/loop, depth={cfg.selfplay_depth})")
    tqdm.write(f"{'='*60}")

    pbar = tqdm(range(cfg.selfplay_loops), desc="SelfPlay Loop",
                ncols=100, leave=True)

    for loop in pbar:
        loop_t0 = time.time()

        tqdm.write(f"\n  --- SelfPlay Loop {loop+1}/{cfg.selfplay_loops} ---")

        # ── Generate new data with current NNUE ──
        try:
            sp_model = nnue_cpp.NNUEModel()
            sp_model.load(str(nnue_path))

            sp_config = nnue_cpp.DataGenConfig()
            sp_config.search_depth = cfg.selfplay_depth
            sp_config.lambda_search = cfg.selfplay_lambda
            sp_config.early_stop_empty = 15

            t0 = time.time()
            new_boards, new_values = nnue_cpp.generate_data(
                sp_model, sp_config,
                num_games=cfg.selfplay_games,
                num_threads=cfg.selfplay_threads,
                seed=42 + loop * 10000,
            )
            sp_elapsed = time.time() - t0
            sp_rate = cfg.selfplay_games / max(1, sp_elapsed)

            tqdm.write(f"  SelfPlay: {len(new_values):,} pos from "
                       f"{cfg.selfplay_games} games ({sp_elapsed:.1f}s, {sp_rate:.1f} g/s)")
            _log(log_path, f"SelfPlay[{loop}]: {len(new_values):,} pos from "
                 f"{cfg.selfplay_games} games ({sp_elapsed:.0f}s, {sp_rate:.1f} g/s)")

            # Merge with existing data
            boards = np.concatenate([boards, new_boards])
            values = np.concatenate([values, new_values])

            # Trim to max_positions (keep newest)
            if len(values) > cfg.max_positions:
                boards = boards[-cfg.max_positions:]
                values = values[-cfg.max_positions:]

            data_state['boards'], data_state['values'] = boards, values

        except Exception as e:
            tqdm.write(f"  SelfPlay failed: {e}")
            _log(log_path, f"SelfPlay[{loop}] FAILED: {e}")
            continue

        # ── Retrain on merged data (warm-start from previous) ──
        model, val_loss, ep_metrics = train_nnue(
            data=(boards, values),
            output_path=str(pt_path),
            train_config=train_cfg,
            device=cfg.device,
            warm_start_path=str(pt_path),
        )
        global_step += len(ep_metrics)

        model.eval()
        export_weights(model, str(nnue_path))

        for m in ep_metrics:
            wandb.log(m, step=global_step - len(ep_metrics) + int(m['train/epoch']))

        # ── Evaluate ──
        eval_metrics = _run_eval(cfg, pt_path)
        eval_metrics['step/phase'] = loop + 1
        eval_metrics['step/val_loss'] = val_loss
        eval_metrics['step/selfplay_pos'] = len(new_values)
        eval_metrics['step/total_pos'] = len(values)
        eval_metrics['step/loop_time_s'] = time.time() - loop_t0
        wandb.log(eval_metrics, step=global_step)

        wr = eval_metrics.get('eval/vs_random_winrate', 0)
        mm2_wr = eval_metrics.get('eval/vs_minimax2_winrate', 0)

        if wr > best_winrate:
            best_winrate = wr
            shutil.copy2(str(pt_path), str(run_dir / "best.pt"))
            shutil.copy2(str(nnue_path), str(run_dir / "best.nnue"))

        pbar.set_postfix_str(
            f"val={val_loss:.5f} mm2={mm2_wr:.0f}% rnd={wr:.0f}% "
            f"pos={len(values):,}")

        _log(log_path, f"Loop[{loop}]: val={val_loss:.6f} "
             f"vs_mm2={mm2_wr:.0f}% vs_rnd={wr:.0f}% pos={len(values):,}")

    pbar.close()

    # ── Final save ──
    _save_checkpoint(boards, values, ckpt_path)
    shutil.copy2(str(pt_path), str(run_dir / "latest.pt"))
    shutil.copy2(str(nnue_path), str(run_dir / "latest.nnue"))

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Pipeline complete! Run: {run_id}")
    tqdm.write(f"  Best vs_random: {best_winrate:.1f}%")
    tqdm.write(f"  Models: {run_dir}")
    tqdm.write(f"{'='*60}")

    _log(log_path, f"Complete. best_wr={best_winrate:.1f}%")
    atexit.unregister(_save_on_exit)
    wandb.finish()


def _run_eval(cfg, pt_path):
    """Run evaluation suite."""
    agent = NNUEAgent(
        model_path=str(pt_path),
        depth=cfg.eval_depth,
    )
    return run_evaluation_suite(
        agent,
        num_games=cfg.eval_games,
        num_games_minimax=cfg.eval_games_minimax,
    )


if __name__ == "__main__":
    run()
