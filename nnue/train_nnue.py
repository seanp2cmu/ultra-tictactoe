#!/usr/bin/env python3
"""
NNUE Training Pipeline — Phase 1 + Phase 2 unified.

Phase 1: Generate initial data from AlphaZero teacher
Phase 2+: Train NNUE, then self-play loop with C++ engine

All settings controlled via PipelineConfig in nnue/config.py.
Run: python -m nnue.train_nnue
"""
import datetime
import shutil
import time
from pathlib import Path

import numpy as np
import wandb
from tqdm import tqdm

import nnue_cpp
from nnue.core.model import NNUE
from nnue.cpp.export_weights import export_weights
from nnue.training.trainer import train_nnue
from nnue.evaluation.evaluator import run_evaluation_suite
from nnue.agent import NNUEAgent
from nnue.config import PipelineConfig, TrainingConfig
from ai.evaluation.elo import EloTracker


ELO_CHECKPOINT_INTERVAL = 10
ELO_GAMES_PER_MATCHUP = 50


# ─── Paths (set per-run in run()) ─────────────────────────────────

MODEL_DIR = Path("nnue/model")
DATA_DIR = Path("nnue/data")
PT_PATH = MODEL_DIR / "nnue_model.pt"
NNUE_PATH = MODEL_DIR / "nnue_model.nnue"


# ─── Logging ──────────────────────────────────────────────────────

def _log(log_path, msg):
    """Append timestamped message to training log."""
    now = datetime.datetime.now().strftime('%H:%M:%S')
    with open(log_path, 'a') as f:
        f.write(f"[{now}] {msg}\n")


# ─── Phase 1: AlphaZero teacher datagen ──────────────────────────

def phase1_generate(cfg: PipelineConfig, log_path=None):
    """Generate initial NNUE training data using AlphaZero teacher."""
    from ai.core import AlphaZeroNet
    from nnue.data import NNUEDataGenerator
    from nnue.config import DataGenConfig

    output_path = DATA_DIR / "phase1_data.npz"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        tqdm.write(f"[Phase 1] Data exists: {output_path}")
        data = np.load(output_path)
        tqdm.write(f"  Positions: {len(data['values']):,}")
        return str(output_path)

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Phase 1: AlphaZero Teacher Data Generation")
    tqdm.write(f"{'='*60}")

    dg_config = DataGenConfig()
    dg_config.num_games = cfg.phase1_games
    dg_config.num_simulations = cfg.alphazero_sims
    dg_config.device = cfg.device
    dg_config.lambda_search = cfg.lambda_search

    tqdm.write(f"Loading AlphaZero from {cfg.alphazero_model}...")
    network = AlphaZeroNet(device=cfg.device)
    if Path(cfg.alphazero_model).exists():
        network.load(cfg.alphazero_model)
        tqdm.write("  Model loaded")
    else:
        tqdm.write(f"  WARNING: {cfg.alphazero_model} not found, using random weights")

    generator = NNUEDataGenerator(
        network=network,
        config=dg_config,
        num_simulations=cfg.alphazero_sims,
        seed=42,
    )

    tqdm.write(f"Generating {cfg.phase1_games} games (sims={cfg.alphazero_sims})...")
    t0 = time.time()
    dataset = generator.generate_dataset(
        num_games=cfg.phase1_games,
        output_path=str(output_path),
        verbose=True,
    )
    elapsed = time.time() - t0

    generator.print_stats()
    msg = f"Phase 1: {len(dataset['values']):,} positions in {elapsed:.0f}s"
    tqdm.write(msg)
    if log_path:
        _log(log_path, msg)

    return str(output_path)


# ─── Train NNUE ──────────────────────────────────────────────────

def train_model(data_path, cfg, train_cfg, log_path=None, global_step=0):
    """Train NNUE model. Returns (model, best_val_loss, epoch_metrics)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model, best_val_loss, epoch_metrics = train_nnue(
        data_path=data_path,
        output_path=str(PT_PATH),
        train_config=train_cfg,
        device=cfg.device,
    )

    # Export to .nnue
    model.eval()
    export_weights(model, str(NNUE_PATH))

    # Log epoch metrics to wandb
    for m in epoch_metrics:
        wandb.log(m, step=global_step + int(m['train/epoch']))

    if log_path:
        _log(log_path, f"Train: loss={epoch_metrics[-1]['train/loss']:.6f} "
             f"val={best_val_loss:.6f}")

    return model, best_val_loss, epoch_metrics


# ─── Evaluation ──────────────────────────────────────────────────

def run_eval(cfg, log_path=None):
    """Evaluate current NNUE model against baselines. Returns metrics dict."""
    agent = NNUEAgent(
        model_path=str(PT_PATH) if PT_PATH.exists() else None,
        depth=cfg.eval_depth,
    )
    eval_metrics = run_evaluation_suite(
        agent,
        num_games=cfg.eval_games,
        num_games_minimax=cfg.eval_games_minimax,
    )
    
    # Log to file
    if log_path:
        parts = []
        for k, v in eval_metrics.items():
            if 'winrate' in k:
                parts.append(f"{k.split('/')[-1]}={v:.1f}%")
        _log(log_path, f"Eval: {' | '.join(parts)}")

    return eval_metrics


# ─── C++ self-play datagen ────────────────────────────────────────

def selfplay_generate(cfg, loop_idx, log_path=None):
    """Generate training data via C++ NNUE self-play with tqdm progress."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"selfplay_{loop_idx:03d}.npz"

    model = nnue_cpp.NNUEModel()
    if NNUE_PATH.exists():
        model.load(str(NNUE_PATH))

    config = nnue_cpp.DataGenConfig()
    config.search_depth = cfg.selfplay_depth
    config.qsearch_mode = 2
    config.lambda_search = cfg.lambda_search
    config.random_move_count = 8
    config.write_minply = 4
    config.write_maxply = 60
    config.eval_limit = 0.9
    config.random_skip_rate = 0.3
    config.skip_noisy = True
    config.skip_noisy_maxply = 30

    # Generate in chunks for tqdm progress
    chunk_size = max(1000, cfg.selfplay_games // 20)
    remaining = cfg.selfplay_games
    seed = 42 + loop_idx * 100000
    all_boards, all_values = [], []

    t0 = time.time()
    pbar = tqdm(total=cfg.selfplay_games, desc=f"DataGen[{loop_idx}]",
                unit="g", ncols=90, leave=False)

    while remaining > 0:
        n = min(chunk_size, remaining)
        b, v = nnue_cpp.generate_data(
            model, config,
            num_games=n,
            num_threads=cfg.selfplay_threads,
            seed=seed,
        )
        all_boards.append(b)
        all_values.append(v)
        remaining -= n
        seed += n
        pbar.update(n)
        elapsed = time.time() - t0
        done = cfg.selfplay_games - remaining
        rate = done / elapsed if elapsed > 0 else 0
        pbar.set_postfix_str(f"{rate:.0f} g/s, {sum(len(v) for v in all_values):,} pos")

    pbar.close()
    elapsed = time.time() - t0

    boards = np.concatenate(all_boards)
    values = np.concatenate(all_values)
    np.savez_compressed(str(output_path), boards=boards, values=values)

    rate = cfg.selfplay_games / elapsed if elapsed > 0 else 0
    msg = (f"SelfPlay[{loop_idx}]: {len(values):,} pos from {cfg.selfplay_games} games "
           f"({elapsed:.1f}s, {rate:.1f} g/s)")
    tqdm.write(f"  {msg}")
    if log_path:
        _log(log_path, msg)

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

    merged_path = DATA_DIR / "merged.npz"
    np.savez_compressed(str(merged_path), boards=boards, values=values)
    tqdm.write(f"  Merged: {len(values):,} positions")
    return str(merged_path)


# ─── Main pipeline ───────────────────────────────────────────────

def run(cfg: PipelineConfig = None, train_cfg: TrainingConfig = None):
    """Run the full NNUE training pipeline."""
    cfg = cfg or PipelineConfig()
    train_cfg = train_cfg or TrainingConfig()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run name ──
    run_name = input("Run name (enter to skip): ").strip() or None

    # ── W&B init ──
    wb_run = wandb.init(
        project="uttt-nnue",
        name=run_name,
        config={
            **{f"pipeline/{k}": v for k, v in cfg.__dict__.items()},
            **{f"training/{k}": v for k, v in train_cfg.__dict__.items()},
        },
    )
    run_id = wb_run.id

    # ── Run directory ──
    run_dir = MODEL_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"

    _log(log_path, f"NNUE Training Pipeline — run {run_id}")
    _log(log_path, f"Pipeline: {cfg}")
    _log(log_path, f"Training: {train_cfg}")
    tqdm.write(f"\nRun: {run_id} | Log: {log_path}")

    data_paths = []
    global_step = 0
    best_winrate = 0.0

    # ── Elo tracker ──
    elo_tracker = EloTracker(
        save_path=str(run_dir / 'elo.json'),
        max_opponents=5,
        num_games=ELO_GAMES_PER_MATCHUP,
    )

    def _make_nnue_agent(ckpt_path):
        """Create NNUE agent from a .pt checkpoint for Elo evaluation."""
        return NNUEAgent(model_path=ckpt_path, depth=cfg.eval_depth)

    # ── Phase 1: AlphaZero teacher ──
    if not cfg.skip_phase1:
        p1_path = phase1_generate(cfg, log_path)
        data_paths.append(p1_path)

        if cfg.phase1_only:
            tqdm.write("Phase 1 complete.")
            wandb.finish()
            return

        _, _, ep_metrics = train_model(p1_path, cfg, train_cfg, log_path, global_step)
        global_step += len(ep_metrics)

        # Initial eval
        eval_metrics = run_eval(cfg, log_path)
        wandb.log(eval_metrics, step=global_step)
    else:
        if not PT_PATH.exists():
            tqdm.write(f"ERROR: No model at {PT_PATH}. Run Phase 1 first.")
            wandb.finish()
            return
        if not NNUE_PATH.exists():
            model = NNUE.load(str(PT_PATH))
            export_weights(model, str(NNUE_PATH))

    # ── Phase 2+: Self-play loop ──
    best_loop_loss = float('inf')
    no_improve_count = 0

    loop_pbar = tqdm(range(cfg.selfplay_loops), desc="Self-Play Loop", ncols=100)
    for loop in loop_pbar:
        loop_t0 = time.time()

        # Generate
        sp_path = selfplay_generate(cfg, loop_idx=loop, log_path=log_path)
        data_paths.append(sp_path)

        # Merge
        merged_path = merge_datasets(data_paths, max_positions=cfg.max_positions)

        # Train
        _, val_loss, ep_metrics = train_model(
            merged_path, cfg, train_cfg, log_path, global_step
        )
        global_step += len(ep_metrics)

        # Evaluate
        eval_metrics = run_eval(cfg, log_path)
        eval_metrics['loop/val_loss'] = val_loss
        eval_metrics['loop/loop_time_s'] = time.time() - loop_t0
        eval_metrics['loop/total_positions'] = sum(
            len(np.load(p)['values']) for p in data_paths if Path(p).exists()
        )

        # Save best model by winrate
        wr = eval_metrics.get('eval/vs_random_winrate', 0)
        if wr > best_winrate:
            best_winrate = wr
            shutil.copy2(str(PT_PATH), str(run_dir / "best.pt"))
            shutil.copy2(str(NNUE_PATH), str(run_dir / "best.nnue"))

        # Elo checkpoint evaluation every N loops
        if (loop + 1) % ELO_CHECKPOINT_INTERVAL == 0:
            ckpt_name = f'loop_{loop+1:03d}'
            ckpt_pt = run_dir / f'{ckpt_name}.pt'
            ckpt_nnue = run_dir / f'{ckpt_name}.nnue'
            shutil.copy2(str(PT_PATH), str(ckpt_pt))
            shutil.copy2(str(NNUE_PATH), str(ckpt_nnue))

            elo_tracker.register_checkpoint(ckpt_name, str(ckpt_pt), iteration=loop)
            try:
                elo_metrics = elo_tracker.evaluate_latest(_make_nnue_agent)
                eval_metrics.update(elo_metrics)
                _log(log_path, f"Elo: {ckpt_name} = {elo_metrics.get('elo/current', 0):.0f} "
                     f"(delta={elo_metrics.get('elo/delta', 0):+.0f})")
                tqdm.write(f"  Elo: {ckpt_name} = {elo_metrics.get('elo/current', 0):.0f}")
            except Exception as e:
                tqdm.write(f"  Elo eval failed: {e}")

        # Log all metrics (including Elo if evaluated this loop)
        wandb.log(eval_metrics, step=global_step)

        # Early stopping
        improved = ""
        if val_loss < best_loop_loss:
            best_loop_loss = val_loss
            no_improve_count = 0
            improved = " ★"
        else:
            no_improve_count += 1

        loop_pbar.set_postfix_str(
            f"val={val_loss:.5f} wr={wr:.0f}%{improved} "
            f"({no_improve_count}/{cfg.early_stop_patience})"
        )

        if cfg.early_stop_patience > 0 and no_improve_count >= cfg.early_stop_patience:
            _log(log_path, f"Early stop at loop {loop+1}")
            tqdm.write(f"\n  Early stopping after {cfg.early_stop_patience} loops without improvement.")
            break

    loop_pbar.close()

    # ── Final save ──
    shutil.copy2(str(PT_PATH), str(run_dir / "latest.pt"))
    shutil.copy2(str(NNUE_PATH), str(run_dir / "latest.nnue"))

    _log(log_path, f"Complete. Best val_loss={best_loop_loss:.6f} best_wr={best_winrate:.1f}%")

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Pipeline complete! Run: {run_id}")
    tqdm.write(f"  Best val_loss: {best_loop_loss:.6f}")
    tqdm.write(f"  Best vs_random: {best_winrate:.1f}%")
    tqdm.write(f"  Models: {run_dir}")
    tqdm.write(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    run()
