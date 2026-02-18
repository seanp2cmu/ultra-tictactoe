import os
import sys
import time
import subprocess
import torch
import torch._inductor.config
import wandb

torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

from tqdm import tqdm
from ai.config import Config
from ai.training import Trainer
from ai.utils import (
    RUNS_FILE, select_run_and_checkpoint, register_run,
    update_run_iteration,
    log_iteration_to_file, collect_wandb_metrics,
    run_and_log_eval, log_training_complete
)
from utils import upload_to_hf


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_extensions_built():
    """Auto-build Cython/C++ extensions if .so files are missing."""
    required_so = [
        'game/board_cy',
        'ai/mcts/node_cy',
        'utils/_board_symmetry_cy',
        'utils/_board_encoder_cy',
        'uttt_cpp',
        'nnue_cpp',
    ]
    missing = []
    for mod in required_so:
        parts = mod.rsplit('/', 1)
        if len(parts) == 2:
            search_dir = os.path.join(ROOT_DIR, parts[0])
            prefix = parts[1]
        else:
            search_dir = ROOT_DIR
            prefix = parts[0]
        found = any(
            f.startswith(prefix) and f.endswith('.so')
            for f in os.listdir(search_dir)
        ) if os.path.isdir(search_dir) else False
        if not found:
            missing.append(mod)

    if not missing:
        return

    print(f"[Setup] Missing extensions: {missing}")

    # Determine which setup scripts to run
    cython_mods = {'game/board_cy', 'ai/mcts/node_cy', 'utils/_board_symmetry_cy', 'utils/_board_encoder_cy'}
    cpp_mods = {'uttt_cpp', 'nnue_cpp'}
    need_cython = bool(set(missing) & cython_mods)
    need_cpp = bool(set(missing) & cpp_mods)

    if need_cython:
        print("[Setup] Building Cython extensions (setup.py) ...")
        ret = subprocess.run(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=ROOT_DIR
        )
        if ret.returncode != 0:
            print("[Setup] WARNING: Cython build failed")

    if need_cpp:
        print("[Setup] Building C++ extensions (setup_cpp.py) ...")
        ret = subprocess.run(
            [sys.executable, 'setup_cpp.py', 'build_ext', '--inplace'],
            cwd=ROOT_DIR
        )
        if ret.returncode != 0:
            print("[Setup] WARNING: C++ build failed")


def _download_from_hf(base_dir: str):
    """Download runs.json and latest checkpoints from HuggingFace if not present locally."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        repo_id = os.environ.get('HF_REPO_ID', 'sean2474/ultra-tictactoe-models')
        api = HfApi()

        # Download runs.json
        runs_path = os.path.join(base_dir, RUNS_FILE)
        if not os.path.exists(runs_path):
            try:
                p = hf_hub_download(repo_id, RUNS_FILE, repo_type='model',
                                    local_dir=base_dir, local_dir_use_symlinks=False)
                print(f"[HF] Downloaded {RUNS_FILE}")
            except Exception:
                print(f"[HF] No {RUNS_FILE} on remote")

        # Download latest.pt for each run listed in runs.json
        if os.path.exists(runs_path):
            import json
            with open(runs_path) as f:
                runs = json.load(f)
            for run_id in runs:
                run_dir = os.path.join(base_dir, run_id)
                latest_local = os.path.join(run_dir, 'latest.pt')
                if not os.path.exists(latest_local):
                    remote_path = f'{run_id}/latest.pt'
                    try:
                        os.makedirs(run_dir, exist_ok=True)
                        hf_hub_download(repo_id, remote_path, repo_type='model',
                                        local_dir=base_dir, local_dir_use_symlinks=False)
                        print(f"[HF] Downloaded {remote_path}")
                    except Exception:
                        pass
    except Exception as e:
        print(f"[HF] Download failed: {e}")


def main():
    # 0. Auto-build extensions if needed
    _ensure_extensions_built()

    config = Config()
    base_dir = config.training.save_dir
    os.makedirs(base_dir, exist_ok=True)

    # 0.5. Download checkpoints from HF if not present locally
    _download_from_hf(base_dir)
    
    # 1. Select or create run
    run_id, run_name, checkpoint_path, is_new_run = select_run_and_checkpoint(base_dir)
    
    # 2. Device
    device = "cuda" if torch.cuda.is_available() else "cpu" if config.gpu.device == "auto" else config.gpu.device
    
    # 3. W&B init
    wandb.init(
        entity="seanp2-carnegie-mellon-university",
        project="ultra-tictactoe",
        name=run_name, id=run_id, resume='allow',
        config={
            'num_res_blocks': config.network.num_res_blocks,
            'num_channels': config.network.num_channels,
            'num_iterations': config.training.num_iterations,
            'num_self_play_games': config.training.num_self_play_games,
            'num_train_epochs': config.training.num_train_epochs,
            'num_simulations': config.training.num_simulations,
            'batch_size': config.training.batch_size,
            'lr': config.training.lr,
            'weight_decay': config.training.weight_decay,
            'replay_buffer_size': config.training.replay_buffer_size,
            'parallel_games': config.gpu.parallel_games,
            'use_amp': config.training.use_amp,
        },
    )
    if is_new_run:
        run_id = wandb.run.id
    print(f"W&B run: {wandb.run.url}")
    
    # 4. Run directory + mapping
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    register_run(base_dir, run_id, run_name)
    upload_to_hf(os.path.join(base_dir, RUNS_FILE), RUNS_FILE)
    
    print("=" * 80)
    print(f"Run: {run_name} ({run_id}) | Dir: {run_dir}")
    print(f"Network: {config.network.num_res_blocks}b/{config.network.num_channels}ch | Device: {device}")
    print(f"Training: {config.training.num_iterations} iters x {config.training.num_self_play_games} games x {config.training.num_simulations} sims")
    print("=" * 80)
    
    # 5. Create trainer
    trainer = Trainer(
        network=None, lr=config.training.lr, weight_decay=config.training.weight_decay,
        batch_size=config.training.batch_size, num_simulations=config.training.num_simulations,
        replay_buffer_size=config.training.replay_buffer_size, device=device,
        use_amp=config.training.use_amp, num_res_blocks=config.network.num_res_blocks,
        num_channels=config.network.num_channels, hot_cache_size=config.dtw.hot_cache_size,
        cold_cache_size=config.dtw.cold_cache_size, total_iterations=config.training.num_iterations,
        inference_batch_size=config.gpu.inference_batch_size,
    )
    
    # 6. Load checkpoint if resuming
    start_iteration = 0
    if checkpoint_path:
        loaded_iter = trainer.load(checkpoint_path)
        if loaded_iter is not None:
            start_iteration = loaded_iter + 1
        print(f"\u2713 Resuming from iteration {start_iteration}")
    
    # Load shared DTW cache
    dtw_cache_path = os.path.join(base_dir, 'dtw_cache.pkl')
    if os.path.exists(dtw_cache_path) and trainer.dtw_calculator and trainer.dtw_calculator.tt:
        trainer.dtw_calculator.tt.load_from_file(dtw_cache_path)
    
    # 7. Training loop
    log_path = os.path.join(run_dir, 'training.log')
    total_iters = config.training.num_iterations
    best_winrate = 0.0
    print("Starting training...\n")
    
    pbar = tqdm(range(start_iteration, total_iters), desc="Training", ncols=100,
                initial=start_iteration, total=total_iters, leave=True, position=0)
    
    for iteration in pbar:
        t0 = time.time()
        
        result = trainer.train_iteration(
            num_self_play_games=config.training.num_self_play_games,
            num_train_epochs=config.training.num_train_epochs,
            temperature=1.0, verbose=False, disable_tqdm=False,
            num_simulations=config.training.num_simulations,
            parallel_games=config.gpu.parallel_games, iteration=iteration,
            num_workers=config.gpu.num_workers,
        )
        
        elapsed = time.time() - t0
        loss = result['avg_loss']
        buf = trainer.replay_buffer.get_stats() if hasattr(trainer.replay_buffer, 'get_stats') else {}
        lr = result.get('learning_rate', 0)
        dtw_stats = trainer.dtw_calculator.get_stats() if trainer.dtw_calculator else None
        
        timing = getattr(trainer, '_mp_timing', {})
        
        # File log + HF
        log_iteration_to_file(log_path, iteration, total_iters, loss, lr,
                              result['num_samples'], buf, timing, elapsed, config, dtw_stats)
        upload_to_hf(log_path, f'{run_id}/training.log')
        
        # W&B metrics + eval
        wm = collect_wandb_metrics(loss, lr, buf, elapsed, dtw_stats)
        run_and_log_eval(log_path, trainer.network, trainer.dtw_calculator, wm)
        wandb.log(wm, step=iteration)
        
        if dtw_stats:
            trainer.dtw_calculator.reset_search_stats()
        
        # Save latest
        p = os.path.join(run_dir, 'latest.pt')
        trainer.save(p, iteration=iteration)
        upload_to_hf(p, f'{run_id}/latest.pt')
        
        # Save best.pt if this is the best eval so far
        current_wr = wm.get('eval/vs_random_winrate', 0)
        if current_wr > best_winrate:
            best_winrate = current_wr
            p = os.path.join(run_dir, 'best.pt')
            trainer.save(p, iteration=iteration)
            upload_to_hf(p, f'{run_id}/best.pt')
        
        # Shared DTW cache
        if trainer.dtw_calculator and trainer.dtw_calculator.tt:
            trainer.dtw_calculator.tt.save_to_file(dtw_cache_path)
            if (iteration + 1) % 10 == 0:
                upload_to_hf(dtw_cache_path, 'dtw_cache.pkl')
        
        update_run_iteration(base_dir, run_id, iteration + 1)
        upload_to_hf(os.path.join(base_dir, RUNS_FILE), RUNS_FILE)
    
    # 8. Finish
    log_training_complete(log_path)
    upload_to_hf(log_path, f'{run_id}/training.log')
    
    if trainer.dtw_calculator and trainer.dtw_calculator.tt:
        trainer.dtw_calculator.tt.save_to_file(dtw_cache_path)
        upload_to_hf(dtw_cache_path, 'dtw_cache.pkl')
    
    upload_to_hf(os.path.join(base_dir, RUNS_FILE), RUNS_FILE)
    wandb.finish()
    print(f"\n\u2713 Training completed!")


if __name__ == '__main__':
    main()
