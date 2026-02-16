import os
import time
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


def main():
    config = Config()
    base_dir = config.training.save_dir
    os.makedirs(base_dir, exist_ok=True)
    
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
        cold_cache_size=config.dtw.cold_cache_size, total_iterations=config.training.num_iterations
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
            parallel_games=config.gpu.parallel_games, iteration=iteration
        )
        
        elapsed = time.time() - t0
        loss = result['avg_loss']
        buf = trainer.replay_buffer.get_stats() if hasattr(trainer.replay_buffer, 'get_stats') else {}
        lr = result.get('learning_rate', 0)
        dtw_stats = trainer.dtw_calculator.get_stats() if trainer.dtw_calculator else None
        
        from ai.training.self_play import get_parallel_timing
        timing = get_parallel_timing()
        
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
