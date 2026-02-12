import os
import time
import torch._inductor.config
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

from tqdm import tqdm
from ai.config import Config
from ai.training import Trainer
from utils import (
    upload_to_hf,
    get_start_iteration
)


def main():
    config = Config()
    
    checkpoint_path, start_iteration = get_start_iteration(config.training.save_dir) 
    
    if config.gpu.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.gpu.device
    
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("AlphaZero Training with DTW + Compressed Transposition Table")
    print("=" * 80)
    print(f"Network:")
    print(f"  Residual Blocks: {config.network.num_res_blocks}")
    print(f"  Channels: {config.network.num_channels}")
    print(f"\nTraining:")
    print(f"  Iterations: {config.training.num_iterations}")
    print(f"  Games per iteration: {config.training.num_self_play_games}")
    print(f"  Training epochs: {config.training.num_train_epochs}")
    print(f"  MCTS simulations: 200 → {config.training.num_simulations}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.lr}")
    print(f"\nDTW (Always Enabled):")
    print(f"  Endgame threshold: {config.dtw.endgame_threshold} cells")
    print(f"  Cache size: {config.dtw.hot_cache_size + config.dtw.cold_cache_size:,} entries")
    print(f"\nGPU:")
    print(f"  Device: {device}")
    print(f"  AMP: {config.training.use_amp}")
    print(f"  Parallel Games: {config.gpu.parallel_games}")
    print("=" * 80)
    
    trainer = Trainer(
        network=None,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        batch_size=config.training.batch_size,
        num_simulations=config.training.num_simulations,
        replay_buffer_size=config.training.replay_buffer_size,
        device=device,
        use_amp=config.training.use_amp,
        num_res_blocks=config.network.num_res_blocks,
        num_channels=config.network.num_channels,
        hot_cache_size=config.dtw.hot_cache_size,
        cold_cache_size=config.dtw.cold_cache_size,
        total_iterations=config.training.num_iterations
    )
    
    best_loss = float('inf')
    if checkpoint_path:
        print(f"\n  Loading checkpoint: {checkpoint_path}")
        loaded_iter = trainer.load(checkpoint_path)
        if loaded_iter is not None:
            start_iteration = loaded_iter + 1
        print(f"✓ Model loaded (will resume from iteration {start_iteration})")
        
        # Load DTW cache if exists
        dtw_cache_path = os.path.join(config.training.save_dir, 'dtw_cache.pkl')
        if os.path.exists(dtw_cache_path) and trainer.dtw_calculator and trainer.dtw_calculator.tt:
            trainer.dtw_calculator.tt.load_from_file(dtw_cache_path)
        print()
    
    # 고정 값 사용 (Lc0 style)
    temp = 1.0  # 첫 8수만 적용됨 (self_play.py에서 처리)
    num_sims = config.training.num_simulations  # 800 고정
    num_games = config.training.num_self_play_games  # 2048 고정
    
    print("Starting training...")
    print("=" * 80 + "\n")
    
    for iteration in tqdm(range(start_iteration, config.training.num_iterations), desc="Training", ncols=100, initial=start_iteration, total=config.training.num_iterations, leave=True, position=0):
        iter_start_time = time.time()
        
        result = trainer.train_iteration(
            num_self_play_games=num_games,
            num_train_epochs=config.training.num_train_epochs,
            temperature=temp,
            verbose=False,
            disable_tqdm=False,
            num_simulations=num_sims,
            parallel_games=config.gpu.parallel_games,
            iteration=iteration
        )
        
        # Compact output - single line per iteration
        iter_elapsed = time.time() - iter_start_time
        loss = result['avg_loss']['total_loss']
        samples = result['num_samples']
        buffer_stats = trainer.replay_buffer.get_stats() if hasattr(trainer.replay_buffer, 'get_stats') else {}
        buffer_total = buffer_stats.get('total', 0)
        
        tqdm.write(f"[{iteration+1:03d}] Loss: {loss:.4f} | Samples: {samples:,} | Buffer: {buffer_total:,} | Time: {iter_elapsed:.0f}s")
        
        if 'dtw_stats' in result:
            trainer.dtw_calculator.reset_search_stats()
        
        current_loss = result['avg_loss']['total_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            save_path = os.path.join(config.training.save_dir, 'best.pt')
            trainer.save(save_path, iteration=iteration)
            tqdm.write(f"  ✓ New best! (loss: {best_loss:.4f})")
            upload_to_hf(save_path, 'best.pt')
        
        if (iteration + 1) % 5 == 0:
            ckpt_path = os.path.join(config.training.save_dir, f'checkpoint_{iteration + 1}.pt')
            trainer.save(ckpt_path, iteration=iteration)
            tqdm.write(f"  ✓ Checkpoint: checkpoint_{iteration + 1}.pt")
            
            old_iter = iteration + 1 - 10
            if old_iter > 0:
                old_path = os.path.join(config.training.save_dir, f'checkpoint_{old_iter}.pt')
                if os.path.exists(old_path):
                    os.remove(old_path)
        
        if (iteration + 1) % 20 == 0:
            model_path = os.path.join(config.training.save_dir, f'model_{iteration + 1}.pt')
            trainer.save(model_path, iteration=iteration)
            tqdm.write(f"  ✓ Model: model_{iteration + 1}.pt → HF")
            upload_to_hf(model_path, f'model_{iteration + 1}.pt')
        
        if trainer.dtw_calculator and trainer.dtw_calculator.tt:
            dtw_cache_path = os.path.join(config.training.save_dir, 'dtw_cache.pkl')
            trainer.dtw_calculator.tt.save_to_file(dtw_cache_path)
            if (iteration + 1) % 10 == 0:
                upload_to_hf(dtw_cache_path, 'dtw_cache.pkl')
    
    print(f"\n✓ Training completed! Best loss: {best_loss:.4f}")
    
    if trainer.dtw_calculator and trainer.dtw_calculator.tt:
        dtw_cache_path = os.path.join(config.training.save_dir, 'dtw_cache.pkl')
        trainer.dtw_calculator.tt.save_to_file(dtw_cache_path)


if __name__ == '__main__':
    main()
