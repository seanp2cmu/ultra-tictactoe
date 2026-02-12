import os
import time
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
    print(f"  Games per iteration: {config.training.num_self_play_games} (adaptive)")
    print(f"  Training epochs: {config.training.num_train_epochs}")
    print(f"  MCTS simulations: 200 → {config.training.num_simulations} (adaptive)")
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
    
    print("=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    for iteration in tqdm(range(start_iteration, config.training.num_iterations), desc="Training Progress", ncols=100, initial=start_iteration, total=config.training.num_iterations):
        iter_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{config.training.num_iterations} (Temp: {temp:.2f}, Sims: {num_sims}, Games: {num_games})")
        print(f"{'='*80}")
        
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
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Samples: {result['num_samples']}")
        print(f"  Total Loss: {result['avg_loss']['total_loss']:.4f}")
        print(f"  Policy Loss: {result['avg_loss']['policy_loss']:.4f}")
        print(f"  Value Loss: {result['avg_loss']['value_loss']:.4f}")
        if 'learning_rate' in result:
            print(f"  Learning Rate: {result['learning_rate']:.6f}")
        
        if hasattr(trainer.replay_buffer, 'get_stats'):
            buffer_stats = trainer.replay_buffer.get_stats()
            if buffer_stats:
                print(f"  Buffer: {buffer_stats.get('total', 0)} samples, {buffer_stats.get('games', 0)} games")
        
        if 'dtw_stats' in result:
            stats = result['dtw_stats']
            print(f"  DTW Cache Hit Rate: {stats.get('hit_rate', 'N/A')}")
            print(f"  DTW Cache Size: {stats.get('total_mb', 0):.2f} MB")
            print(f"  DTW Searches: {stats.get('dtw_searches', 0):,} (Aborted: {stats.get('dtw_aborted', 0)}) Avg: {stats.get('dtw_avg_nodes', 0):.0f} nodes")
            trainer.dtw_calculator.reset_search_stats()
        
        iter_elapsed = time.time() - iter_start_time
        iter_mins, iter_secs = divmod(int(iter_elapsed), 60)
        iter_hours, iter_mins = divmod(iter_mins, 60)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Iteration Time: {iter_hours:02d}:{iter_mins:02d}:{iter_secs:02d}")
        print(f"  Completed at: {current_time}")
        
        current_loss = result['avg_loss']['total_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            save_path = os.path.join(config.training.save_dir, 'best.pt')
            trainer.save(save_path, iteration=iteration)
            print(f"\n✓ New best! (loss: {best_loss:.4f})")
            upload_to_hf(save_path, 'best.pt')
        
        if (iteration + 1) % 5 == 0:
            ckpt_path = os.path.join(config.training.save_dir, f'checkpoint_{iteration + 1}.pt')
            trainer.save(ckpt_path, iteration=iteration)
            print(f"✓ Checkpoint saved: checkpoint_{iteration + 1}.pt")
            
            old_iter = iteration + 1 - 10
            if old_iter > 0:
                old_path = os.path.join(config.training.save_dir, f'checkpoint_{old_iter}.pt')
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"  (Deleted old checkpoint: checkpoint_{old_iter}.pt)")
        
        if (iteration + 1) % 20 == 0:
            model_path = os.path.join(config.training.save_dir, f'model_{iteration + 1}.pt')
            trainer.save(model_path, iteration=iteration)
            print(f"✓ Model saved: model_{iteration + 1}.pt")
            upload_to_hf(model_path, f'model_{iteration + 1}.pt')
        
        if trainer.dtw_calculator and trainer.dtw_calculator.tt:
            dtw_cache_path = os.path.join(config.training.save_dir, 'dtw_cache.pkl')
            trainer.dtw_calculator.tt.save_to_file(dtw_cache_path)
            if (iteration + 1) % 10 == 0:
                upload_to_hf(dtw_cache_path, 'dtw_cache.pkl')
    
    print("\n" + "="*80)
    print(f"Training completed! Best loss: {best_loss:.4f}")
    print("="*80)
    
    if trainer.dtw_calculator and trainer.dtw_calculator.tt:
        dtw_cache_path = os.path.join(config.training.save_dir, 'dtw_cache.pkl')
        trainer.dtw_calculator.tt.save_to_file(dtw_cache_path)
        print(f"✓ DTW cache saved to {dtw_cache_path}")
        
        stats = trainer.dtw_calculator.get_stats()
        if stats:
            print(f"  Final cache size: {stats.get('total_mb', 0):.1f} MB")
            print(f"  Hit rate: {stats.get('hit_rate', 'N/A')}")
    
    if trainer.dtw_calculator:
        final_stats = trainer.dtw_calculator.get_stats()
        print("\nFinal DTW Statistics:")
        print(f"  Total queries: {final_stats.get('total_queries', 0)}")
        print(f"  Hit rate: {final_stats.get('hit_rate', 'N/A')}")
        print(f"  Cache size: {final_stats.get('total_mb', 0):.2f} MB")


if __name__ == '__main__':
    main()
