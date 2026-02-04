"""
Ultimate Tic-Tac-Toe AlphaZero + DTW 학습 스크립트

RTX 5090 (32GB VRAM) 최적화 설정:
- 30 ResNet blocks with SE (512 channels)
- 7-channel input (perspective normalized, last move, valid mask)
- 4096 batch size, 800 simulations
- Cosine Annealing LR (0.002 → 0.00002)
- Gradient clipping (max_norm=1.0)
- 500만/2000만 DTW cache
"""
import os
import re
import glob
import time
from tqdm import tqdm
from config import Config
from ai.training import Trainer
from ai.training.self_play import set_slow_log_file


def find_best_checkpoint(save_dir: str) -> str:
    """Find best.pt checkpoint if exists."""
    best_path = os.path.join(save_dir, 'best.pt')
    if os.path.exists(best_path):
        return best_path
    return None

def find_latest_checkpoint(save_dir: str) -> tuple:
    """Find the latest checkpoint_*.pt and return (path, iteration)."""
    pattern = os.path.join(save_dir, 'checkpoint_*.pt')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None, 0
    
    iterations = []
    for ckpt in checkpoints:
        match = re.search(r'checkpoint_(\d+)\.pt', ckpt)
        if match:
            iterations.append((int(match.group(1)), ckpt))
    
    if not iterations:
        return None, 0
    
    iterations.sort(key=lambda x: x[0], reverse=True)
    return iterations[0][1], iterations[0][0]

def main():
    config = Config()
    
    # 체크포인트 찾기 (checkpoint_*.pt 우선, 없으면 best.pt)
    checkpoint_path, start_iteration = find_latest_checkpoint(config.training.save_dir)
    if not checkpoint_path:
        checkpoint_path = find_best_checkpoint(config.training.save_dir)
        start_iteration = 0 
    
    # 디바이스 자동 설정
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
    
    # 저장 디렉토리 생성
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    # 설정 출력
    print("=" * 80)
    print("AlphaZero Training with DTW + Compressed Transposition Table")
    print("=" * 80)
    print(f"Network:")
    print(f"  Residual Blocks: {config.network.num_res_blocks}")
    print(f"  Channels: {config.network.num_channels}")
    print(f"\nTraining:")
    print(f"  Iterations: {config.training.num_iterations}")
    print(f"  Games per iteration: {config.training.num_self_play_games // 3} → {config.training.num_self_play_games} (adaptive)")
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
    print("=" * 80)
    
    # Trainer 초기화
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
    
    # Initialize slow steps log file
    slow_log_path = os.path.join(config.training.save_dir, 'slow_steps.txt')
    set_slow_log_file(slow_log_path)
    print(f"📝 Slow steps (>5s) will be logged to: {slow_log_path}")
    
    # Load existing model
    best_loss = float('inf')
    if checkpoint_path:
        print(f"\n  Loading checkpoint: {checkpoint_path}")
        loaded_iter = trainer.load(checkpoint_path)
        if loaded_iter:
            start_iteration = loaded_iter
        print(f"✓ Model loaded (will resume from iteration {start_iteration + 1})\n")
    
    # Temperature Schedule
    def get_temperature(iteration, total_iterations):
        progress = iteration / total_iterations
        if progress < 0.3:
            return config.mcts.temperature_start
        elif progress < 0.7:
            return (config.mcts.temperature_start + config.mcts.temperature_end) / 2
        else:
            return config.mcts.temperature_end
    
    # Simulation Schedule
    MIN_SIM = config.training.num_simulations // 4
    MAX_SIM = config.training.num_simulations  
    
    def get_num_simulations(iteration, total_iterations):
        progress = iteration / total_iterations
        min_sim = MIN_SIM
        max_sim = MAX_SIM
        
        if progress < 0.2:
            return min_sim
        elif progress < 0.5:
            return min_sim + int((max_sim - min_sim) * 0.3)
        elif progress < 0.8:
            return min_sim + int((max_sim - min_sim) * 0.6)
        else:
            return max_sim
    
    MIN_GAMES = config.training.num_self_play_games // 3
    MAX_GAMES = config.training.num_self_play_games       
    
    def get_num_games(iteration, total_iterations):
        progress = iteration / total_iterations
        min_games = MIN_GAMES
        max_games = MAX_GAMES
        
        if progress < 0.2:
            return min_games
        elif progress < 0.5:
            return min_games + int((max_games - min_games) * 0.3)
        elif progress < 0.8:
            return min_games + int((max_games - min_games) * 0.6)
        else:
            return max_games
    
    print("=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    for iteration in tqdm(range(start_iteration, config.training.num_iterations), desc="Training Progress", ncols=100, initial=start_iteration, total=config.training.num_iterations):
        iter_start_time = time.time()
        temp = get_temperature(iteration, config.training.num_iterations)
        num_sims = get_num_simulations(iteration, config.training.num_iterations)
        
        print(f"\n{'='*80}")
        num_games = get_num_games(iteration, config.training.num_iterations)
        print(f"Iteration {iteration + 1}/{config.training.num_iterations} (Temp: {temp:.2f}, Sims: {num_sims}, Games: {num_games})")
        print(f"{'='*80}")
        
        # Self-play + Training
        result = trainer.train_iteration(
            num_self_play_games=num_games,
            num_train_epochs=config.training.num_train_epochs,
            temperature=temp,
            verbose=False,
            disable_tqdm=False,
            num_simulations=num_sims
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
                print(f"  Avg Weight: {buffer_stats.get('avg_weight', 0):.2f}")
                dist = buffer_stats.get('distribution', {})
                if 'transition' in dist:
                    print(f"  Transition (26-29칸): {dist['transition']}")
        
        if 'dtw_stats' in result:
            stats = result['dtw_stats']
            print(f"  DTW Cache Hit Rate: {stats.get('hit_rate', 'N/A')}")
            print(f"  DTW Cache Size: {stats.get('total_mb', 0):.2f} MB")
            print(f"  DTW Searches: {stats.get('dtw_searches', 0):,} (Aborted: {stats.get('dtw_aborted', 0)}) Avg: {stats.get('dtw_avg_nodes', 0):.0f} nodes")
            print(f"  Shallow Searches: {stats.get('shallow_searches', 0):,} (Aborted: {stats.get('shallow_aborted', 0)}) Avg: {stats.get('shallow_avg_nodes', 0):.0f} nodes")
            # Reset DTW stats for next iteration
            if trainer.dtw_calculator:
                trainer.dtw_calculator.reset_search_stats()
        
        # Iteration 소요 시간 및 완료 시각 출력
        iter_elapsed = time.time() - iter_start_time
        iter_mins, iter_secs = divmod(int(iter_elapsed), 60)
        iter_hours, iter_mins = divmod(iter_mins, 60)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Iteration Time: {iter_hours:02d}:{iter_mins:02d}:{iter_secs:02d}")
        print(f"  Completed at: {current_time}")
        
        # Save best model
        current_loss = result['avg_loss']['total_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            save_path = os.path.join(config.training.save_dir, 'best.pt')
            trainer.save(save_path, iteration=iteration)
            print(f"\n✓ New best! (loss: {best_loss:.4f})")
        
        # Rolling checkpoint: save every 5, keep only 2 most recent
        if (iteration + 1) % 5 == 0:
            ckpt_path = os.path.join(config.training.save_dir, f'checkpoint_{iteration + 1}.pt')
            trainer.save(ckpt_path, iteration=iteration)
            print(f"✓ Checkpoint saved: checkpoint_{iteration + 1}.pt")
            
            # Delete old checkpoint (keep only 2)
            old_iter = iteration + 1 - 10  # e.g., iter 15 saves -> delete iter 5
            if old_iter > 0:
                old_path = os.path.join(config.training.save_dir, f'checkpoint_{old_iter}.pt')
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"  (Deleted old checkpoint: checkpoint_{old_iter}.pt)")
        
        #  DTW cache clear
        if (iteration + 1) % 20 == 0:
            trainer.clear_dtw_cache()
    
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
