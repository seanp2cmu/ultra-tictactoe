import os
import time
import torch
from tqdm import tqdm
from ai.trainer import AlphaZeroTrainer
from config import Config, get_default_config, get_gpu_optimized_config, get_cpu_config

def train_alphazero(config: Config = None):
    if config is None:
        config = get_default_config()
    
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    device_str = config.gpu.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    
    print("=" * 70)
    print("AlphaZero Ultra Tic-Tac-Toe Training")
    print("=" * 70)
    print(f"Device: {device_str}")
    if device_str == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"\nNetwork Configuration:")
    print(f"  Residual blocks: {config.network.num_res_blocks}")
    print(f"  Channels: {config.network.num_channels}")
    print(f"\nTraining Configuration:")
    print(f"  Iterations: {config.training.num_iterations}")
    print(f"  Self-play games per iteration: {config.training.num_self_play_games}")
    print(f"  Parallel games: {config.training.num_parallel_games}")
    print(f"  Training epochs per iteration: {config.training.num_train_epochs}")
    print(f"  MCTS simulations: {config.training.num_simulations}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.lr}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Replay buffer size: {config.training.replay_buffer_size}")
    print(f"  Mixed precision (AMP): {config.training.use_amp}")
    print(f"  Save directory: {config.training.save_dir}")
    print(f"  Save interval: every {config.training.save_interval} iterations")
    print("=" * 70 + "\n")
    
    trainer = AlphaZeroTrainer(
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        batch_size=config.training.batch_size,
        num_simulations=config.training.num_simulations,
        replay_buffer_size=config.training.replay_buffer_size,
        device=device_str,
        use_amp=config.training.use_amp,
        num_res_blocks=config.network.num_res_blocks,
        num_channels=config.network.num_channels,
        num_parallel_games=config.training.num_parallel_games
    )
    
    start_time = time.time()
    
    pbar = tqdm(range(1, config.training.num_iterations + 1), desc="Training", ncols=120)
    for iteration in pbar:
        iter_start = time.time()
        
        pbar.set_description(f"Iteration {iteration}/{config.training.num_iterations}")
        
        # Progressive temperature
        temp_range = config.mcts.temperature_start - config.mcts.temperature_end
        temperature = max(config.mcts.temperature_end, 
                         config.mcts.temperature_start - (iteration / config.training.num_iterations) * temp_range)
        
        # Progressive MCTS: 초반에는 적은 시뮬레이션, 후반에는 많은 시뮬레이션
        progress = iteration / config.training.num_iterations
        min_sims = 50  # 초반 시뮬레이션 수
        max_sims = config.training.num_simulations  # 최대 시뮬레이션 수
        current_sims = int(min_sims + (max_sims - min_sims) * progress)
        
        result = trainer.train_iteration(
            num_self_play_games=config.training.num_self_play_games,
            num_train_epochs=config.training.num_train_epochs,
            temperature=temperature,
            verbose=False,
            disable_tqdm=False,
            num_simulations=current_sims
        )
        
        iter_elapsed = time.time() - iter_start
        total_elapsed = time.time() - start_time
        
        pbar.set_postfix({
            'loss': f"{result['avg_loss']['total_loss']:.4f}",
            'samples': result['num_samples'],
            'sims': current_sims,
            'time': f"{iter_elapsed:.1f}s"
        })
        
        if iteration % config.training.save_interval == 0:
            model_path = os.path.join(config.training.save_dir, f"model_iter_{iteration}.pth")
            trainer.save(model_path)
            print(f"\n  ✓ Model saved: {model_path}")
        
        latest_path = os.path.join(config.training.save_dir, "model_latest.pth")
        trainer.save(latest_path)
    
    final_path = os.path.join(config.training.save_dir, "model_final.pth")
    trainer.save(final_path)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final buffer size: {len(trainer.replay_buffer)}")
    print(f"Final model saved: {final_path}")
    print(f"Latest model: {latest_path}")
    print("=" * 70)


if __name__ == "__main__":
    import torch
    
    if torch.cuda.is_available():
        print("CUDA GPU detected - using GPU optimized config")
        config = get_gpu_optimized_config()
    elif torch.backends.mps.is_available():
        print("MPS GPU (Apple Silicon) detected - using GPU optimized config")
        config = get_gpu_optimized_config()
        config.gpu.device = "mps"  # MPS 사용
        config.training.use_amp = False  # MPS는 AMP 지원 안함
    else:
        print("No GPU detected - using default config")
        config = get_default_config()
    
    train_alphazero(config)
