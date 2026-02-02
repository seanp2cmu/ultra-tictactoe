"""
Ultimate Tic-Tac-Toe AlphaZero + DTW 학습 스크립트

RTX 5090 (32GB VRAM) 최적화 설정:
- 30 ResNet blocks with SE (512 channels)
- 7-channel input (perspective normalized, last move, valid mask)
- 4096 batch size, 800 simulations
- Cosine Annealing LR (0.002 → 0.00002)
- Gradient clipping (max_norm=1.0)
- 500만/2000만 Tablebase cache
"""
import os
from tqdm import tqdm
from config import Config
from ai.training import Trainer

def main():
    config = Config()
    
    # 기존 모델 로드할 경로 (None이면 새로 시작)
    load_model_path = None  # 예: "./model/model_dtw_iter_50.pth"
    
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
    print(f"  Games per iteration: {config.training.num_self_play_games}")
    print(f"  Training epochs: {config.training.num_train_epochs}")
    print(f"  MCTS simulations: {config.training.num_simulations}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.lr}")
    print(f"  Parallel games: {config.training.num_parallel_games}")
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
        num_parallel_games=config.training.num_parallel_games,
        hot_cache_size=config.dtw.hot_cache_size,
        cold_cache_size=config.dtw.cold_cache_size,
        total_iterations=config.training.num_iterations
    )
    
    # 기존 모델 로드
    if load_model_path:
        print(f"\nLoading model from {load_model_path}")
        trainer.load(load_model_path)
        print("✓ Model loaded successfully\n")
    
    # Temperature 스케줄
    def get_temperature(iteration, total_iterations):
        progress = iteration / total_iterations
        if progress < 0.3:
            return config.mcts.temperature_start
        elif progress < 0.7:
            return (config.mcts.temperature_start + config.mcts.temperature_end) / 2
        else:
            return config.mcts.temperature_end
    
    # 학습 시작
    print("=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    # 학습 루프
    for iteration in tqdm(range(config.training.num_iterations), desc="Training Progress", ncols=100):
        temp = get_temperature(iteration, config.training.num_iterations)
        
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{config.training.num_iterations} (Temperature: {temp:.2f})")
        print(f"{'='*80}")
        
        # Self-play + Training
        result = trainer.train_iteration(
            num_self_play_games=config.training.num_self_play_games,
            num_train_epochs=config.training.num_train_epochs,
            temperature=temp,
            verbose=False,
            disable_tqdm=False,
            num_simulations=config.training.num_simulations
        )
        
        # 결과 출력
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Samples: {result['num_samples']}")
        print(f"  Total Loss: {result['avg_loss']['total_loss']:.4f}")
        print(f"  Policy Loss: {result['avg_loss']['policy_loss']:.4f}")
        print(f"  Value Loss: {result['avg_loss']['value_loss']:.4f}")
        if 'learning_rate' in result:
            print(f"  Learning Rate: {result['learning_rate']:.6f}")
        
        # 샘플 분포 통계
        if hasattr(trainer.replay_buffer, 'get_stats'):
            buffer_stats = trainer.replay_buffer.get_stats()
            if buffer_stats:
                print(f"  Avg Weight: {buffer_stats.get('avg_weight', 0):.2f}")
                dist = buffer_stats.get('distribution', {})
                if 'transition' in dist:
                    print(f"  Transition (26-29칸): {dist['transition']}")
        
        # DTW 통계
        if 'dtw_stats' in result:
            stats = result['dtw_stats']
            print(f"  DTW Cache Hit Rate: {stats.get('hit_rate', 'N/A')}")
            print(f"  DTW Cache Size: {stats.get('total_mb', 0):.2f} MB")
        
        # 주기적으로 모델 저장
        if (iteration + 1) % config.training.save_interval == 0:
            save_path = os.path.join(config.training.save_dir, f'model_dtw_iter_{iteration + 1}.pth')
            trainer.save(save_path)
            print(f"\n✓ Model saved to {save_path}")
            
            # Tablebase도 함께 저장 (중단 시 복구용)
            if trainer.dtw_calculator and trainer.dtw_calculator.tt:
                tablebase_path = os.path.join(config.training.save_dir, 'tablebase.pkl')
                trainer.dtw_calculator.tt.save_to_file(tablebase_path)
                stats = trainer.dtw_calculator.get_stats()
                print(f"✓ Tablebase checkpoint saved ({stats.get('total_mb', 0):.1f} MB, hit rate: {stats.get('hit_rate', 'N/A')})")
        
        # 주기적으로 DTW 캐시 클리어 (메모리 관리)
        if (iteration + 1) % 20 == 0:
            trainer.clear_dtw_cache()
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    # 최종 모델 저장
    final_path = os.path.join(config.training.save_dir, 'model_dtw_final.pth')
    trainer.save(final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    # Tablebase 저장
    if trainer.dtw_calculator and trainer.dtw_calculator.tt:
        tablebase_path = os.path.join(config.training.save_dir, 'tablebase.pkl')
        trainer.dtw_calculator.tt.save_to_file(tablebase_path)
        print(f"✓ Tablebase saved to {tablebase_path}")
        
        stats = trainer.dtw_calculator.get_stats()
        if stats:
            print(f"  Final cache size: {stats.get('total_mb', 0):.1f} MB")
            print(f"  Hit rate: {stats.get('hit_rate', 'N/A')}")
    
    # 최종 통계
    if trainer.dtw_calculator:
        final_stats = trainer.dtw_calculator.get_stats()
        print("\nFinal DTW Statistics:")
        print(f"  Total queries: {final_stats.get('total_queries', 0)}")
        print(f"  Hit rate: {final_stats.get('hit_rate', 'N/A')}")
        print(f"  Cache size: {final_stats.get('total_mb', 0):.2f} MB")


if __name__ == '__main__':
    main()
