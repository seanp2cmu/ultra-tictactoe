import time
from config import Config, NetworkConfig, TrainingConfig
from ai.trainer import AlphaZeroTrainer
import torch

print("Testing Parallel Self-Play...")
print("=" * 70)

# 로컬 테스트용 작은 설정
config = Config()
config.network = NetworkConfig(
    num_res_blocks=3,
    num_channels=64
)
config.training = TrainingConfig(
    num_parallel_games=2,  # 2개 병렬 테스트
    num_simulations=50,
    batch_size=32,
    use_amp=False
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

print(f"\nTest Configuration:")
print(f"  Device: {device}")
print(f"  Network: {config.network.num_res_blocks} blocks, {config.network.num_channels} channels")
print(f"  Parallel games: {config.training.num_parallel_games}")
print(f"  MCTS simulations: {config.training.num_simulations}")
print("=" * 70)

trainer = AlphaZeroTrainer(
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
    batch_size=config.training.batch_size,
    num_simulations=config.training.num_simulations,
    replay_buffer_size=config.training.replay_buffer_size,
    device=device,
    use_amp=config.training.use_amp,
    num_res_blocks=config.network.num_res_blocks,
    num_channels=config.network.num_channels,
    num_parallel_games=2
)

num_test_games = 4

print(f"\nRunning {num_test_games} self-play games with 2 parallel workers...")
print("(Batch statistics will be shown at the end)")
start_time = time.time()
num_samples = trainer.generate_self_play_data(num_games=num_test_games, temperature=1.0, verbose=False)
parallel_time = time.time() - start_time

print(f"\n✓ Parallel execution completed!")
print(f"  Time: {parallel_time:.2f} seconds")
print(f"  Samples collected: {num_samples}")
print(f"  Games per second: {num_test_games / parallel_time:.2f}")

print(f"\nTesting sequential execution for comparison ({num_test_games} games)...")
trainer.num_parallel_games = 1
start_time = time.time()
num_samples_seq = trainer.generate_self_play_data(num_games=num_test_games, temperature=1.0, verbose=False)
sequential_time = time.time() - start_time

print(f"\n✓ Sequential execution completed!")
print(f"  Time: {sequential_time:.2f} seconds")
print(f"  Samples collected: {num_samples_seq}")
print(f"  Games per second: {num_test_games / sequential_time:.2f}")

speedup = sequential_time / parallel_time
print("\n" + "=" * 70)
print(f"Speedup: {speedup:.2f}x faster with parallel execution!")
print("=" * 70)
