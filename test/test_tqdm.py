import torch
from config import Config, NetworkConfig, TrainingConfig
from ai.trainer import AlphaZeroTrainer

print("Testing tqdm progress bars...")
print("=" * 70)

# 작은 설정으로 빠른 테스트
config = Config()
config.network = NetworkConfig(
    num_res_blocks=2,
    num_channels=32
)
config.training = TrainingConfig(
    num_parallel_games=2,
    num_simulations=20,
    batch_size=16,
    use_amp=False
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Device: {device}")
print(f"Running 1 iteration with progress bars...")
print("=" * 70)

trainer = AlphaZeroTrainer(
    lr=0.001,
    weight_decay=1e-4,
    batch_size=16,
    num_simulations=20,
    replay_buffer_size=1000,
    device=device,
    use_amp=False,
    num_res_blocks=2,
    num_channels=32,
    num_parallel_games=2
)

# 1 iteration 테스트
result = trainer.train_iteration(
    num_self_play_games=4,
    num_train_epochs=10,
    temperature=1.0,
    verbose=False,
    disable_tqdm=False
)

print("\n" + "=" * 70)
print("Test complete!")
print(f"Samples collected: {result['num_samples']}")
print(f"Average loss: {result['avg_loss']['total_loss']:.4f}")
print("=" * 70)
