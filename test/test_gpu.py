import torch
import time
import numpy as np
from ai.network import AlphaZeroNet
from game import Board

print("=" * 70)
print("GPU Test - Verifying GPU is actually being used")
print("=" * 70)

# Device 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# Network 생성
net = AlphaZeroNet(num_res_blocks=15, num_channels=256, lr=0.001, device=device)
print(f"\nNetwork created on {net.device}")

if device == "cuda":
    print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# Test 1: Single prediction
print("\n" + "=" * 70)
print("Test 1: Single prediction")
print("=" * 70)

board = Board()
start = time.time()
for i in range(10):
    policy, value = net.predict(board)
end = time.time()
print(f"10 predictions: {(end-start)*1000:.1f}ms")
print(f"Avg per prediction: {(end-start)*100:.1f}ms")

# Test 2: Batch prediction
print("\n" + "=" * 70)
print("Test 2: Batch prediction (32 boards)")
print("=" * 70)

boards = [Board() for _ in range(32)]
start = time.time()
for i in range(10):
    policies, values = net.predict_batch(boards)
end = time.time()
print(f"10 batch predictions: {(end-start)*1000:.1f}ms")
print(f"Avg per batch: {(end-start)*100:.1f}ms")
print(f"Avg per board in batch: {(end-start)*100/32:.1f}ms")

if device == "cuda":
    print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"Max GPU Memory: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")

# Test 3: Training step (GPU 많이 사용)
print("\n" + "=" * 70)
print("Test 3: Training step (should use GPU heavily)")
print("=" * 70)

batch_size = 1024
boards_data = np.random.randn(batch_size, 6, 9, 9).astype(np.float32)
policies_data = np.random.randn(batch_size, 81).astype(np.float32)
values_data = np.random.randn(batch_size).astype(np.float32)

print(f"Training with batch_size={batch_size}...")
start = time.time()
for i in range(10):
    loss = net.train_step(boards_data, policies_data, values_data)
end = time.time()
print(f"10 training steps: {(end-start)*1000:.1f}ms")
print(f"Avg per step: {(end-start)*100:.1f}ms")
print(f"Loss: {loss}")

if device == "cuda":
    print(f"\nPeak GPU Memory during training: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")

print("\n" + "=" * 70)
print("Summary:")
print("- If 'Single prediction' is fast (~1-5ms), GPU is working")
print("- If 'Batch prediction' is faster per board, batching is working")
print("- Training step should show high GPU memory usage")
print("=" * 70)
