#!/usr/bin/env python3
"""
Test optimal parallel games count for GPU.
Run this on GPU server to find best settings for your VRAM.

Usage: python test/test_parallel_gpu_optimal.py
"""
import sys
sys.path.insert(0, '.')

import time
import torch
import gc
import numpy as np

from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from ai.training.self_play import SelfPlayWorker
from ai.training.self_play import SelfPlayWorker


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_max():
    """Get max GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_parallel_games(network, dtw, num_parallel, num_sims=100):
    """Test with specific number of parallel games."""
    clear_gpu_memory()
    
    worker = SelfPlayWorker(
        network=network,
        dtw_calculator=dtw,
        num_simulations=num_sims,
        temperature=1.0,
        parallel_games=num_parallel
    )
    
    # Play enough games to measure
    num_games = max(num_parallel, 4)
    
    t0 = time.perf_counter()
    try:
        data = worker.play_games(num_games)
        elapsed = time.perf_counter() - t0
        
        mem_peak = get_gpu_memory_max()
        samples = len(data)
        games_per_sec = num_games / elapsed
        
        return {
            'success': True,
            'time': elapsed,
            'samples': samples,
            'games_per_sec': games_per_sec,
            'mem_peak_mb': mem_peak,
            'error': None
        }
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            return {
                'success': False,
                'error': 'OOM',
                'mem_peak_mb': get_gpu_memory_max()
            }
        raise


def test_sequential(network, dtw, num_games=4, num_sims=100):
    """Test sequential baseline."""
    clear_gpu_memory()
    
    t0 = time.perf_counter()
    samples = 0
    for _ in range(num_games):
        worker = SelfPlayWorker(
            network=network,
            dtw_calculator=dtw,
            num_simulations=num_sims,
            temperature=1.0
        )
        data = worker.play_game()
        samples += len(data)
    
    elapsed = time.perf_counter() - t0
    mem_peak = get_gpu_memory_max()
    
    return {
        'time': elapsed,
        'samples': samples,
        'games_per_sec': num_games / elapsed,
        'mem_peak_mb': mem_peak
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    
    print("=" * 70)
    print("PARALLEL SELF-PLAY GPU OPTIMIZATION TEST")
    print("=" * 70)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    
    print()
    
    # Load network
    print("Loading network...")
    network = AlphaZeroNet(device=device)
    dtw = DTWCalculator(use_cache=True)
    
    clear_gpu_memory()
    base_mem = get_gpu_memory()
    print(f"Base memory: {base_mem:.0f} MB")
    print()
    
    # Settings
    num_sims = 200  # Realistic training setting
    test_parallel_counts = [1, 2, 4, 8, 12, 16, 24, 32]
    
    print(f"Testing with {num_sims} simulations per move")
    print("-" * 70)
    
    # Sequential baseline
    print("\n[Sequential baseline]")
    seq_result = test_sequential(network, dtw, num_games=2, num_sims=num_sims)
    print(f"  Time: {seq_result['time']:.1f}s for 2 games")
    print(f"  Speed: {seq_result['games_per_sec']:.3f} games/sec")
    print(f"  Memory: {seq_result['mem_peak_mb']:.0f} MB")
    
    baseline_speed = seq_result['games_per_sec']
    
    # Test different parallel counts
    results = []
    print("\n[Parallel tests]")
    print(f"{'Parallel':>10} {'Time':>10} {'Speed':>12} {'Speedup':>10} {'Memory':>12} {'Status':>10}")
    print("-" * 70)
    
    for n in test_parallel_counts:
        result = test_parallel_games(network, dtw, n, num_sims)
        
        if result['success']:
            speedup = result['games_per_sec'] / baseline_speed
            status = "✓ OK"
            results.append({
                'parallel': n,
                'speedup': speedup,
                'mem_mb': result['mem_peak_mb'],
                'games_per_sec': result['games_per_sec']
            })
            print(f"{n:>10} {result['time']:>10.1f}s {result['games_per_sec']:>10.3f}/s {speedup:>10.2f}x {result['mem_peak_mb']:>10.0f} MB {status:>10}")
        else:
            print(f"{n:>10} {'--':>10} {'--':>12} {'--':>10} {result.get('mem_peak_mb', 0):>10.0f} MB {'❌ OOM':>10}")
            break  # Stop if OOM
    
    # Find optimal
    print("\n" + "=" * 70)
    if results:
        best = max(results, key=lambda x: x['speedup'])
        print(f"OPTIMAL: parallel_games = {best['parallel']}")
        print(f"  Speedup: {best['speedup']:.2f}x")
        print(f"  Memory: {best['mem_mb']:.0f} MB")
        
        # Safety margin recommendation
        safe_parallel = best['parallel']
        for r in results:
            if r['speedup'] >= best['speedup'] * 0.9:  # Within 90% of best
                safe_parallel = r['parallel']
                break
        
        print(f"\nRECOMMENDED (with safety margin): parallel_games = {safe_parallel}")
    else:
        print("No successful tests - reduce parallel_games or check VRAM")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
