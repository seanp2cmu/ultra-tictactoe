"""
Checkpoint Utilities
Functions for finding and loading model checkpoints
"""
import os
import re
import glob
import torch


def find_best_checkpoint(save_dir: str) -> str:
    """Find best.pt checkpoint if exists."""
    best_path = os.path.join(save_dir, 'best.pt')
    if os.path.exists(best_path):
        return best_path
    return None


def find_latest_checkpoint(save_dir: str) -> tuple:
    """Find the latest checkpoint_*.pt and return (path, next_iteration).
    checkpoint_5.pt means iterations 0-4 done, returns 5 as next iteration."""
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
    # checkpoint_N.pt: N iterations done (0 to N-1), next is N
    return iterations[0][1], iterations[0][0]


def get_start_iteration(save_dir: str) -> tuple:
    """Get checkpoint path and start iteration.
    Returns (checkpoint_path, start_iteration)
    Prefers latest checkpoint_*.pt over best.pt
    """
    checkpoint_path, start_iteration = find_latest_checkpoint(save_dir)
    
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(save_dir)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            saved_iter = ckpt.get('iteration', None)
            if saved_iter is not None:
                start_iteration = saved_iter + 1
    
    return checkpoint_path, start_iteration
