"""Data Rescoring: re-evaluate NNUE self-play positions with AlphaZero teacher.

Pipeline:
  1. NNUE C++ self-play generates positions quickly (CPU)
  2. AlphaZero GPU batch inference rescores each position with accurate value
  3. Rescored data is saved for NNUE training

This is the Stockfish × Leela approach adapted for UTTT.
"""
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from game import Board
from ai.core import AlphaZeroNet
from ai.mcts import AlphaZeroAgent, Node
from nnue.core.features import board_array_to_features


def _array_to_board(arr):
    """Convert stored board array (92,) back to a Board object."""
    board = Board()
    cells = arr[:81]
    meta = arr[81:90]
    active = int(arr[90])
    current_player = int(arr[91])
    
    # Reconstruct board state by placing pieces
    for idx in range(81):
        v = cells[idx]
        if v == 0:
            continue
        r, c = idx // 9, idx % 9
        sub = (r // 3) * 3 + (c // 3)
        cell = (r % 3) * 3 + (c % 3)
        bit = 1 << cell
        if v == 1:
            board.x_masks[sub] |= bit
        else:
            board.o_masks[sub] |= bit
    
    # Set meta-board
    for i in range(9):
        board.completed_boards[i] = int(meta[i])
    
    # Set current player and active constraint
    board.current_player = current_player
    if active >= 0:
        board.last_move_r = (active // 3)
        board.last_move_c = (active % 3)
        board.has_last_move = True
    else:
        board.has_last_move = False
    
    # Check winner
    board._check_global_winner()
    
    return board


def rescore_with_alphazero(
    data_path: str,
    network: AlphaZeroNet,
    num_simulations: int = 50,
    batch_size: int = 256,
    lambda_rescore: float = 0.75,
    output_path: str = None,
    max_positions: int = None,
) -> dict:
    """Rescore NNUE self-play positions with AlphaZero MCTS.
    
    Args:
        data_path: Path to .npz file with 'boards' and 'values' arrays
        network: Loaded AlphaZero network
        num_simulations: MCTS sims per position for rescoring
        batch_size: Batch size for processing
        lambda_rescore: Blend: λ × alphazero_value + (1-λ) × original_value
        output_path: Where to save rescored data (None = overwrite)
        max_positions: Limit number of positions to rescore (None = all)
    
    Returns:
        dict with 'boards', 'values' arrays and stats
    """
    data = np.load(data_path)
    boards = data['boards']
    original_values = data['values']
    
    if max_positions and len(boards) > max_positions:
        indices = np.random.choice(len(boards), max_positions, replace=False)
        boards = boards[indices]
        original_values = original_values[indices]
    
    n = len(boards)
    rescored_values = np.zeros(n, dtype=np.float32)
    
    agent = AlphaZeroAgent(
        network=network,
        num_simulations=num_simulations,
        temperature=0.01,
    )
    
    t0 = time.time()
    pbar = tqdm(range(0, n, batch_size), desc="Rescoring", ncols=90, leave=False)
    
    for start in pbar:
        end = min(start + batch_size, n)
        
        for i in range(start, end):
            arr = boards[i]
            board = _array_to_board(arr)
            
            if board.winner not in (None, -1):
                # Terminal position — use exact result
                if board.winner == 3:
                    rescored_values[i] = 0.0
                elif board.winner == board.current_player:
                    rescored_values[i] = 1.0
                else:
                    rescored_values[i] = -1.0
                continue
            
            # MCTS search for value
            root = agent.search(board, add_noise=False)
            
            # Best child value from STM perspective
            if root.children:
                best_child = max(root.children.values(), key=lambda c: c.visits)
                az_value = -best_child.value()  # Negate: child value is from opponent's perspective
            else:
                az_value = root.value() if root.visits > 0 else 0.0
            
            rescored_values[i] = az_value
        
        elapsed = time.time() - t0
        done = min(end, n)
        rate = done / elapsed if elapsed > 0 else 0
        pbar.set_postfix_str(f"{rate:.0f} pos/s")
    
    pbar.close()
    
    # Blend: λ × alphazero + (1-λ) × original
    blended_values = lambda_rescore * rescored_values + (1.0 - lambda_rescore) * original_values
    
    # Save
    out_path = output_path or data_path
    np.savez_compressed(out_path, boards=boards, values=blended_values)
    
    elapsed = time.time() - t0
    stats = {
        'num_positions': n,
        'elapsed_s': elapsed,
        'rate_pos_per_s': n / elapsed if elapsed > 0 else 0,
        'value_shift_mean': float(np.mean(np.abs(blended_values - original_values))),
        'value_shift_max': float(np.max(np.abs(blended_values - original_values))),
    }
    
    tqdm.write(f"  Rescored {n:,} positions in {elapsed:.1f}s "
               f"({stats['rate_pos_per_s']:.0f} pos/s, "
               f"mean shift={stats['value_shift_mean']:.4f})")
    
    return {'boards': boards, 'values': blended_values, **stats}
