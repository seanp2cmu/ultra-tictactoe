"""Training logging: file logs, W&B metrics, evaluation."""
import os
import time
import datetime
import wandb

from ai.evaluation.evaluator import run_evaluation_suite


def log_iteration_to_file(log_path, iteration, total_iters, loss, lr, samples,
                          buffer_stats, timing_stats, iter_elapsed, config, dtw_stats=None):
    """Write iteration details to training.log."""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{now}] ITERATION {iteration+1}/{total_iters}\n")
        f.write(f"{'='*60}\n")
        
        total_t = timing_stats['total_time']
        if total_t > 0:
            f.write(f"[Timing] Total: {total_t:.1f}s | Network: {timing_stats['network_time']:.1f}s | Overhead: {timing_stats['mcts_overhead']:.1f}s\n")
            moves_t = timing_stats.get('moves', 0)
            if moves_t > 0:
                f.write(f"  Avg Time/Move: {total_t/moves_t*1000:.2f}ms\n")
        
        f.write(f"[Self-Play] Games: {config.training.num_self_play_games} | Sims: {config.training.num_simulations} | Samples: {samples:,}\n")
        f.write(f"[Buffer] Total: {buffer_stats.get('total', 0):,} | Games: {buffer_stats.get('games', 0):,}\n")
        f.write(f"[Loss] Total: {loss['total_loss']:.6f} | Policy: {loss['policy_loss']:.6f} | Value: {loss['value_loss']:.6f}\n")
        f.write(f"[LR] {lr:.8f}\n")
        f.write(f"[Time] {iter_elapsed:.1f}s | ETA: {iter_elapsed * (total_iters - iteration - 1) / 3600:.1f}h\n")
        
        if dtw_stats:
            f.write(f"[DTW] Queries: {dtw_stats.get('total_queries', 0):,} | Hit: {dtw_stats.get('hit_rate', 'N/A')} | Hot: {dtw_stats.get('hot_entries', 0):,}\n")


def collect_wandb_metrics(loss, lr, buffer_stats, iter_elapsed, dtw_stats=None):
    """Build W&B metrics dict for an iteration."""
    metrics = {
        'train/total_loss': loss['total_loss'],
        'train/policy_loss': loss['policy_loss'],
        'train/value_loss': loss['value_loss'],
        'train/learning_rate': lr,
        'train/replay_buffer_size': buffer_stats.get('total', 0),
        'train/iteration_time_s': iter_elapsed,
    }
    if dtw_stats:
        metrics['dtw/hit_rate'] = float(dtw_stats.get('hit_rate', '0%').rstrip('%')) / 100
        metrics['dtw/hot_entries'] = dtw_stats.get('hot_entries', 0)
    return metrics


def run_and_log_eval(log_path, network, dtw_calculator, wandb_metrics):
    """Run baseline evaluation, append to log and wandb metrics."""
    eval_start = time.time()
    try:
        eval_metrics = run_evaluation_suite(
            network=network,
            num_games=500,
            dtw_calculator=dtw_calculator
        )
        eval_elapsed = time.time() - eval_start
        
        with open(log_path, 'a') as f:
            f.write(f"[Eval] ({eval_elapsed:.1f}s)")
            for k, v in eval_metrics.items():
                if 'winrate' in k or 'drawrate' in k:
                    f.write(f" | {k.split('/')[-1]}: {v:.1f}%")
            f.write("\n")
        
        wandb_metrics.update(eval_metrics)
        wandb_metrics['eval/eval_time_s'] = eval_elapsed
    except Exception as e:
        print(f"Eval failed: {e}")


def log_training_complete(log_path, best_loss):
    """Write final completion message."""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as f:
        f.write(f"\n[{now}] TRAINING COMPLETE | Best Loss: {best_loss:.4f}\n")
