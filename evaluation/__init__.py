"""Unified evaluation module for both AlphaZero (GPU) and NNUE (CPU) agents."""
from .evaluator import (
    MatchResult, play_parallel, play_agents,
    play_networks_parallel, play_sims_parallel,
    evaluate_vs_baseline, run_evaluation_suite,
)
from .elo import EloTracker
from .nnue_evaluator import (
    evaluate_vs_baseline as nnue_evaluate_vs_baseline,
    run_evaluation_suite as nnue_run_evaluation_suite,
)

__all__ = [
    'MatchResult', 'play_parallel', 'play_agents',
    'play_networks_parallel', 'play_sims_parallel',
    'evaluate_vs_baseline', 'run_evaluation_suite',
    'EloTracker',
    'nnue_evaluate_vs_baseline', 'nnue_run_evaluation_suite',
]
