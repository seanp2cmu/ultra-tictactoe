"""Redirect: ai.evaluation → evaluation (root). Kept for backward compatibility."""
from evaluation.evaluator import (
    MatchResult, play_parallel, play_agents,
    play_networks_parallel, play_sims_parallel,
    evaluate_vs_baseline, run_evaluation_suite,
)
from evaluation.elo import EloTracker

__all__ = [
    'MatchResult', 'play_parallel', 'play_agents',
    'play_networks_parallel', 'play_sims_parallel',
    'evaluate_vs_baseline', 'run_evaluation_suite',
    'EloTracker',
]
