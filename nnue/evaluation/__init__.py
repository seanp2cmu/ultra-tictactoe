"""Redirect: nnue.evaluation → evaluation (root). Kept for backward compatibility."""
from evaluation.evaluator import (
    nnue_evaluate_vs_baseline as evaluate_vs_baseline,
    nnue_run_evaluation_suite as run_evaluation_suite,
)
__all__ = ['evaluate_vs_baseline', 'run_evaluation_suite']
