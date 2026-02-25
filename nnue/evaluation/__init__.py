"""NNUE evaluation against baseline opponents."""
"""Redirect: nnue.evaluation → evaluation (root). Kept for backward compatibility."""
from evaluation.nnue_evaluator import (
    evaluate_vs_baseline,
    run_evaluation_suite,
)
__all__ = ['evaluate_vs_baseline', 'run_evaluation_suite']
