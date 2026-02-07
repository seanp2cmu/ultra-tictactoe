"""
Baseline opponents for testing AlphaZero agent.
"""
from .random_agent import RandomAgent
from .heuristic_agent import HeuristicAgent
from .minimax_agent import MinimaxAgent

__all__ = ['RandomAgent', 'HeuristicAgent', 'MinimaxAgent']
