"""
Endgame Tablebase for Ultimate Tic-Tac-Toe

Precomputed perfect play database for positions with ≤15 empty cells.
Uses progressive endgame construction to compute exact game-theoretic values.
"""

from .tablebase import Tablebase, CompactTablebase
from .solver import TablebaseSolver
from .enumerator import PositionEnumerator
from .builder import TablebaseBuilder

__all__ = [
    'Tablebase', 
    'CompactTablebase',
    'TablebaseSolver',
    'PositionEnumerator',
    'TablebaseBuilder'
]
