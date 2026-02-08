"""
Endgame Tablebase for Ultimate Tic-Tac-Toe

Precomputed perfect play database for positions with ≤15 empty cells.
Uses retrograde analysis to compute exact game-theoretic values.
"""

from .generator import TablebaseGenerator
from .tablebase import Tablebase, CompactTablebase
from .meta_board import MetaBoardEnumerator
from .solver import TablebaseSolver, count_empty
from .enumerator import PositionEnumerator
from .builder import TablebaseBuilder

__all__ = [
    'TablebaseGenerator', 
    'Tablebase', 
    'CompactTablebase',
    'MetaBoardEnumerator',
    'TablebaseSolver',
    'count_empty',
    'PositionEnumerator',
    'TablebaseBuilder'
]
