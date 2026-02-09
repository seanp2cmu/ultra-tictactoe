try:
    from .board_cy import BoardCy as Board
    USING_CYTHON = True
except ImportError:
    from .board import Board
    USING_CYTHON = False

from .board import Board as BoardPy

__all__ = ['Board', 'BoardPy', 'USING_CYTHON']