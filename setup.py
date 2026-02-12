from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "game.board_cy",
        ["game/board_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "ai.mcts.node_cy",
        ["ai/mcts/node_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="ultra-tictactoe",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
)
