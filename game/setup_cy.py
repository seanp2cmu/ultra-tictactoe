from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "board_cy",  # 패키지명 없이 모듈명만
    ["board_cy.pyx"],
    include_dirs=[np.get_include()],
)

setup(
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
)
