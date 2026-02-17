"""Build script for C++ extensions (Board + DTW + NNUE)."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class get_pybind_include:
    def __str__(self):
        import pybind11
        return pybind11.get_include()


common_args = ['-std=c++17', '-O3', '-fPIC', '-march=native', '-mavx2', '-mfma']
common_includes = [get_pybind_include(), '.']

ext_modules = [
    Extension(
        'uttt_cpp',
        [
            'game/cpp/board.cpp',
            'ai/endgame/cpp/dtw.cpp',
            'cpp_bindings.cpp',
        ],
        include_dirs=common_includes,
        language='c++',
        extra_compile_args=common_args,
    ),
    Extension(
        'nnue_cpp',
        [
            'game/cpp/board.cpp',
            'nnue/cpp/nnue_search.cpp',
            'nnue/cpp/nnue_datagen.cpp',
            'nnue/cpp/nnue_bindings.cpp',
        ],
        include_dirs=common_includes,
        language='c++',
        extra_compile_args=common_args + ['-pthread'],
        extra_link_args=['-pthread'],
    ),
]

setup(
    name='uttt_cpp',
    version='1.0.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
