from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class get_pybind_include:
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'uttt_cpp',
        ['board.cpp', 'dtw.cpp', 'bindings.cpp'],
        include_dirs=[get_pybind_include()],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-fPIC'],
    ),
]

setup(
    name='uttt_cpp',
    version='1.0.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
