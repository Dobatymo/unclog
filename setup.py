import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

if sys.platform.startswith("linux"):
    cflags = ["-O2"]
    cflags_omp = ["-O2", "-fopenmp"]
elif sys.platform == "win32":
    cflags = ["/O2"]
    cflags_omp = ["/O2", "/openmp"]
elif sys.platform == "darwin":
    cflags = ["-O2"]
    cflags_omp = ["-O2"]
else:
    cflags = []
    cflags_omp = []

extensions = [
    Extension(
        "unclog.math",
        ["unclog/math.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=cflags_omp,
        extra_link_args=[],
    ),
    Extension("*", ["unclog/*.pyx"], extra_compile_args=cflags),
]

setup(
    ext_modules=cythonize(extensions),
)
