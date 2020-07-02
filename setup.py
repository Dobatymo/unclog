from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
	Extension("unclog.math", ["unclog/math.pyx"],
		include_dirs=[np.get_include()],
		extra_compile_args=["/O2", "/openmp"],
		extra_link_args=[]
	),
	Extension("*", ["unclog/*.pyx"],
		extra_compile_args=["/O2"]
	),
]

setup(
	name="unclog",
	packages=["unclog"],
	version="0.0.1",
	ext_modules=cythonize(extensions),
)
