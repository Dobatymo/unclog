from setuptools import setup, Extension
from Cython.Build import build_ext
import numpy as np

setup(
	name = 'unclog',
	version = '0.0.1',
	ext_modules=[
		Extension('unclog',
			sources=['unclog.pyx'],
			include_dirs = [np.get_include()],
			extra_compile_args=['/openmp'],
			extra_link_args=['/openmp'],
		)
	],
	cmdclass = {'build_ext': build_ext}
)
