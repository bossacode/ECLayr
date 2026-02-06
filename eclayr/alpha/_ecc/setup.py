from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np
# import sys

# if sys.platform.startswith("win"):
#     openmp_arg = "/openmp"
# else:
#     openmp_arg = "-fopenmp"

Options.annotate = True

extensions = [
    # Extension(
    #     name="ecc",
    #     sources=["ecc.pyx"],
    #     extra_compile_args=[openmp_arg],
    #     extra_link_args=[openmp_arg],
    #     include_dirs=[np.get_include()]
    #     )
    Extension(
        name="ecc",
        sources=["ecc.pyx"],
        include_dirs=[np.get_include()]
        )
]

setup(
    name="ecc",
    ext_modules=cythonize(extensions, annotate=True)
)