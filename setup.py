from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext = Extension("cmbbest",
        sources=["cmbbest.pyx", "gl_integration.c", "tetrapyd.c", "arrays.c"],
        include_dirs=[numpy.get_include()],
        libraries=["gsl", "gslcblas"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        )

module = cythonize(ext, compiler_directives={"language_level": "3"}, annotate=True)

setup(ext_modules=module)
