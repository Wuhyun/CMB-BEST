from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import os



src_files=["cmbbest.pyx", "tetrapyd.c", "arrays.c"]

ext = Extension("cmbbest",
        sources = ["src/" + src_file for src_file in src_files],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        )

module = cythonize(ext,
                   compiler_directives={"language_level": "3"},
                   build_dir="build",
                   )

setup(
    name="cmbbest",
    version="1.0",
    description="CMB bispectrum estimator for constraining primordial non-Gaussianity",
    author="Wuhyun Sohn",
    url="https://github.com/Wuhyun/CMB-BEST",
    author_email="wuhyun@kasi.re.kr",
    ext_modules=module
)
