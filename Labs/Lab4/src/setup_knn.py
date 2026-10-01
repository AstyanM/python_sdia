import os
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

os.environ["CC"] = "gcc"

extensions = [
    Extension("knn_cy", ["knn_cy.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="knn_cy",
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
