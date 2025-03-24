from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "active_contour",
        ["active_contour.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],  # Optimization level
    ),
    Extension(
        "hough_transform",
        ["hough_transform.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3)
)
