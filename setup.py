from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("curling.stone", ["curling/stone.pyx"])
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
