from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("curling.stone", ["curling/stone.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name="curling",
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
    packages=["curling"],
    package_dir={"curling": "curling"},
    zip_safe=False,
    setup_requires=['Cython'],
)
