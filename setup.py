from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("curling.stone", ["src/stone.pyx"])
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
    extras_require={
        "test": [
            "pytest>=6.2",
            "pytest-timeout>=1.5",
        ],
    },
    packages={
        "src": "curling",
    }
)
