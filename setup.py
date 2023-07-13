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
        "docs": [
            "mkdocs==1.4.3",
            "mkdocs-material==7.3.3",
            "mkdocstrings==0.18.0",
            "mkdocs-git-revision-date-plugin",
            "mkdocs-mermaid2-plugin==0.6.0",
            "mkdocs-include-markdown-plugin==2.0.0",
        ]
    },
    package_dir={
        "curling": "src",
    }
)
