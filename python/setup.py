"""
Minimal setup.py for installing cuProx Python package without C++ build.

This is used when:
1. CUDA is not available
2. User just wants the CPU fallback
3. CI systems without CUDA

For full installation with GPU support, use:
    pip install cuprox

For CPU-only installation:
    cd python && pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="cuprox",
    version="0.1.0",
    description="GPU-accelerated LP/QP solver (Python-only fallback)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "hypothesis>=6.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
    },
)
