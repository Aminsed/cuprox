"""
Minimal setup.py for installing cuProx Python package without C++ build.

This is used when:
1. CUDA is not available
2. User wants the CPU fallback for development
3. CI systems without CUDA

For full installation with GPU support, first build the C++ library:
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

Then install the Python package:
    pip install -e python/
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
