# cuProx Installation Guide

This guide covers how to build and install cuProx from source on your local machine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Install (CPU-only)](#quick-install-cpu-only)
- [Full Install with GPU Support](#full-install-with-gpu-support)
- [Development Setup](#development-setup)
- [Verify Installation](#verify-installation)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required (All Builds)

| Component | Version | Check Command |
|-----------|---------|---------------|
| Python | 3.9+ | `python --version` |
| pip | Latest | `pip --version` |
| NumPy | 1.20+ | `python -c "import numpy; print(numpy.__version__)"` |
| SciPy | 1.7+ | `python -c "import scipy; print(scipy.__version__)"` |

### Required (GPU Builds Only)

| Component | Version | Check Command |
|-----------|---------|---------------|
| CUDA Toolkit | 11.4+ | `nvcc --version` |
| CMake | 3.24+ | `cmake --version` |
| C++ Compiler | GCC 7+ / Clang 8+ | `gcc --version` or `clang --version` |
| NVIDIA GPU | Volta+ (Compute 7.0+) | `nvidia-smi` |

### Installing Prerequisites

**Ubuntu/Debian:**

```bash
# Install build tools
sudo apt update
sudo apt install -y build-essential cmake git

# Install Python dependencies
pip install --upgrade pip numpy scipy
```

**CUDA Toolkit (if building with GPU support):**

```bash
# Check if CUDA is already installed
nvcc --version

# If not, install CUDA Toolkit from NVIDIA:
# https://developer.nvidia.com/cuda-downloads

# Or via apt on Ubuntu (example for CUDA 12.x):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit
```

---

## Quick Install (CPU-only)

For development, testing, or systems without NVIDIA GPUs:

```bash
# 1. Clone the repository
git clone https://github.com/Aminsed/cuprox.git
cd cuprox

# 2. Install the Python package
pip install -e python/

# 3. Verify installation
python -c "import cuprox; print(f'Version: {cuprox.__version__}')"
```

This installs cuProx in "editable" mode using only the Python fallback solver (no GPU acceleration).

---

## Full Install with GPU Support

For GPU-accelerated performance:

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aminsed/cuprox.git
cd cuprox
```

### Step 2: Build the C++ Library

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (use all available cores)
make -j$(nproc)

# Return to project root
cd ..
```

**Build Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type: `Release`, `Debug`, `RelWithDebInfo` |
| `CUPROX_BUILD_TESTS` | OFF | Build C++ unit tests |
| `CUPROX_BUILD_PYTHON` | ON | Build Python bindings |
| `CUPROX_BUILD_BENCHMARKS` | OFF | Build benchmarks |

Example with options:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUPROX_BUILD_TESTS=ON \
  -DCUPROX_BUILD_PYTHON=ON
```

### Step 3: Install the Python Package

```bash
# From project root
pip install -e python/
```

### Step 4: Verify GPU Support

```python
import cuprox
print(f"cuProx version: {cuprox.__version__}")
print(f"CUDA available: {cuprox.__cuda_available__}")  # Should be True
```

---

## Development Setup

For contributors and developers:

### Step 1: Clone and Build

```bash
# Clone the repository
git clone https://github.com/Aminsed/cuprox.git
cd cuprox

# Build C++ library with tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUPROX_BUILD_TESTS=ON
make -j$(nproc)
cd ..
```

### Step 2: Install with Development Dependencies

```bash
pip install -e "python/[dev]"
```

This installs additional tools:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `ruff` - Fast linter
- `mypy` - Type checker

### Step 3: Run Linters

```bash
# Check formatting
black --check python/ tests/python/

# Run linter
ruff check python/ tests/python/

# Type checking (optional, building up types)
mypy python/cuprox --ignore-missing-imports
```

### Step 4: Run Tests

```bash
# Python tests
pytest tests/python/ -v

# C++ tests (if built with CUPROX_BUILD_TESTS=ON)
cd build
ctest --output-on-failure
cd ..
```

---

## Verify Installation

Run this script to verify your installation:

```python
import cuprox
import numpy as np
from scipy import sparse

# Print version info
print(f"cuProx version: {cuprox.__version__}")
print(f"CUDA available: {cuprox.__cuda_available__}")

# Run a simple test
n, m = 100, 50
A = sparse.random(m, n, density=0.1, format='csr')
b = np.random.rand(m)
c = np.random.randn(n)

result = cuprox.solve(c=c, A=A, b=b, lb=np.zeros(n))

print(f"Status: {result.status}")
print(f"Solve time: {result.solve_time:.4f}s")
print(f"Iterations: {result.iterations}")

if result.status == "optimal":
    print("✓ Installation verified successfully!")
else:
    print("✗ Test did not converge - check installation")
```

Save this as `verify_install.py` and run:

```bash
python verify_install.py
```

---

## Running Tests

### Python Tests

```bash
# Run all tests
pytest tests/python/ -v

# Run specific test file
pytest tests/python/test_solver_lp.py -v

# Run tests with coverage
pytest tests/python/ --cov=python/cuprox --cov-report=term-missing

# Skip GPU tests (if no GPU available)
pytest tests/python/ -v -m "not gpu"

# Run only fast tests
pytest tests/python/ -v -m "not slow"
```

### C++ Tests

```bash
# Build with tests enabled
cd build
cmake .. -DCUPROX_BUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Run specific test
./tests/cpp/test_admm
```

---

## Troubleshooting

### CUDA Not Detected

If `cuprox.__cuda_available__` is `False` after building:

1. **Check CUDA installation:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Ensure CUDA is in PATH:**
   ```bash
   echo $CUDA_HOME
   echo $PATH | grep cuda
   ```

3. **Set CUDA_HOME if needed:**
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

4. **Rebuild:**
   ```bash
   rm -rf build
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

### Build Fails with CMake Errors

1. **Update CMake:**
   ```bash
   pip install --upgrade cmake
   # or
   sudo apt install cmake
   ```

2. **Check CMake version:**
   ```bash
   cmake --version  # Need 3.24+
   ```

3. **Clear CMake cache:**
   ```bash
   rm -rf build/CMakeCache.txt build/CMakeFiles
   ```

### Import Errors

1. **Check Python environment:**
   ```bash
   which python
   pip list | grep cuprox
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install --upgrade numpy scipy
   ```

3. **Reinstall cuprox:**
   ```bash
   pip uninstall cuprox
   pip install -e python/
   ```

### GPU Out of Memory

1. **Reduce problem size** for testing
2. **Check GPU memory:**
   ```bash
   nvidia-smi
   ```
3. **Use CPU fallback:**
   ```python
   result = model.solve(params={"device": "cpu"})
   ```

### Slow Performance

1. **Verify GPU is being used:**
   ```python
   print(f"CUDA available: {cuprox.__cuda_available__}")
   ```

2. **Check GPU utilization during solve:**
   ```bash
   watch -n 0.5 nvidia-smi
   ```

3. **Ensure Release build:**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

---

## Uninstall

To remove cuProx:

```bash
pip uninstall cuprox
rm -rf build/
```

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

Recommended platform. Full GPU support.

### macOS

- GPU support not available (no CUDA on macOS)
- CPU fallback works fully
- Use `clang` as the compiler

### Windows

- Experimental support
- Requires Visual Studio 2019+ with C++ tools
- CUDA Toolkit for Windows required

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

---

For more help, please [open an issue](https://github.com/Aminsed/cuprox/issues) on GitHub.
