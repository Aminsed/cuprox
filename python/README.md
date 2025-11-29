# cuProx

<div align="center">

**GPU-Accelerated First-Order LP/QP Solver**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.4+](https://img.shields.io/badge/CUDA-11.4+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

*Solve large-scale Linear Programs and Quadratic Programs 10-100x faster on GPU*

</div>

---

## What is cuProx?

cuProx is a GPU-accelerated optimization solver for **Linear Programs (LP)** and **convex Quadratic Programs (QP)**. It uses first-order proximal methods (PDHG, ADMM) that are perfectly suited for GPU parallelization.

### Key Features

| Feature | Description |
|---------|-------------|
| **Fast** | 10-100x speedup over CPU solvers on large problems |
| **Focused** | LP and QP only — does one thing exceptionally well |
| **Batch Solving** | Solve 1000s of problems in parallel (unique capability) |
| **ML-Ready** | PyTorch integration for differentiable optimization |
| **Fallback** | Automatic CPU fallback if no GPU available |

---

## Installation

This is the Python package for cuProx. For complete build instructions, see the main repository's [INSTALL.md](../INSTALL.md).

### Quick Install (CPU-only / Development)

```bash
# From the project root directory
pip install -e python/

# Or with development dependencies
pip install -e "python/[dev]"
```

### Full Install with GPU Support

For GPU acceleration, you must first build the C++ library:

```bash
# From the project root directory
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Then install the Python package
cd ..
pip install -e python/
```

### Verify Installation

```python
import cuprox
print(f"cuProx version: {cuprox.__version__}")
print(f"CUDA available: {cuprox.__cuda_available__}")
```

---

## Quick Start

### Example 1: Simple LP

```python
import cuprox

# Create model
model = cuprox.Model()

# Add variables (x, y >= 0)
x = model.add_var(lb=0, name="x")
y = model.add_var(lb=0, name="y")

# Add constraints
model.add_constr(x + 2*y <= 20)
model.add_constr(3*x + y <= 30)

# Minimize objective
model.minimize(-5*x - 4*y)

# Solve
result = model.solve()

print(f"Status: {result.status}")
print(f"Optimal objective: {result.objective:.2f}")
print(f"x = {result.get_value(x):.2f}")
print(f"y = {result.get_value(y):.2f}")
```

### Example 2: Large-Scale LP (Matrix Form)

```python
import cuprox
import numpy as np
from scipy import sparse

# Problem: 100K variables, 50K constraints
n, m = 100_000, 50_000

# Random sparse problem
A = sparse.random(m, n, density=0.001, format='csr')
b = np.random.rand(m)
c = np.random.randn(n)

# Solve
result = cuprox.solve(c=c, A=A, b=b, lb=np.zeros(n))

print(f"Solved in {result.solve_time:.3f} seconds")
print(f"Iterations: {result.iterations}")
```

---

## Benchmarks

Performance on an NVIDIA RTX A6000 (48GB):

| Problem | Size | SciPy (CPU) | cuProx (GPU) | Speedup |
|---------|------|-------------|--------------|---------|
| Netlib pilot4 | 410 × 1123 | 50 ms | 10 ms | **5x** |
| pds-20 | 33K × 108K | 30 s | 2 s | **15x** |
| Random LP | 1M × 500K | 5 min | 20 s | **15x** |
| Portfolio QP | 1000 × 1000 | 100 ms | 5 ms | **20x** |
| Batch 10K LP | 100 × 50 each | 60 s | 0.5 s | **120x** |

*Batch solving is where cuProx truly shines — no other solver offers this.*

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the optimization community**

[GitHub](https://github.com/Aminsed/cuprox) •
[Report Bug](https://github.com/Aminsed/cuprox/issues) •
[Request Feature](https://github.com/Aminsed/cuprox/issues)

</div>
