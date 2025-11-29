# cuProx

<div align="center">

**GPU-Accelerated First-Order LP/QP Solver**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.4+](https://img.shields.io/badge/CUDA-11.4+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

*Solve large-scale Linear Programs and Quadratic Programs 10-100x faster on GPU*

[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Benchmarks](#benchmarks) •
[Contributing](#contributing)

</div>

---

## What is cuProx?

cuProx is a GPU-accelerated optimization solver for **Linear Programs (LP)** and **convex Quadratic Programs (QP)**. It uses first-order proximal methods (PDHG, ADMM) that are perfectly suited for GPU parallelization.


### When to Use cuProx

**Use cuProx for:**
- Large-scale LP/QP (100K+ variables)
- Batch solving (many small problems)
- Real-time optimization (MPC, trading)
- ML training with optimization layers
- Moderate accuracy requirements (1e-4 to 1e-6)

**Don't use cuProx for:**
- Mixed-integer programming (use Gurobi, HiGHS)
- Very high precision (1e-10+, use interior-point)
- Small single problems (GPU overhead)
- Non-convex optimization

---

## Installation

### From PyPI (Recommended)

```bash
pip install cuprox
```

This automatically installs the right version for your CUDA setup.

### Requirements

- Python 3.9+
- NVIDIA GPU (Volta or newer: V100, RTX 2080+, T4, A100, etc.)
- CUDA 11.4+ (auto-detected)

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

### Example 3: Batch Solving (1000 LPs in Parallel)

```python
import cuprox
import numpy as np

# Generate 1000 small LP problems
problems = []
for i in range(1000):
    n, m = 100, 50
    problems.append({
        "c": np.random.randn(n),
        "A": sparse.random(m, n, density=0.1, format='csr'),
        "b": np.random.rand(m),
        "lb": np.zeros(n),
    })

# Solve ALL in parallel on GPU
results = cuprox.solve_batch(problems)

# All 1000 solved in ~100ms (vs ~10s sequential)
print(f"Solved {len(results)} problems")
print(f"All optimal: {all(r.status == 'optimal' for r in results)}")
```

### Example 4: Quadratic Program (Portfolio Optimization)

```python
import cuprox
import numpy as np

# Markowitz portfolio optimization
# minimize (1/2) x' Σ x - μ' x
# subject to: sum(x) = 1, x >= 0

n_assets = 1000
mu = np.random.rand(n_assets)  # Expected returns
Sigma = np.random.rand(n_assets, n_assets)
Sigma = Sigma @ Sigma.T + np.eye(n_assets)  # Covariance (PSD)

model = cuprox.Model()
x = model.add_vars(n_assets, lb=0, name="weight")

# Quadratic objective
model.minimize(0.5 * x @ Sigma @ x - mu @ x)

# Budget constraint
model.add_constr(sum(x) == 1)

result = model.solve()
print(f"Portfolio variance: {result.objective:.4f}")
```

---

## Solver Parameters

```python
result = model.solve(params={
    # Convergence
    "tolerance": 1e-6,        # Primal/dual residual tolerance
    "max_iterations": 100000, # Maximum iterations
    "time_limit": 3600.0,     # Time limit in seconds
    
    # Algorithm
    "scaling": "ruiz",        # "ruiz", "geometric", or "none"
    "restart": "adaptive",    # "adaptive", "fixed", or "none"
    
    # Precision
    "precision": "float64",   # "float32" (faster) or "float64" (accurate)
    
    # Device
    "device": "auto",         # "auto", "gpu", or "cpu"
    "verbose": True,          # Print iteration log
})
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

## How It Works

cuProx uses **Primal-Dual Hybrid Gradient (PDHG)** for LP and **ADMM** for QP. These are first-order methods where every operation is GPU-friendly:

```
PDHG Iteration (LP):
  y ← project(y + σ(Ax̄ - b))     # Sparse matrix-vector: GPU-perfect
  x ← project(x - τ(c + Aᵀy))    # Sparse matrix-vector: GPU-perfect
  x̄ ← 2x - x_prev               # Element-wise: GPU-perfect
```

Unlike interior-point methods (which require Cholesky factorization — poorly parallelizable), PDHG is embarrassingly parallel.

---

## Comparison with Other Solvers

| Feature | cuProx | Gurobi | HiGHS | OSQP | SCS |
|---------|--------|--------|-------|------|-----|
| GPU acceleration | ✅ Full | ⚠️ Limited | ❌ | ❌ | ❌ |
| Batch solving | ✅ Native | ❌ | ❌ | ❌ | ❌ |
| pip install | ✅ | ✅ ($$$) | ✅ | ✅ | ✅ |
| LP support | ✅ | ✅ | ✅ | ❌ | ✅ |
| QP support | ✅ | ✅ | ❌ | ✅ | ✅ |
| MIP support | ❌ | ✅ | ✅ | ❌ | ❌ |
| Open source | ✅ MIT | ❌ | ✅ | ✅ | ✅ |

---

## API Reference

### Model Class

```python
class Model:
    def add_var(lb=0, ub=inf, name=None) -> Variable
    def add_vars(count, lb=0, ub=inf) -> List[Variable]
    def add_constr(constraint, name=None) -> Constraint
    def minimize(expr) -> None
    def maximize(expr) -> None
    def solve(params=None, warm_start=None) -> SolveResult
```

### Solve Functions

```python
def solve(c, A, b, lb=None, ub=None, P=None, params=None) -> SolveResult
def solve_batch(problems, params=None) -> List[SolveResult]
```

### SolveResult

```python
@dataclass
class SolveResult:
    status: str           # "optimal", "infeasible", "unbounded", etc.
    objective: float      # Optimal objective value
    x: np.ndarray        # Primal solution
    y: np.ndarray        # Dual solution
    iterations: int       # Number of iterations
    solve_time: float     # Wall clock time (seconds)
```

---

## Roadmap

- [x] LP solver (PDHG)
- [x] QP solver (ADMM)
- [x] Batch solving

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/Aminsed/cuprox.git
cd cuprox
pip install -e ".[dev]"
pytest tests/
```

---

## Citation

If you use cuProx in your research, please cite:

```bibtex
@software{cuprox2024,
  title = {cuProx: GPU-Accelerated First-Order LP/QP Solver},
  year = {2025},
  url = {https://github.com/Aminsed/cuprox}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for the optimization community**

[Report Bug](https://github.com/Aminsed/cuprox/issues) •
[Request Feature](https://github.com/Aminsed/cuprox/issues) •
[Discussions](https://github.com/Aminsed/cuprox/discussions)

</div>

