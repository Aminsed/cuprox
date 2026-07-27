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

### Key Features

| Feature | Description |
|---------|-------------|
| **Fast** | 10-100x speedup over CPU solvers on large problems |
| **Focused** | LP and QP only — does one thing exceptionally well |
| **Batch solving** | Sequential today (see Known limitations) |
| **ML-Ready** | PyTorch integration for differentiable optimization |
| **Fallback** | Automatic CPU fallback if no GPU available |

### When to Use cuProx

**Use cuProx for:**
- Large-scale LP/QP (100K+ variables)
- Batch solving (many small problems)
- Real-time optimization (MPC, trading)
- ML training with optimization layers
- Moderate accuracy requirements (1e-4 to 1e-6)

**Not recommended for:**
- Mixed-integer programming (use Gurobi, HiGHS)
- Very high precision (1e-10+, use interior-point)
- Small single problems (GPU overhead)
- Non-convex optimization

---

## Installation

cuProx is built from source to ensure optimal performance for your specific hardware. See [INSTALL.md](INSTALL.md) for detailed instructions.

### Quick Start (GPU Build)

**Prerequisites:**
- CUDA Toolkit 11.4+
- CMake 3.24+
- Python 3.9+
- C++ compiler (GCC 7+ or Clang)

```bash
# Clone the repository
git clone https://github.com/Aminsed/cuprox.git
cd cuprox

# Build and install in one step, from the repository root
pip install .
```

That single command builds the CUDA library and the extension and installs them
together. Installing from `python/` instead will *not* build the extension: the
package will import, report `__cuda_available__ == False`, and quietly fall back
to SciPy.

### Quick Start (CPU Only)

For development or systems without CUDA:

```bash
git clone https://github.com/Aminsed/cuprox.git
cd cuprox
pip install -e python/
```

### Verify Installation

```python
import cuprox
print(f"cuProx version: {cuprox.__version__}")
print(f"CUDA available: {cuprox.__cuda_available__}")  # True = GPU ready!
```

For comprehensive installation instructions including troubleshooting, see [INSTALL.md](INSTALL.md).

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

### Example 3: Solving Many Problems

```python
import cuprox
import numpy as np
from scipy import sparse

problems = [
    {"c": np.random.randn(100), "A": sparse.random(50, 100, density=0.1, format="csr"),
     "b": np.random.rand(50), "lb": np.zeros(100)}
    for _ in range(1000)
]

results = cuprox.solve_batch(problems)
```

`solve_batch` solves these **one after another**. See
[Known limitations](#known-limitations).

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

## Example Gallery

Each core notebook now ships with a couple of “hero” visuals. Browse the highlights below (all generated directly from the notebooks).

### Notebook 01 — Differentiable QP Layers (PyTorch)

![OptNet Decision-Focused Training](examples/pytorch_decision_focused.png)
- Decision-focused OptNet stack trained end-to-end on GPU, showing large downstream gains.

![QP Gradient Flow](examples/qp_gradient_flow.png)
- Visualizes the implicit Jacobian used for stable backpropagation through QP layers.

### Notebook 02 — Multi-Period Portfolio Optimization

![Efficient Frontier & Capital Market Line](examples/portfolio_frontier.png)
- Proper Markowitz frontier with capital market line, turnover limits, and transaction costs.

![Rolling Backtest Diagnostics](examples/portfolio_backtest.png)
- Realised returns, drawdowns, and turnover over a multi-year simulation.

### Notebook 03 — GPU MPC (Shooting Form)

![Extreme Racetrack Trajectory](examples/mpc_racing.png)
- 440-variable shooting MPC with centimetre-level tracking error.

![Disturbance Rejection Replanning](examples/mpc_disturbance.png)
- 1 kHz replanning loop that absorbs injected velocity impulses while respecting bounds.

### Notebook 04 — Stochastic Programming at Scale

![Energy Portfolio under Renewable Uncertainty](examples/stochastic_energy.png)
- Two-stage SAA model allocating gas, solar, and wind with storage and penalty costs.

![CVaR Frontier](examples/stochastic_cvar.png)
- Risk-averse dispatch showing how CVaR tightening shifts the optimal generation mix.

### Notebook 05 — Finance Stress Lab

![Regime-Aware Frontier](examples/finance_frontier.png)
- Multi-period frontier with regime switching, leverage caps, and borrowing costs.

![GPU vs CPU Stress Benchmarks](examples/finance_benchmark.png)
- Monte Carlo stress testing across thousands of scenarios.

### Notebook 06 — Learn to Race: GPU-Accelerated Racing AI

Complex track with walls, obstacles, and crash physics. Neural policy learns via imitation learning with DAgger.

![Track Layout](examples/track_layout.png)
- Racing track with variable-width walls and strategic obstacles.

![Training Progress](examples/training_progress.gif)
- Training loss convergence with DAgger iterations adding 200K+ training samples.

![Head-to-Head Race](examples/racing_head2head.gif)
- Expert (green) vs learned (pink) racing side-by-side with real-time speed and distance display.

![Control Signals Comparison](examples/controls_comparison.gif)
- Acceleration and steering commands comparison between expert and learned policy.

**Training:** 800+ epochs with DAgger, 200K samples, 803K model parameters.

---

## Solver Parameters

```python
result = model.solve(params={
    # Convergence
    "tolerance": 1e-6,        # Primal/dual residual tolerance
    "max_iterations": 100000, # Maximum iterations
    
    # Device
    "device": "gpu",          # "gpu" (default) or "cpu" to force the fallback
    "verbose": True,          # Print an iteration log
})
```

---

## Benchmarks

Measured on an NVIDIA RTX A6000 against OSQP on a 24-thread i9-12900K, QPs with
`m = n/2`, tolerance `1e-4`, best of 3 runs. Every row was checked against OSQP's
objective; relative disagreement stayed between `1e-7` and `1e-8` throughout.

The result depends strongly on **sparsity pattern**, so both cases are given.

| n | banded `A` | random `A` |
|--:|-----------:|-----------:|
| 1,000 | 0.02x | 0.54x |
| 5,000 | 0.11x | **62x** |
| 20,000 | 0.42x | - |
| 100,000 | **2.05x** | - |
| 400,000 | **3.56x** | - |

Reading it honestly: cuProx is *slower* than a good CPU solver on small or
well-structured problems, and much faster on large or awkwardly-structured ones.

The reason is structural rather than incidental. cuProx is matrix-free: a PDHG or
ADMM iteration costs a few sparse mat-vecs and some element-wise work, and that
cost is the same whatever the sparsity pattern looks like. OSQP factorises, which
is superb when the factor stays sparse and expensive when it fills in. So the
crossover is not really about problem size, it is about how badly the
factorisation fills in - which is why the two columns above diverge so sharply at
the same `n`.

Reproduce with `python benchmarks/benchmark_qp.py`. Note that the GPU must be
otherwise idle: benchmarking against a busy GPU inflated these numbers by 20-40x
in early runs.

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

## Known limitations

- **Batch solving is sequential.** A batched PDHG kernel exists in the C++
  sources but is not exposed. Its sparse operator applied the shared matrix in a
  per-problem loop that allocated scratch inside the iteration, making it slower
  than solving the problems individually; replacing that with a single
  `cusparseSpMM` over the batch is the right fix but does not yet pass
  validation. It is left unexposed rather than shipped unverified.
- **ADMM does not equilibrate.** OSQP applies Ruiz scaling by default; cuProx
  does not for QP. Badly scaled problems can stall, and the fix is to scale the
  problem before handing it over. PDHG does support Ruiz scaling.
- **Small problems are slower than a CPU solver.** Below roughly 10,000
  variables a good CPU solver usually wins; see [Benchmarks](#benchmarks).
- **Moderate accuracy.** First-order methods target `1e-4` to `1e-8`. For tighter
  tolerances use an interior-point solver.

---

## Comparison with Other Solvers

| Feature | cuProx | Gurobi | HiGHS | OSQP | SCS |
|---------|--------|--------|-------|------|-----|
| GPU acceleration | Yes (Full) | Limited | No | No | No |
| Batch solving | Yes (Native) | No | No | No | No |
| LP support | Yes | Yes | Yes | No | Yes |
| QP support | Yes | Yes | No | Yes | Yes |
| MIP support | No | Yes | Yes | No | No |
| Open source | Yes (MIT) | No | Yes | Yes | Yes |

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
- [x] CPU fallback
- [x] PyTorch autograd integration
- [ ] Windows support
- [ ] Multi-GPU support
- [ ] SOCP extension

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/Aminsed/cuprox.git
cd cuprox

# Build C++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUPROX_BUILD_TESTS=ON
make -j$(nproc)

# Editable install, from the repository root
pip install -e ".[dev]"

# Run tests (the Python suite needs a GPU and the installed extension)
pytest tests/python/
ctest --test-dir build --output-on-failure
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

**Built for the optimization community**

[Report Bug](https://github.com/Aminsed/cuprox/issues) •
[Request Feature](https://github.com/Aminsed/cuprox/issues) •
[Discussions](https://github.com/Aminsed/cuprox/discussions)

</div>
