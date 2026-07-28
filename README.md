# cuProx

<div align="center">

**GPU-Accelerated First-Order LP/QP Solver**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Aminsed/cuprox/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.4+](https://img.shields.io/badge/CUDA-11.4+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

*A matrix-free GPU solver for sparse Linear and convex Quadratic Programs*

[Installation](#installation) •
[Quick Start](#quick-start) •
[Benchmarks](#benchmarks) •
[Contributing](#contributing)

</div>

---

## What is cuProx?

cuProx is a GPU-accelerated optimization solver for **Linear Programs (LP)** and **convex Quadratic Programs (QP)**. It uses first-order proximal methods (PDHG, ADMM) that are perfectly suited for GPU parallelization.

### Key Features

| Feature | Description |
|---------|-------------|
| **Matrix-free** | An iteration is a few sparse mat-vecs and some element-wise work, so its cost does not depend on the sparsity *pattern* |
| **Scope** | LP and convex QP |
| **Batch solving** | Sequential today (see [Known limitations](#known-limitations)) |
| **PyTorch** | A QP as a differentiable layer; gradients for the objective parameters `P` and `q` only |
| **Fallback** | CPU fallback via SciPy, which densifies — see [Known limitations](#known-limitations) |

Where it stands relative to other solvers depends heavily on the problem; see
[Benchmarks](#benchmarks) for what has actually been measured, and note that
NVIDIA cuOpt is the more mature GPU LP/QP solver.

### When to Use cuProx

**cuProx may suit you if:**
- Your `A` is sparse but *unstructured*, so a factorising solver fills in badly
- You want a QP inside a PyTorch model and can live with gradients only for
  the objective parameters
- Moderate accuracy is enough (1e-4 to 1e-6); first-order methods reach that
  quickly and tighten slowly

**Look elsewhere if:** your problem is small, banded or otherwise
well-structured — a factorising CPU solver such as OSQP will beat this
comfortably, as the benchmarks below show.

**Not recommended for:**
- Mixed-integer programming (use Gurobi, HiGHS)
- Very high precision (1e-10+, use interior-point)
- Small single problems (GPU overhead)
- Non-convex optimization

---

## Installation

```bash
pip install cuprox
```

The CUDA kernels are compiled for your GPU at install time, so a CUDA toolkit
has to be present. There is no prebuilt wheel, and that is deliberate: a wheel
without the compiled core imports cleanly and then fails on every solve, which
is exactly what the 0.1.0 release did.

**Requirements:** CUDA Toolkit 11.4+, CMake 3.24+, Python 3.9+, and a C++17
compiler. `nvcc` does not need to be on `PATH` -- the standard toolkit
locations are searched, and `CUDACXX` overrides that search. If no CUDA
compiler is found the build fails with instructions rather than quietly
producing a package that cannot solve.

From a checkout:

```bash
git clone https://github.com/Aminsed/cuprox.git
cd cuprox
pip install .
```

### Verify

```python
import cuprox
print(cuprox.__version__)
print(cuprox.__cuda_available__)   # True means the GPU core is present
```

See [INSTALL.md](INSTALL.md) for build options and troubleshooting.

---

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

# The algebraic Model API is linear-only: add_vars() returns a plain list,
# so a quadratic term cannot be written that way. QPs go through the matrix
# interface, which takes P directly.
result = cuprox.solve(
    P=Sigma,                              # quadratic term
    c=-mu,                                # linear term
    A=np.ones((1, n_assets)),             # budget constraint
    b=np.ones(1),
    constraint_senses=["=="],
    lb=np.zeros(n_assets),
)
print(f"Portfolio variance: {result.objective:.4f}")
```

---

## Examples

Four notebooks, each runnable end-to-end on a GPU. Every figure below was
produced by the notebook that names it, and every number quoted in the text is
checked with an `assert` rather than narrated — if a notebook runs, its claims
are true.

| Notebook | What it covers |
|---|---|
| [01 — Getting started](examples/01_getting_started.ipynb) | The API end to end, and how the sparsity pattern — not the problem size — decides whether the GPU wins |
| [02 — Differentiable optimization](examples/02_differentiable_optimization.ipynb) | A QP as a PyTorch layer, with gradients checked against the implicit Jacobian |
| [03 — Model predictive control](examples/03_model_predictive_control.ipynb) | Receding-horizon control of a double integrator, driven onto its input limits |

![Sparsity pattern decides the winner, not problem size](https://raw.githubusercontent.com/Aminsed/cuprox/main/examples/benchmark_sparsity.png)

![Double integrator under MPC, saturating the input constraint](https://raw.githubusercontent.com/Aminsed/cuprox/main/examples/mpc_double_integrator.png)

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

The only performance numbers published here are the ones a committed,
executable file produces. That file is
[examples/01_getting_started.ipynb](examples/01_getting_started.ipynb), which
compares cuProx against OSQP on the same problems, asserts its own results, and
writes the figure below.

![Sparsity pattern decides the winner, not problem size](https://raw.githubusercontent.com/Aminsed/cuprox/main/examples/benchmark_sparsity.png)

The finding is qualitative and robust: **at the same problem size, the sparsity
pattern decides the outcome.** With a banded `A`, cuProx is far *slower* than
OSQP. With an unstructured random `A` of identical dimensions, it is faster —
and the gap widens with size. Read the notebook for the numbers from a run you
can repeat.

Earlier revisions of this README quoted a table reaching 400,000 variables and
a "10-100x" headline. Neither is reproducible from anything in this repository,
so both have been removed rather than left standing.

The reason is structural rather than incidental. cuProx is matrix-free: a PDHG or
ADMM iteration costs a few sparse mat-vecs and some element-wise work, and that
cost is the same whatever the sparsity pattern looks like. OSQP factorises, which
is superb when the factor stays sparse and expensive when it fills in. So the
crossover is not really about problem size, it is about how badly the
factorisation fills in - which is why the two
cases in the figure above diverge so sharply at the same `n`.

`benchmarks/benchmark_qp.py` sweeps 100 to 5,000 variables. Whatever you
measure, make sure the GPU is otherwise idle — a competing process changes
these timings by more than any of the effects being measured.

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
- [ ] Batch solving — `solve_batch` loops in Python today. A batched PDHG
      kernel exists in the C++ tree but is not wired up.
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
