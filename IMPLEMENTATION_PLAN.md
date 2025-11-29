# cuProx Implementation Plan
## GPU-Accelerated First-Order LP/QP Solver

**Target:** Production-quality, pip-installable solver for NVIDIA GPUs (Volta+)  
**Timeline:** 6-9 months | **Approach:** Test-Driven Development  
**Philosophy:** Do one thing exceptionally well — fast proximal methods on GPU

---

### Related Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Public-facing overview, installation, quick start |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data structures, kernel design, API |
| [RESEARCH.md](RESEARCH.md) | Mathematical foundations, algorithms, convergence |
| **IMPLEMENTATION_PLAN.md** | *This document* — Sprint breakdown, TDD strategy, CI/CD |

---

### Why "cuProx"?
- **cu** = CUDA (NVIDIA GPU acceleration)
- **Prox** = Proximal methods (PDHG, ADMM — mathematically principled first-order algorithms)

cuProx solves large-scale Linear Programs (LP) and convex Quadratic Programs (QP) using GPU-accelerated proximal splitting methods. It does NOT attempt to replace general-purpose solvers like Gurobi — instead, it excels at problems where first-order methods shine: **large, sparse, moderate-accuracy problems**.

### Definition of Done (DoD)

A feature is "done" when:
- [ ] All tests pass (unit + integration)
- [ ] Test coverage ≥ 90% for new code
- [ ] Code reviewed and approved
- [ ] Documentation updated
- [ ] No linter warnings
- [ ] Performance benchmarked (if applicable)
- [ ] GPU memory properly managed (no leaks)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Hardware & Software Compatibility](#2-hardware--software-compatibility)
3. [Project Structure](#3-project-structure)
4. [Development Phases](#4-development-phases)
5. [Test-Driven Development Strategy](#5-test-driven-development-strategy)
6. [Build System & Packaging](#6-build-system--packaging)
7. [CI/CD Pipeline](#7-cicd-pipeline)
8. [API Design](#8-api-design)
9. [Sprint Breakdown](#9-sprint-breakdown)
10. [Risk Mitigation](#10-risk-mitigation)

---

## 1. Project Overview

### 1.1 Scope (What We Build)
| Component | Description | Priority |
|-----------|-------------|----------|
| LP Solver (PDHG) | First-order primal-dual method for linear programming | P0 |
| QP Solver (ADMM) | Operator-splitting for convex quadratic programming | P1 |
| Batch Solver | Solve thousands of small LPs in parallel | P2 |
| CPU Fallback | Automatic fallback when no GPU available | P1 |

### 1.2 Out of Scope (What We Don't Build)
- Mixed-integer programming (MIP)
- Simplex method
- Interior-point/barrier methods
- Non-convex optimization
- Multi-GPU support (future version)

### 1.3 Success Criteria
- [ ] 10x+ speedup over SciPy on LP with 100K+ variables
- [ ] Works on CUDA 11.x and 12.x
- [ ] Works on GPUs from Volta (SM 70) to Blackwell (SM 100)
- [ ] `pip install cuprox` works on Linux (manylinux2014)
- [ ] 90%+ test coverage
- [ ] Documentation with examples

---

## 2. Hardware & Software Compatibility

### 2.1 Supported GPU Architectures
| Architecture | Compute Capability | GPUs | Status |
|--------------|-------------------|------|--------|
| Volta | SM 7.0, 7.2 | V100, Titan V | Supported |
| Turing | SM 7.5 | RTX 2080, T4 | Supported |
| Ampere | SM 8.0, 8.6, 8.7 | A100, RTX 3090, A6000 | Primary Target |
| Ada Lovelace | SM 8.9 | RTX 4090, L40 | Supported |
| Hopper | SM 9.0 | H100 | Supported |
| Blackwell | SM 10.0 | B100, B200 | Future |

### 2.2 CUDA Version Strategy
```
Minimum: CUDA 11.4 (for broad compatibility)
Recommended: CUDA 12.x (for latest features)
Build Matrix: 11.8, 12.1, 12.4 (3 wheel variants)
```

### 2.3 Operating System Support
| OS | Support Level | Notes |
|----|--------------|-------|
| Linux (glibc 2.17+) | Full | Primary target, manylinux2014 wheels |
| Windows 10/11 | Planned v1.1 | MSVC build, separate wheels |
| macOS | CPU-only | No NVIDIA GPU support |

### 2.4 Python Version Support
```
Minimum: Python 3.9
Maximum: Python 3.12
Build Matrix: 3.9, 3.10, 3.11, 3.12
```

---

## 3. Project Structure

```
cuprox/
├── CMakeLists.txt              # Root CMake configuration
├── pyproject.toml              # Python package configuration
├── setup.py                    # Fallback setup (if needed)
├── LICENSE                     # MIT License
├── README.md                   # Project overview
├── IMPLEMENTATION_PLAN.md      # This document
│
├── cmake/
│   ├── FindCUDAToolkit.cmake   # CUDA detection
│   ├── CUDAArchitectures.cmake # SM version handling
│   └── Dependencies.cmake      # External dependencies
│
├── src/                        # C++/CUDA source code
│   ├── cuprox/
│   │   ├── core/
│   │   │   ├── types.hpp           # Common type definitions
│   │   │   ├── error.hpp           # Error handling (CUDA + logic)
│   │   │   ├── error.cpp
│   │   │   ├── memory.cuh          # GPU memory management
│   │   │   ├── memory.cu
│   │   │   ├── sparse_matrix.cuh   # CSR sparse matrix
│   │   │   ├── sparse_matrix.cu
│   │   │   ├── dense_vector.cuh    # GPU vector operations
│   │   │   └── dense_vector.cu
│   │   │
│   │   ├── linalg/
│   │   │   ├── spmv.cuh            # Sparse matrix-vector multiply
│   │   │   ├── spmv.cu
│   │   │   ├── blas.cuh            # BLAS wrappers (axpy, dot, nrm2)
│   │   │   ├── blas.cu
│   │   │   ├── projections.cuh     # Box/cone projections
│   │   │   └── projections.cu
│   │   │
│   │   ├── preprocess/
│   │   │   ├── scaling.cuh         # Ruiz equilibration
│   │   │   ├── scaling.cu
│   │   │   ├── validation.hpp      # Input validation
│   │   │   └── validation.cpp
│   │   │
│   │   ├── solvers/
│   │   │   ├── solver_base.hpp     # Abstract solver interface
│   │   │   ├── pdhg.cuh            # PDHG LP solver
│   │   │   ├── pdhg.cu
│   │   │   ├── admm.cuh            # ADMM QP solver
│   │   │   ├── admm.cu
│   │   │   ├── batch_pdhg.cuh      # Batch LP solver
│   │   │   └── batch_pdhg.cu
│   │   │
│   │   ├── utils/
│   │   │   ├── timer.hpp           # Performance timing
│   │   │   ├── logging.hpp         # Debug logging
│   │   │   └── cuda_utils.cuh      # CUDA helpers
│   │   │
│   │   └── cpu_fallback/
│   │       ├── cpu_pdhg.hpp        # CPU implementation
│   │       └── cpu_pdhg.cpp
│   │
│   └── bindings/
│       ├── python_module.cpp       # pybind11 main module
│       ├── py_model.cpp            # Model class bindings
│       ├── py_solver.cpp           # Solver bindings
│       └── py_result.cpp           # Result bindings
│
├── python/
│   └── cuprox/
│       ├── __init__.py             # Package init, version
│       ├── model.py                # User-facing Model class
│       ├── solver.py               # Solver interface
│       ├── result.py               # Solution result class
│       ├── exceptions.py           # Custom exceptions
│       ├── _core.pyi               # Type stubs for C++ module
│       └── utils/
│           ├── __init__.py
│           ├── sparse.py           # Sparse matrix utilities
│           └── validation.py       # Python-side validation
│
├── tests/
│   ├── cpp/                        # C++ unit tests (Google Test)
│   │   ├── CMakeLists.txt
│   │   ├── test_sparse_matrix.cpp
│   │   ├── test_vector_ops.cpp
│   │   ├── test_spmv.cpp
│   │   ├── test_projections.cpp
│   │   ├── test_scaling.cpp
│   │   ├── test_pdhg.cpp
│   │   ├── test_admm.cpp
│   │   └── fixtures/
│   │       ├── small_lp.hpp        # Test problem generators
│   │       └── netlib_loader.hpp   # Netlib problem loader
│   │
│   ├── python/                     # Python tests (pytest)
│   │   ├── conftest.py             # Pytest fixtures
│   │   ├── test_model.py
│   │   ├── test_solver_lp.py
│   │   ├── test_solver_qp.py
│   │   ├── test_batch.py
│   │   ├── test_edge_cases.py
│   │   ├── test_cpu_fallback.py
│   │   └── benchmarks/
│   │       ├── bench_netlib.py
│   │       └── bench_random.py
│   │
│   └── data/                       # Test data files
│       ├── netlib/                 # Netlib LP problems (MPS format)
│       └── generated/              # Generated test problems
│
├── benchmarks/
│   ├── benchmark_suite.py          # Main benchmark runner
│   ├── compare_gurobi.py           # Comparison with Gurobi
│   ├── compare_scipy.py            # Comparison with SciPy
│   └── results/                    # Benchmark results (JSON)
│
├── examples/
│   ├── 01_basic_lp.py              # Simple LP example
│   ├── 02_portfolio_qp.py          # Portfolio optimization
│   ├── 03_batch_solving.py         # Batch LP solving
│   ├── 04_warm_starting.py         # Warm start demonstration
│   └── 05_large_scale.py           # Million-variable LP
│
├── docs/
│   ├── conf.py                     # Sphinx configuration
│   ├── index.rst                   # Documentation home
│   ├── installation.rst            # Installation guide
│   ├── quickstart.rst              # Getting started
│   ├── api/                        # API reference
│   └── tutorials/                  # Step-by-step tutorials
│
└── scripts/
    ├── download_netlib.py          # Download test problems
    ├── check_cuda.py               # CUDA availability checker
    └── run_benchmarks.sh           # Benchmark automation
```

---

## 4. Development Phases

### Phase 0: Foundation (Weeks 1-2)
**Goal:** Project setup, build system, basic GPU infrastructure

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 0.1 | Set up repository, CMake, pyproject.toml | N/A |
| 0.2 | CUDA detection and architecture handling | `test_cuda_available()` |
| 0.3 | GPU memory allocator wrapper | `test_gpu_alloc_free()` |
| 0.4 | Error handling system | `test_cuda_error_throw()` |
| 0.5 | Basic pybind11 module compiles | `test_import_cuprox()` |

**Deliverable:** Empty package installs, detects GPU

### Phase 1: Core Linear Algebra (Weeks 3-6)
**Goal:** GPU vector/matrix operations with full test coverage

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 1.1 | Dense vector class (GPU) | `test_vector_create`, `test_vector_copy` |
| 1.2 | Vector BLAS: axpy, dot, nrm2 | `test_axpy`, `test_dot_product`, `test_norm` |
| 1.3 | CSR sparse matrix class | `test_csr_from_coo`, `test_csr_transpose` |
| 1.4 | SpMV (cuSPARSE wrapper) | `test_spmv_identity`, `test_spmv_random` |
| 1.5 | Box projections | `test_project_box`, `test_project_nonneg` |
| 1.6 | Reduction operations | `test_sum`, `test_max`, `test_min` |

**Deliverable:** Tested linear algebra primitives

### Phase 2: Preprocessing (Weeks 7-8)
**Goal:** Problem scaling and validation

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 2.1 | Input validation (dimensions, values) | `test_validate_dimensions`, `test_detect_nan` |
| 2.2 | Ruiz equilibration scaling | `test_ruiz_identity`, `test_ruiz_improves_condition` |
| 2.3 | Bound normalization | `test_normalize_bounds` |
| 2.4 | Problem structure detection | `test_detect_free_vars`, `test_detect_fixed_vars` |

**Deliverable:** Robust preprocessing pipeline

### Phase 3: PDHG LP Solver (Weeks 9-14)
**Goal:** Working LP solver with convergence guarantees

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 3.1 | PDHG single iteration kernel | `test_pdhg_one_step` |
| 3.2 | Step size computation | `test_stepsize_stable` |
| 3.3 | Residual computation (primal/dual) | `test_residual_zero_at_optimum` |
| 3.4 | Convergence checking | `test_convergence_detection` |
| 3.5 | Adaptive restart (Halpern) | `test_restart_accelerates` |
| 3.6 | Full solve loop | `test_solve_trivial_lp`, `test_solve_netlib_afiro` |
| 3.7 | Infeasibility detection | `test_detect_infeasible`, `test_detect_unbounded` |
| 3.8 | Solution extraction & unscaling | `test_solution_unscale` |

**Deliverable:** LP solver passing Netlib tests

### Phase 4: Python API (Weeks 15-17)
**Goal:** User-friendly Python interface

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 4.1 | Model class (add_var, add_constr) | `test_model_build`, `test_model_from_numpy` |
| 4.2 | Solver class with parameters | `test_solver_params`, `test_solver_timeout` |
| 4.3 | Result class | `test_result_access`, `test_result_status` |
| 4.4 | SciPy sparse matrix integration | `test_scipy_csr_input` |
| 4.5 | NumPy array integration | `test_numpy_dense_input` |
| 4.6 | Error messages and exceptions | `test_exception_messages` |

**Deliverable:** Complete Python API for LP

### Phase 5: QP Solver (Weeks 18-21)
**Goal:** ADMM-based convex QP solver

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 5.1 | Quadratic form evaluation | `test_quadratic_form` |
| 5.2 | ADMM iteration kernel | `test_admm_one_step` |
| 5.3 | Linear system solve (CG) | `test_cg_converges` |
| 5.4 | Full QP solve loop | `test_solve_trivial_qp` |
| 5.5 | Python QP API | `test_model_quadratic_obj` |

**Deliverable:** QP solver working

### Phase 6: CPU Fallback & Robustness (Weeks 22-24)
**Goal:** Works everywhere, handles edge cases

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 6.1 | CPU PDHG implementation | `test_cpu_matches_gpu` |
| 6.2 | Auto-detection and fallback | `test_auto_fallback` |
| 6.3 | Mixed precision (FP32/FP64) | `test_precision_modes` |
| 6.4 | Warm starting | `test_warm_start_faster` |
| 6.5 | Edge case handling | `test_empty_problem`, `test_single_var` |
| 6.6 | Memory limits and cleanup | `test_large_problem_oom` |

**Deliverable:** Robust solver for all environments

### Phase 7: Batch Solving (Weeks 25-26)
**Goal:** Solve many LPs in parallel

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 7.1 | Batch problem storage | `test_batch_create` |
| 7.2 | Parallel PDHG kernel | `test_batch_solve_identical` |
| 7.3 | Varying sizes handling | `test_batch_varying_sizes` |

**Deliverable:** Batch solving capability

### Phase 8: Polish & Release (Weeks 27-30)
**Goal:** Production-ready package

| Task | Description | TDD Test First |
|------|-------------|----------------|
| 8.1 | Performance optimization | Benchmark suite |
| 8.2 | Documentation | N/A |
| 8.3 | Examples | N/A |
| 8.4 | Wheel building (cibuildwheel) | `test_wheel_install` |
| 8.5 | PyPI test release | Manual testing |
| 8.6 | PyPI production release | N/A |

**Deliverable:** `pip install cuprox` works!

---

## 5. Test-Driven Development Strategy

### 5.1 TDD Workflow
```
For each feature:
1. Write failing test that defines expected behavior
2. Run test → confirm it fails
3. Write minimal code to pass test
4. Run test → confirm it passes
5. Refactor if needed (tests still pass)
6. Commit
```

### 5.2 Test Categories

#### Unit Tests (Fast, Isolated)
```cpp
// tests/cpp/test_vector_ops.cpp
TEST(VectorOps, AxpyCorrectResult) {
    // Arrange
    DeviceVector<float> x(1000, 1.0f);
    DeviceVector<float> y(1000, 2.0f);
    float alpha = 0.5f;
    
    // Act
    axpy(alpha, x, y);  // y = alpha*x + y
    
    // Assert
    std::vector<float> result = y.to_host();
    for (float val : result) {
        EXPECT_FLOAT_EQ(val, 2.5f);  // 0.5*1 + 2 = 2.5
    }
}
```

#### Integration Tests (End-to-End)
```python
# tests/python/test_solver_lp.py
def test_solve_simple_lp():
    """
    Solve: minimize -x - y
           subject to: x + y <= 1
                       x, y >= 0
    Optimal: x=0.5, y=0.5, obj=-1
    """
    model = cuprox.Model()
    x = model.add_var(lb=0, name="x")
    y = model.add_var(lb=0, name="y")
    model.add_constr(x + y <= 1)
    model.minimize(-x - y)
    
    result = model.solve()
    
    assert result.status == "optimal"
    assert abs(result.objective - (-1.0)) < 1e-4
    assert abs(result.x[0] - 0.5) < 1e-4
    assert abs(result.x[1] - 0.5) < 1e-4
```

#### Property-Based Tests
```python
# tests/python/test_properties.py
from hypothesis import given, strategies as st

@given(st.integers(min_value=10, max_value=1000))
def test_feasible_lp_always_solves(n):
    """Any feasible LP should return optimal status."""
    # Generate random feasible LP
    A = np.random.randn(n//2, n)
    x_feas = np.abs(np.random.randn(n))
    b = A @ x_feas + 0.1  # Ensure feasibility
    c = np.random.randn(n)
    
    model = cuprox.Model()
    x = model.add_vars(n, lb=0)
    model.add_constrs(A @ x <= b)
    model.minimize(c @ x)
    
    result = model.solve()
    assert result.status in ["optimal", "unbounded"]
```

#### Regression Tests (Known Problems)
```python
# tests/python/test_netlib.py
@pytest.mark.parametrize("problem_name,expected_obj", [
    ("afiro", -464.753),
    ("adlittle", 225494.963),
    ("blend", -30.812),
    ("kb2", -1749.900),
])
def test_netlib_problems(problem_name, expected_obj):
    """Verify correctness on Netlib benchmark problems."""
    model = load_netlib(problem_name)
    result = model.solve()
    
    assert result.status == "optimal"
    assert abs(result.objective - expected_obj) / abs(expected_obj) < 1e-4
```

### 5.3 Test Infrastructure

#### pytest Configuration
```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests/python"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "benchmark: marks benchmark tests",
]
filterwarnings = ["error"]
addopts = "--strict-markers -v"
```

#### Google Test Setup
```cmake
# tests/cpp/CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(cuprox_tests
    test_sparse_matrix.cpp
    test_vector_ops.cpp
    test_spmv.cpp
    test_pdhg.cpp
)
target_link_libraries(cuprox_tests 
    PRIVATE cuprox_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(cuprox_tests)
```

### 5.4 Coverage Requirements
```yaml
# Minimum coverage thresholds
C++ core: 85%
Python API: 95%
Overall: 90%

# Coverage tools
C++: gcov + lcov
Python: pytest-cov

# CI gate: PR blocked if coverage drops
```

---

## 6. Build System & Packaging

### 6.1 CMakeLists.txt (Root)
```cmake
cmake_minimum_required(VERSION 3.24)
project(cuprox VERSION 0.1.0 LANGUAGES CXX CUDA)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_CPU_ONLY "Build without CUDA" OFF)

# CUDA setup
if(NOT BUILD_CPU_ONLY)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # Support multiple architectures
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")
    else()
        message(WARNING "CUDA not found, building CPU-only version")
        set(BUILD_CPU_ONLY ON)
    endif()
endif()

# C++ setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(CUDAToolkit REQUIRED)

# Core library
add_subdirectory(src/cuprox)

# Python bindings
if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(src/bindings)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
```

### 6.2 pyproject.toml
```toml
[build-system]
requires = [
    "scikit-build-core>=0.8",
    "pybind11>=2.11",
    "numpy>=1.20",
]
build-backend = "scikit_build_core.build"

[project]
name = "cuprox"
version = "0.1.0"
description = "GPU-accelerated LP/QP optimization solver"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Mathematics",
]
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "hypothesis",
    "black",
    "ruff",
    "mypy",
]
docs = [
    "sphinx>=6.0",
    "sphinx-rtd-theme",
    "myst-parser",
]
benchmark = [
    "gurobipy>=10.0",  # For comparison
    "osqp",
    "cvxpy",
]

[project.urls]
Homepage = "https://github.com/yourusername/cuprox"
Documentation = "https://cuprox.readthedocs.io"
Repository = "https://github.com/yourusername/cuprox"

[tool.scikit-build]
cmake.build-type = "Release"
wheel.packages = ["python/cuprox"]
wheel.license-files = ["LICENSE"]

[tool.scikit-build.cmake.define]
BUILD_TESTS = "OFF"
BUILD_PYTHON = "ON"

# CUDA architectures - build for multiple
CMAKE_CUDA_ARCHITECTURES = "70;75;80;86;89;90"

[tool.cibuildwheel]
skip = ["*-musllinux*", "pp*", "*-win32", "*_i686"]
build = ["cp39-*", "cp310-*", "cp311-*", "cp312-*"]

[tool.cibuildwheel.linux]
before-all = """
    # Install CUDA toolkit
    yum install -y cuda-toolkit-12-4 || true
"""
environment = { CUDA_HOME="/usr/local/cuda-12.4" }
manylinux-x86_64-image = "manylinux2014"

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "UP", "B"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_ignores = true
```

### 6.3 Multi-CUDA Wheel Strategy
```bash
# Build wheels for different CUDA versions
# Each wheel has CUDA version in filename

cuprox-0.1.0-cp311-cp311-manylinux2014_x86_64.whl          # CUDA 12.x
cuprox-0.1.0+cu118-cp311-cp311-manylinux2014_x86_64.whl   # CUDA 11.8
cuprox-0.1.0+cpu-cp311-cp311-manylinux2014_x86_64.whl     # CPU only
```

---

## 7. CI/CD Pipeline

### 7.1 GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # Fast checks first
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff black mypy
      - run: ruff check python/
      - run: black --check python/
      - run: mypy python/cuprox

  # CPU tests (fast, no GPU needed)
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install numpy scipy pytest pytest-cov hypothesis
          pip install -e . --config-settings=cmake.define.BUILD_CPU_ONLY=ON
      - name: Run CPU tests
        run: pytest tests/python -m "not gpu" --cov=cuprox

  # GPU tests (requires self-hosted runner with GPU)
  test-gpu:
    runs-on: [self-hosted, gpu]
    needs: [lint, test-cpu]
    steps:
      - uses: actions/checkout@v4
      - name: Build with CUDA
        run: |
          pip install -e .
      - name: Run GPU tests
        run: pytest tests/python --cov=cuprox
      - name: Run C++ tests
        run: |
          cmake -B build -DBUILD_TESTS=ON
          cmake --build build
          ctest --test-dir build --output-on-failure

  # Build wheels
  build-wheels:
    runs-on: ubuntu-latest
    needs: [test-gpu]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: pypa/cibuildwheel@v2.17
        env:
          CIBW_ARCHS_LINUX: x86_64
      - uses: actions/upload-artifact@v4
        with:
          name: wheels
          path: wheelhouse/*.whl

  # Benchmark (nightly)
  benchmark:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .[benchmark]
      - run: python benchmarks/benchmark_suite.py --output results.json
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results.json
```

### 7.2 GPU CI Runner Setup
```bash
# Self-hosted runner with GPU requirements
# - NVIDIA GPU (any supported architecture)
# - CUDA toolkit installed
# - Docker with nvidia-container-toolkit

# Example: Run tests in Docker with GPU
docker run --gpus all -v $(pwd):/workspace nvidia/cuda:12.4-devel-ubuntu22.04 \
    bash -c "cd /workspace && pip install -e . && pytest tests/"
```

---

## 8. API Design

### 8.1 Python API (User-Facing)
```python
import cuprox
import numpy as np
from scipy import sparse

# === Basic LP Example ===
model = cuprox.Model()

# Add variables
x = model.add_var(lb=0, ub=10, name="x")
y = model.add_var(lb=0, name="y")

# Add constraints
model.add_constr(x + 2*y <= 20, name="capacity")
model.add_constr(3*x + y <= 30, name="labor")

# Set objective
model.minimize(cost=-5*x - 4*y)

# Solve
result = model.solve(params={
    "tolerance": 1e-6,
    "max_iterations": 10000,
    "verbose": True,
})

print(f"Status: {result.status}")
print(f"Objective: {result.objective:.4f}")
print(f"x = {result.get_value(x):.4f}")
print(f"y = {result.get_value(y):.4f}")


# === Matrix Form (Large-Scale) ===
n_vars = 100000
n_constrs = 50000

# Problem data
A = sparse.random(n_constrs, n_vars, density=0.001, format='csr')
b = np.random.rand(n_constrs)
c = np.random.randn(n_vars)
lb = np.zeros(n_vars)
ub = np.ones(n_vars)

# Create and solve
model = cuprox.Model.from_matrices(
    c=c, 
    A_ub=A, 
    b_ub=b,
    lb=lb,
    ub=ub
)
result = model.solve()


# === Batch Solving ===
problems = [generate_lp(i) for i in range(1000)]
results = cuprox.solve_batch(problems)  # All solved in parallel


# === Warm Starting ===
result1 = model.solve()
model.modify_objective(new_c)
result2 = model.solve(warm_start=result1)  # Faster!


# === QP Example ===
model = cuprox.Model()
x = model.add_vars(10, lb=0)

# Quadratic objective: 0.5 * x^T Q x + c^T x
Q = np.eye(10)  # Identity = simple quadratic
c = np.ones(10)
model.minimize(0.5 * x @ Q @ x + c @ x)

model.add_constr(sum(x) == 1)  # Simplex constraint
result = model.solve()
```

### 8.2 Solver Parameters
```python
DEFAULT_PARAMS = {
    # Convergence
    "tolerance": 1e-6,           # Primal/dual residual tolerance
    "max_iterations": 100000,    # Maximum iterations
    "time_limit": 3600.0,        # Seconds
    
    # Algorithm
    "method": "pdhg",            # "pdhg" or "admm" (for QP)
    "scaling": "ruiz",           # "ruiz", "geometric", or "none"
    "restart": "adaptive",       # "adaptive", "fixed", or "none"
    
    # Precision
    "precision": "float64",      # "float32" or "float64"
    
    # Diagnostics
    "verbose": False,            # Print iteration log
    "log_interval": 100,         # Iterations between log prints
    
    # Device
    "device": "auto",            # "auto", "gpu", or "cpu"
    "gpu_id": 0,                 # Which GPU to use
}
```

### 8.3 Result Object
```python
@dataclass
class SolveResult:
    status: str              # "optimal", "infeasible", "unbounded", "max_iter", "time_limit"
    objective: float         # Optimal objective value
    x: np.ndarray           # Primal solution
    y: np.ndarray           # Dual solution (constraint multipliers)
    
    iterations: int          # Number of iterations
    solve_time: float        # Total solve time (seconds)
    
    primal_residual: float   # Final primal residual
    dual_residual: float     # Final dual residual
    gap: float              # Duality gap
    
    def get_value(self, var) -> float:
        """Get solution value for a variable."""
        ...
    
    def get_dual(self, constr) -> float:
        """Get dual value for a constraint."""
        ...
```

---

## 9. Sprint Breakdown

### Sprint 1-2: Foundation (Weeks 1-2)
```
[ ] Set up git repository with .gitignore, LICENSE, README
[ ] Create directory structure per Section 3
[ ] Write CMakeLists.txt with CUDA detection
[ ] Write pyproject.toml with scikit-build-core
[ ] Create empty Python package that installs
[ ] Set up pytest and Google Test
[ ] Write first test: test_import_cuprox
[ ] CI: GitHub Actions for lint + basic tests
```

### Sprint 3-4: GPU Memory & Vectors (Weeks 3-4)
```
[ ] TEST: test_gpu_malloc_free
[ ] IMPL: memory.cuh - GPU allocator wrapper
[ ] TEST: test_device_vector_create, test_device_vector_copy
[ ] IMPL: dense_vector.cuh - DeviceVector class
[ ] TEST: test_vector_fill, test_vector_to_host
[ ] IMPL: Vector fill and host transfer
[ ] TEST: test_axpy_float, test_axpy_double
[ ] IMPL: BLAS axpy via cuBLAS
[ ] TEST: test_dot_product
[ ] IMPL: BLAS dot via cuBLAS
[ ] TEST: test_nrm2
[ ] IMPL: BLAS nrm2 via cuBLAS
```

### Sprint 5-6: Sparse Matrix & SpMV (Weeks 5-6)
```
[ ] TEST: test_csr_from_triplets
[ ] IMPL: sparse_matrix.cuh - CSRMatrix from COO
[ ] TEST: test_csr_dimensions
[ ] IMPL: CSR dimension accessors
[ ] TEST: test_spmv_identity_matrix
[ ] IMPL: spmv.cuh - cuSPARSE SpMV wrapper
[ ] TEST: test_spmv_random_matrix
[ ] IMPL: Handle general sparse matrices
[ ] TEST: test_spmv_transpose
[ ] IMPL: Transposed SpMV (A^T * x)
[ ] TEST: test_csr_from_scipy
[ ] IMPL: Python binding for scipy.sparse input
```

### Sprint 7-8: Preprocessing (Weeks 7-8)
```
[ ] TEST: test_validate_dimensions_mismatch
[ ] IMPL: validation.hpp - dimension checking
[ ] TEST: test_validate_nan_detection
[ ] IMPL: NaN/Inf detection
[ ] TEST: test_ruiz_scaling_ones (identity should not change)
[ ] IMPL: scaling.cuh - Ruiz equilibration
[ ] TEST: test_ruiz_improves_condition_number
[ ] IMPL: Iterative Ruiz scaling
[ ] TEST: test_unscale_solution
[ ] IMPL: Solution unscaling
```

### Sprint 9-10: PDHG Core (Weeks 9-10)
```
[ ] TEST: test_pdhg_step_dimensions
[ ] IMPL: pdhg.cuh - PDHG iteration struct
[ ] TEST: test_primal_update
[ ] IMPL: Primal step: x = x - tau * A^T * y
[ ] TEST: test_dual_update  
[ ] IMPL: Dual step: y = y + sigma * A * x_bar
[ ] TEST: test_box_projection_bounds
[ ] IMPL: projections.cuh - box projection
[ ] TEST: test_extrapolation_step
[ ] IMPL: x_bar = 2*x_new - x_old
[ ] TEST: test_pdhg_one_full_iteration
[ ] IMPL: Complete iteration combining all steps
```

### Sprint 11-12: PDHG Convergence (Weeks 11-12)
```
[ ] TEST: test_primal_residual_computation
[ ] IMPL: Primal residual: ||Ax - b||
[ ] TEST: test_dual_residual_computation
[ ] IMPL: Dual residual: ||A^T y + c||
[ ] TEST: test_convergence_at_optimum
[ ] IMPL: Convergence checking logic
[ ] TEST: test_solve_trivial_1var_lp
[ ] IMPL: Full solve loop with termination
[ ] TEST: test_solve_trivial_2var_lp
[ ] IMPL: Handle equality constraints
[ ] TEST: test_step_size_stability
[ ] IMPL: Adaptive step size (Malitsky-Pock)
```

### Sprint 13-14: PDHG Robustness (Weeks 13-14)
```
[ ] TEST: test_netlib_afiro
[ ] IMPL: Download and parse Netlib problems
[ ] TEST: test_netlib_blend
[ ] IMPL: Handle various problem structures
[ ] TEST: test_adaptive_restart
[ ] IMPL: Halpern restart acceleration
[ ] TEST: test_infeasibility_detection
[ ] IMPL: Detect infeasible problems
[ ] TEST: test_unbounded_detection
[ ] IMPL: Detect unbounded problems
[ ] TEST: test_max_iterations_terminates
[ ] IMPL: Iteration limit
```

### Sprint 15-16: Python Model API (Weeks 15-16)
```
[ ] TEST: test_model_create_empty
[ ] IMPL: model.py - Model class
[ ] TEST: test_add_variable
[ ] IMPL: add_var method
[ ] TEST: test_add_constraint_le
[ ] IMPL: add_constr for <= constraints
[ ] TEST: test_add_constraint_eq
[ ] IMPL: add_constr for == constraints
[ ] TEST: test_set_objective_minimize
[ ] IMPL: minimize method
[ ] TEST: test_model_to_standard_form
[ ] IMPL: Convert to A, b, c matrices
[ ] TEST: test_solve_via_model
[ ] IMPL: Connect model to C++ solver
```

### Sprint 17: Python Solver & Result (Week 17)
```
[ ] TEST: test_solver_with_params
[ ] IMPL: solver.py - Solver class
[ ] TEST: test_result_status_optimal
[ ] IMPL: result.py - SolveResult class
[ ] TEST: test_result_access_values
[ ] IMPL: get_value, get_dual methods
[ ] TEST: test_solve_from_matrices
[ ] IMPL: Model.from_matrices factory
[ ] TEST: test_scipy_sparse_input
[ ] IMPL: Handle scipy.sparse.csr_matrix
```

### Sprint 18-19: QP ADMM Core (Weeks 18-19)
```
[ ] TEST: test_quadratic_objective_eval
[ ] IMPL: Quadratic form: x^T Q x
[ ] TEST: test_admm_x_update
[ ] IMPL: x-step (solve linear system)
[ ] TEST: test_admm_z_update
[ ] IMPL: z-step (projection)
[ ] TEST: test_admm_u_update
[ ] IMPL: u-step (dual update)
[ ] TEST: test_admm_one_iteration
[ ] IMPL: Complete ADMM iteration
```

### Sprint 20-21: QP Completion (Weeks 20-21)
```
[ ] TEST: test_solve_trivial_qp
[ ] IMPL: QP solve loop
[ ] TEST: test_solve_portfolio_qp
[ ] IMPL: Handle dense Q matrices
[ ] TEST: test_qp_convergence
[ ] IMPL: QP convergence criteria
[ ] TEST: test_python_qp_api
[ ] IMPL: Python Model for QP
```

### Sprint 22-23: CPU Fallback (Weeks 22-23)
```
[ ] TEST: test_cpu_pdhg_matches_gpu
[ ] IMPL: cpu_pdhg.cpp - CPU implementation
[ ] TEST: test_auto_device_selection
[ ] IMPL: Auto GPU/CPU selection
[ ] TEST: test_explicit_cpu_mode
[ ] IMPL: device="cpu" parameter
[ ] TEST: test_no_cuda_install
[ ] IMPL: Handle missing CUDA gracefully
```

### Sprint 24: Edge Cases (Week 24)
```
[ ] TEST: test_empty_problem
[ ] IMPL: Handle n=0 or m=0
[ ] TEST: test_single_variable
[ ] IMPL: Handle n=1
[ ] TEST: test_very_large_bounds
[ ] IMPL: Handle inf bounds correctly
[ ] TEST: test_warm_start
[ ] IMPL: Warm starting from previous solution
[ ] TEST: test_timeout
[ ] IMPL: Time limit termination
```

### Sprint 25-26: Batch Solving (Weeks 25-26)
```
[ ] TEST: test_batch_problem_creation
[ ] IMPL: Batch problem data structure
[ ] TEST: test_batch_solve_identical
[ ] IMPL: Parallel PDHG for batches
[ ] TEST: test_batch_solve_varied
[ ] IMPL: Handle different sized problems
[ ] TEST: test_batch_python_api
[ ] IMPL: solve_batch Python function
```

### Sprint 27-28: Performance & Polish (Weeks 27-28)
```
[ ] Profile with Nsight Compute
[ ] Optimize memory access patterns
[ ] Tune kernel launch configurations
[ ] Add mixed precision option (FP32)
[ ] Benchmark vs Gurobi on Netlib
[ ] Benchmark vs SciPy linprog
[ ] Benchmark vs OSQP (for QP)
[ ] Document performance characteristics
```

### Sprint 29-30: Release (Weeks 29-30)
```
[ ] Write full documentation (Sphinx)
[ ] Create 5 example scripts
[ ] Set up cibuildwheel configuration
[ ] Build wheels for CUDA 11.8, 12.4
[ ] Build CPU-only wheel
[ ] Test wheel installation on fresh systems
[ ] Upload to TestPyPI
[ ] Final testing
[ ] Release to PyPI
[ ] Announce release
```

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PDHG convergence issues on ill-conditioned problems | Medium | High | Implement scaling, adaptive restarts, fallback to CPU |
| cuSPARSE API changes between CUDA versions | Low | Medium | Abstract behind wrapper, test on multiple CUDA versions |
| Memory issues on large problems | Medium | Medium | Implement streaming, memory-efficient representations |
| Slow Python-C++ data transfer | Low | Low | Use buffer protocol, minimize copies |
| Wheel build complexity | High | Medium | Start simple (Linux only), add platforms incrementally |

### 10.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| QP solver takes longer than expected | Medium | Low | QP is P1, can ship LP-only v0.1 |
| Benchmark setup delays | Low | Low | Use simple random problems first |
| Documentation overhead | High | Low | Document as you go, not at end |

### 10.3 Fallback Plan
```
If behind schedule:
1. Week 20+: Drop batch solving (P2)
2. Week 22+: Drop QP solver (P1), release LP-only
3. Week 26+: Drop CPU fallback, require GPU
4. Week 28+: Skip Windows wheels, Linux only

Minimum Viable Product (MVP):
- LP solver (PDHG) on GPU
- Python API (Model class)
- Linux wheel
- Basic documentation
```

---

## Quick Reference: Key Files to Create First

```bash
# Week 1 - Create these files:
touch CMakeLists.txt
touch pyproject.toml
touch README.md
touch LICENSE
mkdir -p src/cuprox/core
touch src/cuprox/core/types.hpp
touch src/cuprox/CMakeLists.txt
mkdir -p python/cuprox
touch python/cuprox/__init__.py
mkdir -p tests/cpp tests/python
touch tests/python/conftest.py
touch tests/python/test_import.py

# First test to write:
# tests/python/test_import.py
def test_import():
    import cuprox
    assert hasattr(cuprox, '__version__')
```

---

**This plan is designed to be actionable.** Each sprint has concrete, testable deliverables. The TDD approach ensures quality at every step. Start with Sprint 1-2 and iterate!

