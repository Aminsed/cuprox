# cuProx Architecture Document
## GPU-Accelerated First-Order Optimization Solver

**Version:** 0.1.0-alpha  
**Last Updated:** November 2024  
**Status:** Design Phase

---

## Executive Summary

cuProx is a GPU-accelerated solver for Linear Programs (LP) and convex Quadratic Programs (QP) using first-order proximal methods. Unlike general-purpose solvers, cuProx focuses exclusively on problems where GPU acceleration provides genuine 10-100x speedups over CPU alternatives.

**Core Philosophy:** Do one thing exceptionally well.

---

## Table of Contents

1. [Market Analysis & Problem Statement](#1-market-analysis--problem-statement)
2. [Competitive Landscape](#2-competitive-landscape)
3. [Technical Differentiation Strategy](#3-technical-differentiation-strategy)
4. [Algorithm Selection & Justification](#4-algorithm-selection--justification)
5. [System Architecture](#5-system-architecture)
6. [Data Structures](#6-data-structures)
7. [Kernel Design](#7-kernel-design)
8. [Memory Management](#8-memory-management)
9. [Python API Design](#9-python-api-design)
10. [Performance Targets](#10-performance-targets)
11. [Quality Assurance](#11-quality-assurance)

---

## 1. Market Analysis & Problem Statement

### 1.1 Who Needs Fast LP/QP Solving?

| Domain | Use Case | Problem Size | Accuracy Need | Frequency |
|--------|----------|--------------|---------------|-----------|
| **Machine Learning** | Differentiable optimization layers (OptNet) | Small-Medium (100-10K vars) | Moderate (1e-4) | Batch: 1000s per forward pass |
| **Finance** | Portfolio optimization, risk | Medium (1K-100K vars) | Moderate (1e-6) | Real-time: 100s per second |
| **Robotics/MPC** | Model Predictive Control | Small (100-1K vars) | Low-Moderate (1e-3) | Real-time: 100-1000 Hz |
| **Energy** | Optimal Power Flow | Large (100K-10M vars) | Moderate (1e-6) | Periodic: minutes |
| **Logistics** | Vehicle routing, scheduling | Large (10K-1M vars) | Moderate (1e-6) | Batch: hourly |
| **Stochastic Programming** | Two-stage, SAA | Many medium LPs | Moderate (1e-6) | Batch: 1000s of scenarios |

### 1.2 The Real Pain Points

1. **ML Integration is Broken**
   - cvxpylayers/OptNet use OSQP (CPU) → training bottleneck
   - No native GPU QP solver with backward pass support
   - Batch solving is an afterthought in all existing solvers

2. **Installation is a Nightmare**
   - cuPDLP requires building from source
   - CUDA version mismatches break everything
   - No `pip install` solution exists

3. **Accuracy vs Speed Tradeoff is Fixed**
   - Users can't choose "give me 1e-4 accuracy fast"
   - Interior-point methods always go for 1e-10+
   - Wastes compute when moderate accuracy suffices

4. **Batch Solving is Unsupported**
   - Existing solvers: loop over problems sequentially
   - GPU parallel batch? Nobody does this well

### 1.3 Our Target Users (Prioritized)

| Priority | User Type | What They Need | Our Value |
|----------|-----------|----------------|-----------|
| **P0** | ML Researchers/Engineers | Fast QP in PyTorch training loop | 100x faster OptNet layers |
| **P1** | Quant Finance | Real-time portfolio rebalancing | Sub-millisecond QP solving |
| **P2** | Robotics Engineers | MPC on Jetson/GPU | 1000 Hz control loops |
| **P3** | Operations Researchers | Large-scale LP | 10x faster than SciPy |

---

## 2. Competitive Landscape

### 2.1 Existing Solutions Analysis

#### LP Solvers

| Solver | Type | GPU? | Batch? | pip install? | Best For |
|--------|------|------|--------|--------------|----------|
| **Gurobi** | Simplex/Barrier | Limited (v13) | No | Yes ($$$) | Everything, if you pay |
| **HiGHS** | Simplex/IPM | No | No | Yes | General LP, free |
| **cuPDLP** | PDHG | Yes | No | No | Large sparse LP |
| **PDLP (Google)** | PDHG | CPU | No | Via OR-Tools | Large LP, research |
| **SciPy linprog** | Simplex/IPM | No | No | Yes | Small LP, convenience |

#### QP Solvers

| Solver | Type | GPU? | Batch? | pip install? | Best For |
|--------|------|------|--------|--------------|----------|
| **OSQP** | ADMM | No | No | Yes | General QP, MPC |
| **qpOASES** | Active Set | No | No | Via interfaces | Dense QP, warm start |
| **ECOS** | IPM | No | No | Yes | SOCP |
| **SCS** | ADMM | No | No | Yes | Large conic |
| **Clarabel** | IPM | No | No | Yes | Modern, conic |

### 2.2 Gap Analysis

**Nobody provides:**
- ✗ GPU-accelerated QP with `pip install`
- ✗ Native batch solving (1000s of problems in parallel)
- ✗ PyTorch integration with backward pass
- ✗ Combined LP + QP in one lightweight package
- ✗ User-configurable accuracy/speed tradeoff

**cuProx fills ALL these gaps.**

### 2.3 Why We Can Win

| Factor | Our Advantage |
|--------|---------------|
| **Focus** | We only do LP/QP with first-order methods — no bloat |
| **Batch-First Design** | Kernel architecture designed for parallel problems |
| **Modern Stack** | C++17/CUDA 12, Python 3.9+, pybind11 |
| **ML-Native** | PyTorch integration from day one |
| **Easy Install** | Single `pip install cuprox` |

---

## 3. Technical Differentiation Strategy

### 3.1 Two Operational Modes

cuProx operates in two distinct modes, each optimized differently:

#### Mode A: Single Large Problem
```
Problem: 1 LP/QP with 100K+ variables
GPU Utilization: Many threads cooperate on one problem
Competition: cuPDLP, Gurobi
Target: 10x faster than HiGHS/SciPy
```

#### Mode B: Batch Many Small Problems
```
Problem: 1000+ LP/QPs with 100-10K variables each
GPU Utilization: One thread block per problem
Competition: Nobody (unique offering)
Target: 100x faster than sequential OSQP
```

### 3.2 Core Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LP Algorithm** | Restarted PDHG | Best convergence rate, proven in cuPDLP |
| **QP Algorithm** | ADMM (OSQP-style) | Robust, well-understood, warm-start friendly |
| **Precision** | FP64 default, FP32 option | FP32 is 2-32x faster depending on GPU† |
| **Sparse Format** | CSR (Compressed Sparse Row) | Best for SpMV, cuSPARSE native |
| **Batch Storage** | Padded arrays | Simple, efficient, fixed overhead |
| **Memory** | Device memory (not unified) | Maximum performance, explicit control |
| **Preconditioning** | Ruiz equilibration | Essential for PDHG convergence |

†**FP32 vs FP64 Performance Note:**  
- Data center GPUs (A100, H100): FP64 is 1/2 speed of FP32
- Professional GPUs (A6000): FP64 is 1/2 speed of FP32
- Consumer GPUs (RTX 3090, 4090): FP64 is 1/32 to 1/64 speed of FP32
- For consumer GPUs, FP32 with iterative refinement may be practical

### 3.3 Non-Goals (Explicit Scope Limits)

We will NOT implement:
- ❌ Mixed-Integer Programming (MIP) — fundamentally sequential
- ❌ Simplex method — poor GPU fit
- ❌ Interior-Point/Barrier — Cholesky bottleneck
- ❌ General conic (SDP, exponential cone) — different algorithms needed
- ❌ Multi-GPU — premature optimization, add in v2.0

---

## 4. Algorithm Selection & Justification

### 4.1 LP: Primal-Dual Hybrid Gradient (PDHG)

#### The Algorithm

PDHG solves the saddle-point formulation of LP:

```
min_x max_y  c^T x + y^T (Ax - b)
subject to:  x ∈ X (box constraints)
             y ∈ Y (sign constraints for inequalities)
```

**Iteration:**
```
y_{k+1} = proj_Y(y_k + σ(A x̄_k - b))     # Dual update
x_{k+1} = proj_X(x_k - τ(c + A^T y_{k+1})) # Primal update  
x̄_{k+1} = 2x_{k+1} - x_k                   # Extrapolation
```

#### Why PDHG for GPU?

| Operation | GPU Suitability | Notes |
|-----------|-----------------|-------|
| SpMV: `A @ x` | ⭐⭐⭐⭐⭐ | Perfect for cuSPARSE |
| SpMV: `A.T @ y` | ⭐⭐⭐⭐⭐ | Perfect for cuSPARSE |
| Projection | ⭐⭐⭐⭐⭐ | Embarrassingly parallel |
| Vector ops | ⭐⭐⭐⭐⭐ | axpy, norms — perfect |

**Every operation is GPU-friendly. No sequential bottlenecks.**

#### Convergence Acceleration: Adaptive Restarts

Standard PDHG has O(1/k) ergodic convergence rate. With adaptive restarts,
we achieve O(1/k²) accelerated convergence.

**Restart Strategy (cuPDLP-style):**
1. Track normalized primal-dual residual r_k
2. Restart averaging when r_{k+1}/r_k > γ (e.g., γ = 0.8)
3. On restart: reset x̄ = x, ȳ = y (discard stale averages)

**Extrapolation (Chambolle-Pock):**
```
x̄_{k+1} = 2 x_{k+1} - x_k   # Over-relaxation with θ = 1
```

Note: We use the term "restarted PDHG" rather than "Halpern iteration" as our
implementation follows Chambolle-Pock extrapolation with adaptive restarts,
not the classical Halpern fixed-point averaging scheme.

#### Step Size Selection

```python
# Conservative step sizes (guaranteed convergence)
# Theory requires: τσ||A||² < 1
# We use margin 0.9 for numerical stability
tau = sigma = 0.9 / ||A||_2

# Balanced step sizes (often faster in practice)
tau = 0.9 / ||A||_{col_max}   # max column 2-norm
sigma = 0.9 / ||A||_{row_max}  # max row 2-norm

# Adaptive (Malitsky-Pock) for faster convergence
# Adjusts based on iteration progress
```

### 4.2 QP: Alternating Direction Method of Multipliers (ADMM)

#### Problem Form

```
minimize    (1/2) x^T P x + q^T x
subject to  l <= Ax <= u
```

#### ADMM Splitting

We introduce auxiliary variable z ∈ ℝᵐ and reformulate as:
```
minimize    (1/2)x'Px + q'x
subject to  Ax = z,  l ≤ z ≤ u
```

The ADMM iteration (with penalty parameter ρ > 0):
```
x_{k+1} = (P + ρ A^T A)^{-1} (ρ A^T z_k - A^T y_k - q)
z_{k+1} = proj_{[l,u]}(A x_{k+1} + (1/ρ) y_k)
y_{k+1} = y_k + ρ (A x_{k+1} - z_{k+1})
```

**Dimension check:**
- x ∈ ℝⁿ, z ∈ ℝᵐ, y ∈ ℝᵐ
- A ∈ ℝᵐˣⁿ, P ∈ ℝⁿˣⁿ, q ∈ ℝⁿ
- RHS of x-update: ρ A^T z_k (n) - A^T y_k (n) - q (n) = ℝⁿ ✓

#### The x-Update Challenge

The x-update requires solving a linear system. Options:

| Method | When to Use | GPU Benefit |
|--------|-------------|-------------|
| **Direct (cached factorization)** | P and A don't change | Factor once, solve fast |
| **Conjugate Gradient (CG)** | Large sparse systems | Good GPU parallelism |
| **Diagonal approx** | P is diagonal/easy | Trivial, very fast |

**Our approach:** 
- Precompute factorization if P, A are constant (batch mode)
- Use CG with Jacobi preconditioner for large problems
- Detect diagonal P and use closed-form

### 4.3 Preconditioning: Ruiz Equilibration

Ruiz equilibration scales the constraint matrix to improve conditioning:

```
D_1 A D_2 → scaled A with better condition number
```

Where D_1, D_2 are diagonal matrices computed by iterating:
```
d1_i = 1 / sqrt(||A[i,:]||_∞)  # Row scaling
d2_j = 1 / sqrt(||A[:,j]||_∞)  # Column scaling
```

**Critical insight:** Without scaling, PDHG often fails to converge on real problems.

### 4.4 Convergence Criteria

We use relative residuals, with different formulations for LP and QP:

#### LP Residuals (PDHG)

For LP, we measure primal feasibility and dual optimality directly:

```python
# Primal residual: constraint violation
# For Ax <= b: how much does Ax exceed b?
primal_res = ||max(Ax - b, 0)||_∞ / max(||b||_∞, 1)

# Dual residual: reduced cost violation  
# At optimum: c + A^T y = 0 for free variables
# For bounded vars: reduced cost should have correct sign
reduced_cost = c + A^T y
dual_res = ||reduced_cost - proj_bounds(reduced_cost)||_∞ / max(||c||_∞, 1)

converged_lp = (primal_res < tol) and (dual_res < tol)
```

#### QP Residuals (ADMM)

For QP with auxiliary variable z (where Ax = z, l ≤ z ≤ u):

```python
# Primal residual: consensus violation
primal_res = ||Ax - z||_∞ / max(||Ax||_∞, ||z||_∞, 1)

# Dual residual: stationarity violation  
dual_res = ||Px + q + A^T y||_∞ / max(||Px||_∞, ||q||_∞, ||A^T y||_∞, 1)

converged_qp = (primal_res < tol) and (dual_res < tol)
```

Note: The QP residuals follow OSQP conventions where z is the ADMM auxiliary variable.

---

## 5. System Architecture

### 5.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER APPLICATION                            │
│                    (PyTorch training, scripts)                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PYTHON API LAYER                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ │
│  │   Model     │ │   Solver    │ │   Result    │ │ torch.autograd│ │
│  │  (build)    │ │  (solve)    │ │  (output)   │ │  integration  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘ │
│                                                                     │
│  python/cuprox/                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PYBIND11 BINDINGS                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  _core module: direct access to C++ solver classes          │   │
│  │  - CsrMatrix, DeviceVector (zero-copy with NumPy/PyTorch)   │   │
│  │  - PDHGSolver, ADMMSolver                                   │   │
│  │  - BatchSolver                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  src/bindings/                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         C++ SOLVER LAYER                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │    PDHGSolver    │  │    ADMMSolver    │  │   BatchSolver    │  │
│  │  (LP: single)    │  │  (QP: single)    │  │  (LP/QP: batch)  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                     │            │
│           └─────────────────────┴─────────────────────┘            │
│                                 │                                   │
│                                 ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    PREPROCESSING                             │   │
│  │  - Ruiz scaling (GPU parallel)                              │   │
│  │  - Bound normalization                                      │   │
│  │  - Validation                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  src/cuprox/solvers/                                                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LINEAR ALGEBRA LAYER                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ │
│  │    SpMV     │ │    BLAS     │ │ Projections │ │   Reductions  │ │
│  │  (A*x, A'y) │ │(axpy,dot,..)│ │ (box, cone) │ │ (norm, sum)   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘ │
│                                                                     │
│  src/cuprox/linalg/                                                 │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CORE DATA LAYER                            │
│  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │     CsrMatrix       │  │    DeviceVector     │                  │
│  │  (GPU sparse CSR)   │  │   (GPU dense vec)   │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
│  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │    MemoryPool       │  │    ErrorHandler     │                  │
│  │  (GPU allocator)    │  │  (CUDA + logic)     │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
│                                                                     │
│  src/cuprox/core/                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CUDA / cuSPARSE / cuBLAS                    │
│                         (NVIDIA Libraries)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Module Dependency Graph

```
cuprox/
├── core/           # No internal dependencies
│   ├── types.hpp       → (standalone)
│   ├── error.hpp       → types
│   ├── memory.cuh      → error
│   ├── dense_vector.cuh → memory, error
│   └── sparse_matrix.cuh → memory, error, dense_vector
│
├── linalg/         # Depends on core/
│   ├── blas.cuh        → core/dense_vector
│   ├── spmv.cuh        → core/sparse_matrix, core/dense_vector
│   └── projections.cuh → core/dense_vector
│
├── preprocess/     # Depends on core/, linalg/
│   ├── scaling.cuh     → linalg/spmv, core/sparse_matrix
│   └── validation.hpp  → core/types
│
├── solvers/        # Depends on all above
│   ├── solver_base.hpp → core/, linalg/, preprocess/
│   ├── pdhg.cuh        → solver_base, linalg/
│   ├── admm.cuh        → solver_base, linalg/
│   └── batch_pdhg.cuh  → pdhg, core/
│
└── cpu_fallback/   # Standalone, mirrors solvers/
    ├── cpu_pdhg.hpp    → Eigen (external)
    └── cpu_admm.hpp    → Eigen (external)
```

### 5.3 Data Flow: Single Problem

```
User Input                 Internal Processing                GPU Execution
───────────                ────────────────────               ─────────────

scipy.sparse.csr    ─────► validate dimensions    
np.ndarray (c, b)          check for NaN/Inf      
bounds (lb, ub)            
        │
        ▼
copy to GPU         ─────► CsrMatrix (device)     ─────►  cudaMemcpy H→D
                           DeviceVector (device)           (one-time cost)
        │
        ▼
preprocess          ─────► Ruiz scaling           ─────►  parallel kernels
                           compute norms                   no host sync
        │
        ▼
solve loop          ─────► PDHG iterations        ─────►  SpMV (cuSPARSE)
(stays on GPU)             check convergence              vector ops (cuBLAS)
                           every N iterations             projections (custom)
        │
        ▼
extract result      ─────► unscale solution       
                           copy x, y to host      ─────►  cudaMemcpy D→H
        │
        ▼
return to Python    ─────► SolveResult dataclass
```

### 5.4 Data Flow: Batch Mode

```
User Input: List of N problems
──────────────────────────────

[Problem_0, Problem_1, ..., Problem_{N-1}]
     │           │               │
     ▼           ▼               ▼
┌─────────────────────────────────────────┐
│         Batch Preprocessing             │
│  - Pad all problems to max size         │
│  - Stack into contiguous arrays         │
│  - Single transfer to GPU               │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         GPU: Parallel Solve             │
│                                         │
│  Grid: (N_problems, 1, 1)               │
│  Block: (threads_per_problem, 1, 1)     │
│                                         │
│  Each block solves one problem          │
│  Shared memory for problem data         │
│  No inter-block communication           │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Extract N Results               │
│  - Single transfer back to host         │
│  - Unpack into list of results          │
└─────────────────────────────────────────┘
```

---

## 6. Data Structures

### 6.1 Sparse Matrix (CSR Format)

```cpp
// src/cuprox/core/sparse_matrix.cuh

template <typename Scalar = double>
class CsrMatrix {
public:
    // Dimensions
    int num_rows;
    int num_cols;
    int nnz;  // number of non-zeros
    
    // CSR arrays (GPU device memory)
    int* row_offsets;      // size: num_rows + 1
    int* col_indices;      // size: nnz
    Scalar* values;        // size: nnz
    
    // cuSPARSE descriptor (created once, reused)
    cusparseSpMatDescr_t descr;
    
    // Construction
    static CsrMatrix from_coo(int m, int n, 
                              const int* rows, 
                              const int* cols,
                              const Scalar* vals,
                              int nnz);
    
    static CsrMatrix from_scipy(py::object scipy_csr);  // pybind11
    
    // Operations
    void spmv(Scalar alpha, const DeviceVector<Scalar>& x,
              Scalar beta, DeviceVector<Scalar>& y) const;  // y = α*A*x + β*y
    
    void spmv_transpose(Scalar alpha, const DeviceVector<Scalar>& y,
                        Scalar beta, DeviceVector<Scalar>& x) const;  // x = α*A'*y + β*x
    
    // Memory management
    ~CsrMatrix();  // Frees GPU memory
    
private:
    bool owns_memory = true;
    cusparseHandle_t sparse_handle;
};
```

### 6.2 Dense Vector

```cpp
// src/cuprox/core/dense_vector.cuh

template <typename Scalar = double>
class DeviceVector {
public:
    int size;
    Scalar* data;  // GPU device pointer
    
    // Construction
    explicit DeviceVector(int n);                    // Allocate uninitialized
    DeviceVector(int n, Scalar fill_value);          // Allocate and fill
    static DeviceVector from_host(const Scalar* host_data, int n);
    static DeviceVector from_numpy(py::array_t<Scalar> arr);  // Copies to GPU (host→device)
    static DeviceVector from_dlpack(py::object tensor);       // Zero-copy from PyTorch/CuPy GPU tensors
    
    // Data movement
    void copy_to_host(Scalar* host_data) const;
    std::vector<Scalar> to_host() const;
    void copy_from(const DeviceVector& other);
    
    // In-place operations (return *this for chaining)
    DeviceVector& fill(Scalar value);
    DeviceVector& axpy(Scalar alpha, const DeviceVector& x);  // this += α*x
    DeviceVector& scale(Scalar alpha);                        // this *= α
    
    // Reductions
    Scalar dot(const DeviceVector& other) const;
    Scalar norm2() const;
    Scalar norm_inf() const;
    Scalar sum() const;
    
    // Element-wise
    void project_box(const DeviceVector& lb, const DeviceVector& ub);
    void project_nonnegative();
    
    // Memory
    ~DeviceVector();
    
private:
    bool owns_memory = true;
    cublasHandle_t blas_handle;
};
```

### 6.3 Problem Data Structure

```cpp
// src/cuprox/core/problem.hpp

template <typename Scalar = double>
struct LPProblem {
    // Standard form: min c'x s.t. Ax <= b, lb <= x <= ub
    CsrMatrix<Scalar> A;
    DeviceVector<Scalar> b;
    DeviceVector<Scalar> c;
    DeviceVector<Scalar> lb;
    DeviceVector<Scalar> ub;
    
    // Dimensions
    int num_vars() const { return A.num_cols; }
    int num_constrs() const { return A.num_rows; }
};

template <typename Scalar = double>
struct QPProblem {
    // Standard form: min (1/2)x'Px + q'x s.t. l <= Ax <= u
    CsrMatrix<Scalar> P;   // Quadratic term (can be nullptr for LP)
    CsrMatrix<Scalar> A;   // Constraint matrix
    DeviceVector<Scalar> q;
    DeviceVector<Scalar> l;  // Constraint lower bounds
    DeviceVector<Scalar> u;  // Constraint upper bounds
    
    int num_vars() const { return A.num_cols; }
    int num_constrs() const { return A.num_rows; }
};
```

### 6.4 Solver State

```cpp
// src/cuprox/solvers/pdhg_state.cuh

template <typename Scalar = double>
struct PDHGState {
    // Primal-dual variables
    DeviceVector<Scalar> x;      // Primal (size: n)
    DeviceVector<Scalar> x_bar;  // Extrapolated primal
    DeviceVector<Scalar> y;      // Dual (size: m)
    
    // Work vectors (pre-allocated to avoid malloc in loop)
    DeviceVector<Scalar> Ax;     // A * x
    DeviceVector<Scalar> ATy;    // A' * y
    DeviceVector<Scalar> x_prev; // Previous x (for extrapolation)
    
    // Step sizes
    Scalar tau;    // Primal step
    Scalar sigma;  // Dual step
    
    // Iteration count
    int iteration;
    
    // Residuals (computed periodically)
    Scalar primal_residual;
    Scalar dual_residual;
    Scalar gap;
    
    // Scaling factors (from Ruiz)
    DeviceVector<Scalar> row_scale;  // D_1
    DeviceVector<Scalar> col_scale;  // D_2
    Scalar obj_scale;
};
```

---

## 7. Kernel Design

### 7.1 PDHG Iteration Kernel (Single Problem)

```cpp
// src/cuprox/solvers/pdhg_kernels.cuh

// Fused primal update + projection kernel
// x_new = proj_box(x - tau * (c + A'y), lb, ub)
template <typename Scalar>
__global__ void pdhg_primal_update_kernel(
    const Scalar* x,
    const Scalar* ATy,
    const Scalar* c,
    const Scalar* lb,
    const Scalar* ub,
    Scalar tau,
    Scalar* x_new,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Scalar val = x[i] - tau * (c[i] + ATy[i]);
        // Box projection (fused)
        val = fmax(val, lb[i]);
        val = fmin(val, ub[i]);
        x_new[i] = val;
    }
}

// Fused dual update + projection kernel  
// y_new = proj_cone(y + sigma * (A*x_bar - b))
template <typename Scalar>
__global__ void pdhg_dual_update_kernel(
    const Scalar* y,
    const Scalar* Ax_bar,
    const Scalar* b,
    Scalar sigma,
    Scalar* y_new,
    int m,
    bool* constraint_is_equality  // nullptr if all inequalities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        Scalar val = y[i] + sigma * (Ax_bar[i] - b[i]);
        // Projection for inequality: y >= 0
        // For equality: no projection
        if (constraint_is_equality == nullptr || !constraint_is_equality[i]) {
            val = fmax(val, Scalar(0));
        }
        y_new[i] = val;
    }
}

// Extrapolation kernel
// x_bar = 2*x_new - x_old
template <typename Scalar>
__global__ void extrapolate_kernel(
    const Scalar* x_new,
    const Scalar* x_old,
    Scalar* x_bar,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x_bar[i] = Scalar(2) * x_new[i] - x_old[i];
    }
}
```

### 7.2 Batch Solve Kernel

```cpp
// src/cuprox/solvers/batch_pdhg_kernel.cuh

// Each thread block solves one problem
// Designed for small problems (n, m < 1024)
template <typename Scalar, int MAX_VARS, int MAX_CONSTRS>
__global__ void batch_pdhg_kernel(
    // Problem data (batched, contiguous)
    const Scalar* __restrict__ all_A_values,      // [batch, nnz_padded]
    const int* __restrict__ all_A_row_offsets,    // [batch, m+1]
    const int* __restrict__ all_A_col_indices,    // [batch, nnz_padded]
    const Scalar* __restrict__ all_b,             // [batch, m]
    const Scalar* __restrict__ all_c,             // [batch, n]
    const Scalar* __restrict__ all_lb,            // [batch, n]
    const Scalar* __restrict__ all_ub,            // [batch, n]
    
    // Dimensions (same for all problems in this call)
    int n,
    int m,
    int nnz,
    
    // Solver parameters
    int max_iterations,
    Scalar tolerance,
    Scalar tau,
    Scalar sigma,
    
    // Output
    Scalar* __restrict__ all_x,                   // [batch, n]
    Scalar* __restrict__ all_y,                   // [batch, m]
    int* __restrict__ all_iterations,             // [batch]
    int* __restrict__ all_status                  // [batch]
) {
    // Each block = one problem
    int problem_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for this problem's state
    extern __shared__ Scalar shared_mem[];
    Scalar* x = shared_mem;                           // [n]
    Scalar* x_bar = x + MAX_VARS;                     // [n]
    Scalar* y = x_bar + MAX_VARS;                     // [m]
    Scalar* Ax = y + MAX_CONSTRS;                     // [m]
    Scalar* ATy = Ax + MAX_CONSTRS;                   // [n]
    
    // Load problem data pointers
    const Scalar* A_values = all_A_values + problem_idx * nnz;
    const int* A_row_offsets = all_A_row_offsets + problem_idx * (m + 1);
    const int* A_col_indices = all_A_col_indices + problem_idx * nnz;
    const Scalar* b = all_b + problem_idx * m;
    const Scalar* c = all_c + problem_idx * n;
    const Scalar* lb = all_lb + problem_idx * n;
    const Scalar* ub = all_ub + problem_idx * n;
    
    // Initialize x, y to zero
    for (int i = tid; i < n; i += blockDim.x) x[i] = Scalar(0);
    for (int i = tid; i < m; i += blockDim.x) y[i] = Scalar(0);
    for (int i = tid; i < n; i += blockDim.x) x_bar[i] = Scalar(0);
    __syncthreads();
    
    // Main PDHG loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        // 1. Compute A * x_bar → Ax (parallel SpMV within block)
        block_spmv(A_row_offsets, A_col_indices, A_values, x_bar, Ax, m, tid, blockDim.x);
        __syncthreads();
        
        // 2. Dual update: y = proj(y + sigma*(Ax - b))
        for (int i = tid; i < m; i += blockDim.x) {
            Scalar val = y[i] + sigma * (Ax[i] - b[i]);
            y[i] = fmax(val, Scalar(0));  // Inequality projection
        }
        __syncthreads();
        
        // 3. Compute A' * y → ATy (parallel SpMV^T within block)
        block_spmv_transpose(A_row_offsets, A_col_indices, A_values, y, ATy, n, m, tid, blockDim.x);
        __syncthreads();
        
        // 4. Primal update: x_new = proj(x - tau*(c + ATy))
        Scalar x_prev_local;
        for (int i = tid; i < n; i += blockDim.x) {
            x_prev_local = x[i];
            Scalar val = x[i] - tau * (c[i] + ATy[i]);
            val = fmax(val, lb[i]);
            val = fmin(val, ub[i]);
            x[i] = val;
            x_bar[i] = Scalar(2) * val - x_prev_local;
        }
        __syncthreads();
        
        // 5. Check convergence every 50 iterations
        if (iter % 50 == 49) {
            // Compute residuals (block reduction)
            // ... (omitted for brevity)
        }
    }
    
    // Write output
    Scalar* out_x = all_x + problem_idx * n;
    Scalar* out_y = all_y + problem_idx * m;
    for (int i = tid; i < n; i += blockDim.x) out_x[i] = x[i];
    for (int i = tid; i < m; i += blockDim.x) out_y[i] = y[i];
    if (tid == 0) {
        all_iterations[problem_idx] = max_iterations;
        all_status[problem_idx] = 0;  // Success
    }
}
```

### 7.3 Kernel Launch Configuration

```cpp
// Heuristics for optimal performance

// Single problem mode
constexpr int BLOCK_SIZE_SINGLE = 256;  // Threads per block

inline dim3 get_grid_size(int n) {
    return dim3((n + BLOCK_SIZE_SINGLE - 1) / BLOCK_SIZE_SINGLE);
}

// Batch mode
// One block per problem, threads within block cooperate
inline int get_batch_block_size(int problem_size) {
    if (problem_size <= 64) return 64;
    if (problem_size <= 128) return 128;
    if (problem_size <= 256) return 256;
    return 512;  // Max for register pressure
}

inline size_t get_batch_shared_mem(int n, int m) {
    // x, x_bar, y, Ax, ATy
    return sizeof(double) * (2*n + 2*m + n);
}
```

---

## 8. Memory Management

### 8.1 Allocation Strategy

```cpp
// src/cuprox/core/memory.cuh

class MemoryPool {
public:
    // Get singleton pool for current device
    static MemoryPool& get();
    
    // Allocate device memory
    template <typename T>
    T* allocate(size_t count);
    
    // Free device memory
    template <typename T>
    void deallocate(T* ptr);
    
    // Get current memory usage
    size_t bytes_allocated() const;
    size_t peak_bytes() const;
    
    // Clear all allocations (for cleanup)
    void clear();
    
private:
    // Use CUDA's built-in memory pool (CUDA 11.2+)
    cudaMemPool_t pool;
    std::atomic<size_t> current_bytes{0};
    std::atomic<size_t> peak_bytes_{0};
};

// RAII wrapper for device memory
template <typename T>
class DevicePtr {
public:
    explicit DevicePtr(size_t count) : size_(count) {
        ptr_ = MemoryPool::get().allocate<T>(count);
    }
    
    ~DevicePtr() {
        if (ptr_) MemoryPool::get().deallocate(ptr_);
    }
    
    // Move-only
    DevicePtr(DevicePtr&& other) noexcept;
    DevicePtr& operator=(DevicePtr&& other) noexcept;
    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};
```

### 8.2 Host-Device Transfer Optimization

```cpp
// Minimize transfers - key performance principle

// BAD: Transfer every iteration
for (int iter = 0; iter < max_iter; ++iter) {
    cudaMemcpy(x_host, x_dev, ...);  // SLOW!
    if (converged(x_host)) break;
}

// GOOD: Check on device, transfer only at end
for (int iter = 0; iter < max_iter; ++iter) {
    pdhg_iteration(state);
    if (iter % check_interval == 0) {
        compute_residuals_on_device(state);  // GPU kernel
        Scalar res = reduce_on_device(state.residual);  // Single scalar transfer
        if (res < tolerance) break;
    }
}
cudaMemcpy(x_host, state.x.data, ...);  // Only once at end
```

### 8.3 Memory Requirements

```
Single Problem Memory:
─────────────────────
Sparse matrix A (CSR):
  - row_offsets: (m+1) * 4 bytes
  - col_indices: nnz * 4 bytes  
  - values: nnz * 8 bytes (FP64)

Vectors:
  - x, x_bar, x_prev: 3 * n * 8 bytes
  - y: m * 8 bytes
  - Ax, ATy: (m + n) * 8 bytes
  - c, b, lb, ub: (2n + m) * 8 bytes
  - Scaling: (m + n) * 8 bytes

Total ≈ 16 * nnz + 48 * max(n, m) bytes

Example 1: 100K vars, 50K constraints, 0.1% density
  Total entries = 50K * 100K = 5×10^9
  nnz = 5×10^9 * 0.001 = 5×10^6 = 5M
  Sparse memory: 16 * 5M = 80 MB
  Vector memory: 48 * 100K = 4.8 MB
  Total ≈ 85 MB

Example 2: 1M vars, 500K constraints, 0.01% density
  Total entries = 500K * 1M = 5×10^11
  nnz = 5×10^11 * 0.0001 = 5×10^7 = 50M
  Sparse memory: 16 * 50M = 800 MB
  Vector memory: 48 * 1M = 48 MB
  Total ≈ 850 MB

RTX A6000 (48 GB) can handle problems ~50x larger than Example 2.
```

---

## 9. Python API Design

### 9.1 High-Level Interface

```python
# python/cuprox/__init__.py

"""
cuProx: GPU-Accelerated First-Order LP/QP Solver

Example:
    >>> import cuprox
    >>> model = cuprox.Model()
    >>> x = model.add_var(lb=0, name="x")
    >>> y = model.add_var(lb=0, name="y")
    >>> model.add_constr(x + 2*y <= 10)
    >>> model.minimize(-x - y)
    >>> result = model.solve()
    >>> print(result.status, result.objective)
    optimal -5.0
"""

from .model import Model
from .solver import solve, solve_batch
from .result import SolveResult, Status
from .exceptions import CuproxError, InfeasibleError, UnboundedError
from ._core import __cuda_available__, __version__

__all__ = [
    "Model",
    "solve",
    "solve_batch", 
    "SolveResult",
    "Status",
    "CuproxError",
    "InfeasibleError", 
    "UnboundedError",
]
```

### 9.2 Model Builder (Algebraic Interface)

```python
# python/cuprox/model.py

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
import numpy as np
from scipy import sparse

@dataclass
class Variable:
    """Decision variable in the model."""
    index: int
    lb: float = 0.0
    ub: float = float('inf')
    name: Optional[str] = None
    
    def __add__(self, other): ...
    def __mul__(self, other): ...
    def __neg__(self): ...
    # Operator overloading for algebraic syntax

@dataclass  
class LinearExpr:
    """Linear expression: sum of coef * var + constant."""
    terms: Dict[int, float]  # var_index -> coefficient
    constant: float = 0.0
    
    def __add__(self, other): ...
    def __le__(self, other): return Constraint(self, "<=", other)
    def __ge__(self, other): return Constraint(self, ">=", other)
    def __eq__(self, other): return Constraint(self, "==", other)

@dataclass
class Constraint:
    """Linear constraint."""
    lhs: LinearExpr
    sense: str  # "<=", ">=", "=="
    rhs: float
    name: Optional[str] = None

class Model:
    """
    Optimization model builder with algebraic syntax.
    
    Example:
        model = Model()
        x = model.add_var(lb=0, ub=10, name="x")
        y = model.add_var(lb=0, name="y")
        model.add_constr(x + 2*y <= 20)
        model.minimize(-5*x - 4*y)
        result = model.solve()
    """
    
    def __init__(self):
        self._vars: List[Variable] = []
        self._constrs: List[Constraint] = []
        self._objective: Optional[LinearExpr] = None
        self._sense: str = "minimize"  # or "maximize"
        
    def add_var(
        self,
        lb: float = 0.0,
        ub: float = float('inf'),
        name: Optional[str] = None
    ) -> Variable:
        """Add a single decision variable."""
        var = Variable(index=len(self._vars), lb=lb, ub=ub, name=name)
        self._vars.append(var)
        return var
    
    def add_vars(
        self,
        count: int,
        lb: Union[float, np.ndarray] = 0.0,
        ub: Union[float, np.ndarray] = float('inf'),
        name_prefix: str = "x"
    ) -> List[Variable]:
        """Add multiple decision variables."""
        ...
        
    def add_constr(
        self,
        constraint: Constraint,
        name: Optional[str] = None
    ) -> Constraint:
        """Add a single constraint."""
        if name:
            constraint.name = name
        self._constrs.append(constraint)
        return constraint
    
    def add_constrs(self, constraints: List[Constraint]) -> List[Constraint]:
        """Add multiple constraints."""
        for c in constraints:
            self._constrs.append(c)
        return constraints
    
    def minimize(self, expr: Union[LinearExpr, Variable, float]) -> None:
        """Set minimization objective."""
        self._objective = self._to_expr(expr)
        self._sense = "minimize"
        
    def maximize(self, expr: Union[LinearExpr, Variable, float]) -> None:
        """Set maximization objective."""
        self._objective = self._to_expr(expr)
        self._sense = "maximize"
    
    def solve(
        self,
        params: Optional[Dict] = None,
        warm_start: Optional["SolveResult"] = None
    ) -> "SolveResult":
        """Solve the model."""
        from .solver import solve
        A, b, c, lb, ub, constraint_senses = self._to_standard_form()
        return solve(
            c=c, A=A, b=b, lb=lb, ub=ub,
            constraint_senses=constraint_senses,
            params=params,
            warm_start=warm_start
        )
    
    @classmethod
    def from_matrices(
        cls,
        c: np.ndarray,
        A_ub: Optional[sparse.csr_matrix] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[sparse.csr_matrix] = None,
        b_eq: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        P: Optional[sparse.csr_matrix] = None,  # For QP
    ) -> "Model":
        """Create model directly from matrices (for large-scale problems)."""
        ...
        
    def _to_standard_form(self):
        """Convert to matrix form for solver."""
        ...
```

### 9.3 Matrix-Based Interface (Large-Scale)

```python
# python/cuprox/solver.py

from typing import Optional, Dict, List, Union
import numpy as np
from scipy import sparse
from . import _core
from .result import SolveResult

def solve(
    c: np.ndarray,
    A: sparse.csr_matrix,
    b: np.ndarray,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    P: Optional[sparse.csr_matrix] = None,  # For QP
    q: Optional[np.ndarray] = None,          # For QP (alias for c)
    constraint_senses: Optional[np.ndarray] = None,  # '<', '=', '>'
    params: Optional[Dict] = None,
    warm_start: Optional[SolveResult] = None,
) -> SolveResult:
    """
    Solve LP or QP in matrix form.
    
    LP: minimize c'x subject to Ax <= b, lb <= x <= ub
    QP: minimize (1/2)x'Px + q'x subject to l <= Ax <= u
    
    Note: For QP, we use two-sided constraints l <= Ax <= u.
    For one-sided Ax <= b, set l = -inf, u = b.
    Variable bounds lb <= x <= ub are handled separately.
    
    Args:
        c: Linear objective coefficients (n,)
        A: Constraint matrix (m, n) in CSR format
        b: Constraint RHS (m,)
        lb: Variable lower bounds (n,), default 0
        ub: Variable upper bounds (n,), default inf
        P: Quadratic objective matrix (n, n), optional (makes it QP)
        q: Alias for c in QP formulation
        constraint_senses: '<', '=', or '>' for each constraint
        params: Solver parameters
        warm_start: Previous solution for warm starting
        
    Returns:
        SolveResult with status, objective, x, y, etc.
    """
    # Validate inputs
    _validate_inputs(c, A, b, lb, ub, P)
    
    # Set defaults
    n = len(c)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, np.inf)
    
    # Merge params with defaults
    params = {**DEFAULT_PARAMS, **(params or {})}
    
    # Select solver based on problem type
    is_qp = P is not None and P.nnz > 0
    
    # Convert to CSR if needed
    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)
    
    # Convert constraint senses to l, u bounds
    # For QP: l <= Ax <= u format
    # '<=' : l = -inf, u = b
    # '>=' : l = b, u = +inf  
    # '==' : l = b, u = b
    l_bound, u_bound = _convert_constraints(b, constraint_senses)
    
    # Call C++ solver
    if is_qp:
        result = _core.solve_qp(
            P_data=P.data, P_indices=P.indices, P_indptr=P.indptr,
            A_data=A.data, A_indices=A.indices, A_indptr=A.indptr,
            q=q if q is not None else c,
            l=l_bound,
            u=u_bound,
            params=params,
            warm_x=warm_start.x if warm_start else None,
            warm_y=warm_start.y if warm_start else None,
        )
    else:
        result = _core.solve_lp(
            A_data=A.data, A_indices=A.indices, A_indptr=A.indptr,
            b=b, c=c, lb=lb, ub=ub,
            params=params,
            warm_x=warm_start.x if warm_start else None,
            warm_y=warm_start.y if warm_start else None,
        )
    
    return SolveResult.from_raw(result)


def solve_batch(
    problems: List[Dict],
    params: Optional[Dict] = None,
) -> List[SolveResult]:
    """
    Solve many LP/QP problems in parallel on GPU.
    
    This is cuProx's killer feature - massive parallelism for
    many small problems (e.g., ML training, Monte Carlo).
    
    Args:
        problems: List of problem dicts with keys:
            - c, A, b (required)
            - lb, ub (optional)
            - P (optional, for QP)
        params: Solver parameters (shared across all problems)
        
    Returns:
        List of SolveResult, one per problem
        
    Example:
        >>> problems = [
        ...     {"c": c1, "A": A1, "b": b1},
        ...     {"c": c2, "A": A2, "b": b2},
        ...     # ... 1000s more
        ... ]
        >>> results = cuprox.solve_batch(problems)
    """
    ...
```

### 9.4 PyTorch Integration (Future P1)

```python
# python/cuprox/torch.py

"""
PyTorch integration for differentiable optimization.

This enables using cuProx as a layer in neural network training,
following the approach of cvxpylayers but GPU-accelerated.
"""

import torch
from torch.autograd import Function
from . import solve_batch

class QPLayer(Function):
    """
    Differentiable QP layer.
    
    Forward pass solves: min (1/2)x'Px + q'x s.t. Ax <= b
    Backward pass computes gradients via implicit differentiation.
    """
    
    @staticmethod
    def forward(ctx, P, q, A, b, x_warmstart=None):
        """
        Args:
            P: (batch, n, n) or (n, n) quadratic cost
            q: (batch, n) linear cost
            A: (batch, m, n) or (m, n) constraint matrix
            b: (batch, m) constraint RHS
            
        Returns:
            x: (batch, n) optimal solutions
        """
        # Convert to numpy, solve batch
        # ... implementation
        
        # Save for backward
        ctx.save_for_backward(P, q, A, b, x_opt, y_opt)
        return x_opt
    
    @staticmethod
    def backward(ctx, grad_x):
        """
        Compute gradients via implicit differentiation of KKT conditions.
        """
        P, q, A, b, x, y = ctx.saved_tensors
        
        # Implicit differentiation of KKT:
        # [P   A'] [dx]   [dq]
        # [A   0 ] [dy] = [db]
        
        # ... implementation using cuProx for the linear system
        
        return dP, dq, dA, db, None
```

---

## 10. Performance Targets

### 10.1 Benchmark Problems

| Category | Problem | Size (n × m) | Density | Source |
|----------|---------|--------------|---------|--------|
| **Netlib LP** | afiro | 27 × 51 | 10% | Standard |
| **Netlib LP** | blend | 83 × 114 | 8% | Standard |
| **Netlib LP** | pilot4 | 410 × 1123 | 2% | Standard |
| **Large LP** | pds-20 | 33K × 108K | 0.01% | MIPLIB |
| **ML QP** | Portfolio 1K | 1000 × 1000 | 100% (dense) | Synthetic |
| **MPC QP** | Robot arm | 120 × 60 | 15% | Control |
| **Batch** | 10K small LP | 100 × 50 each | 10% | Synthetic |

### 10.2 Target Performance

| Benchmark | Baseline (CPU) | cuProx Target | Speedup |
|-----------|---------------|---------------|---------|
| Netlib afiro | 1 ms (HiGHS) | 0.5 ms | 2x (overhead limited) |
| Netlib blend | 2 ms (HiGHS) | 0.8 ms | 2.5x |
| Netlib pilot4 | 50 ms (HiGHS) | 10 ms | 5x |
| pds-20 | 30 s (SciPy) | 2 s | 15x |
| Portfolio 1K QP | 100 ms (OSQP) | 5 ms | 20x |
| MPC QP (1000 solves) | 10 s (OSQP) | 0.1 s | 100x |
| Batch 10K LP | 60 s (loop) | 0.5 s | 120x |

### 10.3 Performance Validation

```python
# benchmarks/benchmark_suite.py

BENCHMARK_SUITE = [
    # Netlib LP problems
    {"name": "netlib_afiro", "file": "afiro.mps", "expected_obj": -464.753},
    {"name": "netlib_blend", "file": "blend.mps", "expected_obj": -30.812},
    # ...
    
    # Synthetic batch
    {"name": "batch_1k_small", "generator": "random_lp", "n": 100, "m": 50, "count": 1000},
    {"name": "batch_10k_small", "generator": "random_lp", "n": 100, "m": 50, "count": 10000},
    
    # QP benchmarks
    {"name": "qp_portfolio_1k", "generator": "portfolio_qp", "n": 1000},
    {"name": "qp_mpc_robot", "generator": "mpc_qp", "horizon": 20, "states": 6},
]

def run_benchmarks():
    results = []
    for benchmark in BENCHMARK_SUITE:
        problem = load_or_generate(benchmark)
        
        # Warm up
        cuprox.solve(**problem)
        
        # Timed runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = cuprox.solve(**problem)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
        results.append({
            "name": benchmark["name"],
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "status": result.status,
            "iterations": result.iterations,
        })
    
    return results
```

---

## 11. Quality Assurance

### 11.1 Correctness Testing Strategy

```
Test Pyramid:
────────────

                    ┌─────────────────┐
                    │  Benchmarks     │  ← Compare to Gurobi/OSQP solutions
                    │  (10 problems)  │
                    └────────┬────────┘
                             │
               ┌─────────────┴─────────────┐
               │   Integration Tests        │  ← Python API, end-to-end
               │   (50 test cases)          │
               └─────────────┬──────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │          Unit Tests                     │  ← Each kernel, function
        │          (200+ test cases)              │
        └─────────────────────────────────────────┘
```

### 11.2 Numerical Validation

```python
# tests/python/test_numerical.py

def test_solution_satisfies_constraints():
    """Verify Ax <= b within tolerance."""
    result = model.solve()
    violation = np.maximum(A @ result.x - b, 0)
    assert np.max(violation) < 1e-6

def test_solution_satisfies_bounds():
    """Verify lb <= x <= ub."""
    result = model.solve()
    assert np.all(result.x >= lb - 1e-8)
    assert np.all(result.x <= ub + 1e-8)

def test_objective_matches_reference():
    """Compare objective to known optimal."""
    result = model.solve()
    assert abs(result.objective - expected_obj) / abs(expected_obj) < 1e-4

def test_duality_gap():
    """Verify duality gap is small."""
    result = model.solve()
    primal_obj = c @ result.x
    dual_obj = b @ result.y
    gap = abs(primal_obj - dual_obj) / (1 + abs(primal_obj))
    assert gap < 1e-4
```

### 11.3 Robustness Testing

```python
# tests/python/test_robustness.py

@pytest.mark.parametrize("scale", [1e-10, 1e-5, 1, 1e5, 1e10])
def test_handles_various_scales(scale):
    """Solver works across wide range of coefficient magnitudes."""
    A_scaled = A * scale
    b_scaled = b * scale
    result = solve(A=A_scaled, b=b_scaled, c=c)
    assert result.status == "optimal"

def test_handles_degenerate_constraints():
    """Handles redundant/degenerate constraints."""
    # Add duplicate constraint
    A_degen = sparse.vstack([A, A[0:1, :]])
    b_degen = np.append(b, b[0])
    result = solve(A=A_degen, b=b_degen, c=c)
    assert result.status == "optimal"

def test_detects_infeasibility():
    """Correctly identifies infeasible problems."""
    # x >= 1 and x <= 0
    result = solve(...)
    assert result.status == "infeasible"

def test_handles_unbounded():
    """Correctly identifies unbounded problems."""
    # min -x with no upper bound
    result = solve(...)
    assert result.status == "unbounded"

def test_respects_time_limit():
    """Terminates within time limit."""
    result = solve(..., params={"time_limit": 0.1})
    assert result.solve_time < 0.2  # Some margin
    assert result.status in ["optimal", "time_limit"]
```

---

## Appendix A: References

### Algorithms
1. Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems.
2. Lu, H., et al. (2023). cuPDLP: GPU-accelerated LP solver using first-order methods.
3. Stellato, B., et al. (2020). OSQP: An operator splitting solver for quadratic programs.
4. Applegate, D., et al. (2021). Practical large-scale linear programming using PDLP.

### Implementation
5. NVIDIA cuSPARSE Documentation
6. NVIDIA cuBLAS Documentation  
7. pybind11 Documentation
8. scikit-build-core Documentation

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **LP** | Linear Program: min c'x s.t. Ax ≤ b |
| **QP** | Quadratic Program: min (1/2)x'Px + q'x s.t. Ax ≤ b |
| **PDHG** | Primal-Dual Hybrid Gradient algorithm |
| **ADMM** | Alternating Direction Method of Multipliers |
| **CSR** | Compressed Sparse Row matrix format |
| **SpMV** | Sparse Matrix-Vector multiplication |
| **Ruiz scaling** | Equilibration method for matrix conditioning |

---

*Document Version: 1.0.0*  
*This architecture is designed for implementation. Each component has been carefully considered for GPU suitability, correctness, and practical usability.*

