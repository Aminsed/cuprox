# cuProx Research & Mathematical Foundations

This document provides the theoretical background for cuProx's algorithms, convergence guarantees, and design decisions.

---

## Table of Contents

1. [Problem Formulations](#1-problem-formulations)
2. [PDHG Algorithm for LP](#2-pdhg-algorithm-for-lp)
3. [ADMM Algorithm for QP](#3-admm-algorithm-for-qp)
4. [Preconditioning & Scaling](#4-preconditioning--scaling)
5. [Convergence Analysis](#5-convergence-analysis)
6. [Infeasibility Detection](#6-infeasibility-detection)
7. [Numerical Considerations](#7-numerical-considerations)
8. [Literature Review](#8-literature-review)

---

## 1. Problem Formulations

### 1.1 Linear Program (LP)

**Standard Form:**
```
minimize    c'x
subject to  Ax ≤ b
            lb ≤ x ≤ ub
```

Where:
- x ∈ ℝⁿ is the decision variable
- c ∈ ℝⁿ is the objective coefficient vector
- A ∈ ℝᵐˣⁿ is the constraint matrix
- b ∈ ℝᵐ is the constraint right-hand side
- lb, ub ∈ ℝⁿ are variable bounds

**Saddle-Point Reformulation:**

PDHG solves LP by reformulating it as a saddle-point problem:

```
min_x max_y  L(x, y) = c'x + y'(Ax - b) + δ_X(x) - δ_Y(y)
```

Where:
- δ_X(x) = 0 if lb ≤ x ≤ ub, +∞ otherwise (box constraint indicator)
- δ_Y(y) = 0 if y ≥ 0, +∞ otherwise (dual non-negativity for inequalities)

### 1.2 Quadratic Program (QP)

**Standard Form:**
```
minimize    (1/2)x'Px + q'x
subject to  l ≤ Ax ≤ u
```

Where:
- P ∈ ℝⁿˣⁿ is positive semidefinite (P ⪰ 0)
- q ∈ ℝⁿ is the linear objective term
- l, u ∈ ℝᵐ are constraint bounds (l can be -∞, u can be +∞)

**KKT Conditions:**
```
Px + q + A'y = 0          (stationarity)
l ≤ Ax ≤ u                 (primal feasibility)
y_i(Ax - u)_i = 0, ∀i     (complementarity for upper)
y_i(Ax - l)_i = 0, ∀i     (complementarity for lower)
```

---

## 2. PDHG Algorithm for LP

### 2.1 Basic PDHG (Chambolle-Pock)

The Primal-Dual Hybrid Gradient method iterates:

```
y_{k+1} = prox_{σF*}(y_k + σK x̄_k)
x_{k+1} = prox_{τG}(x_k - τK' y_{k+1})
x̄_{k+1} = x_{k+1} + θ(x_{k+1} - x_k)
```

For LP with our saddle-point formulation:
- K = A (constraint matrix)
- G(x) = c'x + δ_X(x)
- F*(y) = b'y + δ_Y(y)
- θ = 1 (over-relaxation parameter)

**Expanded Iteration:**
```
y_{k+1} = Π_Y(y_k + σ(A x̄_k - b))        # Dual update with projection
x_{k+1} = Π_X(x_k - τ(c + A' y_{k+1}))   # Primal update with projection
x̄_{k+1} = 2x_{k+1} - x_k                 # Extrapolation
```

Where:
- Π_Y(y) = max(y, 0) for inequality constraints (project to non-negative orthant)
- Π_X(x) = clip(x, lb, ub) for box constraints

### 2.2 Step Size Selection

**Condition for Convergence:**
```
τσ ||A||² < 1
```

Where ||A|| is the spectral norm (largest singular value).

**Practical Choices:**

1. **Conservative (with safety margin):**
   ```
   τ = σ = 0.9 / ||A||₂
   ```
   The 0.9 factor provides margin below the τσ||A||² < 1 boundary.
   Using exactly 1/||A||₂ puts us on the boundary (τσ||A||² = 1).

2. **Balanced:**
   ```
   τ = 0.9 / ||A||_{col}  (max column 2-norm)
   σ = 0.9 / ||A||_{row}  (max row 2-norm)
   ```
   Often faster in practice.

3. **Adaptive (Malitsky-Pock):**
   Updates step sizes based on iteration progress. Can accelerate convergence.

### 2.3 Accelerated PDHG with Restarts

Standard PDHG has O(1/k) ergodic convergence rate. With adaptive restarts, we achieve O(1/k²).

**Chambolle-Pock Extrapolation (what we implement):**

We use over-relaxation with parameter θ = 1:
```
x̄_{k+1} = 2 x_{k+1} - x_k
```

This is combined with adaptive restarts to achieve acceleration.

**Halpern Averaging (alternative approach):**

Classical Halpern iteration uses weighted averaging with an anchor point:
```
x̄_{k+1} = (1 - η_{k+1}) x̄_k + η_{k+1} x_{k+1}
ȳ_{k+1} = (1 - η_{k+1}) ȳ_k + η_{k+1} y_{k+1}
```

With η_k = 2/(k+2), achieving accelerated rates. cuPDLP uses a variant of this.

**Note:** We primarily implement the simpler extrapolation + restarts approach,
not the full Halpern averaging scheme, unless specifically configured.

**Adaptive Restart Criterion:**

Restart the averaging when:
```
||r_{k+1}||/||r_k|| > γ    (e.g., γ = 0.999)
```

Where r_k is the normalized primal-dual residual.

### 2.4 Complete PDHG Algorithm

```python
def pdhg_lp(A, b, c, lb, ub, tol, max_iter):
    # Initialize
    n, m = A.shape[1], A.shape[0]
    x, y = zeros(n), zeros(m)
    x_bar = x.copy()
    
    # Compute step sizes
    tau = sigma = 1.0 / spectral_norm(A)
    
    # Scale problem (Ruiz equilibration)
    A_scaled, b_scaled, c_scaled, D1, D2 = ruiz_scaling(A, b, c)
    
    for k in range(max_iter):
        # 1. Dual update
        Ax_bar = A_scaled @ x_bar
        y_new = project_dual(y + sigma * (Ax_bar - b_scaled))
        
        # 2. Primal update  
        ATy = A_scaled.T @ y_new
        x_new = project_primal(x - tau * (c_scaled + ATy), lb, ub)
        
        # 3. Extrapolation
        x_bar = 2 * x_new - x
        
        # 4. Check convergence (every N iterations)
        if k % 50 == 0:
            primal_res = norm(A_scaled @ x_new - project_dual(A_scaled @ x_new))
            dual_res = norm(x_new - project_primal(x_new - (c_scaled + A_scaled.T @ y_new)))
            if max(primal_res, dual_res) < tol:
                break
        
        x, y = x_new, y_new
    
    # Unscale solution
    x_final = D2 @ x
    y_final = D1 @ y
    
    return x_final, y_final
```

---

## 3. ADMM Algorithm for QP

### 3.1 Problem Splitting

ADMM solves QP by introducing auxiliary variable z:
```
minimize    (1/2)x'Px + q'x
subject to  Ax = z
            l ≤ z ≤ u
```

**Augmented Lagrangian:**
```
L_ρ(x, z, y) = (1/2)x'Px + q'x + y'(Ax - z) + (ρ/2)||Ax - z||²
```

### 3.2 ADMM Iteration

```
x_{k+1} = argmin_x L_ρ(x, z_k, y_k)                    # x-update
z_{k+1} = argmin_z L_ρ(x_{k+1}, z, y_k)                # z-update
y_{k+1} = y_k + ρ(A x_{k+1} - z_{k+1})                 # dual update
```

**Explicit Forms:**

1. **x-update (linear system):**
   ```
   (P + ρA'A) x_{k+1} = ρA'z_k - A'y_k - q
   ```
   
   **Dimension check:** x ∈ ℝⁿ, z ∈ ℝᵐ, y ∈ ℝᵐ, A ∈ ℝᵐˣⁿ
   - LHS: (n×n)(n×1) = n×1 ✓
   - RHS: ρA'z_k (n) - A'y_k (n) - q (n) = n×1 ✓
   
   This requires solving a linear system. Options:
   - Direct: Factorize (P + ρA'A) once if constant
   - Iterative: Conjugate Gradient with preconditioning

2. **z-update (projection):**
   ```
   z_{k+1} = Π_{[l,u]}(A x_{k+1} + y_k/ρ)
   ```
   Simple box projection, embarrassingly parallel.

3. **Dual update:**
   ```
   y_{k+1} = y_k + ρ(A x_{k+1} - z_{k+1})
   ```
   Vector addition, embarrassingly parallel.

### 3.3 OSQP-Style Enhancements

cuProx implements ADMM enhancements from OSQP:

1. **ρ adaptation:** Adjust ρ based on primal/dual residual ratio
2. **Warm starting:** Initialize from previous solution
3. **Polishing:** Final refinement step for active set

### 3.4 Complete ADMM Algorithm

```python
def admm_qp(P, q, A, l, u, tol, max_iter, rho=1.0):
    n, m = A.shape[1], A.shape[0]
    x, z, y = zeros(n), zeros(m), zeros(m)
    
    # Factorize KKT matrix (do once)
    # [P + σI    A']   (regularized for numerical stability)
    # [A       -1/ρI]
    KKT = form_kkt_matrix(P, A, rho)
    factor = ldl_factorize(KKT)
    
    for k in range(max_iter):
        # 1. x-update: solve linear system
        rhs = -q + A.T @ (rho * z - y)
        x_new = kkt_solve(factor, rhs)
        
        # 2. z-update: projection
        z_tilde = A @ x_new + y / rho
        z_new = clip(z_tilde, l, u)
        
        # 3. Dual update
        y_new = y + rho * (A @ x_new - z_new)
        
        # 4. Residuals
        primal_res = norm(A @ x_new - z_new)
        dual_res = rho * norm(A.T @ (z_new - z))
        
        if max(primal_res, dual_res) < tol:
            break
        
        x, z, y = x_new, z_new, y_new
    
    return x, y
```

---

## 4. Preconditioning & Scaling

### 4.1 Why Scaling is Critical

First-order methods are sensitive to problem conditioning. A poorly scaled problem can cause:
- Slow convergence (or divergence)
- Numerical instability
- Suboptimal step size selection

**Example:** If ||A[i,:]|| varies from 1e-6 to 1e6 across rows, standard step sizes will be very conservative.

### 4.2 Ruiz Equilibration

Ruiz equilibration finds diagonal matrices D₁, D₂ such that:
```
Ã = D₁ A D₂
```
has all rows and columns with approximately unit norm.

**Algorithm:**
```python
def ruiz_equilibration(A, max_iter=10):
    D1, D2 = eye(m), eye(n)
    
    for _ in range(max_iter):
        # Row scaling
        row_norms = max(abs(A), axis=1)
        d1 = 1 / sqrt(row_norms + epsilon)
        D1 = diag(d1) @ D1
        A = diag(d1) @ A
        
        # Column scaling
        col_norms = max(abs(A), axis=0)
        d2 = 1 / sqrt(col_norms + epsilon)
        D2 = D2 @ diag(d2)
        A = A @ diag(d2)
    
    return A, D1, D2
```

**GPU Implementation:**
Each iteration requires computing row/column norms (parallel reductions) and element-wise scaling (embarrassingly parallel).

### 4.3 Scaling Application

**Before Solve:**
```
Ã = D₁ A D₂
b̃ = D₁ b
c̃ = D₂ c (with objective scaling factor)
```

**After Solve:**
```
x = D₂ x̃
y = D₁ ỹ (with objective scaling)
```

### 4.4 Other Preconditioning Options

1. **Geometric mean scaling:** Similar to Ruiz but uses geometric mean
2. **Pock-Chambolle preconditioning:** Uses diagonal approximation of A'A
3. **Problem-specific:** For structured problems (e.g., network flow)

---

## 5. Convergence Analysis

### 5.1 PDHG Convergence Rate

**Theorem (Chambolle-Pock 2011):**
For convex-concave saddle-point problems, PDHG with θ=1 converges at rate O(1/k) in the ergodic sense:
```
|L(x̄_k, y*) - L(x*, ȳ_k)| ≤ C/k
```

Where x̄_k, ȳ_k are the running averages.

**With Restarts:**
Adaptive restarts achieve O(1/k²) rate (Roulet & d'Aspremont 2017).

### 5.2 ADMM Convergence Rate

**Theorem (He & Yuan 2012):**
ADMM converges at rate O(1/k) for convex problems:
```
||r_k|| ≤ C/k
```

Where r_k is the primal-dual residual.

**Practical Observation:**
On well-conditioned problems, ADMM often exhibits linear convergence initially, then slows to O(1/k).

### 5.3 Convergence Criteria

We use relative residuals following OSQP conventions:

**Primal Residual:**
```
ε_pri = ||Ax - z||_∞ / max(||Ax||_∞, ||z||_∞, 1)
```

**Dual Residual:**
```
ε_dual = ||Px + q + A'y||_∞ / max(||Px||_∞, ||q||_∞, ||A'y||_∞, 1)
```

**Convergence:**
```
converged = (ε_pri < ε_abs + ε_rel * scale_pri) AND
            (ε_dual < ε_abs + ε_rel * scale_dual)
```

With defaults: ε_abs = 1e-3, ε_rel = 1e-4.

---

## 6. Infeasibility Detection

### 6.1 Primal Infeasibility

A problem is primal infeasible if there exists y ≠ 0 such that:
```
A'y = 0
b'y < 0  (for Ax ≤ b)
y ≥ 0
```

**Detection in PDHG:**
If ||y_k|| → ∞ while ||A'y_k/||y_k|||| → 0 and b'y_k/||y_k|| < 0:
```
The problem is primal infeasible
```

**Practical Test:**
```python
def check_primal_infeasibility(y, A, b, tol):
    y_norm = norm(y)
    if y_norm < 1:
        return False
    
    y_scaled = y / y_norm
    if norm(A.T @ y_scaled) < tol and b @ y_scaled < -tol:
        return True  # Certificate of infeasibility
    return False
```

### 6.2 Dual Infeasibility (Unboundedness)

A problem is unbounded if there exists x ≠ 0 such that:
```
Ax ≤ 0
c'x < 0
x in bounds direction
```

**Detection:**
If ||x_k|| → ∞ while cost decreases unboundedly.

---

## 7. Numerical Considerations

### 7.1 Floating-Point Precision

| Precision | Range | Epsilon | Use Case |
|-----------|-------|---------|----------|
| FP16 | ±65504 | 9.77e-4 | ML training (not for optimization) |
| FP32 | ±3.4e38 | 1.19e-7 | Fast iterations, moderate accuracy |
| FP64 | ±1.8e308 | 2.22e-16 | Default, high accuracy |

**Our Approach:**
- Default: FP64 for robustness
- Option: FP32 for 2x speedup when accuracy permits
- Mixed: FP32 iterations, FP64 residual computation

### 7.2 Numerical Stability Techniques

1. **Scaling:** Ruiz equilibration (essential)
2. **Regularization:** Add small ε to diagonal for near-singular systems
3. **Overflow prevention:** Clamp values to prevent inf
4. **NaN detection:** Check inputs and intermediate results

### 7.3 GPU-Specific Considerations

1. **Warp divergence:** Minimize in projection kernels
2. **Memory coalescing:** Ensure aligned access patterns
3. **Atomic operations:** Avoid in hot loops
4. **Reduction accuracy:** Use Kahan summation for large reductions

---

## 8. Literature Review

### 8.1 Foundational Papers

1. **Chambolle, A., & Pock, T. (2011).** "A first-order primal-dual algorithm for convex problems with applications to imaging." *Journal of Mathematical Imaging and Vision*, 40(1), 120-145.
   - Original PDHG algorithm
   - O(1/k) convergence proof

2. **Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020).** "OSQP: An operator splitting solver for quadratic programs." *Mathematical Programming Computation*, 12(4), 637-672.
   - ADMM for QP with practical enhancements
   - ρ adaptation, polishing, warm starting

3. **Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2023).** "Practical large-scale linear programming using primal-dual hybrid gradient." *NeurIPS*.
   - PDLP/cuPDLP algorithm
   - Halpern iteration, adaptive restarts

### 8.2 GPU Optimization Papers

4. **Lu, H., et al. (2023).** "cuPDLP.jl: A GPU implementation of restarted primal-dual hybrid gradient for linear programming."
   - GPU implementation strategies
   - Benchmark results vs. Gurobi

5. **Bell, N., & Garland, M. (2009).** "Implementing sparse matrix-vector multiplication on throughput-oriented processors." *SC'09*.
   - SpMV on GPU
   - CSR format optimization

### 8.3 Acceleration and Restarts

6. **Roulet, V., & d'Aspremont, A. (2020).** "Sharpness, restart, and acceleration." *NeurIPS*.
   - Restart schemes for acceleration
   - O(1/k²) rates

7. **Applegate, D., Hinder, O., Lu, H., & Lubin, M. (2021).** "Infeasibility detection with primal-dual hybrid gradient for large-scale linear programming."
   - Infeasibility certificates
   - Robust termination

### 8.4 Preconditioning

8. **Ruiz, D. (2001).** "A scaling algorithm to equilibrate both rows and columns norms in matrices." *Technical Report RAL-TR-2001-034*.
   - Ruiz equilibration algorithm

9. **Pock, T., & Chambolle, A. (2011).** "Diagonal preconditioning for first order primal-dual algorithms in convex optimization."
   - Diagonal preconditioners for PDHG

---

## Appendix: Algorithm Pseudocode Summary

### PDHG for LP (Production Version)

```
Input: A ∈ ℝᵐˣⁿ, b ∈ ℝᵐ, c ∈ ℝⁿ, lb, ub ∈ ℝⁿ
Parameters: τ, σ, tol, max_iter

1. Preprocessing:
   (A, b, c, D1, D2) ← RuizScale(A, b, c)
   Estimate ||A|| for step sizes
   
2. Initialize:
   x ← 0, y ← 0, x̄ ← 0
   
3. Main Loop (k = 0, 1, ...):
   # Dual update
   v ← y + σ(A x̄ - b)
   y ← max(v, 0)  [for inequalities]
   
   # Primal update  
   x_prev ← x
   w ← x - τ(c + Aᵀy)
   x ← clip(w, lb, ub)
   
   # Extrapolation with restart check
   x̄ ← 2x - x_prev
   If restart_needed(k):
       Reset averaging
       
   # Convergence check (every 50 iter)
   If mod(k, 50) = 0:
       r_p ← ||Ax - slack||
       r_d ← ||c + Aᵀy - reduced_cost||
       If max(r_p, r_d) < tol: break

4. Postprocessing:
   x ← D2 · x  [unscale]
   y ← D1 · y  [unscale]
   
Output: (x*, y*, status, iterations)
```

### ADMM for QP (Production Version)

```
Input: P ∈ ℝⁿˣⁿ, q ∈ ℝⁿ, A ∈ ℝᵐˣⁿ, l, u ∈ ℝᵐ
Parameters: ρ, tol, max_iter

1. Preprocessing:
   Scale problem
   Form KKT matrix: M = P + ρAᵀA
   Factor M (or setup CG)
   
2. Initialize:
   x ← 0, z ← 0, y ← 0
   
3. Main Loop (k = 0, 1, ...):
   # x-update (linear solve)
   rhs ← -q + Aᵀ(ρz - y)
   x ← M⁻¹ rhs
   
   # z-update (projection)
   z_tilde ← Ax + y/ρ
   z ← clip(z_tilde, l, u)
   
   # y-update
   y ← y + ρ(Ax - z)
   
   # Convergence check
   r_p ← ||Ax - z||
   r_d ← ρ||Aᵀ(z - z_prev)||
   If max(r_p, r_d) < tol: break
   
   # ρ adaptation (every 25 iter)
   If mod(k, 25) = 0:
       ρ ← adapt_rho(r_p, r_d, ρ)
       Refactor if ρ changed significantly

4. Polishing (optional):
   Identify active constraints
   Solve reduced equality-constrained QP
   
Output: (x*, y*, status, iterations)
```

---

*This document serves as the theoretical foundation for cuProx implementation. All algorithms are chosen for their GPU suitability and proven convergence properties.*

