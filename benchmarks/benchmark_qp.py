#!/usr/bin/env python3
"""
cuProx QP Benchmark: Compare against OSQP
"""

import sys
sys.path.insert(0, '../python')

import time
import numpy as np
from scipy import sparse

# Try to import cuprox
try:
    import cuprox
    HAS_CUPROX = True
    print(f"cuProx version: {cuprox.__version__}")
    if cuprox.__cuda_available__:
        from cuprox import _core
        print(f"GPU: {_core.get_device_name()}")
except ImportError:
    HAS_CUPROX = False
    print("cuProx not available")

# Try to import OSQP
try:
    import osqp
    HAS_OSQP = True
    print("OSQP available")
except ImportError:
    HAS_OSQP = False
    print("OSQP not available (pip install osqp)")

print()

def generate_qp(n, m, density=0.1, seed=42):
    """Generate a random sparse QP."""
    np.random.seed(seed)
    
    # Positive definite P (diagonal + sparse)
    P_diag = np.random.rand(n) + 0.1
    P = sparse.diags(P_diag, format='csc')
    
    # Constraint matrix
    A = sparse.random(m, n, density=density, format='csc', dtype=np.float64)
    A = A + sparse.eye(m, n, format='csc') * 0.1
    
    # Linear cost
    q = np.random.randn(n)
    
    # Constraint bounds (two-sided)
    l = -np.ones(m) * 10
    u = np.ones(m) * 10
    
    return P, q, A, l, u

def solve_cuprox_qp(P, q, A, l, u, **kwargs):
    """Solve QP with cuProx."""
    # Convert to CSR for cuProx
    A_csr = A.tocsr()
    P_csr = P.tocsr()
    
    start = time.perf_counter()
    result = cuprox.solve(
        c=q, A=A_csr, b=np.zeros(A.shape[0]), P=P_csr,
        constraint_l=l, constraint_u=u,
        params={'max_iterations': kwargs.get('max_iters', 4000),
                'tolerance': kwargs.get('tol', 1e-4)}
    )
    elapsed = time.perf_counter() - start
    return {
        'time': elapsed,
        'objective': result.objective,
        'status': result.status.value,
        'iterations': result.iterations,
    }

def solve_osqp(P, q, A, l, u, **kwargs):
    """Solve QP with OSQP."""
    start = time.perf_counter()
    
    prob = osqp.OSQP()
    prob.setup(P=P.tocsc(), q=q, A=A.tocsc(), l=l, u=u,
               verbose=False,
               eps_abs=kwargs.get('tol', 1e-4),
               eps_rel=kwargs.get('tol', 1e-4),
               max_iter=kwargs.get('max_iters', 4000))
    result = prob.solve()
    
    elapsed = time.perf_counter() - start
    
    status_map = {
        1: 'optimal',
        -2: 'max_iterations',
        -3: 'primal_infeasible',
        -4: 'dual_infeasible',
    }
    
    return {
        'time': elapsed,
        'objective': result.info.obj_val if result.info.status_val == 1 else float('nan'),
        'status': status_map.get(result.info.status_val, 'unknown'),
        'iterations': result.info.iter,
    }

def benchmark_single(n, m, density=0.1, seed=42):
    """Benchmark a single QP instance."""
    print(f"  Generating QP: n={n}, m={m}, density={density:.1%}")
    P, q, A, l, u = generate_qp(n, m, density, seed)
    
    results = {}
    
    # OSQP
    if HAS_OSQP:
        res = solve_osqp(P, q, A, l, u)
        results['osqp'] = res
        print(f"    OSQP:    {res['time']*1000:8.1f} ms, obj={res['objective']:10.4f}, "
              f"iters={res['iterations']}, status={res['status']}")
    
    # cuProx
    if HAS_CUPROX and cuprox.__cuda_available__:
        res = solve_cuprox_qp(P, q, A, l, u)
        results['cuprox'] = res
        print(f"    cuProx:  {res['time']*1000:8.1f} ms, obj={res['objective']:10.4f}, "
              f"iters={res['iterations']}, status={res['status']}")
    
    return results

def benchmark_scaling():
    """Benchmark across different problem sizes."""
    print("=" * 70)
    print("QP Scaling Benchmark")
    print("=" * 70)
    
    sizes = [
        (100, 50, 0.2),
        (500, 250, 0.1),
        (1000, 500, 0.05),
        (2000, 1000, 0.02),
        (5000, 2500, 0.01),
    ]
    
    all_results = []
    
    for n, m, density in sizes:
        print(f"\nProblem size: {n} vars, {m} constraints")
        res = benchmark_single(n, m, density)
        all_results.append((n, m, res))
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'n':>8} {'m':>8} {'OSQP (ms)':>12} {'cuProx (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    
    for n, m, res in all_results:
        osqp_time = res.get('osqp', {}).get('time', float('nan')) * 1000
        cuprox_time = res.get('cuprox', {}).get('time', float('nan')) * 1000
        speedup = osqp_time / cuprox_time if not np.isnan(cuprox_time) else float('nan')
        print(f"{n:>8} {m:>8} {osqp_time:>12.1f} {cuprox_time:>12.1f} {speedup:>10.2f}x")

def benchmark_portfolio():
    """Benchmark portfolio optimization QP (typical use case)."""
    print("\n" + "=" * 70)
    print("Portfolio Optimization Benchmark")
    print("=" * 70)
    
    # Portfolio optimization:
    # minimize (1/2) x'Σx - μ'x (risk - return)
    # subject to: sum(x) = 1 (fully invested)
    #             x >= 0 (no shorting)
    
    n_assets = 500
    np.random.seed(42)
    
    # Covariance matrix (positive definite)
    A_factor = np.random.randn(n_assets, 50) / np.sqrt(50)
    Sigma = A_factor @ A_factor.T + np.eye(n_assets) * 0.1
    P = sparse.csc_matrix(Sigma)
    
    # Expected returns
    mu = np.random.randn(n_assets) * 0.1
    q = -mu  # Negative because we minimize
    
    # Constraint: sum(x) = 1
    A = sparse.csc_matrix(np.ones((1, n_assets)))
    l = np.array([1.0])
    u = np.array([1.0])
    
    print(f"  Portfolio: {n_assets} assets")
    print(f"  Dense covariance matrix: {Sigma.shape}")
    
    # Note: cuProx currently needs sparse P, so convert
    P_sparse = sparse.csc_matrix(P)
    
    results = benchmark_single(n_assets, 1, density=1.0, seed=42)
    
    # Also test with actual portfolio problem
    if HAS_OSQP:
        res = solve_osqp(P_sparse, q, A, l, u)
        print(f"    OSQP (portfolio): {res['time']*1000:.1f} ms")
    
    if HAS_CUPROX and cuprox.__cuda_available__:
        res = solve_cuprox_qp(P_sparse, q, A, l, u)
        print(f"    cuProx (portfolio): {res['time']*1000:.1f} ms")

if __name__ == "__main__":
    benchmark_scaling()
    benchmark_portfolio()

