#!/usr/bin/env python3
"""
cuProx LP Benchmark: Compare against scipy.optimize.linprog and HiGHS
"""

import sys
sys.path.insert(0, '../python')

import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

# Try to import cuprox
try:
    import cuprox
    HAS_CUPROX = True
    print(f"cuProx version: {cuprox.__version__}")
    if cuprox.__cuda_available__:
        from cuprox import _core
        print(f"GPU: {_core.get_device_name()}")
    else:
        print("cuProx: CPU only mode")
except ImportError:
    HAS_CUPROX = False
    print("cuProx not available")

# Try to import highspy (HiGHS)
try:
    import highspy
    HAS_HIGHS = True
    print("HiGHS available")
except ImportError:
    HAS_HIGHS = False
    print("HiGHS not available (pip install highspy)")

print()

def generate_lp(n, m, density=0.1, seed=42):
    """Generate a random sparse LP."""
    np.random.seed(seed)
    
    # Constraint matrix
    A = sparse.random(m, n, density=density, format='csr', dtype=np.float64)
    A = A + sparse.eye(m, n, format='csr') * 0.1
    
    # Make feasible: start with x = ones, set b = A*ones + slack
    x_feas = np.ones(n)
    slack = np.random.rand(m) * 0.5
    b = A @ x_feas + slack
    
    # Objective
    c = np.random.randn(n)
    
    # Bounds
    lb = np.zeros(n)
    ub = np.ones(n) * 10
    
    return c, A, b, lb, ub

def solve_cuprox(c, A, b, lb, ub, **kwargs):
    """Solve LP with cuProx."""
    start = time.perf_counter()
    result = cuprox.solve(
        c=c, A=A, b=b, lb=lb, ub=ub,
        params={'max_iterations': kwargs.get('max_iters', 10000),
                'tolerance': kwargs.get('tol', 1e-4)}
    )
    elapsed = time.perf_counter() - start
    return {
        'time': elapsed,
        'objective': result.objective,
        'status': result.status.value,
        'iterations': result.iterations,
    }

def solve_scipy(c, A, b, lb, ub, **kwargs):
    """Solve LP with scipy.optimize.linprog (HiGHS backend)."""
    bounds = [(l, u) for l, u in zip(lb, ub)]
    
    start = time.perf_counter()
    result = linprog(
        c, A_ub=A, b_ub=b, bounds=bounds,
        method='highs',
        options={'presolve': True, 'time_limit': 60}
    )
    elapsed = time.perf_counter() - start
    
    return {
        'time': elapsed,
        'objective': result.fun if result.success else float('nan'),
        'status': 'optimal' if result.success else result.message,
        'iterations': result.nit if hasattr(result, 'nit') else 0,
    }

def benchmark_single(n, m, density=0.1, seed=42):
    """Benchmark a single LP instance."""
    print(f"  Generating LP: n={n}, m={m}, density={density:.1%}")
    c, A, b, lb, ub = generate_lp(n, m, density, seed)
    
    results = {}
    
    # cuProx
    if HAS_CUPROX and cuprox.__cuda_available__:
        res = solve_cuprox(c, A, b, lb, ub)
        results['cuprox'] = res
        print(f"    cuProx:  {res['time']*1000:8.1f} ms, obj={res['objective']:10.4f}, "
              f"iters={res['iterations']}, status={res['status']}")
    
    # SciPy (HiGHS)
    res = solve_scipy(c, A, b, lb, ub)
    results['scipy'] = res
    print(f"    SciPy:   {res['time']*1000:8.1f} ms, obj={res['objective']:10.4f}, "
          f"status={res['status']}")
    
    return results

def benchmark_scaling():
    """Benchmark across different problem sizes."""
    print("=" * 70)
    print("LP Scaling Benchmark")
    print("=" * 70)
    
    sizes = [
        (100, 50, 0.2),
        (500, 250, 0.1),
        (1000, 500, 0.05),
        (2000, 1000, 0.02),
        (5000, 2500, 0.01),
        (10000, 5000, 0.005),
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
    print(f"{'n':>8} {'m':>8} {'cuProx (ms)':>12} {'SciPy (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    
    for n, m, res in all_results:
        cuprox_time = res.get('cuprox', {}).get('time', float('nan')) * 1000
        scipy_time = res.get('scipy', {}).get('time', float('nan')) * 1000
        speedup = scipy_time / cuprox_time if not np.isnan(cuprox_time) else float('nan')
        print(f"{n:>8} {m:>8} {cuprox_time:>12.1f} {scipy_time:>12.1f} {speedup:>10.2f}x")

def benchmark_batch():
    """Benchmark batch solving (many small problems)."""
    print("\n" + "=" * 70)
    print("Batch Solving Benchmark (many small LPs)")
    print("=" * 70)
    
    n, m = 50, 25  # Small problem
    batch_sizes = [10, 100, 1000]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}, problem size: {n}x{m}")
        
        # Generate batch of problems
        problems = []
        for i in range(batch_size):
            c, A, b, lb, ub = generate_lp(n, m, density=0.3, seed=42+i)
            problems.append({'c': c, 'A': A, 'b': b, 'lb': lb, 'ub': ub})
        
        # SciPy (sequential)
        start = time.perf_counter()
        for prob in problems:
            solve_scipy(**prob)
        scipy_time = time.perf_counter() - start
        
        print(f"  SciPy (sequential): {scipy_time*1000:.1f} ms total, "
              f"{scipy_time/batch_size*1000:.2f} ms/problem")
        
        # cuProx (sequential for now)
        if HAS_CUPROX and cuprox.__cuda_available__:
            start = time.perf_counter()
            for prob in problems:
                solve_cuprox(**prob)
            cuprox_time = time.perf_counter() - start
            
            print(f"  cuProx (sequential): {cuprox_time*1000:.1f} ms total, "
                  f"{cuprox_time/batch_size*1000:.2f} ms/problem")
            print(f"  Speedup: {scipy_time/cuprox_time:.2f}x")

if __name__ == "__main__":
    benchmark_scaling()
    benchmark_batch()

