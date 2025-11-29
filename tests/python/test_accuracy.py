#!/usr/bin/env python3
"""
Rigorous accuracy tests: cuProx must match CPU solvers within tolerance.
If these tests fail, performance is meaningless.
"""

import sys
sys.path.insert(0, '../../python')

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import pytest

import cuprox

# Tolerance for matching CPU solutions
ATOL = 1e-4  # Absolute tolerance
RTOL = 1e-4  # Relative tolerance

def assert_close(gpu_val, cpu_val, name, atol=ATOL, rtol=RTOL):
    """Assert GPU and CPU values match within tolerance."""
    diff = abs(gpu_val - cpu_val)
    rel_diff = diff / (abs(cpu_val) + 1e-10)
    
    print(f"  {name}: GPU={gpu_val:.6f}, CPU={cpu_val:.6f}, "
          f"diff={diff:.2e}, rel={rel_diff:.2e}")
    
    assert diff < atol or rel_diff < rtol, \
        f"{name} mismatch: GPU={gpu_val}, CPU={cpu_val}, diff={diff}"


class TestLPAccuracy:
    """LP accuracy tests - compare against scipy.optimize.linprog."""
    
    def test_simple_2var_lp(self):
        """
        Simple LP with known optimal solution.
        minimize -x - y
        subject to: x + 2y <= 10
                    3x + y <= 15
                    x, y >= 0
        Optimal: x=4, y=3, obj=-7
        """
        print("\n=== Simple 2-var LP ===")
        
        # CPU (SciPy)
        c_cpu = np.array([-1.0, -1.0])
        A_cpu = np.array([[1.0, 2.0], [3.0, 1.0]])
        b_cpu = np.array([10.0, 15.0])
        
        cpu_result = linprog(c_cpu, A_ub=A_cpu, b_ub=b_cpu, 
                             bounds=[(0, None), (0, None)], method='highs')
        
        assert cpu_result.success, f"CPU solver failed: {cpu_result.message}"
        print(f"  CPU: x={cpu_result.x}, obj={cpu_result.fun}")
        
        # GPU (cuProx) - convert to equality with slacks
        # x + 2y + s1 = 10
        # 3x + y + s2 = 15
        c_gpu = np.array([-1.0, -1.0, 0.0, 0.0])
        A_gpu = sparse.csr_matrix([[1.0, 2.0, 1.0, 0.0], 
                                    [3.0, 1.0, 0.0, 1.0]])
        b_gpu = np.array([10.0, 15.0])
        lb = np.array([0.0, 0.0, 0.0, 0.0])
        ub = np.array([1e20, 1e20, 1e20, 1e20])
        
        gpu_result = cuprox.solve(
            c=c_gpu, A=A_gpu, b=b_gpu, lb=lb, ub=ub,
            params={'max_iterations': 50000, 'tolerance': 1e-6}
        )
        
        print(f"  GPU: x={gpu_result.x[:2]}, obj={gpu_result.objective}, "
              f"status={gpu_result.status}, iters={gpu_result.iterations}")
        
        # Validate
        assert_close(gpu_result.x[0], cpu_result.x[0], "x[0]")
        assert_close(gpu_result.x[1], cpu_result.x[1], "x[1]")
        assert_close(gpu_result.objective, cpu_result.fun, "objective")
        

class TestQPAccuracy:
    """QP accuracy tests - compare against known solutions."""
    
    def test_unconstrained_qp(self):
        """
        Unconstrained QP: minimize (1/2)x'Px + q'x
        Has closed-form solution: x* = -P^{-1}q
        """
        print("\n=== Unconstrained QP ===")
        
        # P = [[2, 0], [0, 2]], q = [-2, -4]
        # x* = [1, 2], obj = -5
        P = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([-2.0, -4.0])
        
        # Analytical solution
        x_opt = np.array([1.0, 2.0])
        obj_opt = 0.5 * x_opt @ P.toarray() @ x_opt + q @ x_opt
        print(f"  Analytical: x={x_opt}, obj={obj_opt}")
        
        # GPU
        A = sparse.csr_matrix((0, 2))
        
        gpu_result = cuprox.solve(
            c=q, A=A, b=np.array([]), P=P,
            lb=np.array([-1e20, -1e20]), ub=np.array([1e20, 1e20]),
            constraint_l=np.array([]), constraint_u=np.array([]),
            params={'max_iterations': 10000, 'tolerance': 1e-6}
        )
        
        print(f"  GPU: x={gpu_result.x}, obj={gpu_result.objective:.6f}, "
              f"status={gpu_result.status}, iters={gpu_result.iterations}")
        
        assert_close(gpu_result.x[0], x_opt[0], "x[0]")
        assert_close(gpu_result.x[1], x_opt[1], "x[1]")
        assert_close(gpu_result.objective, obj_opt, "objective")
        
    def test_box_constrained_qp(self):
        """
        Box-constrained QP: minimize (1/2)x'Px + q'x s.t. 0 <= x <= 1
        """
        print("\n=== Box-constrained QP ===")
        
        # P = 2*I, q = [-3, -3]
        # Without constraints: x* = [1.5, 1.5]
        # With 0 <= x <= 1: x* = [1, 1], obj = -4
        P = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([-3.0, -3.0])
        
        x_opt = np.array([1.0, 1.0])
        obj_opt = 0.5 * x_opt @ P.toarray() @ x_opt + q @ x_opt
        print(f"  Expected: x={x_opt}, obj={obj_opt}")
        
        # GPU
        A = sparse.csr_matrix((0, 2))
        
        gpu_result = cuprox.solve(
            c=q, A=A, b=np.array([]), P=P,
            lb=np.array([0.0, 0.0]), ub=np.array([1.0, 1.0]),
            constraint_l=np.array([]), constraint_u=np.array([]),
            params={'max_iterations': 10000, 'tolerance': 1e-6}
        )
        
        print(f"  GPU: x={gpu_result.x}, obj={gpu_result.objective:.6f}, "
              f"status={gpu_result.status}, iters={gpu_result.iterations}")
        
        assert_close(gpu_result.x[0], x_opt[0], "x[0]")
        assert_close(gpu_result.x[1], x_opt[1], "x[1]")
        assert_close(gpu_result.objective, obj_opt, "objective")


if __name__ == "__main__":
    # Run tests
    lp_tests = TestLPAccuracy()
    lp_tests.test_simple_2var_lp()
    
    qp_tests = TestQPAccuracy()
    qp_tests.test_unconstrained_qp()
    qp_tests.test_box_constrained_qp()
    
    print("\n" + "=" * 60)
    print("All accuracy tests passed!")
    print("=" * 60)
