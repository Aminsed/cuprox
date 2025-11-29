"""
Rigorous accuracy tests: cuProx must match CPU solvers within tolerance.

These tests verify numerical accuracy against scipy/analytical solutions.
If these tests fail, performance optimizations are meaningless.
"""

import numpy as np
import pytest

try:
    from scipy import sparse
    from scipy.optimize import linprog

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from cuprox import solve

# Tolerance for matching CPU solutions
ATOL = 1e-4  # Absolute tolerance
RTOL = 1e-4  # Relative tolerance


def assert_close(gpu_val, cpu_val, name, atol=ATOL, rtol=RTOL):
    """Assert GPU and CPU values match within tolerance."""
    diff = abs(gpu_val - cpu_val)
    rel_diff = diff / (abs(cpu_val) + 1e-10)

    assert (
        diff < atol or rel_diff < rtol
    ), f"{name} mismatch: GPU={gpu_val}, CPU={cpu_val}, diff={diff}"


@pytest.mark.gpu
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
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        # CPU (SciPy)
        c_cpu = np.array([-1.0, -1.0])
        A_cpu = np.array([[1.0, 2.0], [3.0, 1.0]])
        b_cpu = np.array([10.0, 15.0])

        cpu_result = linprog(
            c_cpu, A_ub=A_cpu, b_ub=b_cpu, bounds=[(0, None), (0, None)], method="highs"
        )

        assert cpu_result.success, f"CPU solver failed: {cpu_result.message}"

        # GPU (cuProx) - convert to equality with slacks
        c_gpu = np.array([-1.0, -1.0, 0.0, 0.0])
        A_gpu = sparse.csr_matrix([[1.0, 2.0, 1.0, 0.0], [3.0, 1.0, 0.0, 1.0]])
        b_gpu = np.array([10.0, 15.0])
        lb = np.array([0.0, 0.0, 0.0, 0.0])
        ub = np.array([1e20, 1e20, 1e20, 1e20])

        gpu_result = solve(
            c=c_gpu,
            A=A_gpu,
            b=b_gpu,
            lb=lb,
            ub=ub,
            params={"max_iterations": 50000, "tolerance": 1e-6},
        )

        # Validate
        assert_close(gpu_result.x[0], cpu_result.x[0], "x[0]")
        assert_close(gpu_result.x[1], cpu_result.x[1], "x[1]")
        assert_close(gpu_result.objective, cpu_result.fun, "objective")


@pytest.mark.gpu
class TestQPAccuracy:
    """QP accuracy tests - compare against known analytical solutions."""

    def test_unconstrained_qp(self):
        """
        Unconstrained QP: minimize (1/2)x'Px + q'x
        Has closed-form solution: x* = -P^{-1}q

        P = [[2, 0], [0, 2]], q = [-2, -4]
        x* = [1, 2], obj = -5
        """
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        P = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([-2.0, -4.0])

        # Analytical solution
        x_opt = np.array([1.0, 2.0])
        obj_opt = 0.5 * x_opt @ P.toarray() @ x_opt + q @ x_opt

        # GPU
        A = sparse.csr_matrix((0, 2))

        gpu_result = solve(
            c=q,
            A=A,
            b=np.array([]),
            P=P,
            lb=np.array([-1e20, -1e20]),
            ub=np.array([1e20, 1e20]),
            constraint_l=np.array([]),
            constraint_u=np.array([]),
            params={"max_iterations": 10000, "tolerance": 1e-6},
        )

        # Validate
        assert_close(gpu_result.x[0], x_opt[0], "x[0]")
        assert_close(gpu_result.x[1], x_opt[1], "x[1]")
        assert_close(gpu_result.objective, obj_opt, "objective")

    def test_box_constrained_qp(self):
        """
        Box-constrained QP: minimize (1/2)x'Px + q'x s.t. 0 <= x <= 1

        P = 2*I, q = [-3, -3]
        Without constraints: x* = [1.5, 1.5]
        With 0 <= x <= 1: x* = [1, 1], obj = -4
        """
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        P = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([-3.0, -3.0])

        x_opt = np.array([1.0, 1.0])
        obj_opt = 0.5 * x_opt @ P.toarray() @ x_opt + q @ x_opt

        # GPU
        A = sparse.csr_matrix((0, 2))

        gpu_result = solve(
            c=q,
            A=A,
            b=np.array([]),
            P=P,
            lb=np.array([0.0, 0.0]),
            ub=np.array([1.0, 1.0]),
            constraint_l=np.array([]),
            constraint_u=np.array([]),
            params={"max_iterations": 10000, "tolerance": 1e-6},
        )

        # Validate
        assert_close(gpu_result.x[0], x_opt[0], "x[0]")
        assert_close(gpu_result.x[1], x_opt[1], "x[1]")
        assert_close(gpu_result.objective, obj_opt, "objective")
