"""
Tests for QP solver functionality.
"""

import numpy as np
import pytest

from cuprox import Status, solve

try:
    from scipy import sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestSolveQPSimple:
    """Tests for simple QP problems."""

    def test_solve_unconstrained_qp(self):
        """
        Test solving unconstrained QP.

        minimize (1/2)x'Px + q'x
        where P = 2I, q = [-2, -4]

        Optimal: x = [1, 2], obj = -5
        """
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        P = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([-2.0, -4.0])
        A = sparse.csr_matrix((0, 2))  # No constraints

        result = solve(
            c=q,
            A=A,
            b=np.array([]),
            P=P,
            lb=np.array([-1e20, -1e20]),
            ub=np.array([1e20, 1e20]),
            constraint_l=np.array([]),
            constraint_u=np.array([]),
            params={"tolerance": 1e-6},
        )

        assert result.status == Status.OPTIMAL
        assert abs(result.x[0] - 1.0) < 1e-3
        assert abs(result.x[1] - 2.0) < 1e-3
        assert abs(result.objective - (-5.0)) < 1e-3

    def test_solve_box_constrained_qp(self):
        """
        Test solving box-constrained QP.

        minimize (1/2)x'Px + q'x
        subject to 0 <= x <= 1

        where P = 2I, q = [-3, -3]
        Unconstrained: x* = [1.5, 1.5]
        With box: x* = [1, 1], obj = -4
        """
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        P = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([-3.0, -3.0])
        A = sparse.csr_matrix((0, 2))

        result = solve(
            c=q,
            A=A,
            b=np.array([]),
            P=P,
            lb=np.array([0.0, 0.0]),
            ub=np.array([1.0, 1.0]),
            constraint_l=np.array([]),
            constraint_u=np.array([]),
            params={"tolerance": 1e-6},
        )

        assert result.status == Status.OPTIMAL
        assert abs(result.x[0] - 1.0) < 1e-3
        assert abs(result.x[1] - 1.0) < 1e-3
        assert abs(result.objective - (-4.0)) < 1e-3

    def test_solve_qp_with_constraint(self, simple_qp):
        """Test solving QP with linear constraint."""
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        P = sparse.csr_matrix(simple_qp["P"])
        A = sparse.csr_matrix(simple_qp["A"])

        result = solve(
            c=simple_qp["c"],
            A=A,
            b=simple_qp["b"],
            P=P,
            lb=simple_qp["lb"],
            ub=simple_qp["ub"],
            constraint_l=simple_qp["b"],  # x + y <= 3 as equality
            constraint_u=np.array([1e20]),
            params={"tolerance": 1e-4, "max_iterations": 5000},
        )

        assert result.status in [Status.OPTIMAL, Status.MAX_ITERATIONS]


class TestSolveQPRandom:
    """Tests for random QP problems."""

    @pytest.mark.slow
    def test_solve_random_qp(self):
        """Test solving a random convex QP."""
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        np.random.seed(42)
        n = 50

        # Random positive definite P
        A_factor = np.random.randn(n, n // 2)
        P_dense = A_factor @ A_factor.T + 0.1 * np.eye(n)
        P = sparse.csr_matrix(P_dense)

        q = np.random.randn(n)
        lb = np.zeros(n)
        ub = np.full(n, 10.0)

        # No linear constraints
        A = sparse.csr_matrix((0, n))

        result = solve(
            c=q,
            A=A,
            b=np.array([]),
            P=P,
            lb=lb,
            ub=ub,
            constraint_l=np.array([]),
            constraint_u=np.array([]),
            params={"max_iterations": 2000},
        )

        assert result.status in [Status.OPTIMAL, Status.MAX_ITERATIONS]
        # Check bounds satisfied
        assert np.all(result.x >= lb - 1e-4)
        assert np.all(result.x <= ub + 1e-4)


class TestQPInputValidation:
    """Tests for QP input validation."""

    def test_non_psd_P_warning(self):
        """Test that non-PSD P doesn't crash (may not converge)."""
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        # Indefinite matrix (not positive semidefinite)
        P = sparse.csr_matrix([[-1.0, 0.0], [0.0, 1.0]])
        q = np.array([1.0, 1.0])
        A = sparse.csr_matrix((0, 2))

        # Should not crash, but may not converge
        result = solve(
            c=q,
            A=A,
            b=np.array([]),
            P=P,
            lb=np.array([-10.0, -10.0]),
            ub=np.array([10.0, 10.0]),
            constraint_l=np.array([]),
            constraint_u=np.array([]),
            params={"max_iterations": 100},
        )

        # Just verify it runs without error
        assert result is not None


class TestQPResult:
    """Tests for QP result attributes."""

    def test_qp_result_attributes(self):
        """Test QP result has expected attributes."""
        if not HAS_SCIPY:
            pytest.skip("scipy not available")

        P = sparse.csr_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([-2.0, -4.0])
        A = sparse.csr_matrix((0, 2))

        result = solve(
            c=q,
            A=A,
            b=np.array([]),
            P=P,
            lb=np.array([-1e20, -1e20]),
            ub=np.array([1e20, 1e20]),
            constraint_l=np.array([]),
            constraint_u=np.array([]),
        )

        assert hasattr(result, "status")
        assert hasattr(result, "objective")
        assert hasattr(result, "x")
        assert hasattr(result, "iterations")
        assert hasattr(result, "solve_time")

        assert result.x.shape == (2,)
        assert result.solve_time > 0
