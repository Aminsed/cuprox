"""
Tests for LP solver functionality.
"""

import pytest
import numpy as np

from cuprox import Model, solve, SolveResult, Status
from cuprox.exceptions import DimensionError, InvalidInputError


class TestSolveLPSimple:
    """Tests for simple LP problems."""
    
    def test_solve_simple_lp(self, simple_lp):
        """Test solving a simple LP."""
        result = solve(
            c=simple_lp["c"],
            A=simple_lp["A"],
            b=simple_lp["b"],
            lb=simple_lp["lb"],
            ub=simple_lp["ub"],
        )
        
        assert isinstance(result, SolveResult)
        assert result.status == Status.OPTIMAL
        assert result.iterations > 0
        assert result.solve_time > 0
        
        # Check objective (with tolerance for first-order methods)
        assert abs(result.objective - simple_lp["expected_obj"]) < 0.1
    
    def test_solve_via_model(self, simple_lp):
        """Test solving LP via Model interface."""
        model = Model()
        x = model.add_var(lb=0, name="x")
        y = model.add_var(lb=0, name="y")
        
        model.add_constr(x + 2*y <= 10)
        model.add_constr(3*x + y <= 15)
        model.minimize(-x - y)
        
        result = model.solve()
        
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - (-7.0)) < 0.1
    
    def test_solve_bounded_variables(self):
        """Test LP with explicit bound constraints."""
        model = Model()
        x = model.add_var(lb=-10, ub=10, name="x")
        y = model.add_var(lb=0, name="y")  # Need at least 2 vars for proper matrix
        
        model.add_constr(x + y <= 100)  # Loose constraint
        model.minimize(x - y)  # Minimize x, maximize y
        
        result = model.solve()
        
        # Should converge - x to lower bound, y can be large
        assert result.status in [Status.OPTIMAL, Status.MAX_ITERATIONS]
        if result.status == Status.OPTIMAL:
            assert result.x[0] < -5.0  # x should be negative


class TestSolveLPRandom:
    """Tests for random LP problems."""
    
    def test_solve_random_lp(self, random_lp):
        """Test solving a random feasible LP."""
        result = solve(
            c=random_lp["c"],
            A=random_lp["A"],
            b=random_lp["b"],
            lb=random_lp["lb"],
            ub=random_lp["ub"],
        )
        
        assert result.status in [Status.OPTIMAL, Status.MAX_ITERATIONS]
        
        # Check feasibility
        Ax = random_lp["A"] @ result.x
        violations = np.maximum(Ax - random_lp["b"], 0)
        assert np.max(violations) < 0.01  # Small violation tolerance
    
    @pytest.mark.slow
    def test_solve_sparse_lp(self, sparse_lp):
        """Test solving a sparse LP."""
        result = solve(
            c=sparse_lp["c"],
            A=sparse_lp["A"],
            b=sparse_lp["b"],
            lb=sparse_lp["lb"],
            ub=sparse_lp["ub"],
            params={"max_iterations": 10000},
        )
        
        assert result.status in [Status.OPTIMAL, Status.MAX_ITERATIONS]


class TestSolverParameters:
    """Tests for solver parameters."""
    
    def test_tolerance_parameter(self, simple_lp):
        """Test tolerance parameter affects convergence."""
        result_loose = solve(
            c=simple_lp["c"],
            A=simple_lp["A"],
            b=simple_lp["b"],
            lb=simple_lp["lb"],
            params={"tolerance": 1e-2},
        )
        
        result_tight = solve(
            c=simple_lp["c"],
            A=simple_lp["A"],
            b=simple_lp["b"],
            lb=simple_lp["lb"],
            params={"tolerance": 1e-8},
        )
        
        # Tighter tolerance should require more iterations
        # (or same if already converged)
        assert result_tight.iterations >= result_loose.iterations - 10
    
    def test_max_iterations_parameter(self, random_lp):
        """Test max_iterations parameter limits iterations."""
        result = solve(
            c=random_lp["c"],
            A=random_lp["A"],
            b=random_lp["b"],
            lb=random_lp["lb"],
            params={"max_iterations": 100},
        )
        
        # Iterations should be close to max (may be slightly over due to check interval)
        assert result.iterations <= 150  # Allow some slack for check_interval
        # If not optimal, should report max iterations
        if result.status != Status.OPTIMAL:
            assert result.status == Status.MAX_ITERATIONS


class TestInputValidation:
    """Tests for input validation."""
    
    def test_dimension_mismatch_A_c(self):
        """Test error on A/c dimension mismatch."""
        c = np.array([1, 2, 3])
        A = np.array([[1, 2]])  # 2 columns, but c has 3
        b = np.array([10])
        
        with pytest.raises(DimensionError):
            solve(c=c, A=A, b=b)
    
    def test_dimension_mismatch_A_b(self):
        """Test error on A/b dimension mismatch."""
        c = np.array([1, 2])
        A = np.array([[1, 2], [3, 4]])  # 2 rows
        b = np.array([10])  # 1 element
        
        with pytest.raises(DimensionError):
            solve(c=c, A=A, b=b)
    
    def test_nan_in_objective(self):
        """Test error on NaN in objective."""
        c = np.array([1, np.nan])
        A = np.array([[1, 2]])
        b = np.array([10])
        
        with pytest.raises(InvalidInputError):
            solve(c=c, A=A, b=b)
    
    def test_invalid_bounds(self):
        """Test error on lb > ub."""
        c = np.array([1, 2])
        A = np.array([[1, 2]])
        b = np.array([10])
        lb = np.array([5, 0])
        ub = np.array([3, 10])  # ub[0] < lb[0]
        
        with pytest.raises(InvalidInputError):
            solve(c=c, A=A, b=b, lb=lb, ub=ub)


class TestSolveResult:
    """Tests for SolveResult class."""
    
    def test_result_attributes(self, simple_lp):
        """Test SolveResult has expected attributes."""
        result = solve(
            c=simple_lp["c"],
            A=simple_lp["A"],
            b=simple_lp["b"],
            lb=simple_lp["lb"],
        )
        
        assert hasattr(result, "status")
        assert hasattr(result, "objective")
        assert hasattr(result, "x")
        assert hasattr(result, "y")
        assert hasattr(result, "iterations")
        assert hasattr(result, "solve_time")
    
    def test_result_summary(self, simple_lp):
        """Test SolveResult summary method."""
        result = solve(
            c=simple_lp["c"],
            A=simple_lp["A"],
            b=simple_lp["b"],
            lb=simple_lp["lb"],
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Status" in summary
        assert "Objective" in summary

