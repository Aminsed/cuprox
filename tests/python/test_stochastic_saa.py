"""
Tests for SAA (Sample Average Approximation).

Tests covering:
1. SAA solver
2. Statistical analysis
3. Confidence intervals
"""

import numpy as np


class TestSAASolver:
    """Test SAASolver class."""

    def test_basic_saa(self):
        """Basic SAA solve."""
        from cuprox.stochastic import (
            NormalDistribution,
            SAASolver,
            ScenarioGenerator,
            TwoStageLP,
        )

        # Simple problem
        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        # Generator
        gen = ScenarioGenerator(
            base_q=np.array([2.0]),
            base_W=np.array([[1.0]]),
            base_T=np.array([[-1.0]]),
            base_h=np.array([10.0]),
        )
        gen.set_h_distribution(NormalDistribution(mean_=np.array([10.0]), std=2.0))

        solver = SAASolver(
            problem,
            gen,
            n_samples=50,
            n_replications=3,
        )

        result = solver.solve(seed=42)

        assert result.x is not None
        assert result.objective is not None
        assert result.n_samples == 50
        assert result.n_replications == 3

    def test_saa_result_attributes(self):
        """Check SAA result has all attributes."""
        from cuprox.stochastic import (
            NormalDistribution,
            SAASolver,
            ScenarioGenerator,
            TwoStageLP,
        )

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        gen = ScenarioGenerator(
            base_q=np.array([2.0]),
            base_W=np.array([[1.0]]),
            base_T=np.array([[-1.0]]),
            base_h=np.array([10.0]),
        )
        gen.set_h_distribution(NormalDistribution(mean_=np.array([10.0]), std=1.0))

        solver = SAASolver(problem, gen, n_samples=30, n_replications=3)
        result = solver.solve(seed=42)

        assert hasattr(result, "x")
        assert hasattr(result, "objective")
        assert hasattr(result, "lower_bound")
        assert hasattr(result, "upper_bound")
        assert hasattr(result, "gap_estimate")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "confidence_level")

    def test_gap_estimate_non_negative(self):
        """Gap estimate should be non-negative."""
        from cuprox.stochastic import (
            NormalDistribution,
            SAASolver,
            ScenarioGenerator,
            TwoStageLP,
        )

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        gen = ScenarioGenerator(
            base_q=np.array([2.0]),
            base_W=np.array([[1.0]]),
            base_T=np.array([[-1.0]]),
            base_h=np.array([10.0]),
        )
        gen.set_h_distribution(NormalDistribution(mean_=np.array([10.0]), std=1.0))

        solver = SAASolver(problem, gen, n_samples=50, n_replications=5)
        result = solver.solve(seed=42)

        assert result.gap_estimate >= -0.01  # Allow small numerical error

    def test_summary(self):
        """Test result summary."""
        from cuprox.stochastic import (
            NormalDistribution,
            SAASolver,
            ScenarioGenerator,
            TwoStageLP,
        )

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        gen = ScenarioGenerator(
            base_q=np.array([2.0]),
            base_W=np.array([[1.0]]),
            base_T=np.array([[-1.0]]),
            base_h=np.array([10.0]),
        )
        gen.set_h_distribution(NormalDistribution(mean_=np.array([10.0]), std=1.0))

        solver = SAASolver(problem, gen, n_samples=30, n_replications=3)
        result = solver.solve(seed=42)

        summary = result.summary()

        assert "SAA Solution Summary" in summary
        assert "Confidence" in summary


class TestMonteCarloBound:
    """Test Monte Carlo evaluation."""

    def test_mc_bound(self):
        """Monte Carlo bound estimation."""
        from cuprox.stochastic import TwoStageLP
        from cuprox.stochastic.saa import monte_carlo_bound

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        # Add scenarios
        for demand, prob in [(10, 0.2), (20, 0.5), (30, 0.3)]:
            problem.add_scenario(
                probability=prob,
                q=np.array([2.0]),
                W=np.array([[1.0]]),
                T=np.array([[-1.0]]),
                h=np.array([float(demand)]),
            )

        x = np.array([15.0])

        estimate, ci_lower, ci_upper = monte_carlo_bound(problem, x, confidence_level=0.95)

        assert ci_lower <= estimate <= ci_upper


class TestSolveSAA:
    """Test solve_saa convenience function."""

    def test_solve_saa(self):
        """Solve with SAA subsampling."""
        from cuprox.stochastic import TwoStageLP
        from cuprox.stochastic.saa import solve_saa

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        # Add many scenarios
        np.random.seed(42)
        for i in range(100):
            demand = 10 + np.random.randn() * 5
            problem.add_scenario(
                probability=0.01,
                q=np.array([2.0]),
                W=np.array([[1.0]]),
                T=np.array([[-1.0]]),
                h=np.array([demand]),
            )

        # Solve with subset
        result = solve_saa(problem, n_samples=50, seed=42)

        assert result.x is not None
        assert result.status in ["optimal", "max_iterations"]
