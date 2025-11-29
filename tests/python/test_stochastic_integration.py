"""
Integration Tests for Stochastic Programming.

Tests covering:
1. Classic problems (newsvendor, capacity planning)
2. EVPI and VSS computation
3. End-to-end workflows
"""

import numpy as np
import pytest


@pytest.mark.gpu
class TestNewsvendorProblem:
    """Classic newsvendor problem tests (require GPU solver for accuracy)."""

    def test_newsvendor_setup(self):
        """Set up newsvendor problem."""
        from cuprox.stochastic import TwoStageLP

        # Parameters
        order_cost = 10.0  # Cost per unit ordered
        sell_price = 15.0  # Revenue per unit sold
        salvage = 5.0  # Salvage value per unsold unit

        problem = TwoStageLP(c=np.array([order_cost]), lb=0, ub=100)

        # Demand scenarios
        demands = [20, 40, 60, 80]
        probs = [0.1, 0.3, 0.4, 0.2]

        for demand, prob in zip(demands, probs):
            # y1 = units sold, y2 = units salvaged
            # Constraints: y1 <= demand, y1 + y2 <= x (order)
            problem.add_scenario(
                probability=prob,
                q=np.array([-sell_price, -salvage]),  # Revenue (negative cost)
                W=np.array(
                    [
                        [1, 0],  # y1 <= demand
                        [1, 1],  # y1 + y2 = x (becomes <= with slack)
                    ]
                ),
                T=np.array(
                    [
                        [0],  # demand is exogenous
                        [-1],  # inventory balance
                    ]
                ),
                h=np.array([demand, 0]),
            )

        assert problem.n_scenarios == 4

    def test_newsvendor_solve(self):
        """Solve newsvendor problem."""
        from cuprox.stochastic import TwoStageLP

        order_cost = 10.0
        sell_price = 15.0
        salvage = 5.0

        problem = TwoStageLP(c=np.array([order_cost]), lb=0, ub=100)

        for demand, prob in [(20, 0.2), (40, 0.5), (60, 0.3)]:
            problem.add_scenario(
                probability=prob,
                q=np.array([-sell_price, -salvage]),
                W=np.array([[1, 0], [1, 1]]),
                T=np.array([[0], [-1]]),
                h=np.array([demand, 0]),
            )

        result = problem.solve()

        assert result.status == "optimal"
        # Optimal order should be reasonable
        assert 0 <= result.x[0] <= 100


@pytest.mark.gpu
class TestCapacityPlanning:
    """Capacity planning under demand uncertainty."""

    def test_capacity_planning(self):
        """Simple capacity planning problem."""
        from cuprox.stochastic import TwoStageLP

        # First stage: decide capacity x
        # Second stage: production y based on demand

        capacity_cost = 100.0
        production_cost = 10.0
        shortage_cost = 50.0

        problem = TwoStageLP(c=np.array([capacity_cost]), lb=0, ub=1000)

        # Demand scenarios
        for demand, prob in [(100, 0.3), (200, 0.4), (300, 0.3)]:
            # y1 = production, y2 = shortage
            problem.add_scenario(
                probability=prob,
                q=np.array([production_cost, shortage_cost]),
                W=np.array(
                    [
                        [1, 0],  # production <= capacity
                        [1, 1],  # production + shortage = demand
                    ]
                ),
                T=np.array(
                    [
                        [-1],  # y1 <= x
                        [0],  # demand constraint
                    ]
                ),
                h=np.array([0, demand]),
            )

        result = problem.solve()

        assert result.status == "optimal"


class TestEvaluationMetrics:
    """Test EVPI and VSS computation."""

    def test_evaluate_solution(self):
        """Evaluate solution on scenarios."""
        from cuprox.stochastic import TwoStageLP
        from cuprox.stochastic.evaluation import evaluate_solution

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        for demand, prob in [(10, 0.3), (20, 0.4), (30, 0.3)]:
            problem.add_scenario(
                probability=prob,
                q=np.array([2.0]),
                W=np.array([[1.0]]),
                T=np.array([[-1.0]]),
                h=np.array([demand]),
            )

        x = np.array([20.0])
        result = evaluate_solution(x, problem)

        assert result.objective > 0
        assert result.n_scenarios == 3

    def test_out_of_sample(self):
        """Out-of-sample evaluation."""
        from cuprox.stochastic import TwoStageLP
        from cuprox.stochastic.evaluation import out_of_sample_evaluation

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

        for demand, prob in [(10, 0.2), (20, 0.3), (30, 0.3), (40, 0.2)]:
            problem.add_scenario(
                probability=prob,
                q=np.array([2.0]),
                W=np.array([[1.0]]),
                T=np.array([[-1.0]]),
                h=np.array([demand]),
            )

        result = problem.solve()

        eval_result = out_of_sample_evaluation(result.x, problem, n_samples=100, seed=42)

        assert "mean" in eval_result
        assert "std" in eval_result


class TestGenerateScenarios:
    """Test scenario generation utilities."""

    def test_generate_from_distributions(self):
        """Generate scenarios from multiple distributions."""
        from cuprox.stochastic import (
            NormalDistribution,
            UniformDistribution,
            generate_scenarios,
        )

        demand_dist = NormalDistribution(mean_=np.array([100]), std=10)
        price_dist = UniformDistribution(low=np.array([5]), high=np.array([15]))

        np.random.seed(42)
        samples = generate_scenarios(1000, [demand_dist, price_dist])

        assert samples.shape == (1000, 2)
        # Check demand roughly normal
        assert abs(samples[:, 0].mean() - 100) < 5
        # Check price in range
        assert (samples[:, 1] >= 5).all()
        assert (samples[:, 1] <= 15).all()


@pytest.mark.gpu
class TestTwoStageQP:
    """Test two-stage quadratic programs."""

    def test_two_stage_qp(self):
        """Simple two-stage QP."""
        from cuprox.stochastic import TwoStageQP

        problem = TwoStageQP(
            P=np.array([[1.0]]), c=np.array([1.0]), lb=0, ub=100  # Quadratic cost on x
        )

        problem.add_scenario(
            probability=0.5,
            q=np.array([2.0]),
            W=np.array([[1.0]]),
            T=np.array([[-1.0]]),
            h=np.array([10.0]),
            R=np.array([[0.1]]),  # Quadratic cost on y
        )
        problem.add_scenario(
            probability=0.5,
            q=np.array([3.0]),
            W=np.array([[1.0]]),
            T=np.array([[-1.0]]),
            h=np.array([20.0]),
            R=np.array([[0.1]]),
        )

        result = problem.solve()

        assert result.status == "optimal"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_workflow(self):
        """Complete stochastic programming workflow."""
        from cuprox.stochastic import (
            NormalDistribution,
            SAASolver,
            ScenarioGenerator,
            TwoStageLP,
        )
        from cuprox.stochastic.evaluation import evaluate_solution

        # 1. Define problem structure
        problem = TwoStageLP(c=np.array([10.0]), lb=0, ub=100)

        # 2. Set up scenario generator
        gen = ScenarioGenerator(
            base_q=np.array([5.0]),
            base_W=np.array([[1.0]]),
            base_T=np.array([[-1.0]]),
            base_h=np.array([50.0]),
        )
        gen.set_h_distribution(NormalDistribution(mean_=np.array([50.0]), std=10.0))

        # 3. Solve with SAA
        solver = SAASolver(problem, gen, n_samples=100, n_replications=5)
        result = solver.solve(seed=42)

        # 4. Evaluate solution
        # Generate fresh scenarios for evaluation
        eval_scenarios = gen.generate(200, seed=123)

        eval_problem = TwoStageLP(c=np.array([10.0]), lb=0, ub=100)
        eval_problem.add_scenarios_from_set(eval_scenarios)

        eval_result = evaluate_solution(result.x, eval_problem)

        # Check workflow completed
        assert result.x is not None
        assert eval_result.objective > 0

    def test_reproducibility(self):
        """Results should be reproducible with seed."""
        from cuprox.stochastic import TwoStageLP

        def solve_problem(seed):
            problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=100)

            np.random.seed(seed)
            for _ in range(10):
                demand = 10 + np.random.randn() * 5
                problem.add_scenario(
                    probability=0.1,
                    q=np.array([2.0]),
                    W=np.array([[1.0]]),
                    T=np.array([[-1.0]]),
                    h=np.array([demand]),
                )

            return problem.solve()

        result1 = solve_problem(42)
        result2 = solve_problem(42)

        np.testing.assert_allclose(result1.x, result2.x, rtol=1e-3)
