"""
Tests for Stochastic Programming Problems.

Tests covering:
1. Two-Stage LP formulation
2. Scenario management
3. Deterministic equivalent
4. Problem solving
"""

import numpy as np
import pytest


class TestScenario:
    """Test Scenario class."""

    def test_basic_creation(self):
        """Create basic scenario."""
        from cuprox.stochastic import Scenario

        q = np.array([1, 2])
        W = np.eye(2)
        T = np.array([[1, 0], [0, 1]])
        h = np.array([10, 20])

        s = Scenario(probability=0.5, q=q, W=W, T=T, h=h)

        assert s.probability == 0.5
        assert s.n_second_stage_vars == 2
        assert s.n_second_stage_constraints == 2
        assert s.n_first_stage_vars == 2

    def test_invalid_probability(self):
        """Error on invalid probability."""
        from cuprox.stochastic import Scenario

        with pytest.raises(ValueError, match="Probability"):
            Scenario(probability=1.5, q=[1], W=[[1]], T=[[1]], h=[1])

    def test_dimension_validation(self):
        """Error on dimension mismatch."""
        from cuprox.stochastic import Scenario

        with pytest.raises(ValueError):
            Scenario(
                probability=0.5,
                q=np.array([1, 2, 3]),  # Wrong size
                W=np.eye(2),
                T=np.eye(2),
                h=np.array([1, 2]),
            )

    def test_scale_probability(self):
        """Scale scenario probability."""
        from cuprox.stochastic import Scenario

        s = Scenario(probability=0.5, q=[1], W=[[1]], T=[[1]], h=[1])
        s_scaled = s.scale_probability(2.0)

        assert s_scaled.probability == 1.0
        assert s.probability == 0.5  # Original unchanged


class TestScenarioSet:
    """Test ScenarioSet class."""

    def test_add_scenarios(self):
        """Add multiple scenarios."""
        from cuprox.stochastic import Scenario, ScenarioSet

        scenarios = ScenarioSet()

        for p in [0.2, 0.3, 0.5]:
            scenarios.add(Scenario(p, q=[1], W=[[1]], T=[[1]], h=[10]))

        assert len(scenarios) == 3
        assert scenarios.n_scenarios == 3

    def test_probabilities_sum(self):
        """Probabilities sum to 1."""
        from cuprox.stochastic import Scenario, ScenarioSet

        scenarios = ScenarioSet()
        scenarios.add(Scenario(0.4, q=[1], W=[[1]], T=[[1]], h=[10]))
        scenarios.add(Scenario(0.6, q=[2], W=[[1]], T=[[1]], h=[20]))

        assert abs(scenarios.total_probability - 1.0) < 1e-6
        assert scenarios.validate()

    def test_validate_probability_error(self):
        """Error when probabilities don't sum to 1."""
        from cuprox.stochastic import Scenario, ScenarioSet

        scenarios = ScenarioSet(normalize=False)
        scenarios.add(Scenario(0.3, q=[1], W=[[1]], T=[[1]], h=[10]))
        scenarios.add(Scenario(0.3, q=[2], W=[[1]], T=[[1]], h=[20]))

        with pytest.raises(ValueError, match="Probabilities"):
            scenarios.validate()

    def test_normalize(self):
        """Normalize probabilities."""
        from cuprox.stochastic import Scenario, ScenarioSet

        scenarios = ScenarioSet(normalize=True)
        scenarios.add(Scenario(1.0, q=[1], W=[[1]], T=[[1]], h=[10]))
        scenarios.add(Scenario(1.0, q=[2], W=[[1]], T=[[1]], h=[20]))

        scenarios.validate()

        assert abs(scenarios.total_probability - 1.0) < 1e-6

    def test_sample(self):
        """Sample from scenarios."""
        from cuprox.stochastic import Scenario, ScenarioSet

        scenarios = ScenarioSet()
        scenarios.add(Scenario(0.2, q=[1], W=[[1]], T=[[1]], h=[10]))
        scenarios.add(Scenario(0.8, q=[2], W=[[1]], T=[[1]], h=[20]))

        np.random.seed(42)
        sampled = scenarios.sample(100)

        assert sampled.n_scenarios == 100
        # Each sampled scenario has equal probability
        assert abs(sampled.probabilities[0] - 0.01) < 1e-6

    def test_iteration(self):
        """Iterate over scenarios."""
        from cuprox.stochastic import Scenario, ScenarioSet

        scenarios = ScenarioSet()
        for p in [0.2, 0.3, 0.5]:
            scenarios.add(Scenario(p, q=[1], W=[[1]], T=[[1]], h=[10]))

        probs = [s.probability for s in scenarios]
        assert probs == [0.2, 0.3, 0.5]


class TestTwoStageLP:
    """Test TwoStageLP class."""

    def test_basic_creation(self):
        """Create basic two-stage LP."""
        from cuprox.stochastic import TwoStageLP

        c = np.array([1, 2])
        A = np.array([[1, 1]])
        b = np.array([10])

        problem = TwoStageLP(c=c, A=A, b=b)

        assert problem.n_x == 2
        assert problem.n_scenarios == 0

    def test_add_scenarios(self):
        """Add scenarios to problem."""
        from cuprox.stochastic import TwoStageLP

        problem = TwoStageLP(c=np.array([1]))

        problem.add_scenario(
            probability=0.5, q=np.array([2]), W=np.array([[1]]), T=np.array([[1]]), h=np.array([10])
        )
        problem.add_scenario(
            probability=0.5, q=np.array([3]), W=np.array([[1]]), T=np.array([[1]]), h=np.array([20])
        )

        assert problem.n_scenarios == 2

    def test_newsvendor_problem(self):
        """Classic newsvendor problem."""
        from cuprox.stochastic import TwoStageLP

        # Newsvendor: order quantity x, uncertain demand ξ
        # Cost: c*x + E[revenue - salvage]

        c = np.array([10.0])  # Unit ordering cost

        problem = TwoStageLP(c=c, lb=0, ub=100)

        # Demand scenarios
        demands = [(20, 0.3), (40, 0.5), (60, 0.2)]

        for demand, prob in demands:
            # Second stage: sell (min of order, demand), salvage rest
            # Variables: y1 = sales, y2 = salvage
            # Constraints: y1 <= demand, y1 + y2 = x
            problem.add_scenario(
                probability=prob,
                q=np.array([-15.0, -5.0]),  # Selling price, salvage (negative = revenue)
                W=np.array(
                    [
                        [1, 0],  # y1 <= demand
                        [1, 1],  # y1 + y2 = x
                    ]
                ),
                T=np.array(
                    [
                        [0],  # demand constraint independent of x
                        [-1],  # inventory balance: y1 + y2 = x
                    ]
                ),
                h=np.array([demand, 0]),
            )

        assert problem.n_scenarios == 3

    def test_solve_simple(self):
        """Solve simple two-stage problem."""
        from cuprox.stochastic import TwoStageLP

        # Very simple problem
        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=10)

        # Two scenarios
        problem.add_scenario(
            probability=0.5,
            q=np.array([1.0]),
            W=np.array([[1.0]]),
            T=np.array([[-1.0]]),
            h=np.array([5.0]),  # y = 5 + x
        )
        problem.add_scenario(
            probability=0.5,
            q=np.array([2.0]),
            W=np.array([[1.0]]),
            T=np.array([[-1.0]]),
            h=np.array([3.0]),  # y = 3 + x
        )

        result = problem.solve()

        assert result.status in ["optimal", "max_iterations"]
        assert result.x is not None

    def test_result_attributes(self):
        """Check result has all attributes."""
        from cuprox.stochastic import TwoStageLP

        problem = TwoStageLP(c=np.array([1.0]), lb=0, ub=10)
        problem.add_scenario(0.5, [1], [[1]], [[-1]], [5])
        problem.add_scenario(0.5, [2], [[1]], [[-1]], [3])

        result = problem.solve()

        assert hasattr(result, "x")
        assert hasattr(result, "y")
        assert hasattr(result, "first_stage_cost")
        assert hasattr(result, "expected_recourse")
        assert hasattr(result, "total_cost")
        assert hasattr(result, "status")


class TestDeterministicEquivalent:
    """Test deterministic equivalent construction."""

    def test_build_de(self):
        """Build deterministic equivalent."""
        from cuprox.stochastic import Scenario, ScenarioSet

        scenarios = ScenarioSet()
        scenarios.add(Scenario(0.5, q=[1, 2], W=np.eye(2), T=np.ones((2, 1)), h=[10, 20]))
        scenarios.add(Scenario(0.5, q=[2, 3], W=np.eye(2), T=np.ones((2, 1)), h=[15, 25]))

        c = np.array([5.0])
        A = np.zeros((0, 1))
        b = np.array([])

        de = scenarios.to_deterministic_equivalent(c, A, b)

        # Variables: x (1) + y_1 (2) + y_2 (2) = 5
        assert len(de["c"]) == 5

        # First var cost
        assert de["c"][0] == 5.0

        # Scenario costs (probability weighted)
        assert de["c"][1] == 0.5 * 1  # 0.5 * q[0]
        assert de["c"][2] == 0.5 * 2  # 0.5 * q[1]


class TestDistributions:
    """Test distribution classes."""

    def test_discrete(self):
        """Discrete distribution."""
        from cuprox.stochastic import DiscreteDistribution

        dist = DiscreteDistribution(
            values=np.array([10, 20, 30]), probabilities=np.array([0.2, 0.5, 0.3])
        )

        assert dist.n_outcomes == 3

        np.random.seed(42)
        samples = [dist.sample() for _ in range(100)]
        assert all(s in [10, 20, 30] for s in samples)

    def test_normal(self):
        """Normal distribution."""
        from cuprox.stochastic import NormalDistribution

        dist = NormalDistribution(mean_=np.array([10, 20]), std=2.0)

        np.random.seed(42)
        samples = dist.sample(size=1000)

        assert samples.shape == (1000, 2)
        assert abs(samples[:, 0].mean() - 10) < 0.5

    def test_uniform(self):
        """Uniform distribution."""
        from cuprox.stochastic import UniformDistribution

        dist = UniformDistribution(low=np.array([0, 10]), high=np.array([5, 20]))

        np.random.seed(42)
        samples = dist.sample(size=1000)

        assert (samples[:, 0] >= 0).all()
        assert (samples[:, 0] <= 5).all()
        assert (samples[:, 1] >= 10).all()
        assert (samples[:, 1] <= 20).all()


class TestScenarioGenerator:
    """Test scenario generation."""

    def test_generate_scenarios(self):
        """Generate scenarios from distributions."""
        from cuprox.stochastic import NormalDistribution, ScenarioGenerator

        base_q = np.array([1.0, 2.0])
        base_W = np.eye(2)
        base_T = np.ones((2, 1))
        base_h = np.array([10.0, 20.0])

        gen = ScenarioGenerator(base_q, base_W, base_T, base_h)
        gen.set_h_distribution(NormalDistribution(mean_=base_h, std=2.0))

        scenarios = gen.generate(n_scenarios=50, seed=42)

        assert scenarios.n_scenarios == 50
        assert abs(scenarios.total_probability - 1.0) < 1e-6
