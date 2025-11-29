"""
Sample Average Approximation (SAA)
==================================

SAA solver for stochastic programming with statistical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .problem import TwoStageLP, TwoStageResult
from .scenarios import ScenarioGenerator


@dataclass
class SAAResult:
    """
    SAA solution with statistical analysis.

    Attributes:
        x: Optimal first-stage decision
        objective: SAA objective value
        lower_bound: Statistical lower bound on optimal value
        upper_bound: Statistical upper bound on optimal value
        gap_estimate: Optimality gap estimate
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        confidence_level: Confidence level (e.g., 0.95)
        n_samples: Number of SAA samples
        n_replications: Number of SAA replications
        solve_time: Total solve time
    """

    x: np.ndarray
    objective: float
    lower_bound: float
    upper_bound: float
    gap_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int
    n_replications: int
    solve_time: float

    def __repr__(self) -> str:
        return (
            f"SAAResult(\n"
            f"  objective={self.objective:.4f},\n"
            f"  gap_estimate={self.gap_estimate:.4f},\n"
            f"  ci=[{self.ci_lower:.4f}, {self.ci_upper:.4f}],\n"
            f"  n_samples={self.n_samples}\n"
            f")"
        )

    def summary(self) -> str:
        """Formatted summary."""
        return (
            "=" * 50 + "\n"
            "SAA Solution Summary\n"
            "=" * 50 + "\n"
            f"Objective value:     {self.objective:.4f}\n"
            f"Lower bound:         {self.lower_bound:.4f}\n"
            f"Upper bound:         {self.upper_bound:.4f}\n"
            f"Optimality gap:      {self.gap_estimate:.4f}\n"
            "-" * 50 + "\n"
            f"Confidence level:    {self.confidence_level:.0%}\n"
            f"Confidence interval: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            "-" * 50 + "\n"
            f"SAA samples:         {self.n_samples}\n"
            f"Replications:        {self.n_replications}\n"
            f"Total solve time:    {self.solve_time:.4f}s\n"
            "=" * 50
        )


class SAASolver:
    """
    Sample Average Approximation solver.

    SAA replaces the expectation E[Q(x,ξ)] with a sample average:

        (1/N) Σ_{i=1}^N Q(x, ξ_i)

    This creates a deterministic equivalent that can be solved efficiently.

    Statistical analysis is performed using multiple replications to
    estimate the optimality gap and construct confidence intervals.

    Args:
        base_problem: TwoStageLP with base parameters
        scenario_generator: Generator for random scenarios
        n_samples: Number of samples per SAA problem
        n_replications: Number of independent SAA replications
        confidence_level: Confidence level for intervals

    Example:
        >>> # Create base problem
        >>> problem = TwoStageLP(c=c, A=A, b=b)
        >>>
        >>> # Create scenario generator
        >>> gen = ScenarioGenerator(q_base, W, T, h_base)
        >>> gen.set_h_distribution(NormalDistribution(h_mean, h_std))
        >>>
        >>> # Solve with SAA
        >>> solver = SAASolver(problem, gen, n_samples=500, n_replications=20)
        >>> result = solver.solve()
        >>> print(f"Gap estimate: {result.gap_estimate:.4f}")
    """

    def __init__(
        self,
        base_problem: TwoStageLP,
        scenario_generator: ScenarioGenerator,
        n_samples: int = 1000,
        n_replications: int = 10,
        confidence_level: float = 0.95,
    ) -> None:
        self.base_problem = base_problem
        self.generator = scenario_generator
        self.n_samples = n_samples
        self.n_replications = n_replications
        self.confidence_level = confidence_level

    def solve(
        self,
        max_iters: int = 50000,
        tolerance: float = 1e-5,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> SAAResult:
        """
        Solve using SAA with multiple replications.

        Args:
            max_iters: Maximum iterations per SAA problem
            tolerance: Convergence tolerance
            verbose: Print progress
            seed: Random seed for reproducibility

        Returns:
            SAAResult with solution and statistical analysis
        """
        import time

        if seed is not None:
            np.random.seed(seed)

        start_time = time.time()

        # Store results from each replication
        objectives: List[float] = []
        solutions: List[np.ndarray] = []

        for rep in range(self.n_replications):
            if verbose:
                print(f"Replication {rep + 1}/{self.n_replications}")

            # Generate scenarios
            scenarios = self.generator.generate(self.n_samples)

            # Create SAA problem
            problem = TwoStageLP(
                c=self.base_problem.c,
                A=self.base_problem.A,
                b=self.base_problem.b,
                sense=self.base_problem.sense,
                lb=self.base_problem.lb,
                ub=self.base_problem.ub,
            )
            problem.add_scenarios_from_set(scenarios)

            # Solve
            result = problem.solve(
                method="extensive",
                max_iters=max_iters,
                tolerance=tolerance,
                verbose=False,
            )

            objectives.append(result.total_cost)
            solutions.append(result.x.copy())

        solve_time = time.time() - start_time

        # Statistical analysis
        objectives_arr = np.array(objectives)

        # Lower bound: average of SAA objectives
        lower_bound = objectives_arr.mean()

        # Best solution
        best_idx = np.argmin(objectives_arr)
        x_best = solutions[best_idx]

        # Upper bound: evaluate best solution on larger sample
        upper_bound = self._estimate_upper_bound(x_best, seed)

        # Gap estimate
        gap_estimate = upper_bound - lower_bound

        # Confidence interval
        ci_lower, ci_upper = self._compute_confidence_interval(lower_bound, objectives_arr.std())

        return SAAResult(
            x=x_best,
            objective=objectives_arr[best_idx],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            gap_estimate=gap_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_samples=self.n_samples,
            n_replications=self.n_replications,
            solve_time=solve_time,
        )

    def _estimate_upper_bound(
        self,
        x: np.ndarray,
        seed: Optional[int],
    ) -> float:
        """Estimate upper bound by evaluating on fresh sample."""
        if seed is not None:
            np.random.seed(seed + 1000)

        # Generate large independent sample
        n_eval = self.n_samples * 2
        scenarios = self.generator.generate(n_eval)

        # Evaluate
        first_stage = float(self.base_problem.c @ x)
        recourse = scenarios.expected_recourse(x)

        return first_stage + recourse

    def _compute_confidence_interval(
        self,
        mean: float,
        std: float,
    ) -> tuple:
        """Compute confidence interval."""
        from scipy import stats

        n = self.n_replications
        alpha = 1 - self.confidence_level

        # t-distribution critical value
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)

        margin = t_crit * std / np.sqrt(n)

        return (mean - margin, mean + margin)


def solve_saa(
    problem: TwoStageLP,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    **kwargs,
) -> TwoStageResult:
    """
    Convenience function to solve with SAA.

    Args:
        problem: TwoStageLP with scenarios already added
        n_samples: Number of samples (if problem has more, subsample)
        seed: Random seed
        **kwargs: Additional solver arguments

    Returns:
        TwoStageResult
    """
    if seed is not None:
        np.random.seed(seed)

    if problem.n_scenarios > n_samples:
        # Subsample
        sampled = problem.scenarios.sample(n_samples)

        new_problem = TwoStageLP(
            c=problem.c,
            A=problem.A,
            b=problem.b,
            sense=problem.sense,
            lb=problem.lb,
            ub=problem.ub,
        )
        new_problem.add_scenarios_from_set(sampled)

        return new_problem.solve(**kwargs)
    else:
        return problem.solve(**kwargs)


def monte_carlo_bound(
    problem: TwoStageLP,
    x: np.ndarray,
    n_samples: int = 10000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> tuple:
    """
    Monte Carlo estimation of objective with confidence bounds.

    Args:
        problem: Problem with generator attached
        x: First-stage decision to evaluate
        n_samples: Number of Monte Carlo samples
        confidence_level: Confidence level
        seed: Random seed

    Returns:
        Tuple of (estimate, ci_lower, ci_upper)
    """
    from scipy import stats

    if seed is not None:
        np.random.seed(seed)

    # First stage cost (deterministic)
    first_stage = float(problem.c @ x)

    # Monte Carlo for recourse
    recourse_samples = []

    for s in problem.scenarios:
        Q = s.evaluate_recourse(x)
        recourse_samples.append(Q)

    recourse_samples = np.array(recourse_samples)
    probs = problem.scenarios.probabilities

    # Weighted estimate
    estimate = first_stage + (recourse_samples * probs).sum()

    # Variance estimate
    variance = ((recourse_samples - estimate) ** 2 * probs).sum()
    std = np.sqrt(variance)

    # Confidence interval
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)
    margin = z * std / np.sqrt(len(recourse_samples))

    return estimate, estimate - margin, estimate + margin
