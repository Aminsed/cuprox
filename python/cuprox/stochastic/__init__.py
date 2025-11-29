"""
cuProx Stochastic Programming
=============================

GPU-accelerated stochastic optimization for decision-making under uncertainty.

This module provides tools for formulating and solving stochastic programming
problems, with focus on two-stage problems and Sample Average Approximation (SAA).

Two-Stage Stochastic Programming
--------------------------------
A two-stage problem has:
- **First stage**: "Here and now" decisions (x) before uncertainty is revealed
- **Second stage**: "Recourse" decisions (y) after observing random outcome ξ

Standard form:
    minimize    c'x + E[Q(x, ξ)]
    subject to  Ax = b, x ≥ 0

    where Q(x, ξ) = min  q(ξ)'y
                   s.t. W(ξ)y = h(ξ) - T(ξ)x, y ≥ 0

>>> from cuprox.stochastic import TwoStageLP, ScenarioSet
>>>
>>> # Define first stage
>>> problem = TwoStageLP(
...     c=first_stage_cost,
...     A=first_stage_constraints,
...     b=first_stage_rhs,
... )
>>>
>>> # Add scenarios for second stage
>>> for prob, q, W, T, h in scenarios:
...     problem.add_scenario(prob, q=q, W=W, T=T, h=h)
>>>
>>> result = problem.solve()

Sample Average Approximation (SAA)
----------------------------------
SAA replaces the expectation with a sample average, creating a deterministic
equivalent that can be solved efficiently:

>>> from cuprox.stochastic import SAASolver
>>>
>>> solver = SAASolver(problem, n_samples=1000)
>>> result = solver.solve()
>>>
>>> # Statistical analysis
>>> print(f"Optimality gap: {result.gap_estimate:.4f}")
>>> print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

Applications
------------
- Supply chain planning under demand uncertainty
- Portfolio optimization with stochastic returns
- Energy system dispatch with renewable uncertainty
- Production planning with uncertain yield/demand

Classes
-------
TwoStageLP
    Two-stage stochastic linear program
TwoStageQP
    Two-stage stochastic quadratic program
Scenario
    Single scenario with probability and parameters
ScenarioSet
    Collection of scenarios with validation
SAASolver
    Sample Average Approximation solver
SAAResult
    SAA solution with statistical analysis

References
----------
- Birge & Louveaux (2011): "Introduction to Stochastic Programming"
- Shapiro, Dentcheva & Ruszczyński (2014): "Lectures on Stochastic Programming"
- Kleywegt, Shapiro & Homem-de-Mello (2002): "The Sample Average Approximation Method"
"""

from .distributions import (
    DiscreteDistribution,
    NormalDistribution,
    UniformDistribution,
    generate_scenarios,
)
from .evaluation import (
    compute_evpi,
    compute_vss,
    evaluate_solution,
    out_of_sample_evaluation,
)
from .problem import TwoStageLP, TwoStageQP, TwoStageResult
from .saa import SAAResult, SAASolver
from .scenarios import Scenario, ScenarioGenerator, ScenarioSet

__all__ = [
    # Problems
    "TwoStageLP",
    "TwoStageQP",
    "TwoStageResult",
    # Scenarios
    "Scenario",
    "ScenarioSet",
    "ScenarioGenerator",
    # SAA
    "SAASolver",
    "SAAResult",
    # Evaluation
    "evaluate_solution",
    "compute_evpi",
    "compute_vss",
    "out_of_sample_evaluation",
    # Distributions
    "DiscreteDistribution",
    "NormalDistribution",
    "UniformDistribution",
    "generate_scenarios",
]
