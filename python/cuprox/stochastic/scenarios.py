"""
Scenario Management
===================

Classes for managing scenarios in stochastic programming.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class Scenario:
    """
    A single scenario in a stochastic program.

    Represents one possible realization of the uncertain parameters
    with an associated probability.

    For two-stage problems:
        Q(x, ξ) = min  q'y
                 s.t. Wy = h - Tx, y ≥ 0

    Args:
        probability: Scenario probability (should sum to 1 across scenarios)
        q: Second-stage cost vector (n_y,)
        W: Second-stage constraint matrix (m_2, n_y)
        T: Technology matrix linking stages (m_2, n_x)
        h: Second-stage RHS (m_2,)
        name: Optional scenario identifier
        data: Optional additional data

    Example:
        >>> scenario = Scenario(
        ...     probability=0.25,
        ...     q=np.array([1, 2]),
        ...     W=np.eye(2),
        ...     T=np.array([[1, 0], [0, 1]]),
        ...     h=np.array([10, 20])
        ... )
    """

    probability: float
    q: np.ndarray
    W: np.ndarray
    T: np.ndarray
    h: np.ndarray
    name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and convert arrays."""
        self.q = np.asarray(self.q, dtype=np.float64)
        self.W = np.asarray(self.W, dtype=np.float64)
        self.T = np.asarray(self.T, dtype=np.float64)
        self.h = np.asarray(self.h, dtype=np.float64)

        self._validate()

    def _validate(self):
        """Validate scenario dimensions."""
        if not (0 <= self.probability <= 1):
            raise ValueError(f"Probability must be in [0,1], got {self.probability}")

        if self.W.ndim != 2:
            raise ValueError(f"W must be 2D, got shape {self.W.shape}")

        m_2, n_y = self.W.shape

        if self.q.shape != (n_y,):
            raise ValueError(f"q must have shape ({n_y},), got {self.q.shape}")

        if self.h.shape != (m_2,):
            raise ValueError(f"h must have shape ({m_2},), got {self.h.shape}")

        if self.T.shape[0] != m_2:
            raise ValueError(f"T rows must match W rows ({m_2}), got {self.T.shape[0]}")

    @property
    def n_second_stage_vars(self) -> int:
        """Number of second-stage variables."""
        return self.W.shape[1]

    @property
    def n_second_stage_constraints(self) -> int:
        """Number of second-stage constraints."""
        return self.W.shape[0]

    @property
    def n_first_stage_vars(self) -> int:
        """Number of first-stage variables (from T matrix)."""
        return self.T.shape[1]

    def scale_probability(self, factor: float) -> Scenario:
        """Return scenario with scaled probability."""
        return Scenario(
            probability=self.probability * factor,
            q=self.q.copy(),
            W=self.W.copy(),
            T=self.T.copy(),
            h=self.h.copy(),
            name=self.name,
            data=self.data,
        )

    def evaluate_recourse(self, x: np.ndarray) -> float:
        """
        Evaluate the recourse cost Q(x, ξ) for given first-stage decision.

        Args:
            x: First-stage decision

        Returns:
            Optimal second-stage cost (or inf if infeasible)
        """
        from scipy.optimize import linprog

        # Solve: min q'y s.t. Wy = h - Tx, y >= 0
        b_eq = self.h - self.T @ x

        result = linprog(c=self.q, A_eq=self.W, b_eq=b_eq, bounds=(0, None), method="highs")

        if result.success:
            return result.fun
        else:
            return float("inf")


class ScenarioSet:
    """
    Collection of scenarios for stochastic programming.

    Manages a set of scenarios and ensures they form a valid
    probability distribution.

    Args:
        scenarios: Initial list of scenarios
        normalize: Whether to normalize probabilities to sum to 1

    Example:
        >>> scenarios = ScenarioSet()
        >>> scenarios.add(Scenario(0.3, q1, W1, T1, h1))
        >>> scenarios.add(Scenario(0.7, q2, W2, T2, h2))
        >>>
        >>> for s in scenarios:
        ...     print(f"Scenario: prob={s.probability:.2f}")
    """

    def __init__(
        self,
        scenarios: Optional[List[Scenario]] = None,
        normalize: bool = False,
    ) -> None:
        self._scenarios: List[Scenario] = []
        self._normalize = normalize

        if scenarios:
            for s in scenarios:
                self.add(s)

    def add(self, scenario: Scenario) -> None:
        """Add a scenario to the set."""
        self._scenarios.append(scenario)

    def add_scenario(
        self,
        probability: float,
        q: np.ndarray,
        W: np.ndarray,
        T: np.ndarray,
        h: np.ndarray,
        name: Optional[str] = None,
    ) -> None:
        """Convenience method to add scenario from arrays."""
        self.add(Scenario(probability, q, W, T, h, name))

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios."""
        return len(self._scenarios)

    @property
    def probabilities(self) -> np.ndarray:
        """Array of scenario probabilities."""
        return np.array([s.probability for s in self._scenarios])

    @property
    def total_probability(self) -> float:
        """Sum of probabilities."""
        return self.probabilities.sum()

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self._scenarios)

    def __getitem__(self, idx: int) -> Scenario:
        return self._scenarios[idx]

    def validate(self) -> bool:
        """
        Validate scenario set.

        Checks:
        1. Probabilities sum to 1 (within tolerance)
        2. All scenarios have consistent dimensions

        Returns:
            True if valid

        Raises:
            ValueError: If invalid
        """
        if len(self._scenarios) == 0:
            raise ValueError("No scenarios defined")

        # Check probabilities
        total = self.total_probability
        if abs(total - 1.0) > 1e-6:
            if self._normalize:
                self._normalize_probabilities()
            else:
                raise ValueError(f"Probabilities sum to {total}, not 1.0")

        # Check dimensions
        first = self._scenarios[0]
        n_y = first.n_second_stage_vars
        m_2 = first.n_second_stage_constraints
        n_x = first.n_first_stage_vars

        for i, s in enumerate(self._scenarios[1:], 2):
            if s.n_second_stage_vars != n_y:
                raise ValueError(f"Scenario {i}: n_y={s.n_second_stage_vars} != {n_y}")
            if s.n_second_stage_constraints != m_2:
                raise ValueError(f"Scenario {i}: m_2={s.n_second_stage_constraints} != {m_2}")
            if s.n_first_stage_vars != n_x:
                raise ValueError(f"Scenario {i}: n_x={s.n_first_stage_vars} != {n_x}")

        return True

    def _normalize_probabilities(self) -> None:
        """Normalize probabilities to sum to 1."""
        total = self.total_probability
        if total > 0:
            for i in range(len(self._scenarios)):
                self._scenarios[i] = self._scenarios[i].scale_probability(1.0 / total)

    def sample(self, n: int, replace: bool = True) -> ScenarioSet:
        """
        Sample scenarios according to their probabilities.

        Args:
            n: Number of samples
            replace: Whether to sample with replacement

        Returns:
            New ScenarioSet with sampled scenarios (equal probabilities)
        """
        probs = self.probabilities
        probs = probs / probs.sum()  # Normalize

        indices = np.random.choice(len(self._scenarios), size=n, replace=replace, p=probs)

        sampled = []
        equal_prob = 1.0 / n

        for idx in indices:
            s = self._scenarios[idx]
            sampled.append(
                Scenario(
                    probability=equal_prob,
                    q=s.q.copy(),
                    W=s.W.copy(),
                    T=s.T.copy(),
                    h=s.h.copy(),
                    name=s.name,
                )
            )

        return ScenarioSet(sampled)

    def expected_recourse(self, x: np.ndarray) -> float:
        """
        Compute expected recourse cost E[Q(x, ξ)].

        Args:
            x: First-stage decision

        Returns:
            Expected second-stage cost
        """
        total = 0.0
        for s in self._scenarios:
            total += s.probability * s.evaluate_recourse(x)
        return total

    def to_deterministic_equivalent(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Build deterministic equivalent (extensive form).

        Combines first stage and all scenarios into a single large LP.

        Args:
            c: First-stage cost
            A: First-stage constraint matrix
            b: First-stage RHS

        Returns:
            Dictionary with keys: c_full, A_full, b_full, lb_full, ub_full
        """
        n_x = len(c)
        n_s = self.n_scenarios

        if n_s == 0:
            raise ValueError("No scenarios")

        n_y = self._scenarios[0].n_second_stage_vars
        m_1 = A.shape[0]
        m_2 = self._scenarios[0].n_second_stage_constraints

        # Total variables: x (n_x) + y_s for each scenario (n_s * n_y)
        n_total = n_x + n_s * n_y

        # Total constraints: first stage (m_1) + second stage per scenario (n_s * m_2)
        m_total = m_1 + n_s * m_2

        # Build cost vector
        c_full = np.zeros(n_total)
        c_full[:n_x] = c

        for i, s in enumerate(self._scenarios):
            start = n_x + i * n_y
            c_full[start : start + n_y] = s.probability * s.q

        # Build constraint matrix
        A_full = np.zeros((m_total, n_total))
        b_full = np.zeros(m_total)

        # First stage constraints
        A_full[:m_1, :n_x] = A
        b_full[:m_1] = b

        # Second stage constraints for each scenario
        for i, s in enumerate(self._scenarios):
            row_start = m_1 + i * m_2
            y_start = n_x + i * n_y

            # T_i @ x + W_i @ y_i = h_i
            A_full[row_start : row_start + m_2, :n_x] = s.T
            A_full[row_start : row_start + m_2, y_start : y_start + n_y] = s.W
            b_full[row_start : row_start + m_2] = s.h

        return {
            "c": c_full,
            "A": A_full,
            "b": b_full,
            "n_x": n_x,
            "n_y": n_y,
            "n_scenarios": n_s,
        }


class ScenarioGenerator:
    """
    Generate scenarios from distributions.

    Args:
        base_q: Base second-stage cost
        base_W: Base technology matrix (typically fixed)
        base_T: Base linking matrix (typically fixed)
        base_h: Base second-stage RHS

    Example:
        >>> gen = ScenarioGenerator(q_base, W, T, h_base)
        >>> gen.set_h_distribution(NormalDistribution(mean=h_base, std=2.0))
        >>> scenarios = gen.generate(n_scenarios=100)
    """

    def __init__(
        self,
        base_q: np.ndarray,
        base_W: np.ndarray,
        base_T: np.ndarray,
        base_h: np.ndarray,
    ) -> None:
        self.base_q = np.asarray(base_q, dtype=np.float64)
        self.base_W = np.asarray(base_W, dtype=np.float64)
        self.base_T = np.asarray(base_T, dtype=np.float64)
        self.base_h = np.asarray(base_h, dtype=np.float64)

        self._q_generator: Optional[Callable] = None
        self._h_generator: Optional[Callable] = None

    def set_q_distribution(self, distribution) -> ScenarioGenerator:
        """Set distribution for q (second-stage costs)."""
        self._q_generator = distribution.sample
        return self

    def set_h_distribution(self, distribution) -> ScenarioGenerator:
        """Set distribution for h (second-stage RHS)."""
        self._h_generator = distribution.sample
        return self

    def generate(
        self,
        n_scenarios: int,
        seed: Optional[int] = None,
    ) -> ScenarioSet:
        """
        Generate scenarios.

        Args:
            n_scenarios: Number of scenarios to generate
            seed: Random seed

        Returns:
            ScenarioSet with generated scenarios
        """
        if seed is not None:
            np.random.seed(seed)

        scenarios = []
        prob = 1.0 / n_scenarios

        for i in range(n_scenarios):
            # Generate random parameters
            q = self._q_generator() if self._q_generator else self.base_q.copy()
            h = self._h_generator() if self._h_generator else self.base_h.copy()

            scenarios.append(
                Scenario(
                    probability=prob,
                    q=q,
                    W=self.base_W.copy(),
                    T=self.base_T.copy(),
                    h=h,
                    name=f"scenario_{i}",
                )
            )

        return ScenarioSet(scenarios)
