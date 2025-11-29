"""
Two-Stage Stochastic Programming Problems
==========================================

Classes for defining and solving two-stage stochastic programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from .scenarios import ScenarioSet


@dataclass
class TwoStageResult:
    """
    Result of solving a two-stage stochastic program.

    Attributes:
        x: First-stage decision
        y: Second-stage decisions (dict: scenario -> y)
        first_stage_cost: c'x
        expected_recourse: E[Q(x, ξ)]
        total_cost: c'x + E[Q(x, ξ)]
        status: Solver status
        solve_time: Solution time
        n_scenarios: Number of scenarios
    """

    x: np.ndarray
    y: Optional[Dict[int, np.ndarray]]
    first_stage_cost: float
    expected_recourse: float
    total_cost: float
    status: str
    solve_time: float
    n_scenarios: int

    def __repr__(self) -> str:
        return (
            f"TwoStageResult(\n"
            f"  status={self.status},\n"
            f"  first_stage_cost={self.first_stage_cost:.4f},\n"
            f"  expected_recourse={self.expected_recourse:.4f},\n"
            f"  total_cost={self.total_cost:.4f},\n"
            f"  n_scenarios={self.n_scenarios}\n"
            f")"
        )

    def summary(self) -> str:
        """Formatted summary."""
        lines = [
            "=" * 50,
            "Two-Stage Stochastic Program Solution",
            "=" * 50,
            f"Status:            {self.status}",
            f"Scenarios:         {self.n_scenarios}",
            f"Solve time:        {self.solve_time:.4f}s",
            "-" * 50,
            f"First-stage cost:  {self.first_stage_cost:.4f}",
            f"Expected recourse: {self.expected_recourse:.4f}",
            f"Total cost:        {self.total_cost:.4f}",
            "-" * 50,
            "First-stage decision x:",
        ]

        for i, xi in enumerate(self.x):
            if abs(xi) > 1e-6:
                lines.append(f"  x[{i}] = {xi:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)


class TwoStageLP:
    """
    Two-Stage Stochastic Linear Program.

    Standard form:
        minimize    c'x + E[Q(x, ξ)]
        subject to  Ax ≤ b (or = b)
                    x ≥ lb

        where Q(x, ξ) = min  q(ξ)'y
                       s.t. W(ξ)y ≥ h(ξ) - T(ξ)x  (or =)
                            y ≥ 0

    Args:
        c: First-stage cost vector (n_x,)
        A: First-stage constraint matrix (m_1, n_x)
        b: First-stage RHS (m_1,)
        sense: Constraint sense ('<=', '>=', '=')
        lb: Variable lower bounds (default: 0)
        ub: Variable upper bounds (default: inf)

    Example:
        >>> # Newsvendor problem
        >>> problem = TwoStageLP(
        ...     c=np.array([10]),  # ordering cost
        ...     A=np.zeros((0, 1)),
        ...     b=np.array([]),
        ...     lb=0, ub=100
        ... )
        >>>
        >>> # Add demand scenarios
        >>> for demand, prob in [(20, 0.3), (40, 0.5), (60, 0.2)]:
        ...     problem.add_scenario(
        ...         probability=prob,
        ...         q=np.array([-15, 5]),  # selling price, salvage
        ...         W=np.array([[1, 0], [0, 1], [-1, -1]]),
        ...         T=np.array([[0], [0], [1]]),
        ...         h=np.array([demand, 0, 0])  # demand, non-neg, inventory
        ...     )
        >>>
        >>> result = problem.solve()
    """

    def __init__(
        self,
        c: np.ndarray,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        sense: str = "<=",
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> None:
        self.c = np.asarray(c, dtype=np.float64)
        self.n_x = len(self.c)

        if A is not None:
            self.A = np.asarray(A, dtype=np.float64)
            self.b = np.asarray(b, dtype=np.float64)
        else:
            self.A = np.zeros((0, self.n_x))
            self.b = np.array([])

        self.sense = sense

        # Bounds
        if lb is None:
            self.lb = np.zeros(self.n_x)
        elif np.isscalar(lb):
            self.lb = np.full(self.n_x, lb)
        else:
            self.lb = np.asarray(lb, dtype=np.float64)

        if ub is None:
            self.ub = np.full(self.n_x, 1e20)
        elif np.isscalar(ub):
            self.ub = np.full(self.n_x, ub)
        else:
            self.ub = np.asarray(ub, dtype=np.float64)

        self._scenarios = ScenarioSet()

    @property
    def scenarios(self) -> ScenarioSet:
        """Access scenario set."""
        return self._scenarios

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios."""
        return len(self._scenarios)

    def add_scenario(
        self,
        probability: float,
        q: np.ndarray,
        W: np.ndarray,
        T: np.ndarray,
        h: np.ndarray,
        name: Optional[str] = None,
    ) -> None:
        """
        Add a scenario.

        Args:
            probability: Scenario probability
            q: Second-stage cost
            W: Second-stage constraint matrix
            T: Technology matrix
            h: Second-stage RHS
            name: Scenario name
        """
        self._scenarios.add_scenario(probability, q, W, T, h, name)

    def add_scenarios_from_set(self, scenario_set: ScenarioSet) -> None:
        """Add all scenarios from a ScenarioSet."""
        for s in scenario_set:
            self._scenarios.add(s)

    def solve(
        self,
        method: str = "extensive",
        max_iters: int = 50000,
        tolerance: float = 1e-5,
        verbose: bool = False,
    ) -> TwoStageResult:
        """
        Solve the two-stage stochastic program.

        Args:
            method: Solution method ('extensive', 'benders')
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            TwoStageResult
        """
        if method == "extensive":
            return self._solve_extensive(max_iters, tolerance, verbose)
        elif method == "benders":
            return self._solve_benders(max_iters, tolerance, verbose)
        else:
            raise ValueError(f"Unknown method '{method}'")

    def _solve_extensive(
        self,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> TwoStageResult:
        """Solve using extensive (deterministic equivalent) form."""
        import time

        from .. import solve as cuprox_solve

        start_time = time.time()

        # Validate scenarios
        self._scenarios.validate()

        # Build deterministic equivalent
        de = self._scenarios.to_deterministic_equivalent(self.c, self.A, self.b)

        n_x = de["n_x"]
        n_y = de["n_y"]
        n_s = de["n_scenarios"]
        n_total = len(de["c"])

        # Build bounds
        lb = np.zeros(n_total)
        ub = np.full(n_total, 1e20)

        lb[:n_x] = self.lb
        ub[:n_x] = self.ub

        # Constraint bounds (equality)
        m = len(de["b"])
        constraint_l = de["b"].copy()
        constraint_u = de["b"].copy()

        # If first stage is inequality
        if self.sense == "<=":
            constraint_l[: len(self.b)] = -1e20
        elif self.sense == ">=":
            constraint_u[: len(self.b)] = 1e20

        # Solve
        result = cuprox_solve(
            c=de["c"],
            A=sparse.csr_matrix(de["A"]),
            b=de["b"],
            lb=lb,
            ub=ub,
            constraint_l=constraint_l,
            constraint_u=constraint_u,
            params={
                "max_iterations": max_iters,
                "tolerance": tolerance,
                "verbose": verbose,
            },
        )

        solve_time = time.time() - start_time

        # Extract solution
        z = result.x
        x = z[:n_x]

        # Extract second-stage solutions
        y_dict = {}
        for i in range(n_s):
            start = n_x + i * n_y
            y_dict[i] = z[start : start + n_y]

        # Compute costs
        first_stage_cost = float(self.c @ x)
        expected_recourse = float(de["c"][n_x:] @ z[n_x:])

        status = str(result.status.value) if hasattr(result.status, "value") else str(result.status)

        return TwoStageResult(
            x=x,
            y=y_dict,
            first_stage_cost=first_stage_cost,
            expected_recourse=expected_recourse,
            total_cost=first_stage_cost + expected_recourse,
            status=status,
            solve_time=solve_time,
            n_scenarios=n_s,
        )

    def _solve_benders(
        self,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> TwoStageResult:
        """Solve using Benders decomposition (L-shaped method)."""
        # TODO: Implement Benders decomposition
        raise NotImplementedError("Benders decomposition not yet implemented")

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate objective for given first-stage decision.

        Args:
            x: First-stage decision

        Returns:
            Total cost c'x + E[Q(x, ξ)]
        """
        first_stage = float(self.c @ x)
        recourse = self._scenarios.expected_recourse(x)
        return first_stage + recourse


class TwoStageQP(TwoStageLP):
    """
    Two-Stage Stochastic Quadratic Program.

    First stage:
        minimize    (1/2) x'Px + c'x + E[Q(x, ξ)]
        subject to  Ax ≤ b
                    x ≥ 0

    Second stage:
        Q(x, ξ) = min  (1/2) y'Ry + q(ξ)'y
                 s.t. W(ξ)y ≥ h(ξ) - T(ξ)x
                      y ≥ 0

    Args:
        P: First-stage quadratic cost (n_x, n_x)
        c: First-stage linear cost (n_x,)
        A: First-stage constraints (m_1, n_x)
        b: First-stage RHS (m_1,)
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        P: np.ndarray,
        c: np.ndarray,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(c, A, b, **kwargs)
        self.P = np.asarray(P, dtype=np.float64)

        # Store second-stage quadratic costs
        self._R_scenarios: List[np.ndarray] = []

    def add_scenario(
        self,
        probability: float,
        q: np.ndarray,
        W: np.ndarray,
        T: np.ndarray,
        h: np.ndarray,
        R: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        """Add scenario with optional quadratic cost."""
        super().add_scenario(probability, q, W, T, h, name)

        if R is not None:
            self._R_scenarios.append(np.asarray(R, dtype=np.float64))
        else:
            # Default: no quadratic term
            n_y = W.shape[1]
            self._R_scenarios.append(np.zeros((n_y, n_y)))

    def _solve_extensive(
        self,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> TwoStageResult:
        """Solve extensive form of QP."""
        import time

        from .. import solve as cuprox_solve

        start_time = time.time()

        self._scenarios.validate()

        # Build deterministic equivalent
        de = self._scenarios.to_deterministic_equivalent(self.c, self.A, self.b)

        n_x = de["n_x"]
        n_y = de["n_y"]
        n_s = de["n_scenarios"]
        n_total = len(de["c"])

        # Build full P matrix
        P_full = np.zeros((n_total, n_total))
        P_full[:n_x, :n_x] = self.P

        for i, R in enumerate(self._R_scenarios):
            start = n_x + i * n_y
            prob = self._scenarios[i].probability
            P_full[start : start + n_y, start : start + n_y] = prob * R

        # Bounds
        lb = np.zeros(n_total)
        ub = np.full(n_total, 1e20)
        lb[:n_x] = self.lb
        ub[:n_x] = self.ub

        m = len(de["b"])
        constraint_l = de["b"].copy()
        constraint_u = de["b"].copy()

        result = cuprox_solve(
            c=de["c"],
            A=sparse.csr_matrix(de["A"]),
            b=de["b"],
            P=sparse.csr_matrix(P_full),
            lb=lb,
            ub=ub,
            constraint_l=constraint_l,
            constraint_u=constraint_u,
            params={
                "max_iterations": max_iters,
                "tolerance": tolerance,
                "verbose": verbose,
            },
        )

        solve_time = time.time() - start_time

        z = result.x
        x = z[:n_x]

        y_dict = {}
        for i in range(n_s):
            start = n_x + i * n_y
            y_dict[i] = z[start : start + n_y]

        first_stage_cost = float(0.5 * x @ self.P @ x + self.c @ x)
        expected_recourse = float(de["c"][n_x:] @ z[n_x:])

        # Add quadratic recourse terms
        for i in range(n_s):
            prob = self._scenarios[i].probability
            y_i = y_dict[i]
            expected_recourse += 0.5 * prob * float(y_i @ self._R_scenarios[i] @ y_i)

        status = str(result.status.value) if hasattr(result.status, "value") else str(result.status)

        return TwoStageResult(
            x=x,
            y=y_dict,
            first_stage_cost=first_stage_cost,
            expected_recourse=expected_recourse,
            total_cost=first_stage_cost + expected_recourse,
            status=status,
            solve_time=solve_time,
            n_scenarios=n_s,
        )
