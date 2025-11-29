"""
cuProx Result Classes
=====================

Data classes for solver results and status.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

import numpy as np


class Status(Enum):
    """
    Solver status codes.
    
    Attributes:
        OPTIMAL: Solution found within tolerance
        PRIMAL_INFEASIBLE: Problem has no feasible solution
        DUAL_INFEASIBLE: Problem is unbounded (objective → -∞)
        MAX_ITERATIONS: Maximum iteration limit reached
        TIME_LIMIT: Time limit exceeded
        NUMERICAL_ERROR: Numerical issues encountered
        UNSOLVED: Problem not yet solved
    """
    OPTIMAL = "optimal"
    PRIMAL_INFEASIBLE = "primal_infeasible"
    DUAL_INFEASIBLE = "dual_infeasible"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    NUMERICAL_ERROR = "numerical_error"
    UNSOLVED = "unsolved"
    INVALID_INPUT = "invalid_input"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_successful(self) -> bool:
        """True if an optimal solution was found."""
        return self == Status.OPTIMAL
    
    @property
    def has_solution(self) -> bool:
        """True if a (possibly suboptimal) solution is available."""
        return self in (
            Status.OPTIMAL,
            Status.MAX_ITERATIONS,
            Status.TIME_LIMIT,
        )


@dataclass
class SolveResult:
    """
    Result of solving an LP/QP problem.
    
    Attributes:
        status: Solver status
        objective: Optimal objective value
        x: Primal solution vector
        y: Dual solution vector (Lagrange multipliers)
        iterations: Number of iterations performed
        solve_time: Wall clock time in seconds
        primal_residual: Final primal residual (feasibility)
        dual_residual: Final dual residual (optimality)
        gap: Duality gap
    
    Example:
        >>> result = model.solve()
        >>> if result.status == Status.OPTIMAL:
        ...     print(f"Optimal value: {result.objective}")
        ...     print(f"Solution: {result.x}")
    """
    
    status: Status
    objective: float
    x: np.ndarray
    y: np.ndarray
    iterations: int
    solve_time: float
    
    # Convergence metrics
    primal_residual: float = 0.0
    dual_residual: float = 0.0
    gap: float = 0.0
    
    # Optional metadata
    setup_time: float = 0.0
    problem_info: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"SolveResult(status={self.status}, "
            f"objective={self.objective:.6g}, "
            f"iterations={self.iterations}, "
            f"time={self.solve_time:.4f}s)"
        )
    
    def get_value(self, var: "Variable") -> float:
        """
        Get the solution value for a specific variable.
        
        Args:
            var: Variable object from the model
            
        Returns:
            Optimal value of the variable
        """
        return float(self.x[var.index])
    
    def get_values(self, vars: list) -> np.ndarray:
        """
        Get solution values for multiple variables.
        
        Args:
            vars: List of Variable objects
            
        Returns:
            Array of optimal values
        """
        indices = [v.index for v in vars]
        return self.x[indices]
    
    def get_dual(self, constr: "Constraint") -> float:
        """
        Get the dual value (shadow price) for a constraint.
        
        Args:
            constr: Constraint object from the model
            
        Returns:
            Dual value of the constraint
        """
        return float(self.y[constr.index])
    
    def summary(self) -> str:
        """Return a formatted summary of the solve result."""
        lines = [
            "=" * 50,
            "cuProx Solve Summary",
            "=" * 50,
            f"Status:           {self.status}",
            f"Objective:        {self.objective:.10g}",
            f"Iterations:       {self.iterations}",
            f"Solve time:       {self.solve_time:.4f} s",
            f"Setup time:       {self.setup_time:.4f} s",
            "-" * 50,
            f"Primal residual:  {self.primal_residual:.6e}",
            f"Dual residual:    {self.dual_residual:.6e}",
            f"Duality gap:      {self.gap:.6e}",
            "=" * 50,
        ]
        return "\n".join(lines)
    
    @classmethod
    def from_raw(cls, raw_result: dict) -> "SolveResult":
        """
        Create SolveResult from raw C++ solver output.
        
        This is used internally by the solver bindings.
        """
        status_map = {
            0: Status.OPTIMAL,
            1: Status.PRIMAL_INFEASIBLE,
            2: Status.DUAL_INFEASIBLE,
            3: Status.MAX_ITERATIONS,
            4: Status.TIME_LIMIT,
            5: Status.NUMERICAL_ERROR,
            -1: Status.UNSOLVED,
        }
        
        return cls(
            status=status_map.get(raw_result.get("status", -1), Status.UNSOLVED),
            objective=raw_result.get("objective", 0.0),
            x=np.asarray(raw_result.get("x", [])),
            y=np.asarray(raw_result.get("y", [])),
            iterations=raw_result.get("iterations", 0),
            solve_time=raw_result.get("solve_time", 0.0),
            primal_residual=raw_result.get("primal_residual", 0.0),
            dual_residual=raw_result.get("dual_residual", 0.0),
            gap=raw_result.get("gap", 0.0),
            setup_time=raw_result.get("setup_time", 0.0),
        )


# Type alias for Variable (defined in model.py)
# This avoids circular imports
Variable = Any
Constraint = Any

