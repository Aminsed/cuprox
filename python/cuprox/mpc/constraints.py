"""
MPC Constraints
===============

Constraint handling for Model Predictive Control.

Supports:
- Box constraints (lb <= x <= ub)
- Polytope constraints (A @ x <= b)
- Mixed constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class BoxConstraints:
    """
    Box (bound) constraints.
    
    Represents: lb <= x <= ub
    
    Args:
        lower: Lower bound (scalar or vector)
        upper: Upper bound (scalar or vector)
        dim: Dimension (required if bounds are scalar)
    
    Example:
        >>> # Scalar bounds for 3D
        >>> box = BoxConstraints(-1.0, 1.0, dim=3)
        >>> 
        >>> # Per-dimension bounds
        >>> box = BoxConstraints(
        ...     lower=np.array([-1, -2, -3]),
        ...     upper=np.array([1, 2, 3])
        ... )
    """
    lower: Union[float, np.ndarray]
    upper: Union[float, np.ndarray]
    dim: Optional[int] = None
    
    def __post_init__(self):
        """Process bounds."""
        if np.isscalar(self.lower):
            if self.dim is None:
                raise ValueError("dim required when bounds are scalar")
            self.lower = np.full(self.dim, self.lower)
        else:
            self.lower = np.asarray(self.lower, dtype=np.float64)
            if self.dim is None:
                self.dim = len(self.lower)
        
        if np.isscalar(self.upper):
            self.upper = np.full(self.dim, self.upper)
        else:
            self.upper = np.asarray(self.upper, dtype=np.float64)
        
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper must have same length")
    
    @property
    def lb(self) -> np.ndarray:
        """Lower bounds."""
        return self.lower
    
    @property
    def ub(self) -> np.ndarray:
        """Upper bounds."""
        return self.upper
    
    def is_satisfied(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if x satisfies constraints."""
        return (x >= self.lower - tol).all() and (x <= self.upper + tol).all()
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project x onto feasible set."""
        return np.clip(x, self.lower, self.upper)
    
    def violation(self, x: np.ndarray) -> float:
        """Compute maximum constraint violation."""
        lower_viol = np.maximum(self.lower - x, 0).max()
        upper_viol = np.maximum(x - self.upper, 0).max()
        return max(lower_viol, upper_viol)
    
    @classmethod
    def unbounded(cls, dim: int) -> "BoxConstraints":
        """Create unbounded constraints."""
        return cls(
            lower=np.full(dim, -np.inf),
            upper=np.full(dim, np.inf),
            dim=dim
        )
    
    @classmethod
    def positive(cls, dim: int) -> "BoxConstraints":
        """Create non-negativity constraints."""
        return cls(lower=0.0, upper=np.inf, dim=dim)


@dataclass
class PolytopeConstraints:
    """
    Polytope (linear inequality) constraints.
    
    Represents: A @ x <= b
    
    Args:
        A: Constraint matrix (m, n)
        b: Constraint bounds (m,)
    
    Example:
        >>> # Single constraint: x1 + x2 <= 1
        >>> A = np.array([[1, 1]])
        >>> b = np.array([1])
        >>> poly = PolytopeConstraints(A, b)
    """
    A: np.ndarray
    b: np.ndarray
    
    def __post_init__(self):
        """Validate dimensions."""
        self.A = np.asarray(self.A, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {self.A.shape}")
        if self.b.ndim != 1:
            raise ValueError(f"b must be 1D, got shape {self.b.shape}")
        if self.A.shape[0] != len(self.b):
            raise ValueError("A rows must match b length")
    
    @property
    def n_constraints(self) -> int:
        """Number of constraints."""
        return len(self.b)
    
    @property
    def dim(self) -> int:
        """Dimension of constrained variable."""
        return self.A.shape[1]
    
    def is_satisfied(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if x satisfies constraints."""
        return (self.A @ x <= self.b + tol).all()
    
    def violation(self, x: np.ndarray) -> float:
        """Compute maximum constraint violation."""
        return np.maximum(self.A @ x - self.b, 0).max()
    
    @classmethod
    def from_box(cls, box: BoxConstraints) -> "PolytopeConstraints":
        """
        Convert box constraints to polytope form.
        
        lb <= x <= ub becomes:
        -I @ x <= -lb
         I @ x <= ub
        """
        n = box.dim
        A = np.vstack([-np.eye(n), np.eye(n)])
        b = np.concatenate([-box.lower, box.upper])
        return cls(A, b)


@dataclass
class StateInputConstraints:
    """
    Combined state and input constraints for MPC.
    
    Args:
        x_min: State lower bounds
        x_max: State upper bounds
        u_min: Input lower bounds
        u_max: Input upper bounds
        x_polytope: Additional state polytope constraints
        u_polytope: Additional input polytope constraints
    """
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    x_polytope: Optional[PolytopeConstraints] = None
    u_polytope: Optional[PolytopeConstraints] = None
    
    def get_state_box(self, n_x: int) -> BoxConstraints:
        """Get state box constraints."""
        lower = self.x_min if self.x_min is not None else np.full(n_x, -np.inf)
        upper = self.x_max if self.x_max is not None else np.full(n_x, np.inf)
        return BoxConstraints(lower, upper)
    
    def get_input_box(self, n_u: int) -> BoxConstraints:
        """Get input box constraints."""
        lower = self.u_min if self.u_min is not None else np.full(n_u, -np.inf)
        upper = self.u_max if self.u_max is not None else np.full(n_u, np.inf)
        return BoxConstraints(lower, upper)


def terminal_constraint_set(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    x_box: BoxConstraints,
    u_box: BoxConstraints,
    n_vertices: int = 100,
) -> PolytopeConstraints:
    """
    Compute maximum positive invariant terminal set.
    
    For a stabilizing controller u = K @ x, finds the largest set
    where the closed-loop system remains feasible.
    
    Args:
        A: State matrix
        B: Input matrix
        K: Feedback gain
        x_box: State constraints
        u_box: Input constraints
        n_vertices: Maximum vertices for approximation
    
    Returns:
        Polytope constraints for terminal set
    
    Note:
        This is a simplified approximation. For rigorous MPI sets,
        use specialized tools like MPT3.
    """
    n_x = A.shape[0]
    
    # Closed-loop dynamics
    A_cl = A + B @ K
    
    # Start with state and input constraints
    # u = Kx constraints: u_min <= Kx <= u_max
    A_constraints = [
        -np.eye(n_x),       # -x <= -x_min
        np.eye(n_x),        # x <= x_max
        -K,                 # -Kx <= -u_min
        K,                  # Kx <= u_max
    ]
    b_constraints = [
        -x_box.lower,
        x_box.upper,
        -u_box.lower,
        u_box.upper,
    ]
    
    A_poly = np.vstack(A_constraints)
    b_poly = np.concatenate(b_constraints)
    
    # Remove infinite bounds
    finite_mask = np.isfinite(b_poly)
    A_poly = A_poly[finite_mask]
    b_poly = b_poly[finite_mask]
    
    return PolytopeConstraints(A_poly, b_poly)

