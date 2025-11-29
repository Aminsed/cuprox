"""
Differentiable Optimization Layers
==================================

PyTorch nn.Module wrappers for optimization problems that support
backpropagation through the solver.

Classes:
    QPLayer: Differentiable quadratic programming layer
    LPLayer: Differentiable linear programming layer
"""

from __future__ import annotations

from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = type("Module", (), {})  # Placeholder

from .utils import check_torch_available
from .functions import QPFunction, LPFunction


class QPLayer(nn.Module if HAS_TORCH else object):
    """
    Differentiable Quadratic Programming Layer.
    
    Solves:
        minimize    (1/2) x' P x + q' x
        subject to  A x = b           (equality)
                    G x <= h          (inequality)
                    lb <= x <= ub     (bounds)
    
    Supports backpropagation through the optimal solution using
    implicit differentiation of the KKT conditions.
    
    Args:
        n_vars: Number of decision variables
        n_eq: Number of equality constraints (default: 0)
        n_ineq: Number of inequality constraints (default: 0)
        max_iters: Maximum solver iterations (default: 10000)
        eps: Convergence tolerance (default: 1e-5)
        verbose: Print solver progress (default: False)
        differentiable: Enable gradients (default: True)
    
    Example:
        >>> layer = QPLayer(n_vars=2)
        >>> P = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
        >>> q = torch.tensor([-2., -4.], requires_grad=True)
        >>> x = layer(P, q)  # x = [1, 2]
        >>> x.sum().backward()
        >>> print(q.grad)  # gradient w.r.t. q
    """
    
    def __init__(
        self,
        n_vars: int,
        n_eq: int = 0,
        n_ineq: int = 0,
        max_iters: int = 10000,
        eps: float = 1e-5,
        verbose: bool = False,
        differentiable: bool = True,
    ) -> None:
        check_torch_available()
        super().__init__()
        
        # Validate
        if n_vars <= 0:
            raise ValueError(f"n_vars must be positive, got {n_vars}")
        if n_eq < 0:
            raise ValueError(f"n_eq must be non-negative, got {n_eq}")
        if n_ineq < 0:
            raise ValueError(f"n_ineq must be non-negative, got {n_ineq}")
        
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.max_iters = max_iters
        self.eps = eps
        self.verbose = verbose
        self.differentiable = differentiable
    
    def forward(
        self,
        P: Tensor,
        q: Tensor,
        A: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        G: Optional[Tensor] = None,
        h: Optional[Tensor] = None,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Solve QP and return optimal solution.
        
        Args:
            P: Quadratic cost (n, n) - must be positive semidefinite
            q: Linear cost (n,)
            A: Equality constraints (n_eq, n), optional
            b: Equality RHS (n_eq,), optional
            G: Inequality constraints (n_ineq, n), optional
            h: Inequality RHS (n_ineq,), optional
            lb: Variable lower bounds (n,), optional
            ub: Variable upper bounds (n,), optional
        
        Returns:
            Optimal solution x* with shape (n,)
        
        Raises:
            ValueError: If dimensions are inconsistent
        """
        self._validate(P, q, A, b, G, h, lb, ub)
        
        # Defaults
        n = self.n_vars
        device, dtype = q.device, q.dtype
        
        if lb is None:
            lb = torch.full((n,), -1e20, device=device, dtype=dtype)
        if ub is None:
            ub = torch.full((n,), 1e20, device=device, dtype=dtype)
        
        # Solve
        if self.differentiable:
            return QPFunction.apply(
                P, q, A, b, G, h, lb, ub,
                self.max_iters, self.eps, self.verbose
            )
        else:
            with torch.no_grad():
                return QPFunction.apply(
                    P, q, A, b, G, h, lb, ub,
                    self.max_iters, self.eps, self.verbose
                )
    
    def _validate(
        self,
        P: Tensor,
        q: Tensor,
        A: Optional[Tensor],
        b: Optional[Tensor],
        G: Optional[Tensor],
        h: Optional[Tensor],
        lb: Optional[Tensor],
        ub: Optional[Tensor],
    ) -> None:
        """Validate input dimensions."""
        n = self.n_vars
        
        if P.shape[-2:] != (n, n):
            raise ValueError(f"P must have shape (..., {n}, {n}), got {P.shape}")
        if q.shape[-1] != n:
            raise ValueError(f"q must have shape (..., {n}), got {q.shape}")
        
        if self.n_eq > 0:
            if A is None or b is None:
                raise ValueError(f"A and b required for {self.n_eq} equality constraints")
            if A.shape[-2:] != (self.n_eq, n):
                raise ValueError(f"A must have shape (..., {self.n_eq}, {n}), got {A.shape}")
        
        if self.n_ineq > 0:
            if G is None or h is None:
                raise ValueError(f"G and h required for {self.n_ineq} inequality constraints")
            if G.shape[-2:] != (self.n_ineq, n):
                raise ValueError(f"G must have shape (..., {self.n_ineq}, {n}), got {G.shape}")
    
    def extra_repr(self) -> str:
        """String representation."""
        return f"n_vars={self.n_vars}, n_eq={self.n_eq}, n_ineq={self.n_ineq}, eps={self.eps}"


class LPLayer(nn.Module if HAS_TORCH else object):
    """
    Differentiable Linear Programming Layer.
    
    Solves:
        minimize    c' x
        subject to  A x = b           (equality)
                    G x <= h          (inequality)
                    lb <= x <= ub     (bounds)
    
    Note:
        LP gradients are less reliable than QP gradients due to
        the non-smooth nature of LP solutions. Consider using
        a small quadratic regularization (QPLayer with small P)
        for more stable gradients.
    
    Args:
        n_vars: Number of decision variables
        n_eq: Number of equality constraints
        n_ineq: Number of inequality constraints
        max_iters: Maximum solver iterations
        eps: Convergence tolerance
        verbose: Print solver progress
        differentiable: Enable gradients
    
    Example:
        >>> layer = LPLayer(n_vars=2, n_eq=1)
        >>> c = torch.tensor([1., 1.])
        >>> A = torch.tensor([[1., 1.]])
        >>> b = torch.tensor([2.])
        >>> x = layer(c, A=A, b=b, lb=torch.zeros(2))
    """
    
    def __init__(
        self,
        n_vars: int,
        n_eq: int = 0,
        n_ineq: int = 0,
        max_iters: int = 50000,
        eps: float = 1e-5,
        verbose: bool = False,
        differentiable: bool = True,
    ) -> None:
        check_torch_available()
        super().__init__()
        
        if n_vars <= 0:
            raise ValueError(f"n_vars must be positive, got {n_vars}")
        
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.max_iters = max_iters
        self.eps = eps
        self.verbose = verbose
        self.differentiable = differentiable
    
    def forward(
        self,
        c: Tensor,
        A: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        G: Optional[Tensor] = None,
        h: Optional[Tensor] = None,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Solve LP and return optimal solution.
        
        Args:
            c: Cost vector (n,)
            A: Equality constraints (n_eq, n), optional
            b: Equality RHS (n_eq,), optional
            G: Inequality constraints (n_ineq, n), optional
            h: Inequality RHS (n_ineq,), optional
            lb: Variable lower bounds (n,), optional
            ub: Variable upper bounds (n,), optional
        
        Returns:
            Optimal solution x* with shape (n,)
        """
        n = self.n_vars
        device, dtype = c.device, c.dtype
        
        if lb is None:
            lb = torch.zeros(n, device=device, dtype=dtype)
        if ub is None:
            ub = torch.full((n,), 1e20, device=device, dtype=dtype)
        
        if self.differentiable:
            return LPFunction.apply(
                c, A, b, G, h, lb, ub,
                self.max_iters, self.eps, self.verbose
            )
        else:
            with torch.no_grad():
                return LPFunction.apply(
                    c, A, b, G, h, lb, ub,
                    self.max_iters, self.eps, self.verbose
                )
    
    def extra_repr(self) -> str:
        """String representation."""
        return f"n_vars={self.n_vars}, n_eq={self.n_eq}, n_ineq={self.n_ineq}"
