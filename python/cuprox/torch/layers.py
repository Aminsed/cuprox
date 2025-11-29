"""
Differentiable Optimization Layers for PyTorch
===============================================

This module provides nn.Module wrappers for optimization problems,
enabling end-to-end differentiable optimization in neural networks.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Union
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = Any
    nn = None

from .functions import QPFunction, LPFunction


def _check_torch():
    """Raise ImportError if torch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for cuprox.torch. "
            "Install with: pip install torch"
        )


class OptLayer(nn.Module if HAS_TORCH else object):
    """
    Base class for differentiable optimization layers.
    
    This class provides common functionality for QP and LP layers,
    including parameter validation, device handling, and configuration.
    
    Attributes:
        n_vars: Number of decision variables
        n_eq: Number of equality constraints  
        n_ineq: Number of inequality constraints
        max_iters: Maximum solver iterations
        eps: Convergence tolerance
        verbose: Whether to print solver progress
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
    ):
        """
        Initialize optimization layer.
        
        Args:
            n_vars: Number of decision variables
            n_eq: Number of equality constraints (Ax = b)
            n_ineq: Number of inequality constraints (Gx <= h)
            max_iters: Maximum solver iterations
            eps: Convergence tolerance
            verbose: Print solver progress
            differentiable: Enable gradient computation (True by default)
        """
        _check_torch()
        super().__init__()
        
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
        
        # Solver statistics (updated after each solve)
        self._last_solve_info: Dict[str, Any] = {}
    
    @property
    def solve_info(self) -> Dict[str, Any]:
        """Information from the last solve (iterations, time, status)."""
        return self._last_solve_info
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"n_vars={self.n_vars}, n_eq={self.n_eq}, n_ineq={self.n_ineq}, "
            f"eps={self.eps}"
        )


class QPLayer(OptLayer):
    """
    Differentiable Quadratic Programming Layer.
    
    Solves problems of the form:
    
        minimize    (1/2) x' P x + q' x
        subject to  A x = b           (equality constraints)
                    G x <= h          (inequality constraints)
                    lb <= x <= ub     (variable bounds)
    
    The layer computes gradients with respect to all problem parameters
    (P, q, A, b, G, h, lb, ub) using implicit differentiation of the
    KKT conditions.
    
    Example:
        >>> import torch
        >>> from cuprox.torch import QPLayer
        >>> 
        >>> # Unconstrained QP: min (1/2)x'Px + q'x
        >>> layer = QPLayer(n_vars=2)
        >>> 
        >>> P = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
        >>> q = torch.tensor([-2., -4.], requires_grad=True)
        >>> 
        >>> x = layer(P, q)
        >>> print(x)  # tensor([1., 2.])
        >>> 
        >>> # Backpropagate
        >>> x.sum().backward()
        >>> print(q.grad)
        
        >>> # With inequality constraints: Gx <= h
        >>> layer = QPLayer(n_vars=2, n_ineq=1)
        >>> G = torch.tensor([[1., 1.]])
        >>> h = torch.tensor([1.])
        >>> x = layer(P, q, G=G, h=h)
    
    Note:
        For best performance with batched problems, use batch dimension
        as the first dimension of all inputs.
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
    ):
        """
        Initialize QP layer.
        
        Args:
            n_vars: Number of decision variables (n)
            n_eq: Number of equality constraints (m_eq)
            n_ineq: Number of inequality constraints (m_ineq)
            max_iters: Maximum solver iterations
            eps: Convergence tolerance
            verbose: Print solver progress
            differentiable: Enable gradient computation
        """
        super().__init__(
            n_vars=n_vars,
            n_eq=n_eq,
            n_ineq=n_ineq,
            max_iters=max_iters,
            eps=eps,
            verbose=verbose,
            differentiable=differentiable,
        )
    
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
        Solve the QP and return the optimal solution.
        
        Args:
            P: Quadratic cost matrix (n, n) or (batch, n, n)
            q: Linear cost vector (n,) or (batch, n)
            A: Equality constraint matrix (m_eq, n) or (batch, m_eq, n)
            b: Equality constraint RHS (m_eq,) or (batch, m_eq)
            G: Inequality constraint matrix (m_ineq, n) or (batch, m_ineq, n)
            h: Inequality constraint RHS (m_ineq,) or (batch, m_ineq)
            lb: Variable lower bounds (n,) or (batch, n)
            ub: Variable upper bounds (n,) or (batch, n)
        
        Returns:
            Optimal solution x* with shape (n,) or (batch, n)
        
        Raises:
            ValueError: If input dimensions are inconsistent
            RuntimeError: If solver fails to converge
        """
        # Validate inputs
        self._validate_inputs(P, q, A, b, G, h, lb, ub)
        
        # Determine if batched
        is_batched = q.dim() == 2
        
        # Set defaults for optional inputs
        n = self.n_vars
        device = q.device
        dtype = q.dtype
        
        if lb is None:
            lb = torch.full((n,), -1e20, device=device, dtype=dtype)
        if ub is None:
            ub = torch.full((n,), 1e20, device=device, dtype=dtype)
        
        # Call the autograd function
        if self.differentiable:
            x = QPFunction.apply(
                P, q, A, b, G, h, lb, ub,
                self.max_iters, self.eps, self.verbose
            )
        else:
            with torch.no_grad():
                x = QPFunction.apply(
                    P, q, A, b, G, h, lb, ub,
                    self.max_iters, self.eps, self.verbose
                )
        
        return x
    
    def _validate_inputs(
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
        """Validate input dimensions and types."""
        n = self.n_vars
        
        # Check P
        if P.shape[-2:] != (n, n):
            raise ValueError(
                f"P must have shape (..., {n}, {n}), got {P.shape}"
            )
        
        # Check q
        if q.shape[-1] != n:
            raise ValueError(
                f"q must have shape (..., {n}), got {q.shape}"
            )
        
        # Check equality constraints
        if self.n_eq > 0:
            if A is None or b is None:
                raise ValueError(
                    f"A and b required for {self.n_eq} equality constraints"
                )
            if A.shape[-2:] != (self.n_eq, n):
                raise ValueError(
                    f"A must have shape (..., {self.n_eq}, {n}), got {A.shape}"
                )
            if b.shape[-1] != self.n_eq:
                raise ValueError(
                    f"b must have shape (..., {self.n_eq}), got {b.shape}"
                )
        
        # Check inequality constraints
        if self.n_ineq > 0:
            if G is None or h is None:
                raise ValueError(
                    f"G and h required for {self.n_ineq} inequality constraints"
                )
            if G.shape[-2:] != (self.n_ineq, n):
                raise ValueError(
                    f"G must have shape (..., {self.n_ineq}, {n}), got {G.shape}"
                )
            if h.shape[-1] != self.n_ineq:
                raise ValueError(
                    f"h must have shape (..., {self.n_ineq}), got {h.shape}"
                )


class LPLayer(OptLayer):
    """
    Differentiable Linear Programming Layer.
    
    Solves problems of the form:
    
        minimize    c' x
        subject to  A x = b           (equality constraints)
                    G x <= h          (inequality constraints)
                    lb <= x <= ub     (variable bounds)
    
    The layer computes gradients with respect to all problem parameters
    using implicit differentiation.
    
    Example:
        >>> import torch
        >>> from cuprox.torch import LPLayer
        >>> 
        >>> # LP: min c'x s.t. Gx <= h, x >= 0
        >>> layer = LPLayer(n_vars=2, n_ineq=2)
        >>> 
        >>> c = torch.tensor([-1., -1.], requires_grad=True)
        >>> G = torch.tensor([[1., 2.], [3., 1.]])
        >>> h = torch.tensor([10., 15.])
        >>> lb = torch.zeros(2)
        >>> 
        >>> x = layer(c, G=G, h=h, lb=lb)
        >>> x.sum().backward()
    
    Note:
        LP gradients are computed by treating the LP as a QP with P=0
        and using the same implicit differentiation framework.
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
    ):
        """
        Initialize LP layer.
        
        Args:
            n_vars: Number of decision variables (n)
            n_eq: Number of equality constraints
            n_ineq: Number of inequality constraints
            max_iters: Maximum solver iterations
            eps: Convergence tolerance
            verbose: Print solver progress
            differentiable: Enable gradient computation
        """
        super().__init__(
            n_vars=n_vars,
            n_eq=n_eq,
            n_ineq=n_ineq,
            max_iters=max_iters,
            eps=eps,
            verbose=verbose,
            differentiable=differentiable,
        )
    
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
        Solve the LP and return the optimal solution.
        
        Args:
            c: Cost vector (n,) or (batch, n)
            A: Equality constraint matrix (m_eq, n) or (batch, m_eq, n)
            b: Equality constraint RHS (m_eq,) or (batch, m_eq)
            G: Inequality constraint matrix (m_ineq, n) or (batch, m_ineq, n)
            h: Inequality constraint RHS (m_ineq,) or (batch, m_ineq)
            lb: Variable lower bounds (n,) or (batch, n)
            ub: Variable upper bounds (n,) or (batch, n)
        
        Returns:
            Optimal solution x* with shape (n,) or (batch, n)
        """
        self._validate_inputs(c, A, b, G, h, lb, ub)
        
        n = self.n_vars
        device = c.device
        dtype = c.dtype
        
        if lb is None:
            lb = torch.zeros(n, device=device, dtype=dtype)
        if ub is None:
            ub = torch.full((n,), 1e20, device=device, dtype=dtype)
        
        if self.differentiable:
            x = LPFunction.apply(
                c, A, b, G, h, lb, ub,
                self.max_iters, self.eps, self.verbose
            )
        else:
            with torch.no_grad():
                x = LPFunction.apply(
                    c, A, b, G, h, lb, ub,
                    self.max_iters, self.eps, self.verbose
                )
        
        return x
    
    def _validate_inputs(
        self,
        c: Tensor,
        A: Optional[Tensor],
        b: Optional[Tensor],
        G: Optional[Tensor],
        h: Optional[Tensor],
        lb: Optional[Tensor],
        ub: Optional[Tensor],
    ) -> None:
        """Validate input dimensions."""
        n = self.n_vars
        
        if c.shape[-1] != n:
            raise ValueError(
                f"c must have shape (..., {n}), got {c.shape}"
            )
        
        if self.n_eq > 0:
            if A is None or b is None:
                raise ValueError(
                    f"A and b required for {self.n_eq} equality constraints"
                )
        
        if self.n_ineq > 0:
            if G is None or h is None:
                raise ValueError(
                    f"G and h required for {self.n_ineq} inequality constraints"
                )

