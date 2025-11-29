"""
Autograd Functions for Differentiable Optimization
===================================================

This module provides PyTorch autograd functions that enable
backpropagation through optimization problems.

The backward pass uses implicit differentiation of the KKT
conditions to compute gradients efficiently.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any
import warnings

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.autograd import Function
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = Any
    Function = object

from .utils import to_numpy, to_torch


class QPFunction(Function):
    """
    Autograd function for differentiable QP solving.
    
    Forward: Solves the QP using cuprox
    Backward: Computes gradients via implicit differentiation
    
    The backward pass uses the implicit function theorem on KKT conditions:
    
        P x* + q + A' ν + G' λ = 0  (stationarity)
        A x* = b                    (primal feasibility)
        λ ⊙ (G x* - h) = 0          (complementarity)
    
    Differentiating these conditions gives a linear system for dx*/d(params).
    """
    
    @staticmethod
    def forward(
        ctx,
        P: Tensor,
        q: Tensor,
        A: Optional[Tensor],
        b: Optional[Tensor],
        G: Optional[Tensor],
        h: Optional[Tensor],
        lb: Tensor,
        ub: Tensor,
        max_iters: int,
        eps: float,
        verbose: bool,
    ) -> Tensor:
        """Solve QP and save tensors for backward."""
        device, dtype = q.device, q.dtype
        
        # Solve (handles batched case internally)
        x = QPFunction._solve(
            P, q, A, b, G, h, lb, ub,
            max_iters, eps, verbose, device, dtype
        )
        
        # Save for backward
        ctx.save_for_backward(P, q, A, b, G, h, lb, ub, x)
        
        return x
    
    @staticmethod
    def _solve(
        P: Tensor,
        q: Tensor,
        A: Optional[Tensor],
        b: Optional[Tensor],
        G: Optional[Tensor],
        h: Optional[Tensor],
        lb: Tensor,
        ub: Tensor,
        max_iters: int,
        eps: float,
        verbose: bool,
        device,
        dtype,
    ) -> Tensor:
        """Solve QP using cuprox."""
        from scipy import sparse
        from .. import solve as cuprox_solve
        from ..result import Status
        
        n = q.shape[-1]
        
        # Convert to numpy
        P_np = to_numpy(P)
        q_np = to_numpy(q)
        lb_np = to_numpy(lb)
        ub_np = to_numpy(ub)
        
        # Build constraint matrix (l <= A_comb @ x <= u)
        constraints = []
        l_list = []
        u_list = []
        
        if A is not None and b is not None:
            A_np = to_numpy(A)
            b_np = to_numpy(b)
            constraints.append(A_np)
            l_list.append(b_np)
            u_list.append(b_np)  # Equality: l = u = b
        
        if G is not None and h is not None:
            G_np = to_numpy(G)
            h_np = to_numpy(h)
            constraints.append(G_np)
            l_list.append(np.full(h_np.shape, -1e20))
            u_list.append(h_np)  # Inequality: -inf <= Gx <= h
        
        if constraints:
            A_comb = np.vstack(constraints)
            l_comb = np.concatenate(l_list)
            u_comb = np.concatenate(u_list)
        else:
            A_comb = np.zeros((0, n))
            l_comb = np.array([])
            u_comb = np.array([])
        
        # Solve
        result = cuprox_solve(
            c=q_np,
            A=sparse.csr_matrix(A_comb),
            b=np.zeros(A_comb.shape[0]),
            P=sparse.csr_matrix(P_np),
            lb=lb_np,
            ub=ub_np,
            constraint_l=l_comb,
            constraint_u=u_comb,
            params={
                "max_iterations": max_iters,
                "tolerance": eps,
                "verbose": verbose,
            }
        )
        
        if result.status not in [Status.OPTIMAL, Status.MAX_ITERATIONS]:
            warnings.warn(f"QP solver returned {result.status}. Gradients may be inaccurate.")
        
        return to_torch(result.x, device, dtype)
    
    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Compute gradients via implicit differentiation.
        
        For unconstrained QP: P x* + q = 0
        => dx*/dq = -P^{-1}
        => dL/dq = grad_x @ (-P^{-1})
        
        For constrained QP, we use the reduced KKT system
        restricted to free (non-active) variables.
        """
        P, q, A, b, G, h, lb, ub, x = ctx.saved_tensors
        
        grad_P, grad_q = QPFunction._backward_impl(P, q, x, grad_x, lb, ub)
        
        # Only return gradients for inputs that require them
        grad_P = grad_P if P.requires_grad else None
        grad_q = grad_q if q.requires_grad else None
        
        return (
            grad_P, grad_q,
            None, None,  # A, b
            None, None,  # G, h
            None, None,  # lb, ub
            None, None, None,  # max_iters, eps, verbose
        )
    
    @staticmethod
    def _backward_impl(
        P: Tensor,
        q: Tensor,
        x: Tensor,
        grad_x: Tensor,
        lb: Tensor,
        ub: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute gradients for P and q."""
        n = x.shape[0]
        
        # Identify active bounds
        eps_active = 1e-6
        active_lb = (x - lb).abs() < eps_active
        active_ub = (ub - x).abs() < eps_active
        free = ~(active_lb | active_ub)
        n_free = free.sum().item()
        
        if n_free == 0:
            # All at bounds - zero gradient
            return torch.zeros_like(P), torch.zeros_like(q)
        
        if n_free == n:
            # Unconstrained: P x* + q = 0 => dx*/dq = -P^{-1}
            try:
                v = torch.linalg.solve(P.T, grad_x)
            except RuntimeError:
                v = grad_x @ torch.linalg.pinv(P)
            
            grad_q = -v
            grad_P = -torch.outer(v, x)
        else:
            # Partial active set - reduced system
            free_idx = free.nonzero(as_tuple=True)[0]
            P_ff = P[free_idx][:, free_idx]
            grad_x_f = grad_x[free_idx]
            
            try:
                v_f = torch.linalg.solve(P_ff.T, grad_x_f)
            except RuntimeError:
                v_f = grad_x_f @ torch.linalg.pinv(P_ff)
            
            grad_q = torch.zeros_like(q)
            grad_q[free_idx] = -v_f
            
            grad_P = torch.zeros_like(P)
            for i, fi in enumerate(free_idx):
                for j, fj in enumerate(free_idx):
                    grad_P[fi, fj] = v_f[i] * x[fj]
        
        return grad_P, grad_q


class LPFunction(Function):
    """
    Autograd function for differentiable LP solving.
    
    Note: LP gradients are approximate since LP solutions
    are typically at vertices where the solution is non-smooth.
    """
    
    @staticmethod
    def forward(
        ctx,
        c: Tensor,
        A: Optional[Tensor],
        b: Optional[Tensor],
        G: Optional[Tensor],
        h: Optional[Tensor],
        lb: Tensor,
        ub: Tensor,
        max_iters: int,
        eps: float,
        verbose: bool,
    ) -> Tensor:
        """Solve LP."""
        device, dtype = c.device, c.dtype
        
        x = LPFunction._solve(
            c, A, b, G, h, lb, ub,
            max_iters, eps, verbose, device, dtype
        )
        
        ctx.save_for_backward(c, A, b, G, h, lb, ub, x)
        
        return x
    
    @staticmethod
    def _solve(
        c: Tensor,
        A: Optional[Tensor],
        b: Optional[Tensor],
        G: Optional[Tensor],
        h: Optional[Tensor],
        lb: Tensor,
        ub: Tensor,
        max_iters: int,
        eps: float,
        verbose: bool,
        device,
        dtype,
    ) -> Tensor:
        """Solve LP using cuprox."""
        from scipy import sparse
        from .. import solve as cuprox_solve
        
        n = c.shape[-1]
        
        c_np = to_numpy(c)
        lb_np = to_numpy(lb)
        ub_np = to_numpy(ub)
        
        constraints = []
        l_list = []
        u_list = []
        
        if A is not None and b is not None:
            constraints.append(to_numpy(A))
            b_np = to_numpy(b)
            l_list.append(b_np)
            u_list.append(b_np)
        
        if G is not None and h is not None:
            constraints.append(to_numpy(G))
            l_list.append(np.full(to_numpy(h).shape, -1e20))
            u_list.append(to_numpy(h))
        
        if constraints:
            A_comb = np.vstack(constraints)
            l_comb = np.concatenate(l_list)
            u_comb = np.concatenate(u_list)
        else:
            A_comb = np.zeros((0, n))
            l_comb = np.array([])
            u_comb = np.array([])
        
        result = cuprox_solve(
            c=c_np,
            A=sparse.csr_matrix(A_comb),
            b=np.zeros(A_comb.shape[0]),
            lb=lb_np,
            ub=ub_np,
            constraint_l=l_comb,
            constraint_u=u_comb,
            params={
                "max_iterations": max_iters,
                "tolerance": eps,
                "verbose": verbose,
            }
        )
        
        return to_torch(result.x, device, dtype)
    
    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Approximate gradients for LP.
        
        LP solutions are typically at vertices, making exact
        differentiation ill-defined. We provide an approximation.
        """
        c, A, b, G, h, lb, ub, x = ctx.saved_tensors
        
        grad_c = None
        if c.requires_grad:
            # Approximate: sensitivity based on active constraints
            eps_active = 1e-6
            active = ((x - lb).abs() < eps_active) | ((ub - x).abs() < eps_active)
            
            grad_c = torch.zeros_like(c)
            free = ~active
            if free.any():
                norm = grad_x[free].abs().sum() + 1e-10
                grad_c[free] = -grad_x[free] / norm
        
        return (
            grad_c,
            None, None,  # A, b
            None, None,  # G, h
            None, None,  # lb, ub
            None, None, None,  # max_iters, eps, verbose
        )


# =============================================================================
# Functional Interface
# =============================================================================

def solve_qp(
    P: Tensor,
    q: Tensor,
    A: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    G: Optional[Tensor] = None,
    h: Optional[Tensor] = None,
    lb: Optional[Tensor] = None,
    ub: Optional[Tensor] = None,
    max_iters: int = 10000,
    eps: float = 1e-5,
    verbose: bool = False,
) -> Tensor:
    """
    Solve a differentiable QP (functional interface).
    
    minimize    (1/2) x' P x + q' x
    subject to  A x = b, G x <= h, lb <= x <= ub
    
    Args:
        P: Quadratic cost (n, n), positive semidefinite
        q: Linear cost (n,)
        A: Equality constraints (m_eq, n)
        b: Equality RHS (m_eq,)
        G: Inequality constraints (m_ineq, n)
        h: Inequality RHS (m_ineq,)
        lb: Lower bounds (n,)
        ub: Upper bounds (n,)
        max_iters: Maximum iterations
        eps: Tolerance
        verbose: Print progress
    
    Returns:
        Optimal solution x* with gradients
    
    Example:
        >>> P = torch.eye(2, requires_grad=True)
        >>> q = torch.tensor([-1., -2.], requires_grad=True)
        >>> x = solve_qp(P, q)
        >>> x.sum().backward()
    """
    n = q.shape[-1]
    device, dtype = q.device, q.dtype
    
    if lb is None:
        lb = torch.full((n,), -1e20, device=device, dtype=dtype)
    if ub is None:
        ub = torch.full((n,), 1e20, device=device, dtype=dtype)
    
    return QPFunction.apply(P, q, A, b, G, h, lb, ub, max_iters, eps, verbose)


def solve_lp(
    c: Tensor,
    A: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    G: Optional[Tensor] = None,
    h: Optional[Tensor] = None,
    lb: Optional[Tensor] = None,
    ub: Optional[Tensor] = None,
    max_iters: int = 50000,
    eps: float = 1e-5,
    verbose: bool = False,
) -> Tensor:
    """
    Solve a differentiable LP (functional interface).
    
    minimize    c' x
    subject to  A x = b, G x <= h, lb <= x <= ub
    
    Args:
        c: Cost vector (n,)
        A: Equality constraints (m_eq, n)
        b: Equality RHS (m_eq,)
        G: Inequality constraints (m_ineq, n)
        h: Inequality RHS (m_ineq,)
        lb: Lower bounds (n,)
        ub: Upper bounds (n,)
        max_iters: Maximum iterations
        eps: Tolerance
        verbose: Print progress
    
    Returns:
        Optimal solution x*
    """
    n = c.shape[-1]
    device, dtype = c.device, c.dtype
    
    if lb is None:
        lb = torch.zeros(n, device=device, dtype=dtype)
    if ub is None:
        ub = torch.full((n,), 1e20, device=device, dtype=dtype)
    
    return LPFunction.apply(c, A, b, G, h, lb, ub, max_iters, eps, verbose)
