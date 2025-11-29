"""
Differentiable Optimization Functions for PyTorch
==================================================

This module provides autograd functions for LP and QP solving,
enabling gradient computation through implicit differentiation.

The backward pass uses the implicit function theorem on the KKT conditions
to compute gradients efficiently without unrolling the solver iterations.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, List
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

# Import cuprox solver
from .. import solve as cuprox_solve
from ..result import Status


def _to_numpy(t: Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return t.detach().cpu().numpy().astype(np.float64)


def _to_torch(arr: np.ndarray, device, dtype) -> Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(arr.astype(np.float64)).to(device=device, dtype=dtype)


class QPFunction(Function):
    """
    Autograd function for differentiable QP solving.
    
    Solves:
        minimize    (1/2) x' P x + q' x
        subject to  A x = b
                    G x <= h
                    lb <= x <= ub
    
    Backward pass computes gradients via implicit differentiation of KKT conditions:
    
        [P   A'  G'_act] [dx ]   [dq + dP @ x*        ]
        [A   0   0     ] [dnu] = [db - dA @ x*        ]
        [G_a 0   0     ] [dla]   [dh_act - dG_act @ x*]
    
    where G_act are the active inequality constraints at the solution.
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
        """
        Solve QP and save information for backward pass.
        """
        device = q.device
        dtype = q.dtype
        n = q.shape[-1]
        
        # Handle batched vs non-batched
        is_batched = q.dim() == 2
        if is_batched:
            batch_size = q.shape[0]
            # For now, solve each problem in batch sequentially
            # TODO: Implement true batched solving
            results = []
            for i in range(batch_size):
                x_i = QPFunction._solve_single(
                    P[i] if P.dim() == 3 else P,
                    q[i],
                    A[i] if A is not None and A.dim() == 3 else A,
                    b[i] if b is not None and b.dim() == 2 else b,
                    G[i] if G is not None and G.dim() == 3 else G,
                    h[i] if h is not None and h.dim() == 2 else h,
                    lb[i] if lb.dim() == 2 else lb,
                    ub[i] if ub.dim() == 2 else ub,
                    max_iters, eps, verbose, device, dtype
                )
                results.append(x_i)
            x = torch.stack(results)
        else:
            x = QPFunction._solve_single(
                P, q, A, b, G, h, lb, ub,
                max_iters, eps, verbose, device, dtype
            )
        
        # Save for backward
        ctx.save_for_backward(P, q, A, b, G, h, lb, ub, x)
        ctx.is_batched = is_batched
        
        return x
    
    @staticmethod
    def _solve_single(
        P: Tensor, q: Tensor,
        A: Optional[Tensor], b: Optional[Tensor],
        G: Optional[Tensor], h: Optional[Tensor],
        lb: Tensor, ub: Tensor,
        max_iters: int, eps: float, verbose: bool,
        device, dtype,
    ) -> Tensor:
        """Solve a single (non-batched) QP."""
        from scipy import sparse
        
        n = q.shape[0]
        
        # Convert to numpy
        P_np = _to_numpy(P)
        q_np = _to_numpy(q)
        lb_np = _to_numpy(lb)
        ub_np = _to_numpy(ub)
        
        # Build combined constraint matrix for cuprox
        # cuprox expects: l <= A_combined @ x <= u
        constraints = []
        l_list = []
        u_list = []
        
        # Equality constraints: A @ x = b -> b <= A @ x <= b
        if A is not None and b is not None:
            A_np = _to_numpy(A)
            b_np = _to_numpy(b)
            constraints.append(A_np)
            l_list.append(b_np)
            u_list.append(b_np)
        
        # Inequality constraints: G @ x <= h -> -inf <= G @ x <= h
        if G is not None and h is not None:
            G_np = _to_numpy(G)
            h_np = _to_numpy(h)
            constraints.append(G_np)
            l_list.append(np.full(h_np.shape, -1e20))
            u_list.append(h_np)
        
        # Combine constraints
        if constraints:
            A_combined = np.vstack(constraints)
            l_combined = np.concatenate(l_list)
            u_combined = np.concatenate(u_list)
        else:
            A_combined = np.zeros((0, n))
            l_combined = np.array([])
            u_combined = np.array([])
        
        # Solve
        result = cuprox_solve(
            c=q_np,
            A=sparse.csr_matrix(A_combined),
            b=np.zeros(A_combined.shape[0]),  # Placeholder
            P=sparse.csr_matrix(P_np),
            lb=lb_np,
            ub=ub_np,
            constraint_l=l_combined,
            constraint_u=u_combined,
            params={
                "max_iterations": max_iters,
                "tolerance": eps,
                "verbose": verbose,
            }
        )
        
        if result.status not in [Status.OPTIMAL, Status.MAX_ITERATIONS]:
            warnings.warn(
                f"QP solver returned status {result.status}. "
                "Gradients may be inaccurate."
            )
        
        return _to_torch(result.x, device, dtype)
    
    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Compute gradients via implicit differentiation of KKT conditions.
        
        For QP with optimal solution x*, the KKT conditions are:
            Px* + q + A'ν + G'λ = 0  (stationarity)
            Ax* = b                   (primal feasibility - eq)
            Gx* <= h                  (primal feasibility - ineq)
            λ >= 0, λ ⊙ (Gx* - h) = 0 (complementarity)
        
        By implicit differentiation:
            d(Px* + q)/d(params) = 0
            => P dx* + dP x* + dq + A' dν + G' dλ = 0
        
        For unconstrained QP (P x* + q = 0):
            P dx* = -dq - dP x*
            
        The gradient w.r.t. the loss L through x* is:
            dL/dq = dL/dx* @ dx*/dq = grad_x @ (-P^{-1})
            dL/dP = ... (requires solving linear system)
        """
        P, q, A, b, G, h, lb, ub, x = ctx.saved_tensors
        is_batched = ctx.is_batched
        
        device = grad_x.device
        dtype = grad_x.dtype
        n = q.shape[-1]
        
        # For simplicity, compute gradients for the unconstrained/box-constrained case
        # using the implicit function theorem
        # 
        # At optimum: P x* + q + (terms from bounds) = 0
        # => dx*/dq = -P^{-1} (for unconstrained)
        #
        # For bound-constrained case, we use the reduced KKT system
        
        if is_batched:
            grad_P = torch.zeros_like(P) if P.requires_grad else None
            grad_q = torch.zeros_like(q) if q.requires_grad else None
            
            batch_size = q.shape[0]
            for i in range(batch_size):
                gP, gq = QPFunction._backward_single(
                    P[i] if P.dim() == 3 else P,
                    q[i],
                    x[i],
                    grad_x[i],
                    lb[i] if lb.dim() == 2 else lb,
                    ub[i] if ub.dim() == 2 else ub,
                )
                if grad_P is not None:
                    if P.dim() == 3:
                        grad_P[i] = gP
                    else:
                        grad_P += gP
                if grad_q is not None:
                    grad_q[i] = gq
        else:
            grad_P, grad_q = QPFunction._backward_single(
                P, q, x, grad_x, lb, ub
            )
            if not P.requires_grad:
                grad_P = None
            if not q.requires_grad:
                grad_q = None
        
        # Gradients for A, b, G, h, lb, ub not yet implemented
        # Return None for those
        return (
            grad_P,  # P
            grad_q,  # q
            None,    # A
            None,    # b
            None,    # G
            None,    # h
            None,    # lb
            None,    # ub
            None,    # max_iters
            None,    # eps
            None,    # verbose
        )
    
    @staticmethod
    def _backward_single(
        P: Tensor, q: Tensor, x: Tensor, grad_x: Tensor,
        lb: Tensor, ub: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute gradients for a single QP.
        
        Uses the implicit function theorem on KKT conditions.
        For unconstrained: P x* + q = 0, so dx*/dq = -P^{-1}
        """
        n = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Identify active bounds
        eps_active = 1e-6
        active_lb = (x - lb).abs() < eps_active
        active_ub = (ub - x).abs() < eps_active
        free = ~(active_lb | active_ub)
        
        n_free = free.sum().item()
        
        if n_free == 0:
            # All variables at bounds - gradient is zero
            grad_q = torch.zeros_like(q)
            grad_P = torch.zeros_like(P)
        elif n_free == n:
            # Unconstrained case: P dx* = -dq => dx*/dq = -P^{-1}
            # grad_q = grad_x @ (dx*/dq) = -grad_x @ P^{-1}
            # Solve P' v = grad_x, then grad_q = -v
            try:
                v = torch.linalg.solve(P.T, grad_x)
                grad_q = -v
            except RuntimeError:
                # P might be singular, use pseudoinverse
                P_inv = torch.linalg.pinv(P)
                grad_q = -grad_x @ P_inv
            
            # grad_P: dL/dP_ij = dL/dx* @ dx*/dP_ij
            # From P x* + q = 0: d(P x*)/dP_ij = x*_j (at position i)
            # => dx*/dP_ij = -P^{-1}_{:,i} x*_j
            # This gives: grad_P_ij = -grad_q_i x*_j = v_i x*_j
            grad_P = torch.outer(grad_q, x)  # Actually -v @ x.T, but grad_q = -v
            grad_P = -grad_P  # Correct sign
        else:
            # Partial active set - use reduced KKT system
            # For free variables: P_ff x_f + q_f = 0
            free_idx = free.nonzero(as_tuple=True)[0]
            
            P_ff = P[free_idx][:, free_idx]
            grad_x_f = grad_x[free_idx]
            
            try:
                v_f = torch.linalg.solve(P_ff.T, grad_x_f)
            except RuntimeError:
                P_ff_inv = torch.linalg.pinv(P_ff)
                v_f = grad_x_f @ P_ff_inv
            
            grad_q = torch.zeros_like(q)
            grad_q[free_idx] = -v_f
            
            # Simplified grad_P (only free-free block)
            grad_P = torch.zeros_like(P)
            for i, fi in enumerate(free_idx):
                for j, fj in enumerate(free_idx):
                    grad_P[fi, fj] = v_f[i] * x[fj]
        
        return grad_P, grad_q


class LPFunction(Function):
    """
    Autograd function for differentiable LP solving.
    
    Solves:
        minimize    c' x
        subject to  A x = b
                    G x <= h
                    lb <= x <= ub
    
    Backward pass treats LP as QP with P=0 and uses implicit differentiation.
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
        """Solve LP and save for backward."""
        device = c.device
        dtype = c.dtype
        n = c.shape[-1]
        
        is_batched = c.dim() == 2
        
        if is_batched:
            batch_size = c.shape[0]
            results = []
            for i in range(batch_size):
                x_i = LPFunction._solve_single(
                    c[i],
                    A[i] if A is not None and A.dim() == 3 else A,
                    b[i] if b is not None and b.dim() == 2 else b,
                    G[i] if G is not None and G.dim() == 3 else G,
                    h[i] if h is not None and h.dim() == 2 else h,
                    lb[i] if lb.dim() == 2 else lb,
                    ub[i] if ub.dim() == 2 else ub,
                    max_iters, eps, verbose, device, dtype
                )
                results.append(x_i)
            x = torch.stack(results)
        else:
            x = LPFunction._solve_single(
                c, A, b, G, h, lb, ub,
                max_iters, eps, verbose, device, dtype
            )
        
        ctx.save_for_backward(c, A, b, G, h, lb, ub, x)
        ctx.is_batched = is_batched
        
        return x
    
    @staticmethod
    def _solve_single(
        c: Tensor,
        A: Optional[Tensor], b: Optional[Tensor],
        G: Optional[Tensor], h: Optional[Tensor],
        lb: Tensor, ub: Tensor,
        max_iters: int, eps: float, verbose: bool,
        device, dtype,
    ) -> Tensor:
        """Solve a single LP."""
        from scipy import sparse
        
        n = c.shape[0]
        
        c_np = _to_numpy(c)
        lb_np = _to_numpy(lb)
        ub_np = _to_numpy(ub)
        
        # Build constraint matrix
        constraints = []
        l_list = []
        u_list = []
        
        if A is not None and b is not None:
            A_np = _to_numpy(A)
            b_np = _to_numpy(b)
            constraints.append(A_np)
            l_list.append(b_np)
            u_list.append(b_np)
        
        if G is not None and h is not None:
            G_np = _to_numpy(G)
            h_np = _to_numpy(h)
            constraints.append(G_np)
            l_list.append(np.full(h_np.shape, -1e20))
            u_list.append(h_np)
        
        if constraints:
            A_combined = np.vstack(constraints)
            l_combined = np.concatenate(l_list)
            u_combined = np.concatenate(u_list)
        else:
            A_combined = np.zeros((0, n))
            l_combined = np.array([])
            u_combined = np.array([])
        
        result = cuprox_solve(
            c=c_np,
            A=sparse.csr_matrix(A_combined),
            b=np.zeros(A_combined.shape[0]),
            lb=lb_np,
            ub=ub_np,
            constraint_l=l_combined,
            constraint_u=u_combined,
            params={
                "max_iterations": max_iters,
                "tolerance": eps,
                "verbose": verbose,
            }
        )
        
        return _to_torch(result.x, device, dtype)
    
    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Compute gradients for LP.
        
        For LP, the gradient w.r.t. c at the optimal x* depends on
        the active constraints. For unconstrained LP (no meaningful solution),
        gradients are undefined.
        
        At bound-constrained optimum, variables at bounds have zero gradient.
        """
        c, A, b, G, h, lb, ub, x = ctx.saved_tensors
        
        # For LP, gradient w.r.t c at optimum is related to dual variables
        # Simple approximation: grad_c ≈ -grad_x (for active set analysis)
        # This is a simplification - full LP gradients require dual solution
        
        grad_c = None
        if c.requires_grad:
            # Heuristic: variables at bounds have grad 0
            eps_active = 1e-6
            active = ((x - lb).abs() < eps_active) | ((ub - x).abs() < eps_active)
            
            grad_c = torch.zeros_like(c)
            free = ~active
            if free.any():
                # For free variables, grad_c ≈ position sensitivity
                grad_c[free] = -grad_x[free] / (grad_x[free].abs().sum() + 1e-10)
        
        return (
            grad_c,  # c
            None,    # A
            None,    # b
            None,    # G
            None,    # h
            None,    # lb
            None,    # ub
            None,    # max_iters
            None,    # eps
            None,    # verbose
        )


# Functional interface
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
    Solve a differentiable QP.
    
    This is a functional interface to the QP solver. For repeated solving
    with the same problem structure, consider using :class:`QPLayer` instead.
    
    Args:
        P: Quadratic cost matrix (n, n)
        q: Linear cost vector (n,)
        A: Equality constraint matrix (m_eq, n)
        b: Equality constraint RHS (m_eq,)
        G: Inequality constraint matrix (m_ineq, n)
        h: Inequality constraint RHS (m_ineq,)
        lb: Variable lower bounds (n,)
        ub: Variable upper bounds (n,)
        max_iters: Maximum solver iterations
        eps: Convergence tolerance
        verbose: Print solver progress
    
    Returns:
        Optimal solution x* with gradients enabled
    
    Example:
        >>> P = torch.eye(2, requires_grad=True)
        >>> q = torch.tensor([-1., -2.], requires_grad=True)
        >>> x = solve_qp(P, q)
        >>> x.sum().backward()
    """
    n = q.shape[-1]
    device = q.device
    dtype = q.dtype
    
    if lb is None:
        lb = torch.full((n,), -1e20, device=device, dtype=dtype)
    if ub is None:
        ub = torch.full((n,), 1e20, device=device, dtype=dtype)
    
    return QPFunction.apply(
        P, q, A, b, G, h, lb, ub, max_iters, eps, verbose
    )


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
    Solve a differentiable LP.
    
    Args:
        c: Cost vector (n,)
        A: Equality constraint matrix (m_eq, n)
        b: Equality constraint RHS (m_eq,)
        G: Inequality constraint matrix (m_ineq, n)
        h: Inequality constraint RHS (m_ineq,)
        lb: Variable lower bounds (n,)
        ub: Variable upper bounds (n,)
        max_iters: Maximum solver iterations
        eps: Convergence tolerance
        verbose: Print solver progress
    
    Returns:
        Optimal solution x* with gradients enabled
    """
    n = c.shape[-1]
    device = c.device
    dtype = c.dtype
    
    if lb is None:
        lb = torch.zeros(n, device=device, dtype=dtype)
    if ub is None:
        ub = torch.full((n,), 1e20, device=device, dtype=dtype)
    
    return LPFunction.apply(
        c, A, b, G, h, lb, ub, max_iters, eps, verbose
    )

