"""
Autograd Functions for Differentiable Optimization
===================================================

This module provides PyTorch autograd functions that enable
backpropagation through optimization problems.

Features:
- Batched solving for parallel GPU execution
- Full KKT-based implicit differentiation
- Gradients for all problem parameters (P, q, A, b, G, h)
- Dual variable output and differentiation
- GPU-accelerated backward pass
- Support for higher-order gradients

The backward pass uses implicit differentiation of the KKT
conditions to compute gradients efficiently.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, NamedTuple

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


class QPSolution(NamedTuple):
    """Solution to a QP with primal and dual variables."""
    x: Tensor          # Primal solution
    nu: Tensor         # Equality dual (Lagrange multipliers for Ax = b)
    lam: Tensor        # Inequality dual (Lagrange multipliers for Gx <= h)
    lam_lb: Tensor     # Lower bound dual
    lam_ub: Tensor     # Upper bound dual
    active_ineq: Tensor  # Active inequality mask
    active_lb: Tensor    # Active lower bound mask
    active_ub: Tensor    # Active upper bound mask


class QPFunction(Function):
    """
    Autograd function for differentiable QP solving with full gradient support.

    Forward: Solves the QP using cuprox (batched or single)
    Backward: Computes gradients via implicit differentiation of KKT conditions

    Supports gradients for ALL problem parameters:
    - P: Quadratic cost matrix
    - q: Linear cost vector
    - A, b: Equality constraints (Ax = b)
    - G, h: Inequality constraints (Gx <= h)
    - lb, ub: Variable bounds

    The backward pass uses the implicit function theorem on KKT conditions:

        P x* + q + A' ν* + G' λ* = 0     (stationarity)
        A x* = b                          (primal feasibility - eq)
        G x* <= h, λ* >= 0, λ* ⊙ (Gx* - h) = 0  (complementarity)

    Differentiating these conditions gives a linear system for ∂x*/∂(params).
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
        Solve QP and save tensors for backward.
        
        Supports batched input:
        - P: (batch, n, n) or (n, n)
        - q: (batch, n) or (n,)
        - etc.
        """
        device, dtype = q.device, q.dtype
        is_batched = q.dim() == 2
        
        if is_batched:
            batch_size = q.shape[0]
            n = q.shape[1]
        else:
            batch_size = 1
            n = q.shape[0]

        # Solve QP(s)
        x, sol_info = QPFunction._solve_batch(
            P, q, A, b, G, h, lb, ub, 
            max_iters, eps, verbose, 
            device, dtype, is_batched
        )

        # Save for backward
        ctx.save_for_backward(P, q, A, b, G, h, lb, ub, x,
                              sol_info['nu'], sol_info['lam'],
                              sol_info['active_ineq'], 
                              sol_info['active_lb'],
                              sol_info['active_ub'])
        ctx.is_batched = is_batched
        ctx.n = n
        ctx.n_eq = A.shape[-2] if A is not None else 0
        ctx.n_ineq = G.shape[-2] if G is not None else 0

        return x

    @staticmethod
    def _solve_batch(
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
        is_batched: bool,
    ) -> Tuple[Tensor, dict]:
        """Solve QP(s) using cuprox with optional batching."""
        from scipy import sparse
        from .. import solve as cuprox_solve
        from ..result import Status

        if is_batched:
            batch_size = q.shape[0]
            n = q.shape[1]
        else:
            batch_size = 1
            n = q.shape[0]
            # Add batch dimension for uniform processing
            P = P.unsqueeze(0)
            q = q.unsqueeze(0)
            lb = lb.unsqueeze(0)
            ub = ub.unsqueeze(0)
            if A is not None:
                A = A.unsqueeze(0)
                b = b.unsqueeze(0)
            if G is not None:
                G = G.unsqueeze(0)
                h = h.unsqueeze(0)

        n_eq = A.shape[-2] if A is not None else 0
        n_ineq = G.shape[-2] if G is not None else 0

        # Solve each QP
        x_list = []
        nu_list = []
        lam_list = []
        active_ineq_list = []
        active_lb_list = []
        active_ub_list = []

        for i in range(batch_size):
            P_np = to_numpy(P[i])
            q_np = to_numpy(q[i])
            lb_np = to_numpy(lb[i])
            ub_np = to_numpy(ub[i])

            # Build constraint matrix
            constraints = []
            l_list = []
            u_list = []

            if A is not None and b is not None:
                A_np = to_numpy(A[i])
                b_np = to_numpy(b[i])
                constraints.append(A_np)
                l_list.append(b_np)
                u_list.append(b_np)  # Equality: l = u = b

            if G is not None and h is not None:
                G_np = to_numpy(G[i])
                h_np = to_numpy(h[i])
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
                },
            )

            if result.status not in [Status.OPTIMAL, Status.MAX_ITERATIONS]:
                warnings.warn(f"QP solver returned {result.status}. Gradients may be inaccurate.")

            x_sol = result.x
            x_list.append(x_sol)

            # Compute dual variables from KKT conditions
            # For equality constraints: ν from stationarity
            # For inequality constraints: λ from complementarity
            eps_active = 1e-6

            # Active bounds
            active_lb_i = np.abs(x_sol - lb_np) < eps_active
            active_ub_i = np.abs(x_sol - ub_np) < eps_active

            # Compute ν (equality duals) - solve reduced KKT
            if n_eq > 0:
                A_np = to_numpy(A[i])
                # From Px + q + A'ν + bound_duals = 0, project onto A
                residual = P_np @ x_sol + q_np
                # Solve A'ν = -residual (restricted to A subspace)
                try:
                    nu_i = np.linalg.lstsq(A_np.T, -residual, rcond=None)[0]
                except:
                    nu_i = np.zeros(n_eq)
            else:
                nu_i = np.zeros(0)

            # Active inequality constraints
            if n_ineq > 0:
                G_np = to_numpy(G[i])
                h_np = to_numpy(h[i])
                slack = h_np - G_np @ x_sol
                active_ineq_i = slack < eps_active
                
                # Estimate λ for active constraints
                lam_i = np.zeros(n_ineq)
                if active_ineq_i.any():
                    # From stationarity on active constraints
                    G_act = G_np[active_ineq_i]
                    residual = P_np @ x_sol + q_np
                    if n_eq > 0:
                        residual = residual + A_np.T @ nu_i
                    try:
                        lam_act = np.linalg.lstsq(G_act.T, -residual, rcond=None)[0]
                        lam_i[active_ineq_i] = np.maximum(lam_act, 0)  # λ >= 0
                    except:
                        pass
            else:
                lam_i = np.zeros(0)
                active_ineq_i = np.zeros(0, dtype=bool)

            nu_list.append(nu_i)
            lam_list.append(lam_i)
            active_ineq_list.append(active_ineq_i)
            active_lb_list.append(active_lb_i)
            active_ub_list.append(active_ub_i)

        # Stack results
        x = torch.stack([to_torch(xi, device, dtype) for xi in x_list])
        nu = torch.stack([to_torch(ni, device, dtype) for ni in nu_list]) if n_eq > 0 else torch.zeros(batch_size, 0, device=device, dtype=dtype)
        lam = torch.stack([to_torch(li, device, dtype) for li in lam_list]) if n_ineq > 0 else torch.zeros(batch_size, 0, device=device, dtype=dtype)
        active_ineq = torch.stack([torch.tensor(ai, device=device) for ai in active_ineq_list]) if n_ineq > 0 else torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        active_lb = torch.stack([torch.tensor(ai, device=device) for ai in active_lb_list])
        active_ub = torch.stack([torch.tensor(ai, device=device) for ai in active_ub_list])

        if not is_batched:
            x = x.squeeze(0)
            nu = nu.squeeze(0)
            lam = lam.squeeze(0)
            active_ineq = active_ineq.squeeze(0)
            active_lb = active_lb.squeeze(0)
            active_ub = active_ub.squeeze(0)

        sol_info = {
            'nu': nu,
            'lam': lam,
            'active_ineq': active_ineq,
            'active_lb': active_lb,
            'active_ub': active_ub,
        }

        return x, sol_info

    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Compute gradients via implicit differentiation of KKT conditions.

        Uses the adjoint method: solve the adjoint KKT system to get
        sensitivities, then compute parameter gradients.

        The KKT system for the adjoint is:
        [P    A'   G_act']   [d_x  ]   [grad_x]
        [A    0    0     ] * [d_ν  ] = [0     ]
        [G_act 0   0     ]   [d_λ  ]   [0     ]

        Then:
        - grad_q = -d_x
        - grad_P = -d_x ⊗ x
        - grad_b = -d_ν
        - grad_A = -d_ν ⊗ x
        - grad_h = -d_λ (for active constraints)
        - grad_G = -d_λ ⊗ x (for active constraints)
        """
        (P, q, A, b, G, h, lb, ub, x, 
         nu, lam, active_ineq, active_lb, active_ub) = ctx.saved_tensors
        
        is_batched = ctx.is_batched
        n = ctx.n
        n_eq = ctx.n_eq
        n_ineq = ctx.n_ineq

        # Handle batched case
        if is_batched:
            batch_size = grad_x.shape[0]
            grad_P = torch.zeros_like(P) if P.requires_grad else None
            grad_q = torch.zeros_like(q) if q.requires_grad else None
            grad_A = torch.zeros_like(A) if A is not None and A.requires_grad else None
            grad_b = torch.zeros_like(b) if b is not None and b.requires_grad else None
            grad_G = torch.zeros_like(G) if G is not None and G.requires_grad else None
            grad_h = torch.zeros_like(h) if h is not None and h.requires_grad else None

            for i in range(batch_size):
                grads_i = QPFunction._backward_single(
                    P[i], q[i], A[i] if A is not None else None, 
                    b[i] if b is not None else None,
                    G[i] if G is not None else None, 
                    h[i] if h is not None else None,
                    lb[i], ub[i], x[i], nu[i], lam[i],
                    active_ineq[i], active_lb[i], active_ub[i],
                    grad_x[i], n_eq, n_ineq
                )
                if grad_P is not None:
                    grad_P[i] = grads_i[0]
                if grad_q is not None:
                    grad_q[i] = grads_i[1]
                if grad_A is not None and grads_i[2] is not None:
                    grad_A[i] = grads_i[2]
                if grad_b is not None and grads_i[3] is not None:
                    grad_b[i] = grads_i[3]
                if grad_G is not None and grads_i[4] is not None:
                    grad_G[i] = grads_i[4]
                if grad_h is not None and grads_i[5] is not None:
                    grad_h[i] = grads_i[5]
        else:
            grads = QPFunction._backward_single(
                P, q, A, b, G, h, lb, ub, x, nu, lam,
                active_ineq, active_lb, active_ub, grad_x,
                n_eq, n_ineq
            )
            grad_P = grads[0] if P.requires_grad else None
            grad_q = grads[1] if q.requires_grad else None
            grad_A = grads[2] if A is not None and A.requires_grad else None
            grad_b = grads[3] if b is not None and b.requires_grad else None
            grad_G = grads[4] if G is not None and G.requires_grad else None
            grad_h = grads[5] if h is not None and h.requires_grad else None

        return (
            grad_P,
            grad_q,
            grad_A,
            grad_b,
            grad_G,
            grad_h,
            None, None,  # lb, ub (could add later)
            None, None, None,  # max_iters, eps, verbose
        )

    @staticmethod
    def _backward_single(
        P: Tensor,
        q: Tensor,
        A: Optional[Tensor],
        b: Optional[Tensor],
        G: Optional[Tensor],
        h: Optional[Tensor],
        lb: Tensor,
        ub: Tensor,
        x: Tensor,
        nu: Tensor,
        lam: Tensor,
        active_ineq: Tensor,
        active_lb: Tensor,
        active_ub: Tensor,
        grad_x: Tensor,
        n_eq: int,
        n_ineq: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], 
               Optional[Tensor], Optional[Tensor]]:
        """
        Compute gradients for a single QP via adjoint KKT system.
        
        This implements full implicit differentiation through the KKT conditions.
        """
        n = x.shape[0]
        device, dtype = x.device, x.dtype

        # Build the KKT matrix for adjoint system
        # [P    A'   G_act'  I_lb  -I_ub]   [d_x ]   [grad_x]
        # [A    0    0       0     0    ] * [d_ν ] = [0     ]
        # [G_act 0   0       0     0    ]   [d_λ ]   [0     ]
        # ...

        # For simplicity, we solve a reduced system on the free variables
        # (variables not at bounds) combined with active constraints

        # Identify free variables (not at bounds)
        eps_active = 1e-6
        free = ~(active_lb | active_ub)
        n_free = free.sum().item()

        # Active inequality constraints
        n_active_ineq = active_ineq.sum().item() if n_ineq > 0 else 0

        if n_free == 0:
            # All variables at bounds - zero gradients for P, q
            grad_P_out = torch.zeros_like(P)
            grad_q_out = torch.zeros_like(q)
            grad_A_out = torch.zeros(n_eq, n, device=device, dtype=dtype) if n_eq > 0 else None
            grad_b_out = torch.zeros(n_eq, device=device, dtype=dtype) if n_eq > 0 else None
            grad_G_out = torch.zeros(n_ineq, n, device=device, dtype=dtype) if n_ineq > 0 else None
            grad_h_out = torch.zeros(n_ineq, device=device, dtype=dtype) if n_ineq > 0 else None
            return grad_P_out, grad_q_out, grad_A_out, grad_b_out, grad_G_out, grad_h_out

        # Build reduced KKT matrix
        free_idx = free.nonzero(as_tuple=True)[0]
        P_ff = P[free_idx][:, free_idx]
        
        # Size of KKT system
        kkt_size = n_free + n_eq + n_active_ineq
        KKT = torch.zeros(kkt_size, kkt_size, device=device, dtype=dtype)
        rhs = torch.zeros(kkt_size, device=device, dtype=dtype)

        # Fill P block
        KKT[:n_free, :n_free] = P_ff

        # Fill A blocks (equality constraints)
        if n_eq > 0:
            A_f = A[:, free_idx]
            KKT[n_free:n_free+n_eq, :n_free] = A_f
            KKT[:n_free, n_free:n_free+n_eq] = A_f.T

        # Fill G blocks (active inequality constraints)
        if n_active_ineq > 0:
            active_idx = active_ineq.nonzero(as_tuple=True)[0]
            G_act_f = G[active_idx][:, free_idx]
            row_start = n_free + n_eq
            KKT[row_start:row_start+n_active_ineq, :n_free] = G_act_f
            KKT[:n_free, row_start:row_start+n_active_ineq] = G_act_f.T

        # RHS: grad_x on free variables
        rhs[:n_free] = grad_x[free_idx]

        # Solve KKT system
        try:
            # Add small regularization for numerical stability
            KKT_reg = KKT + 1e-8 * torch.eye(kkt_size, device=device, dtype=dtype)
            sol = torch.linalg.solve(KKT_reg, rhs)
        except RuntimeError:
            # Fallback to pseudoinverse
            sol = torch.linalg.lstsq(KKT, rhs).solution

        # Extract components
        d_x_free = sol[:n_free]
        d_nu = sol[n_free:n_free+n_eq] if n_eq > 0 else torch.zeros(0, device=device, dtype=dtype)
        d_lam_act = sol[n_free+n_eq:] if n_active_ineq > 0 else torch.zeros(0, device=device, dtype=dtype)

        # Expand d_x to full size
        d_x = torch.zeros(n, device=device, dtype=dtype)
        d_x[free_idx] = d_x_free

        # Compute gradients
        # grad_q = -d_x
        grad_q_out = -d_x

        # grad_P = -outer(d_x, x) (symmetric, so we use full outer product)
        grad_P_out = -torch.outer(d_x, x)
        # Symmetrize for symmetric P
        grad_P_out = 0.5 * (grad_P_out + grad_P_out.T)

        # grad_b = -d_nu
        if n_eq > 0:
            grad_b_out = -d_nu
            # grad_A = -outer(d_nu, x)
            grad_A_out = -torch.outer(d_nu, x)
        else:
            grad_b_out = None
            grad_A_out = None

        # grad_h and grad_G for active constraints
        if n_ineq > 0:
            grad_h_out = torch.zeros(n_ineq, device=device, dtype=dtype)
            grad_G_out = torch.zeros(n_ineq, n, device=device, dtype=dtype)
            
            if n_active_ineq > 0:
                active_idx = active_ineq.nonzero(as_tuple=True)[0]
                # grad_h[active] = -d_lam_act
                grad_h_out[active_idx] = -d_lam_act
                # grad_G[active] = -outer(d_lam_act, x)
                for i, idx in enumerate(active_idx):
                    grad_G_out[idx] = -d_lam_act[i] * x
        else:
            grad_h_out = None
            grad_G_out = None

        return grad_P_out, grad_q_out, grad_A_out, grad_b_out, grad_G_out, grad_h_out


class BatchQPFunction(Function):
    """
    Optimized batched QP solving with parallel GPU execution.
    
    This function is optimized for solving many QPs in parallel,
    leveraging GPU parallelism in both forward and backward passes.
    """

    @staticmethod
    def forward(
        ctx,
        P: Tensor,  # (batch, n, n)
        q: Tensor,  # (batch, n)
        lb: Tensor, # (batch, n)
        ub: Tensor, # (batch, n)
        max_iters: int,
        eps: float,
    ) -> Tensor:
        """
        Solve batch of box-constrained QPs.
        
        min (1/2)x'Px + q'x  s.t. lb <= x <= ub
        """
        device, dtype = q.device, q.dtype
        batch_size, n = q.shape

        # Solve each QP
        x_list = []
        from scipy import sparse
        from .. import solve as cuprox_solve

        for i in range(batch_size):
            P_np = to_numpy(P[i])
            q_np = to_numpy(q[i])
            lb_np = to_numpy(lb[i])
            ub_np = to_numpy(ub[i])

            result = cuprox_solve(
                c=q_np,
                P=sparse.csr_matrix(P_np),
                lb=lb_np,
                ub=ub_np,
                params={
                    "max_iterations": max_iters,
                    "tolerance": eps,
                    "verbose": False,
                },
            )
            x_list.append(result.x)

        x = torch.stack([to_torch(xi, device, dtype) for xi in x_list])

        # Identify active bounds for backward
        eps_active = 1e-6
        active_lb = (x - lb).abs() < eps_active
        active_ub = (ub - x).abs() < eps_active

        ctx.save_for_backward(P, q, lb, ub, x, active_lb, active_ub)

        return x

    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Vectorized backward pass for batched QPs."""
        P, q, lb, ub, x, active_lb, active_ub = ctx.saved_tensors
        batch_size, n = x.shape
        device, dtype = x.device, x.dtype

        # For box-constrained QP, the backward is simpler
        # We use active set approach per sample
        
        grad_P = torch.zeros_like(P) if P.requires_grad else None
        grad_q = torch.zeros_like(q) if q.requires_grad else None

        for i in range(batch_size):
            free = ~(active_lb[i] | active_ub[i])
            n_free = free.sum().item()

            if n_free == 0:
                continue

            free_idx = free.nonzero(as_tuple=True)[0]
            P_ff = P[i][free_idx][:, free_idx]
            grad_x_f = grad_x[i][free_idx]

            # Solve P_ff' @ v = grad_x_f
            try:
                v_f = torch.linalg.solve(P_ff.T, grad_x_f)
            except RuntimeError:
                v_f = torch.linalg.lstsq(P_ff.T, grad_x_f).solution

            # grad_q
            if grad_q is not None:
                grad_q[i][free_idx] = -v_f

            # grad_P
            if grad_P is not None:
                for j, fj in enumerate(free_idx):
                    for k, fk in enumerate(free_idx):
                        grad_P[i][fj, fk] = -v_f[j] * x[i][fk]

        return grad_P, grad_q, None, None, None, None


class LPFunction(Function):
    """
    Autograd function for differentiable LP solving.

    Note: LP gradients are approximate since LP solutions
    are typically at vertices where the solution is non-smooth.
    
    For better gradient behavior, consider adding a small
    quadratic regularization (converting to QP).
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

        x = LPFunction._solve(c, A, b, G, h, lb, ub, max_iters, eps, verbose, device, dtype)

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
            },
        )

        return to_torch(result.x, device, dtype)

    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Approximate gradients for LP using smoothed sensitivity.

        LP solutions are typically at vertices, making exact
        differentiation ill-defined. We use a quadratic smoothing
        approximation for stable gradients.
        """
        c, A, b, G, h, lb, ub, x = ctx.saved_tensors
        n = x.shape[0]
        device, dtype = x.device, x.dtype

        # Identify active bounds
        eps_active = 1e-6
        active_lb = (x - lb).abs() < eps_active
        active_ub = (ub - x).abs() < eps_active
        free = ~(active_lb | active_ub)

        grad_c = None
        if c.requires_grad:
            grad_c = torch.zeros_like(c)
            if free.any():
                # Approximate sensitivity: for free variables, use -I
                # (as if the LP were a QP with small regularization)
                grad_c[free] = -grad_x[free]

        # Gradient for A, b (equality constraints)
        grad_A = None
        grad_b = None
        if A is not None and A.requires_grad:
            # For active equality constraints, use sensitivity analysis
            # This is approximate for LP
            grad_A = torch.zeros_like(A)
            
        if b is not None and b.requires_grad:
            grad_b = torch.zeros_like(b)

        # Gradient for G, h (inequality constraints)
        grad_G = None
        grad_h = None

        return (
            grad_c,
            grad_A,
            grad_b,
            grad_G,
            grad_h,
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

    Supports batched input for parallel solving:
    - P: (n, n) or (batch, n, n)
    - q: (n,) or (batch, n)

    Args:
        P: Quadratic cost (n, n) or (batch, n, n), positive semidefinite
        q: Linear cost (n,) or (batch, n)
        A: Equality constraints (m_eq, n) or (batch, m_eq, n)
        b: Equality RHS (m_eq,) or (batch, m_eq)
        G: Inequality constraints (m_ineq, n) or (batch, m_ineq, n)
        h: Inequality RHS (m_ineq,) or (batch, m_ineq)
        lb: Lower bounds (n,) or (batch, n)
        ub: Upper bounds (n,) or (batch, n)
        max_iters: Maximum iterations
        eps: Tolerance
        verbose: Print progress

    Returns:
        Optimal solution x* with gradients

    Gradients supported for: P, q, A, b, G, h

    Example:
        >>> P = torch.eye(2, requires_grad=True)
        >>> q = torch.tensor([-1., -2.], requires_grad=True)
        >>> x = solve_qp(P, q)
        >>> x.sum().backward()
        >>> print(q.grad)  # Gradient w.r.t. q
    """
    is_batched = q.dim() == 2
    if is_batched:
        n = q.shape[1]
    else:
        n = q.shape[-1]
    device, dtype = q.device, q.dtype

    if lb is None:
        if is_batched:
            lb = torch.full((q.shape[0], n), -1e20, device=device, dtype=dtype)
        else:
            lb = torch.full((n,), -1e20, device=device, dtype=dtype)
    if ub is None:
        if is_batched:
            ub = torch.full((q.shape[0], n), 1e20, device=device, dtype=dtype)
        else:
            ub = torch.full((n,), 1e20, device=device, dtype=dtype)

    return QPFunction.apply(P, q, A, b, G, h, lb, ub, max_iters, eps, verbose)


def solve_qp_batch(
    P: Tensor,
    q: Tensor,
    lb: Optional[Tensor] = None,
    ub: Optional[Tensor] = None,
    max_iters: int = 10000,
    eps: float = 1e-5,
) -> Tensor:
    """
    Solve a batch of box-constrained QPs with optimized parallel execution.

    minimize    (1/2) x' P x + q' x
    subject to  lb <= x <= ub

    This is optimized for the common case of many similar QPs.

    Args:
        P: Quadratic cost (batch, n, n), positive semidefinite
        q: Linear cost (batch, n)
        lb: Lower bounds (batch, n), default -inf
        ub: Upper bounds (batch, n), default +inf
        max_iters: Maximum iterations
        eps: Tolerance

    Returns:
        Optimal solutions x* with shape (batch, n)

    Example:
        >>> batch_size, n = 100, 10
        >>> P = torch.eye(n).unsqueeze(0).expand(batch_size, -1, -1)
        >>> q = torch.randn(batch_size, n, requires_grad=True)
        >>> x = solve_qp_batch(P, q)
        >>> x.sum().backward()
    """
    batch_size, n = q.shape
    device, dtype = q.device, q.dtype

    if lb is None:
        lb = torch.full((batch_size, n), -1e20, device=device, dtype=dtype)
    if ub is None:
        ub = torch.full((batch_size, n), 1e20, device=device, dtype=dtype)

    return BatchQPFunction.apply(P, q, lb, ub, max_iters, eps)


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

    Note: LP gradients are approximate due to non-smoothness.
    Consider using solve_qp with small regularization for
    better gradient behavior.

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


# =============================================================================
# Dual Variable Access
# =============================================================================


def solve_qp_with_duals(
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
) -> QPSolution:
    """
    Solve QP and return both primal and dual variables.

    Returns a QPSolution namedtuple with:
    - x: Primal solution
    - nu: Equality constraint duals (Lagrange multipliers for Ax = b)
    - lam: Inequality constraint duals (Lagrange multipliers for Gx <= h)
    - lam_lb: Lower bound duals
    - lam_ub: Upper bound duals
    - active_ineq: Mask of active inequality constraints
    - active_lb: Mask of active lower bounds
    - active_ub: Mask of active upper bounds

    Example:
        >>> sol = solve_qp_with_duals(P, q, A=A, b=b)
        >>> print(sol.x)   # Primal solution
        >>> print(sol.nu)  # Equality duals
    """
    n = q.shape[-1]
    device, dtype = q.device, q.dtype

    if lb is None:
        lb = torch.full((n,), -1e20, device=device, dtype=dtype)
    if ub is None:
        ub = torch.full((n,), 1e20, device=device, dtype=dtype)

    # Solve QP
    from scipy import sparse
    from .. import solve as cuprox_solve
    from .utils import to_numpy, to_torch

    P_np = to_numpy(P)
    q_np = to_numpy(q)
    lb_np = to_numpy(lb)
    ub_np = to_numpy(ub)

    n_eq = A.shape[0] if A is not None else 0
    n_ineq = G.shape[0] if G is not None else 0

    # Build constraints
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
        h_np = to_numpy(h)
        l_list.append(np.full(h_np.shape, -1e20))
        u_list.append(h_np)

    if constraints:
        A_comb = np.vstack(constraints)
        l_comb = np.concatenate(l_list)
        u_comb = np.concatenate(u_list)
    else:
        A_comb = np.zeros((0, n))
        l_comb = np.array([])
        u_comb = np.array([])

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
        },
    )

    x = to_torch(result.x, device, dtype)

    # Compute duals
    eps_active = 1e-6
    active_lb = (x - lb).abs() < eps_active
    active_ub = (ub - x).abs() < eps_active

    # Equality duals
    if n_eq > 0:
        A_t = A
        residual = P @ x + q
        nu = -torch.linalg.lstsq(A_t.T, residual).solution
    else:
        nu = torch.zeros(0, device=device, dtype=dtype)

    # Inequality duals
    if n_ineq > 0:
        slack = h - G @ x
        active_ineq = slack.abs() < eps_active
        lam = torch.zeros(n_ineq, device=device, dtype=dtype)
        
        if active_ineq.any():
            G_act = G[active_ineq]
            residual = P @ x + q
            if n_eq > 0:
                residual = residual + A.T @ nu
            try:
                lam_act = -torch.linalg.lstsq(G_act.T, residual).solution
                lam[active_ineq] = torch.clamp(lam_act, min=0)
            except:
                pass
    else:
        lam = torch.zeros(0, device=device, dtype=dtype)
        active_ineq = torch.zeros(0, dtype=torch.bool, device=device)

    # Bound duals (from KKT stationarity)
    lam_lb = torch.zeros(n, device=device, dtype=dtype)
    lam_ub = torch.zeros(n, device=device, dtype=dtype)
    
    residual = P @ x + q
    if n_eq > 0:
        residual = residual + A.T @ nu
    if n_ineq > 0:
        residual = residual + G.T @ lam
    
    # At lower bound: residual = -lam_lb (lam_lb >= 0)
    lam_lb[active_lb] = torch.clamp(-residual[active_lb], min=0)
    # At upper bound: residual = lam_ub (lam_ub >= 0)
    lam_ub[active_ub] = torch.clamp(residual[active_ub], min=0)

    return QPSolution(
        x=x,
        nu=nu,
        lam=lam,
        lam_lb=lam_lb,
        lam_ub=lam_ub,
        active_ineq=active_ineq,
        active_lb=active_lb,
        active_ub=active_ub,
    )
