"""
Differentiable Optimization Layers
==================================

PyTorch nn.Module wrappers for optimization problems that support
backpropagation through the solver.

Layers:
    QPLayer: Basic differentiable QP layer
    LPLayer: Basic differentiable LP layer
    OptNetLayer: Full OptNet with learnable constraint parameters
    ParametricQPLayer: QP with learnable P matrix
    BatchQPLayer: Optimized layer for batch solving
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = type("Module", (), {})  # Placeholder

from .functions import (
    LPFunction, 
    QPFunction, 
    BatchQPFunction,
    solve_qp,
    solve_qp_batch,
    QPSolution,
)
from .utils import check_torch_available


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

    Gradients are computed for: P, q, A, b, G, h

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
        is_batched = q.dim() == 2

        if lb is None:
            if is_batched:
                lb = torch.full((q.shape[0], n), -1e20, device=device, dtype=dtype)
            else:
                lb = torch.full((n,), -1e20, device=device, dtype=dtype)
        elif is_batched and lb.dim() == 1:
            lb = lb.unsqueeze(0).expand(q.shape[0], -1)
            
        if ub is None:
            if is_batched:
                ub = torch.full((q.shape[0], n), 1e20, device=device, dtype=dtype)
            else:
                ub = torch.full((n,), 1e20, device=device, dtype=dtype)
        elif is_batched and ub.dim() == 1:
            ub = ub.unsqueeze(0).expand(q.shape[0], -1)

        # Expand P if needed for batch
        if is_batched and P.dim() == 2:
            P = P.unsqueeze(0).expand(q.shape[0], -1, -1)

        # Solve
        if self.differentiable:
            return QPFunction.apply(
                P, q, A, b, G, h, lb, ub, self.max_iters, self.eps, self.verbose
            )
        else:
            with torch.no_grad():
                return QPFunction.apply(
                    P, q, A, b, G, h, lb, ub, self.max_iters, self.eps, self.verbose
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


class OptNetLayer(nn.Module if HAS_TORCH else object):
    """
    Full OptNet Layer with Learnable Constraint Parameters.

    This implements the OptNet architecture from Amos & Kolter (2017),
    where all QP parameters (P, q, A, b, G, h) can be learnable.

    The layer learns to map input features to optimal decisions by
    learning the constraint structure that produces good solutions.

    Solves:
        minimize    (1/2) x' P(θ) x + q(θ)' x
        subject to  A(θ) x = b(θ)
                    G(θ) x <= h(θ)

    where θ are the learnable parameters.

    Args:
        n_features: Input feature dimension
        n_vars: Number of decision variables
        n_eq: Number of equality constraints
        n_ineq: Number of inequality constraints
        learn_P: Learn the quadratic cost matrix (default: True)
        learn_A: Learn the equality constraint matrix (default: True)
        learn_G: Learn the inequality constraint matrix (default: True)
        eps: Solver tolerance
        max_iters: Maximum solver iterations

    Example:
        >>> layer = OptNetLayer(n_features=10, n_vars=5, n_ineq=3)
        >>> x = torch.randn(32, 10)  # Batch of 32, 10 features
        >>> z = layer(x)  # Optimal decisions, shape (32, 5)
        >>> loss = z.sum()
        >>> loss.backward()  # Gradients flow to layer parameters!
    """

    def __init__(
        self,
        n_features: int,
        n_vars: int,
        n_eq: int = 0,
        n_ineq: int = 0,
        learn_P: bool = True,
        learn_A: bool = True,
        learn_G: bool = True,
        eps: float = 1e-5,
        max_iters: int = 10000,
    ) -> None:
        check_torch_available()
        super().__init__()

        self.n_features = n_features
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.eps = eps
        self.max_iters = max_iters

        # Learnable P matrix (lower triangular for PSD guarantee)
        if learn_P:
            self.L = nn.Parameter(torch.eye(n_vars) * 0.1)
        else:
            self.register_buffer("L", torch.eye(n_vars))

        # Feature to q mapping
        self.fc_q = nn.Linear(n_features, n_vars)

        # Equality constraints
        if n_eq > 0:
            if learn_A:
                self.A = nn.Parameter(torch.randn(n_eq, n_vars) * 0.1)
            else:
                self.register_buffer("A", torch.randn(n_eq, n_vars) * 0.1)
            self.fc_b = nn.Linear(n_features, n_eq)
        else:
            self.A = None
            self.fc_b = None

        # Inequality constraints
        if n_ineq > 0:
            if learn_G:
                self.G = nn.Parameter(torch.randn(n_ineq, n_vars) * 0.1)
            else:
                self.register_buffer("G", torch.randn(n_ineq, n_vars) * 0.1)
            self.fc_h = nn.Linear(n_features, n_ineq)
        else:
            self.G = None
            self.fc_h = None

    def forward(
        self,
        x: Tensor,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass: map features to optimal decisions.

        Args:
            x: Input features (batch, n_features) or (n_features,)
            lb: Variable lower bounds (optional)
            ub: Variable upper bounds (optional)

        Returns:
            Optimal decisions z* with shape (batch, n_vars) or (n_vars,)
        """
        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype

        # Compute P = L @ L.T (guaranteed PSD)
        P = self.L @ self.L.T + 1e-4 * torch.eye(self.n_vars, device=device, dtype=dtype)
        P = P.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute q from features
        q = self.fc_q(x)

        # Compute constraint RHS from features
        A = self.A.unsqueeze(0).expand(batch_size, -1, -1) if self.A is not None else None
        b = self.fc_b(x) if self.fc_b is not None else None
        G = self.G.unsqueeze(0).expand(batch_size, -1, -1) if self.G is not None else None
        h = self.fc_h(x) if self.fc_h is not None else None

        # Default bounds
        if lb is None:
            lb = torch.full((batch_size, self.n_vars), -1e20, device=device, dtype=dtype)
        if ub is None:
            ub = torch.full((batch_size, self.n_vars), 1e20, device=device, dtype=dtype)

        # Solve batch of QPs
        z = solve_qp(P, q, A, b, G, h, lb, ub, self.max_iters, self.eps)

        if not is_batched:
            z = z.squeeze(0)

        return z

    def extra_repr(self) -> str:
        return (f"n_features={self.n_features}, n_vars={self.n_vars}, "
                f"n_eq={self.n_eq}, n_ineq={self.n_ineq}")


class ParametricQPLayer(nn.Module if HAS_TORCH else object):
    """
    Parametric QP Layer with Learnable Quadratic Cost.

    This layer learns a positive semi-definite matrix P and optionally
    a constraint structure, useful for learning optimization objectives
    from data.

    Solves:
        minimize    (1/2) x' P x + q' x
        subject to  lb <= x <= ub

    where P = L @ L.T is learned (guaranteed PSD).

    Args:
        n_vars: Number of decision variables
        rank: Rank of P factorization (default: n_vars for full rank)
        init_scale: Initial scale for L matrix

    Example:
        >>> layer = ParametricQPLayer(n_vars=5)
        >>> q = torch.randn(32, 5, requires_grad=True)
        >>> x = layer(q)  # Shape (32, 5)
        >>> loss = ((x - target) ** 2).sum()
        >>> loss.backward()  # Learns optimal P!
    """

    def __init__(
        self,
        n_vars: int,
        rank: Optional[int] = None,
        init_scale: float = 0.1,
        eps: float = 1e-5,
        max_iters: int = 10000,
    ) -> None:
        check_torch_available()
        super().__init__()

        self.n_vars = n_vars
        self.rank = rank if rank is not None else n_vars
        self.eps = eps
        self.max_iters = max_iters

        # L such that P = L @ L.T
        self.L = nn.Parameter(torch.randn(n_vars, self.rank) * init_scale)

    @property
    def P(self) -> Tensor:
        """Get the current P matrix."""
        return self.L @ self.L.T + 1e-6 * torch.eye(self.n_vars, device=self.L.device)

    def forward(
        self,
        q: Tensor,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Solve QP with learned P and given q.

        Args:
            q: Linear cost (batch, n) or (n,)
            lb: Lower bounds
            ub: Upper bounds

        Returns:
            Optimal solution x*
        """
        is_batched = q.dim() == 2
        device, dtype = q.device, q.dtype

        # Compute P
        P = self.P.to(dtype=dtype)

        if is_batched:
            batch_size = q.shape[0]
            P = P.unsqueeze(0).expand(batch_size, -1, -1)
            if lb is None:
                lb = torch.full((batch_size, self.n_vars), -1e20, device=device, dtype=dtype)
            if ub is None:
                ub = torch.full((batch_size, self.n_vars), 1e20, device=device, dtype=dtype)
            return solve_qp_batch(P, q, lb, ub, self.max_iters, self.eps)
        else:
            if lb is None:
                lb = torch.full((self.n_vars,), -1e20, device=device, dtype=dtype)
            if ub is None:
                ub = torch.full((self.n_vars,), 1e20, device=device, dtype=dtype)
            return solve_qp(P, q, lb=lb, ub=ub, max_iters=self.max_iters, eps=self.eps)

    def extra_repr(self) -> str:
        return f"n_vars={self.n_vars}, rank={self.rank}"


class BatchQPLayer(nn.Module if HAS_TORCH else object):
    """
    Optimized QP Layer for Batch Solving.

    This layer is optimized for solving many box-constrained QPs
    in parallel, common in reinforcement learning and MPC.

    Solves:
        minimize    (1/2) x' P x + q' x
        subject to  lb <= x <= ub

    Args:
        n_vars: Number of decision variables
        eps: Solver tolerance
        max_iters: Maximum solver iterations

    Example:
        >>> layer = BatchQPLayer(n_vars=10)
        >>> P = torch.eye(10).unsqueeze(0).expand(100, -1, -1)
        >>> q = torch.randn(100, 10)
        >>> x = layer(P, q)  # Shape (100, 10)
    """

    def __init__(
        self,
        n_vars: int,
        eps: float = 1e-5,
        max_iters: int = 10000,
    ) -> None:
        check_torch_available()
        super().__init__()

        self.n_vars = n_vars
        self.eps = eps
        self.max_iters = max_iters

    def forward(
        self,
        P: Tensor,
        q: Tensor,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Solve batch of QPs.

        Args:
            P: Quadratic cost (batch, n, n)
            q: Linear cost (batch, n)
            lb: Lower bounds (batch, n), optional
            ub: Upper bounds (batch, n), optional

        Returns:
            Optimal solutions x* with shape (batch, n)
        """
        return solve_qp_batch(P, q, lb, ub, self.max_iters, self.eps)

    def extra_repr(self) -> str:
        return f"n_vars={self.n_vars}, eps={self.eps}"


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
            return LPFunction.apply(c, A, b, G, h, lb, ub, self.max_iters, self.eps, self.verbose)
        else:
            with torch.no_grad():
                return LPFunction.apply(
                    c, A, b, G, h, lb, ub, self.max_iters, self.eps, self.verbose
                )

    def extra_repr(self) -> str:
        """String representation."""
        return f"n_vars={self.n_vars}, n_eq={self.n_eq}, n_ineq={self.n_ineq}"


class DecisionFocusedLayer(nn.Module if HAS_TORCH else object):
    """
    Decision-Focused Learning Layer.

    This layer implements end-to-end decision-focused learning where
    a neural network predicts optimization parameters, and the loss
    is computed on the resulting decisions rather than the predictions.

    Architecture:
        features -> neural_net -> QP_params -> QP_solver -> decisions -> task_loss

    This enables training the neural network to make predictions that
    lead to good decisions, even if those predictions don't match
    ground truth parameters.

    Args:
        predictor: Neural network that maps features to QP parameters
        n_vars: Number of decision variables
        param_type: Which QP parameter to predict ("q", "P", or "both")

    Example:
        >>> predictor = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
        >>> layer = DecisionFocusedLayer(predictor, n_vars=5, param_type="q")
        >>> x = torch.randn(32, 10)
        >>> z = layer(x)  # Decisions
        >>> task_loss = ((z - target_z) ** 2).sum()
        >>> task_loss.backward()  # Trains predictor for good decisions!
    """

    def __init__(
        self,
        predictor: nn.Module,
        n_vars: int,
        param_type: str = "q",
        eps: float = 1e-5,
        max_iters: int = 10000,
    ) -> None:
        check_torch_available()
        super().__init__()

        self.predictor = predictor
        self.n_vars = n_vars
        self.param_type = param_type
        self.eps = eps
        self.max_iters = max_iters

        if param_type not in ["q", "P", "both"]:
            raise ValueError(f"param_type must be 'q', 'P', or 'both', got {param_type}")

        # Default P if only predicting q
        if param_type == "q":
            self.register_buffer("P", torch.eye(n_vars))

    def forward(
        self,
        x: Tensor,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass: predict parameters and solve.

        Args:
            x: Input features
            lb: Variable lower bounds
            ub: Variable upper bounds

        Returns:
            Optimal decisions
        """
        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype

        # Predict parameters
        pred = self.predictor(x)

        if self.param_type == "q":
            q = pred
            P = self.P.unsqueeze(0).expand(batch_size, -1, -1)
        elif self.param_type == "P":
            # Predict lower triangular, reconstruct P = L @ L.T
            L = pred.view(batch_size, self.n_vars, self.n_vars).tril()
            P = L @ L.transpose(-1, -2) + 1e-4 * torch.eye(self.n_vars, device=device)
            q = torch.zeros(batch_size, self.n_vars, device=device, dtype=dtype)
        else:  # both
            q = pred[:, :self.n_vars]
            L = pred[:, self.n_vars:].view(batch_size, self.n_vars, self.n_vars).tril()
            P = L @ L.transpose(-1, -2) + 1e-4 * torch.eye(self.n_vars, device=device)

        # Solve
        if lb is None:
            lb = torch.full((batch_size, self.n_vars), -1e20, device=device, dtype=dtype)
        if ub is None:
            ub = torch.full((batch_size, self.n_vars), 1e20, device=device, dtype=dtype)

        z = solve_qp(P, q, lb=lb, ub=ub, max_iters=self.max_iters, eps=self.eps)

        if not is_batched:
            z = z.squeeze(0)

        return z

    def extra_repr(self) -> str:
        return f"n_vars={self.n_vars}, param_type={self.param_type}"
