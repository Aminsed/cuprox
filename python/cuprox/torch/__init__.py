"""
cuProx PyTorch Integration
==========================

Differentiable optimization layers for PyTorch that enable
backpropagation through LP and QP solvers.

Features
--------
- Full KKT-based implicit differentiation
- Gradients for all QP parameters (P, q, A, b, G, h)
- Batched solving for parallel GPU execution
- OptNet-style layers with learnable constraints
- Dual variable access
- Decision-focused learning support

Quick Start
-----------
>>> from cuprox.torch import QPLayer, solve_qp
>>>
>>> # Using nn.Module layer
>>> layer = QPLayer(n_vars=2)
>>> x = layer(P, q)  # differentiable!
>>>
>>> # Using functional API
>>> x = solve_qp(P, q)
>>> x.sum().backward()

Layers
------
QPLayer
    nn.Module for differentiable quadratic programming.
    Solves: min (1/2)x'Px + q'x s.t. Ax=b, Gx<=h, lb<=x<=ub

LPLayer
    nn.Module for differentiable linear programming.
    Solves: min c'x s.t. Ax=b, Gx<=h, lb<=x<=ub

OptNetLayer
    Full OptNet with learnable constraint parameters (P, A, G).
    Maps input features to optimal decisions end-to-end.

ParametricQPLayer
    QP layer with learnable P matrix (guaranteed PSD via L@L.T).

BatchQPLayer
    Optimized layer for solving batches of box-constrained QPs.

DecisionFocusedLayer
    End-to-end decision-focused learning layer.

Functions
---------
solve_qp
    Functional interface to differentiable QP solver.
    Supports batched input for parallel solving.

solve_qp_batch
    Optimized batch solving for box-constrained QPs.

solve_lp
    Functional interface to differentiable LP solver.

solve_qp_with_duals
    Solve QP and return both primal and dual variables.

Example: OptNet-style Network
-----------------------------
>>> import torch.nn as nn
>>> from cuprox.torch import OptNetLayer
>>>
>>> class DecisionNet(nn.Module):
...     def __init__(self, n_features, n_vars, n_ineq):
...         super().__init__()
...         self.optnet = OptNetLayer(
...             n_features=n_features,
...             n_vars=n_vars,
...             n_ineq=n_ineq,
...         )
...
...     def forward(self, x):
...         return self.optnet(x)
>>>
>>> model = DecisionNet(10, 5, 3)
>>> x = torch.randn(32, 10)  # Batch of 32 samples
>>> z = model(x)  # Optimal decisions
>>> loss = ((z - target) ** 2).sum()
>>> loss.backward()  # Gradients through QP solver!

Example: Learning Constraint Structure
--------------------------------------
>>> from cuprox.torch import ParametricQPLayer
>>>
>>> # Layer learns optimal P matrix from data
>>> layer = ParametricQPLayer(n_vars=5)
>>> q = torch.randn(100, 5)  # Batch of q vectors
>>> z = layer(q)
>>> loss = task_loss(z)
>>> loss.backward()  # Updates layer.L parameter

Example: Batched MPC
--------------------
>>> from cuprox.torch import solve_qp_batch
>>>
>>> # Solve 1000 QPs in parallel
>>> batch_size = 1000
>>> P = torch.eye(n).unsqueeze(0).expand(batch_size, -1, -1)
>>> q = torch.randn(batch_size, n, requires_grad=True)
>>> x = solve_qp_batch(P, q)
>>> # Gradients through all 1000 solutions!
>>> x.sum().backward()

See Also
--------
- Amos & Kolter (2017): "OptNet: Differentiable Optimization..."
- Agrawal et al. (2019): "Differentiable Convex Optimization Layers"
"""

from .functions import (
    # Autograd functions
    QPFunction,
    LPFunction,
    BatchQPFunction,
    # Functional API
    solve_qp,
    solve_qp_batch,
    solve_lp,
    solve_qp_with_duals,
    # Data structures
    QPSolution,
)
from .layers import (
    # Basic layers
    QPLayer,
    LPLayer,
    # Advanced layers
    OptNetLayer,
    ParametricQPLayer,
    BatchQPLayer,
    DecisionFocusedLayer,
)

__all__ = [
    # nn.Module layers
    "QPLayer",
    "LPLayer",
    "OptNetLayer",
    "ParametricQPLayer",
    "BatchQPLayer",
    "DecisionFocusedLayer",
    # Functional interface
    "solve_qp",
    "solve_qp_batch",
    "solve_lp",
    "solve_qp_with_duals",
    # Advanced: autograd functions
    "QPFunction",
    "LPFunction",
    "BatchQPFunction",
    # Data structures
    "QPSolution",
]
