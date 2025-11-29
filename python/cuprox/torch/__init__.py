"""
cuProx PyTorch Integration
==========================

Differentiable optimization layers for PyTorch that enable
backpropagation through LP and QP solvers.

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

Classes
-------
QPLayer
    nn.Module for differentiable quadratic programming.
    Solves: min (1/2)x'Px + q'x s.t. Ax=b, Gx<=h, lb<=x<=ub

LPLayer
    nn.Module for differentiable linear programming.
    Solves: min c'x s.t. Ax=b, Gx<=h, lb<=x<=ub

Functions
---------
solve_qp
    Functional interface to differentiable QP solver.

solve_lp
    Functional interface to differentiable LP solver.

Example: OptNet-style Network
-----------------------------
>>> import torch.nn as nn
>>> from cuprox.torch import QPLayer
>>>
>>> class OptNet(nn.Module):
...     def __init__(self, n_features, n_hidden):
...         super().__init__()
...         self.fc = nn.Linear(n_features, n_hidden)
...         self.qp = QPLayer(n_vars=n_hidden)
...         self.P = nn.Parameter(torch.eye(n_hidden))
...
...     def forward(self, x):
...         q = self.fc(x)
...         return self.qp(self.P, q)
>>>
>>> model = OptNet(10, 5)
>>> x = torch.randn(10)
>>> y = model(x)
>>> y.sum().backward()  # Gradients through QP!

See Also
--------
- Amos & Kolter (2017): "OptNet: Differentiable Optimization..."
- Agrawal et al. (2019): "Differentiable Convex Optimization Layers"
"""

from .functions import LPFunction, QPFunction, solve_lp, solve_qp
from .layers import LPLayer, QPLayer

__all__ = [
    # nn.Module layers
    "QPLayer",
    "LPLayer",
    # Functional interface
    "solve_qp",
    "solve_lp",
    # Advanced: autograd functions
    "QPFunction",
    "LPFunction",
]
