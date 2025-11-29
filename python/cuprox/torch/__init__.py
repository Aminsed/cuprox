"""
cuProx PyTorch Integration
==========================

Differentiable optimization layers for PyTorch.

This module provides GPU-accelerated LP/QP solvers that can be used as
differentiable layers in neural networks, enabling end-to-end training
through optimization problems.

Example:
    >>> import torch
    >>> from cuprox.torch import QPLayer
    >>> 
    >>> # Create a differentiable QP layer
    >>> layer = QPLayer(n_vars=2)
    >>> 
    >>> # Problem parameters (can have gradients)
    >>> P = torch.eye(2, requires_grad=True)
    >>> q = torch.tensor([-1.0, -2.0], requires_grad=True)
    >>> 
    >>> # Solve and backpropagate
    >>> x = layer(P, q)
    >>> loss = x.sum()
    >>> loss.backward()
    >>> print(q.grad)  # Gradients w.r.t. problem parameters

See Also:
    - :class:`QPLayer`: Differentiable quadratic programming layer
    - :class:`LPLayer`: Differentiable linear programming layer
    - :func:`solve_qp`: Functional interface for differentiable QP
    - :func:`solve_lp`: Functional interface for differentiable LP
"""

from .layers import QPLayer, LPLayer, OptLayer
from .functions import solve_qp, solve_lp, QPFunction, LPFunction

__all__ = [
    # Layers (nn.Module)
    "QPLayer",
    "LPLayer", 
    "OptLayer",
    # Functional interface
    "solve_qp",
    "solve_lp",
    # Autograd functions (advanced)
    "QPFunction",
    "LPFunction",
]

