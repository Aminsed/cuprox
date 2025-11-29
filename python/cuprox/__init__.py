"""
cuProx: GPU-Accelerated First-Order LP/QP Solver
=================================================

cuProx solves Linear Programs (LP) and convex Quadratic Programs (QP)
using GPU-accelerated proximal methods (PDHG, ADMM).

Quick Start
-----------
>>> import cuprox
>>> model = cuprox.Model()
>>> x = model.add_var(lb=0, name="x")
>>> y = model.add_var(lb=0, name="y")
>>> model.add_constr(x + 2*y <= 10)
>>> model.minimize(-x - y)
>>> result = model.solve()
>>> print(result.status, result.objective)
optimal -5.0

For large-scale problems, use the matrix interface:

>>> import cuprox
>>> import numpy as np
>>> from scipy import sparse
>>> 
>>> A = sparse.random(1000, 5000, density=0.01, format='csr')
>>> b = np.random.rand(1000)
>>> c = np.random.randn(5000)
>>> result = cuprox.solve(c=c, A=A, b=b, lb=np.zeros(5000))
"""

__version__ = "0.1.0"
__author__ = "cuProx Contributors"

# Check for CUDA availability
try:
    from . import _core
    __cuda_available__ = _core.cuda_available
except ImportError:
    # C++ extension not built yet
    _core = None
    __cuda_available__ = False

# Import public API
from .model import Model, Variable, Constraint, LinearExpr
from .solver import solve, solve_batch
from .result import SolveResult, Status
from .exceptions import (
    CuproxError,
    InfeasibleError,
    UnboundedError,
    NumericalError,
    TimeoutError,
    DimensionError,
    InvalidInputError,
    DeviceError,
)

__all__ = [
    # Version
    "__version__",
    "__cuda_available__",
    
    # Model building
    "Model",
    "Variable",
    "Constraint",
    "LinearExpr",
    
    # Solving
    "solve",
    "solve_batch",
    
    # Results
    "SolveResult",
    "Status",
    
    # Exceptions
    "CuproxError",
    "InfeasibleError",
    "UnboundedError",
    "NumericalError",
    "TimeoutError",
    "DimensionError",
    "InvalidInputError",
    "DeviceError",
]


def info() -> str:
    """Return information about the cuProx installation."""
    import platform
    
    lines = [
        f"cuProx version: {__version__}",
        f"Python version: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"CUDA available: {__cuda_available__}",
    ]
    
    if __cuda_available__:
        try:
            from ._core import get_device_info
            device_info = get_device_info()
            lines.append(f"GPU: {device_info['name']}")
            lines.append(f"CUDA version: {device_info['cuda_version']}")
            lines.append(f"Memory: {device_info['memory_gb']:.1f} GB")
        except Exception:
            pass
    
    return "\n".join(lines)

