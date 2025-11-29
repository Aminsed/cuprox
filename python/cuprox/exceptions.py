"""
cuProx Exception Classes
========================

Custom exceptions for cuProx error handling.
"""

from typing import Optional


class CuproxError(Exception):
    """Base exception for all cuProx errors."""
    
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InfeasibleError(CuproxError):
    """
    Raised when the problem is primal infeasible.
    
    This means there is no x that satisfies all constraints.
    """
    
    def __init__(self, message: str = "Problem is infeasible") -> None:
        super().__init__(message)


class UnboundedError(CuproxError):
    """
    Raised when the problem is unbounded (dual infeasible).
    
    This means the objective can be made arbitrarily small (for minimization).
    """
    
    def __init__(self, message: str = "Problem is unbounded") -> None:
        super().__init__(message)


class NumericalError(CuproxError):
    """
    Raised when numerical issues are encountered.
    
    This may indicate ill-conditioning, overflow, or other numerical problems.
    """
    
    def __init__(self, message: str = "Numerical error encountered") -> None:
        super().__init__(message)


class TimeoutError(CuproxError):
    """
    Raised when the solver exceeds the time limit.
    
    The solver may have a partial solution available.
    """
    
    def __init__(
        self,
        message: str = "Time limit exceeded",
        iterations: Optional[int] = None,
    ) -> None:
        self.iterations = iterations
        super().__init__(message)


class DimensionError(CuproxError):
    """
    Raised when matrix/vector dimensions are incompatible.
    """
    
    def __init__(self, message: str) -> None:
        super().__init__(f"Dimension mismatch: {message}")


class InvalidInputError(CuproxError):
    """
    Raised when input data is invalid.
    
    Examples: NaN values, negative counts, invalid constraint sense.
    """
    
    def __init__(self, message: str) -> None:
        super().__init__(f"Invalid input: {message}")


class DeviceError(CuproxError):
    """
    Raised when there's an issue with GPU device.
    
    Examples: No GPU available, out of memory, driver issues.
    """
    
    def __init__(self, message: str) -> None:
        super().__init__(f"Device error: {message}")

