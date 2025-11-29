"""
Test that cuProx can be imported and basic functionality works.

This is the first test to write - validates the package structure.
"""

import pytest


def test_import_cuprox():
    """Verify cuprox package can be imported."""
    import cuprox
    assert hasattr(cuprox, "__version__")


def test_version_format():
    """Verify version string is properly formatted."""
    import cuprox
    version = cuprox.__version__
    
    # Should be semver format
    parts = version.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() or "-" in p for p in parts)


def test_cuda_available_attribute():
    """Verify __cuda_available__ attribute exists."""
    import cuprox
    assert hasattr(cuprox, "__cuda_available__")
    assert isinstance(cuprox.__cuda_available__, bool)


def test_import_model():
    """Verify Model class can be imported."""
    from cuprox import Model
    assert Model is not None


def test_import_solve():
    """Verify solve function can be imported."""
    from cuprox import solve
    assert callable(solve)


def test_import_solve_batch():
    """Verify solve_batch function can be imported."""
    from cuprox import solve_batch
    assert callable(solve_batch)


def test_import_result():
    """Verify SolveResult class can be imported."""
    from cuprox import SolveResult, Status
    assert SolveResult is not None
    assert Status is not None


def test_import_exceptions():
    """Verify exception classes can be imported."""
    from cuprox import (
        CuproxError,
        InfeasibleError,
        UnboundedError,
        NumericalError,
    )
    
    # Verify inheritance
    assert issubclass(InfeasibleError, CuproxError)
    assert issubclass(UnboundedError, CuproxError)
    assert issubclass(NumericalError, CuproxError)


def test_info_function():
    """Verify info() function works."""
    import cuprox
    info = cuprox.info()
    
    assert isinstance(info, str)
    assert "cuProx version" in info
    assert "Python version" in info

