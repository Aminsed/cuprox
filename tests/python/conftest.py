"""
pytest configuration and fixtures for cuProx tests.
"""

import pathlib

import numpy as np
import pytest

try:
    from scipy import sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Determinism
# ============================================================================


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Seed every RNG before each test.

    Several tests draw random problem data and then assert on a tolerance.
    Without a fixed seed those assertions pass or fail by luck, and a failure
    cannot be reproduced from the report alone.
    """
    import random

    random.seed(0)
    np.random.seed(0)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(0)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_lp():
    """
    Simple LP problem for testing.

    minimize: -x - y
    subject to: x + 2y <= 10
                3x + y <= 15
                x, y >= 0

    Optimal: x=4, y=3, obj=-7
    """
    c = np.array([-1.0, -1.0])
    A = np.array(
        [
            [1.0, 2.0],
            [3.0, 1.0],
        ]
    )
    b = np.array([10.0, 15.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([np.inf, np.inf])

    return {
        "c": c,
        "A": A,
        "b": b,
        "lb": lb,
        "ub": ub,
        "expected_obj": -7.0,
        "expected_x": np.array([4.0, 3.0]),
    }


@pytest.fixture
def random_lp():
    """Generate a random feasible LP."""
    np.random.seed(42)
    n, m = 100, 50

    # Generate random feasible problem
    A = np.random.randn(m, n)
    x_feas = np.abs(np.random.randn(n))  # Feasible point
    b = A @ x_feas + 0.1  # Ensure strict feasibility
    c = np.random.randn(n)
    lb = np.zeros(n)
    ub = np.full(n, np.inf)

    return {
        "c": c,
        "A": A,
        "b": b,
        "lb": lb,
        "ub": ub,
    }


@pytest.fixture
def sparse_lp():
    """Generate a sparse LP."""
    if not HAS_SCIPY:
        pytest.skip("scipy not available")

    np.random.seed(42)
    n, m = 1000, 500

    A = sparse.random(m, n, density=0.01, format="csr")
    x_feas = np.abs(np.random.randn(n))
    b = A @ x_feas + 0.1
    c = np.random.randn(n)
    lb = np.zeros(n)
    ub = np.full(n, np.inf)

    return {
        "c": c,
        "A": A,
        "b": b,
        "lb": lb,
        "ub": ub,
    }


@pytest.fixture
def simple_qp():
    """
    Simple QP problem for testing.

    minimize: (1/2)(2x^2 + 2y^2) - 2x - 4y = x^2 + y^2 - 2x - 4y
    subject to: x + y <= 3
                x, y >= 0

    Unconstrained optimal: x=1, y=2
    Constraint x+y<=3 is not binding at (1,2) since 1+2=3, so optimal is (1,2), obj=-5
    """
    P = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    q = np.array([-2.0, -4.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([3.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([np.inf, np.inf])

    return {
        "P": P,
        "c": q,
        "A": A,
        "b": b,
        "lb": lb,
        "ub": ub,
        "expected_obj": -5.0,
        "expected_x": np.array([1.0, 2.0]),
    }


@pytest.fixture
def unconstrained_qp():
    """
    Unconstrained QP with closed-form solution.

    minimize: (1/2)x'Px + q'x
    where P = 2I, q = [-2, -4]

    Solution: x = P^{-1}(-q) = [1, 2], obj = -5
    """
    P = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    q = np.array([-2.0, -4.0])

    return {
        "P": P,
        "c": q,
        "expected_obj": -5.0,
        "expected_x": np.array([1.0, 2.0]),
    }


# ============================================================================
# Markers
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    config.addinivalue_line("markers", "integration: marks integration tests")


# ============================================================================
# Skip Conditions
# ============================================================================


@pytest.fixture
def requires_gpu():
    """Skip test if GPU is not available."""
    try:
        import cuprox

        if not cuprox.__cuda_available__:
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("cuprox C++ extension not built")


@pytest.fixture
def requires_scipy():
    """Skip test if scipy is not available."""
    if not HAS_SCIPY:
        pytest.skip("scipy not available")


# Suites that do not depend on which solver is underneath: pure Python API,
# model construction, risk maths, imports. Everything else asserts on solver
# behaviour -- statuses, iteration counts, achieved accuracy, training
# convergence -- and the CPU fallback is a different solver (scipy), so those
# cannot pass without a device.
_SOLVER_INDEPENDENT = (
    "test_import.py",
    "test_model.py",
    "test_finance_risk.py",
    "test_mpc_dynamics.py",
)


def pytest_collection_modifyitems(config, items):
    """Skip solver-dependent tests when there is no CUDA device.

    Keeping the CPU job to the solver-independent suites makes a green run mean
    something. Previously the whole suite ran against the scipy fallback and
    passed, which is how a completely non-functional CUDA path went unnoticed.
    """
    import cuprox

    if cuprox.__cuda_available__:
        return
    skip = pytest.mark.skip(reason="requires a CUDA device (running the CPU fallback)")
    for item in items:
        name = pathlib.Path(str(item.fspath)).name
        if name not in _SOLVER_INDEPENDENT:
            item.add_marker(skip)
