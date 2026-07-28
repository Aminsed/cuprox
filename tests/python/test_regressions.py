"""One test per defect that shipped in an earlier release.

Kept in its own file so it stays obvious why each of these exists, and so that
deleting one is a deliberate act rather than a cleanup.

Several of these describe the same underlying failure: the CUDA path was never
executed, so nothing downstream of it had ever run. The tests are written
against observable behaviour, not against the internals that happened to be
wrong at the time.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import cuprox
from cuprox import Model, Status, solve
from cuprox.exceptions import DimensionError, InvalidInputError

# --------------------------------------------------------------------------
# The GPU path
# --------------------------------------------------------------------------


def test_extension_is_present_and_used():
    """The compiled extension must be importable from the installed package.

    `pip install -e python/` never built it, so `_core` was absent,
    `__cuda_available__` was False, and every solve fell through to scipy.
    """
    from cuprox import _core

    assert _core is not None, "the compiled extension is missing"
    assert cuprox.__cuda_available__ is True
    assert _core.get_device_name()


def test_solver_calls_names_the_extension_actually_exports():
    """solver.py called _core.solve_lp / _core.solve_qp; the module exports
    solve_lp_pdhg / solve_qp_admm. Every GPU solve raised AttributeError and was
    silently swallowed by a bare `except Exception`."""
    from cuprox import _core

    assert hasattr(_core, "solve_lp_pdhg")
    assert hasattr(_core, "solve_qp_admm")


def test_gpu_failure_is_not_silently_swallowed(monkeypatch):
    """A failing GPU solve must raise, not quietly return a scipy answer
    labelled as GPU output."""
    from cuprox import solver as S

    def boom(*_a, **_k):
        raise RuntimeError("synthetic GPU failure")

    monkeypatch.setattr(S, "_solve_gpu", boom)
    with pytest.raises(RuntimeError, match="synthetic GPU failure"):
        solve(
            c=np.array([1.0, 1.0]),
            A=sparse.csr_matrix(np.array([[1.0, 1.0]])),
            b=np.array([1.0]),
            lb=np.zeros(2),
            ub=np.full(2, np.inf),
        )


# --------------------------------------------------------------------------
# Results coming back intact
# --------------------------------------------------------------------------


def test_solution_vector_is_not_a_constant():
    """pybind11 < 2.12 against NumPy >= 2 returns stride-0 arrays, so every
    element aliases element 0.

    This was especially hard to notice here: the objective is computed in C++
    and stayed correct while x came back as x[0] repeated.
    """
    rng = np.random.default_rng(0)
    n, m = 6, 4
    A = rng.random((m, n))
    c = -rng.random(n)
    r = solve(
        c=c,
        A=sparse.csr_matrix(A),
        constraint_l=np.full(m, 0.5),
        constraint_u=np.full(m, 3.0),
        lb=np.zeros(n),
        ub=np.full(n, 10.0),
        params={"tolerance": 1e-8, "max_iterations": 200_000},
    )
    x = np.asarray(r.x)
    assert x.strides == (x.itemsize,), "returned array has degenerate strides"
    assert np.std(x) > 0
    # The objective must be consistent with the vector it is reported alongside.
    assert abs(r.objective - float(c @ x)) < 1e-4


# --------------------------------------------------------------------------
# Constraint handling
# --------------------------------------------------------------------------


def test_lp_honours_inequality_rows():
    """PDHG only ever launched an equality-constraint kernel, and the binding
    copied b into both l and u. Any inequality was solved as an equality.

    The third row here is inactive at the optimum, so treating it as an
    equality gives a visibly different answer.
    """
    A = sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 1.0], [1.0, 1.0]]))
    r = solve(
        c=np.array([-1.0, -1.0]),
        A=A,
        constraint_l=np.full(3, -np.inf),
        constraint_u=np.array([10.0, 15.0, 100.0]),
        lb=np.zeros(2),
        ub=np.full(2, np.inf),
        params={"tolerance": 1e-8, "max_iterations": 200_000},
    )
    assert r.status is Status.OPTIMAL
    np.testing.assert_allclose(np.asarray(r.x), [4.0, 3.0], atol=1e-3)


def test_qp_honours_variable_bounds():
    """ADMM enforced only l <= Ax <= u; the variable box was ignored entirely,
    so solutions came back outside the bounds the caller declared while the
    status still said 'optimal'."""
    rng = np.random.default_rng(0)
    n, m = 6, 4
    A = rng.random((m, n))
    M = rng.standard_normal((n, n))
    P = M @ M.T + np.eye(n) * 0.5
    lb = np.zeros(n)
    ub = np.full(n, 10.0)
    r = solve(
        c=-rng.random(n),
        P=sparse.csr_matrix(P),
        A=sparse.csr_matrix(A),
        constraint_l=np.full(m, 0.5),
        constraint_u=np.full(m, 3.0),
        lb=lb,
        ub=ub,
        params={"tolerance": 1e-8, "max_iterations": 200_000},
    )
    x = np.asarray(r.x)
    assert np.all(x >= lb - 1e-9)
    assert np.all(x <= ub + 1e-9)


@pytest.mark.parametrize(("m", "n"), [(3, 2), (8, 4), (16, 8), (32, 4)])
def test_qp_accepts_more_constraints_than_variables(m: int, n: int):
    """ADMM sized its A^T workspace max(m, n) instead of n, so cuSPARSE rejected
    every problem with m > n -- which is most real QPs."""
    rng = np.random.default_rng(m * 100 + n)
    A = sparse.csr_matrix(rng.standard_normal((m, n)))
    P = sparse.csr_matrix(np.eye(n))
    r = solve(
        c=rng.standard_normal(n),
        P=P,
        A=A,
        constraint_l=np.full(m, -1.0),
        constraint_u=np.full(m, 1.0),
        lb=np.full(n, -10.0),
        ub=np.full(n, 10.0),
        params={"tolerance": 1e-6, "max_iterations": 50_000},
    )
    assert r.status in (Status.OPTIMAL, Status.MAX_ITERATIONS)
    assert np.isfinite(r.objective)


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------


def test_model_solve_accepts_warm_start():
    """Model.solve() passed warm_start= to a solve() that did not accept it, so
    the README's very first example died with TypeError."""
    model = Model()
    x = model.add_var(lb=0, name="x")
    y = model.add_var(lb=0, name="y")
    model.add_constr(x + 2 * y <= 10)
    model.add_constr(3 * x + y <= 15)
    model.minimize(-x - y)
    r = model.solve()
    assert r.status is Status.OPTIMAL
    assert abs(r.objective - (-7.0)) < 1e-2


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"c": np.array([1.0, np.nan])}, InvalidInputError),
        (
            {"c": np.array([1.0, 2.0]), "lb": np.array([5.0, 0.0]), "ub": np.array([3.0, 10.0])},
            InvalidInputError,
        ),
    ],
)
def test_bad_input_is_rejected(kwargs, exc):
    base = {"A": sparse.csr_matrix(np.array([[1.0, 2.0]])), "b": np.array([10.0])}
    with pytest.raises(exc):
        solve(**{**base, **kwargs})


def test_dimension_mismatch_is_rejected():
    with pytest.raises(DimensionError):
        solve(
            c=np.array([1.0, 2.0]),
            A=sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]])),
            b=np.array([10.0]),
        )


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


def test_maximize_reports_the_right_sign():
    """Model.maximize returned the negated objective.

    _to_standard_form negates c to turn a maximisation into the minimisation the
    solver expects, and nothing negated the objective back, so every maximize()
    reported the correct magnitude with the wrong sign. x was always right,
    which made it easy to miss.
    """
    model = Model()
    x = model.add_var(lb=0, name="x")
    y = model.add_var(lb=0, name="y")
    model.add_constr(x + 2 * y <= 10)
    model.add_constr(3 * x + y <= 15)
    model.maximize(5 * x + 4 * y)

    r = model.fit if False else model.solve()
    assert r.status is Status.OPTIMAL
    assert r.objective > 0, "maximize returned a negated objective"
    assert abs(r.objective - 32.0) < 1e-2
    np.testing.assert_allclose(np.asarray(r.x), [4.0, 3.0], atol=1e-2)
    # And the objective must agree with the point it is reported alongside.
    assert abs(r.objective - float(5 * r.x[0] + 4 * r.x[1])) < 1e-3


def test_badly_scaled_max_sharpe_converges():
    """Daily excess returns are ~1e-3, which forced the transformed variable to
    ~1e3 against a covariance of ~1e-4. ADMM does not equilibrate, so it stalled
    and returned a portfolio concentrated in a negative-return asset."""
    from cuprox.finance import Portfolio

    # Same generator as the finance fixtures, so this is the data that actually
    # triggered the stall: one asset with positive excess return, one negative.
    np.random.seed(42)
    r = np.random.randn(252, 3) * 0.02
    r[:, 0] += 0.0005
    r[:, 1] += 0.0003
    r[:, 2] += 0.0001

    port = Portfolio(r)
    assert (port.expected_returns > 0).any() and (
        port.expected_returns < 0
    ).any(), "fixture must contain both signs for this regression to be meaningful"
    best = port.optimize(method="max_sharpe")
    assert best.status == "optimal"
    # A max-Sharpe portfolio must not beat itself with minimum variance.
    assert best.sharpe_ratio >= port.optimize(method="min_variance").sharpe_ratio
    # ... and must not hold an asset whose excess return is negative.
    negative = port.expected_returns < 0
    assert np.all(np.asarray(best.weights)[negative] < 1e-6)


def test_qp_gradients_match_closed_form():
    """The differentiable layer's headline feature. For an unconstrained QP the
    solution is x = -P^-1 q and the Jacobian is exactly -P^-1."""
    torch = pytest.importorskip("torch")
    from cuprox.torch import solve_qp

    torch.manual_seed(0)
    n = 4
    M = torch.randn(n, n, dtype=torch.float64)
    P = M @ M.T + 3.0 * torch.eye(n, dtype=torch.float64)
    q = torch.randn(n, dtype=torch.float64, requires_grad=True)

    x = solve_qp(P, q)
    expected = -np.linalg.solve(P.numpy(), q.detach().numpy())
    np.testing.assert_allclose(x.detach().numpy(), expected, atol=1e-6)

    jac = np.zeros((n, n))
    for i in range(n):
        qi = q.clone().detach().requires_grad_(True)
        seed = torch.zeros(n, dtype=torch.float64)
        seed[i] = 1.0
        solve_qp(P, qi).backward(seed)
        jac[i] = qi.grad.detach().numpy()
    np.testing.assert_allclose(jac, -np.linalg.inv(P.numpy()), atol=1e-6)


# ---------------------------------------------------------------------------
# Defects found in external review, 2026-07-28
# ---------------------------------------------------------------------------


def test_cpu_lp_reports_optimal_not_numerical_error():
    """The CPU LP path mapped SciPy codes onto Status members that do not exist.

    The resulting AttributeError was raised inside a broad ``except Exception``
    and converted to NUMERICAL_ERROR, so this feasible LP -- whose optimum is
    -7 at (4, 3) -- came back with a nan objective and a zero vector.
    """
    result = cuprox.solve(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, 2.0], [3.0, 1.0]]),
        b=np.array([10.0, 15.0]),
        lb=np.zeros(2),
        params={"device": "cpu"},
    )
    assert result.status == Status.OPTIMAL
    assert result.objective == pytest.approx(-7.0, abs=1e-6)


def test_cpu_path_does_not_mask_programming_errors():
    """A bug in the fallback must not be reported as a numerical failure.

    The broad except is why the broken status map went unnoticed: every LP came
    back "numerical error", which reads as a hard problem rather than a defect.
    """
    import inspect

    import cuprox.solver as solver_module

    source = inspect.getsource(solver_module)
    assert "except (AttributeError, TypeError, NameError, ImportError):" in source
    # And the names that never existed must not be referenced as code.
    assert "2: Status.INFEASIBLE" not in source
    assert "3: Status.UNBOUNDED" not in source


def test_warm_start_is_rejected_rather_than_ignored():
    """warm_start was accepted by the signature and read by nothing."""
    with pytest.raises(NotImplementedError):
        cuprox.solve(
            c=np.zeros(2),
            A=np.eye(2),
            b=np.ones(2),
            warm_start=np.zeros(2),
        )


def test_from_matrices_keeps_P():
    """Model.from_matrices stored its matrices where solve() never looked.

    solve() always rebuilt a linear model through _to_standard_form(), which
    has no algebraic objective for such a model and drops P, so this QP raised
    a raw cuSPARSE error instead of returning (0.5, 0.5).
    """
    model = cuprox.Model.from_matrices(
        c=np.zeros(2),
        A_ub=np.array([[-1.0, -1.0]]),
        b_ub=np.array([-1.0]),
        P=np.eye(2),
    )
    result = model.solve()
    assert result.status == Status.OPTIMAL
    assert result.objective == pytest.approx(0.25, abs=1e-3)
    assert result.x == pytest.approx([0.5, 0.5], abs=1e-2)


def test_pdhg_iteration_count_is_not_doubled():
    """PDHG incremented iter_ in iterate() and again in the loop calling it.

    The solver therefore ran half the requested iterations and reported a count
    that skipped. A capped, deliberately unconverged run must report the cap.
    """
    rng = np.random.default_rng(0)
    cap = 100
    result = cuprox.solve(
        c=rng.standard_normal(200),
        A=rng.standard_normal((100, 200)),
        b=np.ones(100),
        lb=np.full(200, -1.0),
        ub=np.full(200, 1.0),
        params={"max_iterations": cap, "tolerance": 1e-14},
    )
    assert result.iterations == cap


def test_torch_layer_does_not_claim_gradients_it_does_not_return():
    """backward returns None for every constraint parameter.

    The module and class docstrings advertised gradients for A, b, G, h, lb and
    ub. Only P and q are differentiated.
    """
    import inspect

    from cuprox.torch import functions as fn

    source = inspect.getsource(fn)
    assert "Gradients for all problem parameters" not in source
    assert "Supports gradients for ALL problem parameters" not in source
