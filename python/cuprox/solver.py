"""cuProx Solver Interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import sparse

from .exceptions import DimensionError, InvalidInputError
from .result import SolveResult, Status

# The extension reports status as a string; Status values happen to use the same
# spellings, so the mapping is a lookup rather than a translation table. The
# previous code mapped integer codes onto Status.INFEASIBLE and Status.UNBOUNDED,
# neither of which exists on the enum -- another reason the GPU path could never
# have completed a single solve.
_STATUS_FROM_CORE = {s.value: s for s in Status}


def solve(
    c: np.ndarray,
    A: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
    b: Optional[np.ndarray] = None,
    P: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    constraint_l: Optional[np.ndarray] = None,
    constraint_u: Optional[np.ndarray] = None,
    constraint_senses: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    warm_start: Optional[np.ndarray] = None,
) -> SolveResult:
    """Solve LP or QP."""
    import time

    start_time = time.perf_counter()
    params = params or {}
    max_iters = params.get("max_iterations", params.get("max_iters", 10000))
    tol = params.get("tolerance", params.get("tol", 1e-4))
    verbose = params.get("verbose", False)
    device = params.get("device", "gpu")

    c = np.asarray(c, dtype=np.float64).ravel()
    n = len(c)
    if n == 0:
        raise InvalidInputError("objective vector c is empty")
    if not np.all(np.isfinite(c)):
        raise InvalidInputError("objective vector c contains NaN or infinity")
    lb = np.zeros(n) if lb is None else np.asarray(lb, dtype=np.float64).ravel()
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=np.float64).ravel()

    if len(lb) != n or len(ub) != n:
        raise DimensionError(f"Bounds mismatch: lb={len(lb)}, ub={len(ub)}, n={n}")
    if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
        raise InvalidInputError("variable bounds contain NaN")
    if np.any(lb > ub):
        bad = int(np.argmax(lb > ub))
        raise InvalidInputError(
            f"variable bounds cross at index {bad}: lb={lb[bad]} > ub={ub[bad]}"
        )

    if A is not None:
        A = A.tocsr() if sparse.issparse(A) else np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        m = A.shape[0]
        if A.shape[1] != n:
            raise DimensionError(f"A columns {A.shape[1]} != n={n}")
    else:
        m = 0
        A = sparse.csr_matrix((0, n))

    if constraint_senses is not None:
        if b is None:
            raise InvalidInputError("b required with constraint_senses")
        b = np.asarray(b, dtype=np.float64).ravel()
        if len(b) != m:
            raise DimensionError(f"b has {len(b)} entries but A has {m} rows")
        constr_l = np.full(m, -np.inf)
        constr_u = np.full(m, np.inf)
        for i, sense in enumerate(constraint_senses):
            if sense in ("=", "=="):
                constr_l[i] = constr_u[i] = b[i]
            elif sense in ("<=", "<"):
                constr_u[i] = b[i]
            elif sense in (">=", ">"):
                constr_l[i] = b[i]
    elif constraint_l is not None or constraint_u is not None:
        constr_l = (
            np.asarray(constraint_l, dtype=np.float64)
            if constraint_l is not None
            else np.full(m, -np.inf)
        )
        constr_u = (
            np.asarray(constraint_u, dtype=np.float64)
            if constraint_u is not None
            else np.full(m, np.inf)
        )
    elif b is not None:
        b = np.asarray(b, dtype=np.float64).ravel()
        if len(b) != m:
            raise DimensionError(f"b has {len(b)} entries but A has {m} rows")
        if not np.all(np.isfinite(b)):
            raise InvalidInputError("right-hand side b contains NaN or infinity")
        constr_l = constr_u = b
    else:
        constr_l = constr_u = np.array([])

    is_qp = P is not None
    if is_qp:
        P = P.tocsr() if sparse.issparse(P) else np.asarray(P, dtype=np.float64)
        if P.shape != (n, n):
            raise DimensionError(f"P must be ({n},{n}), got {P.shape}")

    from . import _core

    use_gpu = _core is not None and getattr(_core, "cuda_available", False) and device != "cpu"

    if use_gpu:
        result = _solve_gpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp)
    else:
        result = _solve_cpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp)

    result.solve_time = time.perf_counter() - start_time
    return result


def _solve_gpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp):
    """Solve on the GPU.

    Calls the compiled extension directly. Any failure propagates: silently
    falling back to scipy here is how this library previously reported
    "GPU-accelerated" results that were computed on the CPU.
    """
    from . import _core

    n = len(c)
    A_csr = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(A)
    m = A_csr.shape[0]

    # The kernels use a large finite sentinel for "unbounded"; an actual +/-inf
    # would propagate NaN through the projections.
    big = 1e20
    lb_d = np.where(np.isneginf(lb), -big, np.where(np.isposinf(lb), big, lb)).astype(np.float64)
    ub_d = np.where(np.isposinf(ub), big, np.where(np.isneginf(ub), -big, ub)).astype(np.float64)
    l_d = np.where(
        np.isneginf(constr_l), -big, np.where(np.isposinf(constr_l), big, constr_l)
    ).astype(np.float64)
    u_d = np.where(
        np.isposinf(constr_u), big, np.where(np.isneginf(constr_u), -big, constr_u)
    ).astype(np.float64)
    l_d = np.ascontiguousarray(np.atleast_1d(l_d))
    u_d = np.ascontiguousarray(np.atleast_1d(u_d))

    A_ro = A_csr.indptr.astype(np.int32)
    A_ci = A_csr.indices.astype(np.int32)
    A_va = A_csr.data.astype(np.float64)

    if is_qp:
        # ADMM enforces l <= Ax <= u only; it has no separate handling for
        # variable bounds. Encode them the way OSQP does, as identity rows
        # appended to A, so lb <= x <= ub becomes part of the row system.
        # Passing them as var_lb/var_ub instead silently returned points
        # outside the box while reporting "optimal".
        # Only variables that actually have a bound need a row. Adding a row for
        # a free variable contributes nothing and drags a huge sentinel into the
        # residual norms, which wrecks the convergence test.
        bounded = np.flatnonzero(np.isfinite(lb) | np.isfinite(ub))
        if bounded.size:
            box = sparse.csr_matrix(
                (np.ones(bounded.size), (np.arange(bounded.size), bounded)),
                shape=(bounded.size, n),
            )
            A_aug = sparse.vstack([A_csr, box], format="csr")
            l_aug = np.concatenate([l_d, lb_d[bounded]])
            u_aug = np.concatenate([u_d, ub_d[bounded]])
        else:
            A_aug, l_aug, u_aug = A_csr, l_d, u_d
        A_ro = A_aug.indptr.astype(np.int32)
        A_ci = A_aug.indices.astype(np.int32)
        A_va = A_aug.data.astype(np.float64)
        m_aug = A_aug.shape[0]

        P_csr = P.tocsr() if sparse.issparse(P) else sparse.csr_matrix(P)
        out = _core.solve_qp_admm(
            P_row_offsets=P_csr.indptr.astype(np.int32),
            P_col_indices=P_csr.indices.astype(np.int32),
            P_values=P_csr.data.astype(np.float64),
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            A_values=A_va,
            q=np.ascontiguousarray(c, dtype=np.float64),
            l=l_aug,
            u=u_aug,
            # The box is already in A, so leave the solver's own box inert.
            var_lb=np.full(n, -big),
            var_ub=np.full(n, big),
            P_n=n,
            A_m=m_aug,
            A_n=n,
            max_iters=max_iters,
            eps_abs=tol,
            eps_rel=tol,
            verbose=verbose,
        )
    else:
        out = _core.solve_lp_pdhg(
            row_offsets=A_ro,
            col_indices=A_ci,
            values=A_va,
            c=np.ascontiguousarray(c, dtype=np.float64),
            l=l_d,
            u=u_d,
            lb=lb_d,
            ub=ub_d,
            num_rows=m,
            num_cols=n,
            max_iters=max_iters,
            eps_abs=tol,
            eps_rel=tol,
            verbose=verbose,
        )

    # Variable bounds are enforced as rows of A, so they are satisfied only to
    # the primal tolerance. Callers reasonably expect a point inside the box they
    # declared, so project onto it; the move is at most one tolerance and cannot
    # make the box constraint worse.
    x_out = np.clip(np.asarray(out["x"]), lb, ub)

    status = _STATUS_FROM_CORE.get(out["status"], Status.UNSOLVED)
    # For a QP the dual has an entry per augmented row; trim the identity block
    # so y matches the constraints the caller supplied.
    y_out = np.asarray(out["y"])[:m]
    return SolveResult(
        status=status,
        objective=out["objective"],
        x=x_out,
        y=y_out,
        iterations=out["iterations"],
        solve_time=out["solve_time"],
    )


def _solve_cpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp):
    n, m = len(c), A.shape[0] if hasattr(A, "shape") and A.shape[0] > 0 else 0

    if is_qp:
        try:
            from scipy.optimize import minimize

            P_dense = P.toarray() if sparse.issparse(P) else np.asarray(P)
            bounds = list(zip(lb, ub))
            constraints = []
            if m > 0:
                A_dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
                eq_mask = np.abs(constr_l - constr_u) < 1e-10
                if eq_mask.any():
                    A_eq, b_eq = A_dense[eq_mask], constr_l[eq_mask]
                    constraints.append(
                        {
                            "type": "eq",
                            "fun": lambda x, A=A_eq, b=b_eq: A @ x - b,
                            "jac": lambda x, A=A_eq: A,
                        }
                    )
                ineq_mask = ~eq_mask
                if ineq_mask.any():
                    A_ineq = A_dense[ineq_mask]
                    l_ineq, u_ineq = constr_l[ineq_mask], constr_u[ineq_mask]
                    if not np.all(np.isinf(l_ineq)):
                        constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda x, A=A_ineq, l=l_ineq: A @ x - l,
                                "jac": lambda x, A=A_ineq: A,
                            }
                        )
                    if not np.all(np.isinf(u_ineq)):
                        constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda x, A=A_ineq, u=u_ineq: u - A @ x,
                                "jac": lambda x, A=A_ineq: -A,
                            }
                        )
            x0 = np.clip(np.zeros(n), lb, ub)
            x0 = np.where(np.isinf(lb), 0, x0)
            x0 = np.where(np.isinf(ub), x0, np.minimum(x0, ub))
            result = minimize(
                lambda x: 0.5 * x @ P_dense @ x + c @ x,
                x0,
                method="SLSQP",
                jac=lambda x: P_dense @ x + c,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iters, "ftol": tol},
            )
            return SolveResult(
                status=Status.OPTIMAL if result.success else Status.MAX_ITERATIONS,
                objective=result.fun,
                x=result.x,
                y=np.zeros(m),
                iterations=result.nit,
                solve_time=0.0,
            )
        except Exception as e:
            if verbose:
                print(f"scipy QP failed: {e}")
            return SolveResult(
                status=Status.NUMERICAL_ERROR,
                objective=float("nan"),
                x=np.zeros(n),
                y=np.zeros(m),
                iterations=0,
                solve_time=0.0,
            )
    else:
        try:
            from scipy.optimize import linprog

            A_ub, b_ub, A_eq, b_eq = None, None, None, None
            if m > 0:
                A_dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
                eq_mask = np.abs(constr_l - constr_u) < 1e-10
                if eq_mask.any():
                    A_eq, b_eq = A_dense[eq_mask], constr_l[eq_mask]
                ineq_mask = ~eq_mask
                if ineq_mask.any():
                    A_ineq = A_dense[ineq_mask]
                    l_ineq, u_ineq = constr_l[ineq_mask], constr_u[ineq_mask]
                    rows, rhs = [], []
                    if (~np.isinf(u_ineq)).any():
                        rows.append(A_ineq[~np.isinf(u_ineq)])
                        rhs.append(u_ineq[~np.isinf(u_ineq)])
                    if (~np.isinf(l_ineq)).any():
                        rows.append(-A_ineq[~np.isinf(l_ineq)])
                        rhs.append(-l_ineq[~np.isinf(l_ineq)])
                    if rows:
                        A_ub, b_ub = np.vstack(rows), np.concatenate(rhs)
            bounds = [
                (l if not np.isinf(l) else None, u if not np.isinf(u) else None)
                for l, u in zip(lb, ub)
            ]
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"maxiter": max_iters, "tol": tol},
            )
            status_map = {
                0: Status.OPTIMAL,
                1: Status.MAX_ITERATIONS,
                2: Status.INFEASIBLE,
                3: Status.UNBOUNDED,
            }
            return SolveResult(
                status=status_map.get(result.status, Status.NUMERICAL_ERROR),
                objective=result.fun if result.success else float("nan"),
                x=result.x if result.x is not None else np.zeros(n),
                y=np.zeros(m),
                iterations=getattr(result, "nit", 0),
                solve_time=0.0,
            )
        except Exception as e:
            if verbose:
                print(f"scipy LP failed: {e}")
            return SolveResult(
                status=Status.NUMERICAL_ERROR,
                objective=float("nan"),
                x=np.zeros(n),
                y=np.zeros(m),
                iterations=0,
                solve_time=0.0,
            )


def solve_batch(
    problems: List[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
) -> List[SolveResult]:
    """Solve many problems, in one GPU launch where possible.

    Problems are solved one after another. A batched PDHG kernel exists and is
    reachable with ``params={"batched": True}``, but it is **experimental and
    currently slower and less reliable** than the sequential path: it reports
    status through a `cudaMemset` that writes bytes rather than ints, its
    convergence test ignores problem scale, and its residual is computed from
    an aliased buffer. Until those are fixed the default stays sequential,
    which is correct.

    Parameters
    ----------
    problems : list of dict
        Each entry takes the same keys as :func:`solve`.
    params : dict, optional
        Solver parameters, applied to every problem in the batch.

    Returns
    -------
    list of SolveResult
        One result per input problem, in order.
    """
    params = params or {}
    if not problems:
        return []

    if params.get("batched"):
        batched = _try_batched_lp(problems, params)
        if batched is not None:
            return batched

    return [
        solve(
            c=p.get("c"),
            A=p.get("A"),
            b=p.get("b"),
            P=p.get("P"),
            lb=p.get("lb"),
            ub=p.get("ub"),
            constraint_l=p.get("constraint_l"),
            constraint_u=p.get("constraint_u"),
            constraint_senses=p.get("constraint_senses"),
            params=params,
        )
        for p in problems
    ]


def _shares_structure(problems: List[Dict[str, Any]]) -> bool:
    """True when every problem is an equality LP over the same A, lb and ub."""
    first = problems[0]
    if first.get("A") is None or first.get("b") is None:
        return False
    A0 = first["A"].tocsr() if sparse.issparse(first["A"]) else sparse.csr_matrix(first["A"])
    for p in problems:
        if p.get("P") is not None or p.get("constraint_senses") is not None:
            return False
        if p.get("constraint_l") is not None or p.get("constraint_u") is not None:
            return False
        if p.get("A") is None or p.get("b") is None or p.get("c") is None:
            return False
        A = p["A"].tocsr() if sparse.issparse(p["A"]) else sparse.csr_matrix(p["A"])
        if A.shape != A0.shape or A.nnz != A0.nnz:
            return False
        if not (
            np.array_equal(A.indptr, A0.indptr)
            and np.array_equal(A.indices, A0.indices)
            and np.allclose(A.data, A0.data)
        ):
            return False
        if not _same_bounds(p, first):
            return False
    return True


def _same_bounds(p: Dict[str, Any], first: Dict[str, Any]) -> bool:
    for key in ("lb", "ub"):
        a, b = p.get(key), first.get(key)
        if (a is None) != (b is None):
            return False
        if a is not None and not np.allclose(np.ravel(a), np.ravel(b)):
            return False
    return True


def _try_batched_lp(
    problems: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Optional[List[SolveResult]]:
    """Run the whole batch through one kernel, or return None if it does not fit.

    Experimental; see solve_batch for the outstanding defects in the kernel.
    """
    from . import _core

    if _core is None or not getattr(_core, "cuda_available", False):
        return None
    if params.get("device") == "cpu" or len(problems) < 2:
        return None
    if not hasattr(_core, "solve_batch_lp_pdhg") or not _shares_structure(problems):
        return None

    first = problems[0]
    A = first["A"].tocsr() if sparse.issparse(first["A"]) else sparse.csr_matrix(first["A"])
    m, n = A.shape
    k = len(problems)

    c_batch = np.ascontiguousarray(
        np.stack([np.asarray(p["c"], dtype=np.float64).ravel() for p in problems])
    )
    b_batch = np.ascontiguousarray(
        np.stack([np.asarray(p["b"], dtype=np.float64).ravel() for p in problems])
    )
    if c_batch.shape != (k, n) or b_batch.shape != (k, m):
        return None

    big = 1e20
    lb = first.get("lb")
    ub = first.get("ub")
    lb = np.zeros(n) if lb is None else np.asarray(lb, dtype=np.float64).ravel()
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=np.float64).ravel()
    lb_d = np.where(np.isneginf(lb), -big, lb).astype(np.float64)
    ub_d = np.where(np.isposinf(ub), big, ub).astype(np.float64)

    out = _core.solve_batch_lp_pdhg(
        row_offsets=A.indptr.astype(np.int32),
        col_indices=A.indices.astype(np.int32),
        values=A.data.astype(np.float64),
        c_batch=c_batch,
        b_batch=b_batch,
        lb=lb_d,
        ub=ub_d,
        batch_size=k,
        num_rows=m,
        num_cols=n,
        max_iters=params.get("max_iterations", params.get("max_iters", 5000)),
        eps_abs=params.get("tolerance", params.get("tol", 1e-5)),
        eps_rel=params.get("tolerance", params.get("tol", 1e-5)),
        verbose=params.get("verbose", False),
    )

    xs = np.asarray(out["x"])
    objs = np.asarray(out["objectives"])
    stats = np.asarray(out["statuses"])
    iters = np.asarray(out["iterations"])
    per_problem = float(out["solve_time"]) / k
    codes = list(Status)
    return [
        SolveResult(
            status=codes[int(stats[i])] if 0 <= int(stats[i]) < len(codes) else Status.UNSOLVED,
            objective=float(objs[i]),
            x=np.clip(xs[i], lb, ub),
            y=np.zeros(m),
            iterations=int(iters[i]),
            solve_time=per_problem,
        )
        for i in range(k)
    ]
