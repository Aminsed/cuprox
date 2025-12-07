"""cuProx Solver Interface."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy import sparse
from .exceptions import InvalidInputError, DimensionError
from .result import SolveResult, Status


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
) -> SolveResult:
    """Solve LP or QP."""
    import time
    start_time = time.perf_counter()
    params = params or {}
    max_iters = params.get('max_iterations', params.get('max_iters', 10000))
    tol = params.get('tolerance', params.get('tol', 1e-4))
    verbose = params.get('verbose', False)
    device = params.get('device', 'gpu')

    c = np.asarray(c, dtype=np.float64).ravel()
    n = len(c)
    lb = np.zeros(n) if lb is None else np.asarray(lb, dtype=np.float64).ravel()
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=np.float64).ravel()

    if len(lb) != n or len(ub) != n:
        raise DimensionError(f"Bounds mismatch: lb={len(lb)}, ub={len(ub)}, n={n}")

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
        constr_l = np.full(m, -np.inf)
        constr_u = np.full(m, np.inf)
        for i, sense in enumerate(constraint_senses):
            if sense in ('=', '=='):
                constr_l[i] = constr_u[i] = b[i]
            elif sense in ('<=', '<'):
                constr_u[i] = b[i]
            elif sense in ('>=', '>'):
                constr_l[i] = b[i]
    elif constraint_l is not None or constraint_u is not None:
        constr_l = np.asarray(constraint_l, dtype=np.float64) if constraint_l is not None else np.full(m, -np.inf)
        constr_u = np.asarray(constraint_u, dtype=np.float64) if constraint_u is not None else np.full(m, np.inf)
    elif b is not None:
        b = np.asarray(b, dtype=np.float64).ravel()
        constr_l = constr_u = b
    else:
        constr_l = constr_u = np.array([])

    is_qp = P is not None
    if is_qp:
        P = P.tocsr() if sparse.issparse(P) else np.asarray(P, dtype=np.float64)
        if P.shape != (n, n):
            raise DimensionError(f"P must be ({n},{n}), got {P.shape}")

    use_gpu = False
    try:
        from . import _core
        if _core.cuda_available and device != 'cpu':
            use_gpu = True
    except ImportError:
        pass

    if use_gpu:
        result = _solve_gpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp)
    else:
        result = _solve_cpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp)

    result.solve_time = time.perf_counter() - start_time
    return result


def _solve_gpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp):
    from . import _core
    n, m = len(c), A.shape[0]
    A_csr = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(A)
    row_offsets = A_csr.indptr.astype(np.int32)
    col_indices = A_csr.indices.astype(np.int32)
    values = A_csr.data.astype(np.float64)
    lb_safe = np.where(np.isinf(lb), -1e20, lb)
    ub_safe = np.where(np.isinf(ub), 1e20, ub)

    if is_qp:
        P_csr = P.tocsr() if sparse.issparse(P) else sparse.csr_matrix(P)
        P_row = P_csr.indptr.astype(np.int32)
        P_col = P_csr.indices.astype(np.int32)
        P_val = P_csr.data.astype(np.float64)
        qp_l = np.where(np.isinf(constr_l), -1e20, constr_l).astype(np.float64)
        qp_u = np.where(np.isinf(constr_u), 1e20, constr_u).astype(np.float64)
        try:
            x, y, obj, iters, status_code = _core.solve_qp(
                m, n, row_offsets, col_indices, values,
                P_row, P_col, P_val, c, qp_l, qp_u, lb_safe, ub_safe,
                max_iters, tol, tol, verbose
            )
        except Exception:
            return _solve_cpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp)
    else:
        lp_l = np.where(np.isinf(constr_l), -1e20, constr_l).astype(np.float64)
        lp_u = np.where(np.isinf(constr_u), 1e20, constr_u).astype(np.float64)
        try:
            x, y, obj, iters, status_code = _core.solve_lp(
                m, n, row_offsets, col_indices, values,
                c, lp_l, lp_u, lb_safe, ub_safe, max_iters, tol, tol, verbose
            )
        except Exception:
            return _solve_cpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp)

    status_map = {0: Status.OPTIMAL, 1: Status.MAX_ITERATIONS, 2: Status.INFEASIBLE, 3: Status.UNBOUNDED, 4: Status.NUMERICAL_ERROR}
    return SolveResult(status=status_map.get(status_code, Status.UNKNOWN), objective=obj, x=x, y=y, iterations=iters, solve_time=0.0)


def _solve_cpu(c, A, P, lb, ub, constr_l, constr_u, max_iters, tol, verbose, is_qp):
    n, m = len(c), A.shape[0] if hasattr(A, 'shape') and A.shape[0] > 0 else 0

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
                    constraints.append({'type': 'eq', 'fun': lambda x, A=A_eq, b=b_eq: A @ x - b, 'jac': lambda x, A=A_eq: A})
                ineq_mask = ~eq_mask
                if ineq_mask.any():
                    A_ineq = A_dense[ineq_mask]
                    l_ineq, u_ineq = constr_l[ineq_mask], constr_u[ineq_mask]
                    if not np.all(np.isinf(l_ineq)):
                        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_ineq, l=l_ineq: A @ x - l, 'jac': lambda x, A=A_ineq: A})
                    if not np.all(np.isinf(u_ineq)):
                        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_ineq, u=u_ineq: u - A @ x, 'jac': lambda x, A=A_ineq: -A})
            x0 = np.clip(np.zeros(n), lb, ub)
            x0 = np.where(np.isinf(lb), 0, x0)
            x0 = np.where(np.isinf(ub), x0, np.minimum(x0, ub))
            result = minimize(lambda x: 0.5 * x @ P_dense @ x + c @ x, x0, method='SLSQP',
                              jac=lambda x: P_dense @ x + c, bounds=bounds, constraints=constraints,
                              options={'maxiter': max_iters, 'ftol': tol})
            return SolveResult(status=Status.OPTIMAL if result.success else Status.MAX_ITERATIONS,
                               objective=result.fun, x=result.x, y=np.zeros(m), iterations=result.nit, solve_time=0.0)
        except Exception as e:
            if verbose: print(f"scipy QP failed: {e}")
            return SolveResult(status=Status.NUMERICAL_ERROR, objective=float('nan'), x=np.zeros(n), y=np.zeros(m), iterations=0, solve_time=0.0)
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
            bounds = [(l if not np.isinf(l) else None, u if not np.isinf(u) else None) for l, u in zip(lb, ub)]
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                             options={'maxiter': max_iters, 'tol': tol})
            status_map = {0: Status.OPTIMAL, 1: Status.MAX_ITERATIONS, 2: Status.INFEASIBLE, 3: Status.UNBOUNDED}
            return SolveResult(status=status_map.get(result.status, Status.NUMERICAL_ERROR),
                               objective=result.fun if result.success else float('nan'),
                               x=result.x if result.x is not None else np.zeros(n),
                               y=np.zeros(m), iterations=getattr(result, 'nit', 0), solve_time=0.0)
        except Exception as e:
            if verbose: print(f"scipy LP failed: {e}")
            return SolveResult(status=Status.NUMERICAL_ERROR, objective=float('nan'), x=np.zeros(n), y=np.zeros(m), iterations=0, solve_time=0.0)


def solve_batch(problems: List[Dict[str, Any]], params: Optional[Dict[str, Any]] = None) -> List[SolveResult]:
    """Solve multiple problems in batch."""
    return [solve(c=p.get('c'), A=p.get('A'), b=p.get('b'), P=p.get('P'), lb=p.get('lb'), ub=p.get('ub'),
                  constraint_l=p.get('constraint_l'), constraint_u=p.get('constraint_u'),
                  constraint_senses=p.get('constraint_senses'), params=params) for p in problems]
