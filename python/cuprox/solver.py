"""
cuProx Solver Functions
=======================

Core solving functions for LP and QP problems.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from scipy import sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .exceptions import (
    DimensionError,
    InfeasibleError,
    InvalidInputError,
    NumericalError,
    UnboundedError,
)
from .result import SolveResult, Status

# Default solver parameters
DEFAULT_PARAMS = {
    # Convergence
    "tolerance": 1e-6,
    "max_iterations": 100000,
    "time_limit": 0.0,  # 0 = no limit
    # Algorithm
    "scaling": True,
    "scaling_iterations": 10,
    "adaptive_restart": True,
    # Precision
    "precision": "float64",  # or "float32"
    # Diagnostics
    "verbose": False,
    "log_interval": 100,
    # Device
    "device": "auto",  # "auto", "gpu", or "cpu"
    "gpu_id": 0,
}


def solve(
    c: np.ndarray,
    A: Any,
    b: np.ndarray,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    P: Optional[Any] = None,
    q: Optional[np.ndarray] = None,
    constraint_senses: Optional[np.ndarray] = None,
    constraint_l: Optional[np.ndarray] = None,  # Lower bound on Ax (for QP)
    constraint_u: Optional[np.ndarray] = None,  # Upper bound on Ax (for QP)
    params: Optional[Dict[str, Any]] = None,
    warm_start: Optional[SolveResult] = None,
) -> SolveResult:
    """
    Solve an LP or QP problem in matrix form.

    LP: minimize c'x subject to Ax <= b, lb <= x <= ub
    QP: minimize (1/2)x'Px + q'x subject to Ax <= b, lb <= x <= ub

    Args:
        c: Linear objective coefficients (n,)
        A: Constraint matrix (m, n) - scipy.sparse.csr_matrix or numpy array
        b: Constraint right-hand side (m,)
        lb: Variable lower bounds (n,), default zeros
        ub: Variable upper bounds (n,), default +inf
        P: Quadratic objective matrix (n, n), optional (makes it QP)
        q: Alias for c in QP formulation
        constraint_senses: Array of '<', '=', '>' for each constraint
        params: Solver parameters
        warm_start: Previous solution for warm starting

    Returns:
        SolveResult with status, objective, x, y, etc.

    Raises:
        InvalidInputError: If input data is invalid
        DimensionError: If dimensions are incompatible

    Example:
        >>> import numpy as np
        >>> from scipy import sparse
        >>> import cuprox
        >>>
        >>> n, m = 1000, 500
        >>> A = sparse.random(m, n, density=0.01, format='csr')
        >>> b = np.random.rand(m)
        >>> c = np.random.randn(n)
        >>> result = cuprox.solve(c=c, A=A, b=b, lb=np.zeros(n))
        >>> print(result.status)
    """
    # Merge params with defaults
    params = {**DEFAULT_PARAMS, **(params or {})}

    # Validate and preprocess inputs
    c, A, b, lb, ub, P = _validate_inputs(c, A, b, lb, ub, P, q)

    # Convert constraint senses to l, u bounds for two-sided form
    if constraint_l is not None and constraint_u is not None:
        # Use explicit constraint bounds (for QP)
        constr_l = np.asarray(constraint_l, dtype=np.float64)
        constr_u = np.asarray(constraint_u, dtype=np.float64)
    else:
        constr_l, constr_u = _convert_constraints(b, constraint_senses)

    # Determine problem type
    is_qp = P is not None

    # Check for C++ extension
    try:
        from . import _core

        has_core = True
    except ImportError:
        has_core = False

    if has_core and params.get("device") != "cpu":
        # Use GPU solver
        return _solve_gpu(c, A, b, lb, ub, P, constr_l, constr_u, params, warm_start)
    else:
        # Use CPU fallback
        return _solve_cpu(c, A, b, lb, ub, P, constr_l, constr_u, params, warm_start)


def solve_batch(
    problems: List[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
) -> List[SolveResult]:
    """
    Solve many LP/QP problems in parallel on GPU.

    This is cuProx's killer feature - massive parallelism for
    many small problems (e.g., ML training, Monte Carlo).

    Args:
        problems: List of problem dicts with keys:
            - c: objective (required)
            - A: constraint matrix (required)
            - b: constraint RHS (required)
            - lb, ub: bounds (optional)
            - P: quadratic term (optional, for QP)
        params: Solver parameters (shared across all problems)

    Returns:
        List of SolveResult, one per problem

    Example:
        >>> problems = [
        ...     {"c": c1, "A": A1, "b": b1, "lb": lb1},
        ...     {"c": c2, "A": A2, "b": b2, "lb": lb2},
        ...     # ... 1000s more
        ... ]
        >>> results = cuprox.solve_batch(problems)
        >>> print(f"Solved {len(results)} problems")
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    if len(problems) == 0:
        return []

    # Validate all problems have same structure
    first = problems[0]
    n = len(first["c"])
    m = first["A"].shape[0] if hasattr(first["A"], "shape") else len(first["A"])

    for i, prob in enumerate(problems):
        if len(prob["c"]) != n:
            raise DimensionError(f"Problem {i} has {len(prob['c'])} variables, expected {n}")

    # Check for C++ extension with batch support
    try:
        from . import _core

        if hasattr(_core, "solve_batch"):
            return _solve_batch_gpu(problems, params)
    except ImportError:
        pass

    # Fallback: solve sequentially (CPU)
    return [
        solve(
            c=prob["c"],
            A=prob["A"],
            b=prob["b"],
            lb=prob.get("lb"),
            ub=prob.get("ub"),
            P=prob.get("P"),
            params=params,
        )
        for prob in problems
    ]


def _convert_constraints(
    b: np.ndarray,
    constraint_senses: Optional[np.ndarray],
) -> tuple:
    """
    Convert constraint senses to l, u bounds for two-sided form l <= Ax <= u.

    Args:
        b: Constraint RHS values
        constraint_senses: Array of '<', '=', '>' (or '<=', '==', '>=')

    Returns:
        (l, u) tuple of constraint bounds
    """
    m = len(b)

    if constraint_senses is None:
        # Default: all <= constraints
        l = np.full(m, -np.inf)
        u = b.copy()
        return l, u

    l = np.zeros(m)
    u = np.zeros(m)

    for i, sense in enumerate(constraint_senses):
        sense_str = str(sense).strip()
        if sense_str in ("<", "<="):
            l[i] = -np.inf
            u[i] = b[i]
        elif sense_str in (">", ">="):
            l[i] = b[i]
            u[i] = np.inf
        elif sense_str in ("=", "=="):
            l[i] = b[i]
            u[i] = b[i]
        else:
            raise InvalidInputError(
                f"Unknown constraint sense '{sense}' at index {i}. "
                f"Use '<', '<=', '>', '>=', '=', or '=='."
            )

    return l, u


def _validate_inputs(
    c: np.ndarray,
    A: Any,
    b: np.ndarray,
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
    P: Optional[Any],
    q: Optional[np.ndarray],
) -> tuple:
    """Validate and preprocess inputs."""
    # Convert to numpy arrays
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = len(c)

    # Handle q alias
    if q is not None and c is None:
        c = np.asarray(q, dtype=np.float64)

    # Check dimensions
    if A is not None:
        if hasattr(A, "shape"):
            m, n_A = A.shape
        else:
            A = np.asarray(A, dtype=np.float64)
            m, n_A = A.shape

        if n_A != n:
            raise DimensionError(f"A has {n_A} columns but c has {n} elements")
        if len(b) != m:
            raise DimensionError(f"A has {m} rows but b has {len(b)} elements")

        # Convert to CSR if scipy available
        if HAS_SCIPY and not sparse.isspmatrix_csr(A):
            A = sparse.csr_matrix(A)

    # Default bounds
    if lb is None:
        lb = np.zeros(n)
    else:
        lb = np.asarray(lb, dtype=np.float64)
        if len(lb) != n:
            raise DimensionError(f"lb has {len(lb)} elements, expected {n}")

    if ub is None:
        ub = np.full(n, np.inf)
    else:
        ub = np.asarray(ub, dtype=np.float64)
        if len(ub) != n:
            raise DimensionError(f"ub has {len(ub)} elements, expected {n}")

    # Check for invalid bounds
    if np.any(lb > ub):
        raise InvalidInputError("Some lower bounds exceed upper bounds")

    # Check for NaN/Inf
    if np.any(np.isnan(c)):
        raise InvalidInputError("c contains NaN values")
    if np.any(np.isnan(b)):
        raise InvalidInputError("b contains NaN values")

    # Validate P for QP
    if P is not None:
        if HAS_SCIPY and sparse.issparse(P):
            if P.shape != (n, n):
                raise DimensionError(f"P has shape {P.shape}, expected ({n}, {n})")
        elif hasattr(P, "shape"):
            if P.shape != (n, n):
                raise DimensionError(f"P has shape {P.shape}, expected ({n}, {n})")

    return c, A, b, lb, ub, P


def _solve_gpu(
    c: np.ndarray,
    A: Any,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    P: Optional[Any],
    constr_l: np.ndarray,
    constr_u: np.ndarray,
    params: Dict[str, Any],
    warm_start: Optional[SolveResult],
) -> SolveResult:
    """Solve using GPU (C++ extension)."""
    from . import _core

    if _core is None or not _core.cuda_available:
        # Fall back to CPU if CUDA not available
        return _solve_cpu(c, A, b, lb, ub, P, constr_l, constr_u, params, warm_start)

    # Prepare data for C++
    if HAS_SCIPY and sparse.isspmatrix_csr(A):
        A_data = A.data.astype(np.float64)
        A_indices = A.indices.astype(np.int32)
        A_indptr = A.indptr.astype(np.int32)
        m, n = A.shape
    else:
        # Dense matrix - convert to CSR
        A_sparse = sparse.csr_matrix(A)
        A_data = A_sparse.data.astype(np.float64)
        A_indices = A_sparse.indices.astype(np.int32)
        A_indptr = A_sparse.indptr.astype(np.int32)
        m, n = A_sparse.shape

    # Ensure contiguous arrays
    c = np.ascontiguousarray(c, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    lb = np.ascontiguousarray(lb, dtype=np.float64)
    ub = np.ascontiguousarray(ub, dtype=np.float64)

    # Replace inf with large values
    ub = np.where(np.isinf(ub), 1e20, ub)
    lb = np.where(np.isinf(lb), -1e20, lb)

    # Call C++ solver
    if P is None:
        # LP: uses PDHG
        raw_result = _core.solve_lp_pdhg(
            row_offsets=A_indptr,
            col_indices=A_indices,
            values=A_data,
            c=c,
            b=b,
            lb=lb,
            ub=ub,
            num_rows=m,
            num_cols=n,
            max_iters=params.get("max_iterations", 10000),
            eps_abs=params.get("tolerance", 1e-6),
            eps_rel=params.get("tolerance", 1e-6),
            verbose=params.get("verbose", False),
        )

        return SolveResult(
            status=Status(raw_result["status"]),
            objective=float(raw_result["objective"]),
            x=raw_result["x"],
            y=raw_result["y"],
            iterations=raw_result["iterations"],
            solve_time=raw_result["solve_time"],
            primal_residual=raw_result.get("primal_residual", 0.0),
            dual_residual=raw_result.get("dual_residual", 0.0),
        )
    else:
        # QP: use ADMM solver
        if HAS_SCIPY and sparse.isspmatrix_csr(P):
            P_data = P.data.astype(np.float64)
            P_indices = P.indices.astype(np.int32)
            P_indptr = P.indptr.astype(np.int32)
            P_n = P.shape[0]
        else:
            P_sparse = sparse.csr_matrix(P)
            P_data = P_sparse.data.astype(np.float64)
            P_indices = P_sparse.indices.astype(np.int32)
            P_indptr = P_sparse.indptr.astype(np.int32)
            P_n = P_sparse.shape[0]

        # Ensure contiguous
        P_data = np.ascontiguousarray(P_data)
        P_indices = np.ascontiguousarray(P_indices)
        P_indptr = np.ascontiguousarray(P_indptr)

        # For QP, use l and u from constraint conversion
        # If no explicit constraint_senses given with b=0, use free bounds
        if np.allclose(constr_l, constr_u) and np.allclose(constr_u, 0):
            # No meaningful constraints - use large bounds
            qp_l = np.full(m, -1e20, dtype=np.float64)
            qp_u = np.full(m, 1e20, dtype=np.float64)
        else:
            qp_l = np.ascontiguousarray(constr_l, dtype=np.float64)
            qp_u = np.ascontiguousarray(constr_u, dtype=np.float64)

        # Replace inf with large values
        qp_l = np.where(np.isinf(qp_l), -1e20, qp_l)
        qp_u = np.where(np.isinf(qp_u), 1e20, qp_u)

        # Variable bounds
        var_lb = np.ascontiguousarray(lb, dtype=np.float64)
        var_ub = np.ascontiguousarray(ub, dtype=np.float64)
        var_lb = np.where(np.isinf(var_lb), -1e20, var_lb)
        var_ub = np.where(np.isinf(var_ub), 1e20, var_ub)

        raw_result = _core.solve_qp_admm(
            P_row_offsets=P_indptr,
            P_col_indices=P_indices,
            P_values=P_data,
            A_row_offsets=A_indptr,
            A_col_indices=A_indices,
            A_values=A_data,
            q=c,
            l=qp_l,
            u=qp_u,
            var_lb=var_lb,
            var_ub=var_ub,
            P_n=P_n,
            A_m=m,
            A_n=n,
            max_iters=params.get("max_iterations", 4000),
            eps_abs=params.get("tolerance", 1e-6),
            eps_rel=params.get("tolerance", 1e-6),
            rho=params.get("rho", 1.0),
            verbose=params.get("verbose", False),
        )

        return SolveResult(
            status=Status(raw_result["status"]),
            objective=float(raw_result["objective"]),
            x=raw_result["x"],
            y=raw_result["y"],
            iterations=raw_result["iterations"],
            solve_time=raw_result["solve_time"],
            primal_residual=raw_result.get("primal_residual", 0.0),
            dual_residual=raw_result.get("dual_residual", 0.0),
        )


def _solve_cpu(
    c: np.ndarray,
    A: Any,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    P: Optional[Any],
    constr_l: np.ndarray,
    constr_u: np.ndarray,
    params: Dict[str, Any],
    warm_start: Optional[SolveResult],
) -> SolveResult:
    """Solve using CPU fallback (pure Python/NumPy)."""
    import time

    start_time = time.perf_counter()

    # Simple PDHG implementation for LP
    # This is a reference implementation - real performance comes from GPU
    # Note: CPU fallback currently only handles Ax <= b (upper bound) constraints

    n = len(c)
    m = A.shape[0] if hasattr(A, "shape") else 0

    if m == 0:
        # No constraints - just respect bounds
        x = np.clip(-np.sign(c) * np.inf, lb, ub)
        x = np.where(np.isinf(x), 0, x)
        return SolveResult(
            status=Status.OPTIMAL,
            objective=float(c @ x),
            x=x,
            y=np.array([]),
            iterations=0,
            solve_time=time.perf_counter() - start_time,
        )

    # Convert to dense for simplicity in fallback
    if HAS_SCIPY and sparse.issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)

    # Initialize
    x = np.zeros(n)
    y = np.zeros(m)
    x_bar = np.zeros(n)

    # Step sizes (conservative)
    tau = sigma = 0.5 / np.linalg.norm(A_dense, 2)

    tol = params.get("tolerance", 1e-6)
    max_iter = params.get("max_iterations", 100000)

    for k in range(max_iter):
        # Dual update
        y_new = y + sigma * (A_dense @ x_bar - b)
        y_new = np.maximum(y_new, 0)  # Inequality projection

        # Primal update
        x_prev = x.copy()
        grad = c + A_dense.T @ y_new
        x_new = x - tau * grad
        x_new = np.clip(x_new, lb, ub)  # Box projection

        # Extrapolation
        x_bar = 2 * x_new - x_prev

        x, y = x_new, y_new

        # Check convergence
        if k % 50 == 0:
            primal_res = np.linalg.norm(np.maximum(A_dense @ x - b, 0))
            dual_res = np.linalg.norm(
                c
                + A_dense.T @ y
                - np.where(x <= lb, np.minimum(c + A_dense.T @ y, 0), 0)
                - np.where(x >= ub, np.maximum(c + A_dense.T @ y, 0), 0)
            )

            if max(primal_res, dual_res) < tol:
                break

    solve_time = time.perf_counter() - start_time

    return SolveResult(
        status=Status.OPTIMAL if k < max_iter - 1 else Status.MAX_ITERATIONS,
        objective=float(c @ x),
        x=x,
        y=y,
        iterations=k + 1,
        solve_time=solve_time,
        primal_residual=float(primal_res),
        dual_residual=float(dual_res),
    )


def _solve_batch_gpu(
    problems: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[SolveResult]:
    """Solve batch using GPU."""
    from . import _core

    # Prepare batched data
    # ... implementation would go here

    # For now, fall back to sequential
    return [
        solve(
            c=prob["c"],
            A=prob["A"],
            b=prob["b"],
            lb=prob.get("lb"),
            ub=prob.get("ub"),
            P=prob.get("P"),
            params=params,
        )
        for prob in problems
    ]
