"""Input validation utilities."""

from typing import Any, Optional, Tuple

import numpy as np


def validate_problem(
    c: np.ndarray,
    A: Any,
    b: np.ndarray,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    """
    Validate problem data.

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        n = len(c)

        if A is not None:
            m, n_A = A.shape
            if n_A != n:
                return False, f"A has {n_A} columns but c has {n} elements"
            if len(b) != m:
                return False, f"A has {m} rows but b has {len(b)} elements"

        if lb is not None and len(lb) != n:
            return False, f"lb has {len(lb)} elements, expected {n}"

        if ub is not None and len(ub) != n:
            return False, f"ub has {len(ub)} elements, expected {n}"

        if np.any(np.isnan(c)):
            return False, "c contains NaN values"

        if np.any(np.isnan(b)):
            return False, "b contains NaN values"

        return True, ""

    except Exception as e:
        return False, str(e)
