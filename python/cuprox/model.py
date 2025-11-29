"""
cuProx Model Builder
====================

Algebraic interface for building LP/QP models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy import sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .exceptions import DimensionError


@dataclass
class Variable:
    """
    Decision variable in an optimization model.

    Attributes:
        index: Internal index in the model
        lb: Lower bound (default: 0)
        ub: Upper bound (default: +inf)
        name: Optional name for the variable

    Example:
        >>> model = Model()
        >>> x = model.add_var(lb=0, ub=10, name="x")
        >>> y = model.add_var(lb=0, name="y")
    """

    index: int
    lb: float = 0.0
    ub: float = float("inf")
    name: Optional[str] = None

    def __repr__(self) -> str:
        if self.name:
            return f"Variable({self.name})"
        return f"Variable(x_{self.index})"

    # Operator overloading for algebraic syntax
    def __add__(self, other: Union["Variable", "LinearExpr", float]) -> "LinearExpr":
        return LinearExpr.from_var(self) + other

    def __radd__(self, other: Union["Variable", "LinearExpr", float]) -> "LinearExpr":
        return self.__add__(other)

    def __sub__(self, other: Union["Variable", "LinearExpr", float]) -> "LinearExpr":
        return LinearExpr.from_var(self) - other

    def __rsub__(self, other: Union["Variable", "LinearExpr", float]) -> "LinearExpr":
        return (-1) * LinearExpr.from_var(self) + other

    def __mul__(self, other: float) -> "LinearExpr":
        return LinearExpr.from_var(self, coef=other)

    def __rmul__(self, other: float) -> "LinearExpr":
        return self.__mul__(other)

    def __neg__(self) -> "LinearExpr":
        return self.__mul__(-1)

    def __truediv__(self, other: float) -> "LinearExpr":
        return self.__mul__(1.0 / other)

    # Comparison operators for constraints
    def __le__(self, other: Union["Variable", "LinearExpr", float]) -> "Constraint":
        return LinearExpr.from_var(self) <= other

    def __ge__(self, other: Union["Variable", "LinearExpr", float]) -> "Constraint":
        return LinearExpr.from_var(self) >= other

    def __eq__(self, other: Union["Variable", "LinearExpr", float]) -> "Constraint":
        return LinearExpr.from_var(self).__eq__(other)


@dataclass
class LinearExpr:
    """
    Linear expression: sum of coefficient * variable + constant.

    Example:
        >>> expr = 2*x + 3*y + 5
        >>> print(expr)
        2*x + 3*y + 5
    """

    terms: Dict[int, float] = field(default_factory=dict)  # var_index -> coefficient
    constant: float = 0.0
    _var_names: Dict[int, str] = field(default_factory=dict)  # For pretty printing

    @classmethod
    def from_var(cls, var: Variable, coef: float = 1.0) -> "LinearExpr":
        """Create expression from a single variable."""
        expr = cls()
        expr.terms[var.index] = coef
        if var.name:
            expr._var_names[var.index] = var.name
        return expr

    def __repr__(self) -> str:
        parts = []
        for idx, coef in sorted(self.terms.items()):
            name = self._var_names.get(idx, f"x_{idx}")
            if coef == 1:
                parts.append(name)
            elif coef == -1:
                parts.append(f"-{name}")
            else:
                parts.append(f"{coef}*{name}")
        if self.constant != 0 or not parts:
            parts.append(str(self.constant))
        return " + ".join(parts).replace("+ -", "- ")

    def __add__(self, other: Union[Variable, "LinearExpr", float]) -> "LinearExpr":
        result = LinearExpr(dict(self.terms), self.constant, dict(self._var_names))
        if isinstance(other, Variable):
            result.terms[other.index] = result.terms.get(other.index, 0) + 1
            if other.name:
                result._var_names[other.index] = other.name
        elif isinstance(other, LinearExpr):
            for idx, coef in other.terms.items():
                result.terms[idx] = result.terms.get(idx, 0) + coef
            result._var_names.update(other._var_names)
            result.constant += other.constant
        else:
            result.constant += float(other)
        return result

    def __radd__(self, other: Union[Variable, "LinearExpr", float]) -> "LinearExpr":
        return self.__add__(other)

    def __sub__(self, other: Union[Variable, "LinearExpr", float]) -> "LinearExpr":
        if isinstance(other, Variable):
            return self + (-1) * other
        elif isinstance(other, LinearExpr):
            neg_other = LinearExpr(
                {k: -v for k, v in other.terms.items()}, -other.constant, dict(other._var_names)
            )
            return self + neg_other
        else:
            return self + (-float(other))

    def __rsub__(self, other: Union[Variable, "LinearExpr", float]) -> "LinearExpr":
        return (-1) * self + other

    def __mul__(self, other: float) -> "LinearExpr":
        return LinearExpr(
            {k: v * other for k, v in self.terms.items()},
            self.constant * other,
            dict(self._var_names),
        )

    def __rmul__(self, other: float) -> "LinearExpr":
        return self.__mul__(other)

    def __neg__(self) -> "LinearExpr":
        return self.__mul__(-1)

    def __truediv__(self, other: float) -> "LinearExpr":
        return self.__mul__(1.0 / other)

    # Comparison operators for constraints
    def __le__(self, other: Union[Variable, "LinearExpr", float]) -> "Constraint":
        if isinstance(other, Variable):
            other = LinearExpr.from_var(other)
        elif not isinstance(other, LinearExpr):
            other = LinearExpr(constant=float(other))
        lhs = self - other
        return Constraint(lhs, "<=", 0.0)

    def __ge__(self, other: Union[Variable, "LinearExpr", float]) -> "Constraint":
        if isinstance(other, Variable):
            other = LinearExpr.from_var(other)
        elif not isinstance(other, LinearExpr):
            other = LinearExpr(constant=float(other))
        lhs = self - other
        return Constraint(lhs, ">=", 0.0)

    def __eq__(self, other: Union[Variable, "LinearExpr", float]) -> "Constraint":
        if isinstance(other, Variable):
            other = LinearExpr.from_var(other)
        elif not isinstance(other, LinearExpr):
            other = LinearExpr(constant=float(other))
        lhs = self - other
        return Constraint(lhs, "==", 0.0)


@dataclass
class Constraint:
    """
    Linear constraint in an optimization model.

    Represents: lhs sense rhs (e.g., 2*x + 3*y <= 10)

    Attributes:
        lhs: Left-hand side linear expression
        sense: Constraint sense ("<=", ">=", "==")
        rhs: Right-hand side constant
        name: Optional constraint name
        index: Internal index (set when added to model)
    """

    lhs: LinearExpr
    sense: str  # "<=", ">=", "=="
    rhs: float
    name: Optional[str] = None
    index: int = -1

    def __repr__(self) -> str:
        name_str = f"{self.name}: " if self.name else ""
        return f"{name_str}{self.lhs} {self.sense} {self.rhs}"


class Model:
    """
    Optimization model builder with algebraic syntax.

    Supports Linear Programs (LP) and Quadratic Programs (QP).

    Example:
        >>> model = Model()
        >>> x = model.add_var(lb=0, ub=10, name="x")
        >>> y = model.add_var(lb=0, name="y")
        >>> model.add_constr(x + 2*y <= 20, name="capacity")
        >>> model.add_constr(3*x + y <= 30, name="labor")
        >>> model.minimize(-5*x - 4*y)
        >>> result = model.solve()
        >>> print(result.objective)
        -46.0
    """

    def __init__(self, name: str = ""):
        """
        Create a new optimization model.

        Args:
            name: Optional model name
        """
        self.name = name
        self._vars: List[Variable] = []
        self._constrs: List[Constraint] = []
        self._objective: Optional[LinearExpr] = None
        self._sense: str = "minimize"
        self._quadratic_obj: Optional[Any] = None  # For QP

    @property
    def num_vars(self) -> int:
        """Number of variables in the model."""
        return len(self._vars)

    @property
    def num_constrs(self) -> int:
        """Number of constraints in the model."""
        return len(self._constrs)

    def add_var(
        self,
        lb: float = 0.0,
        ub: float = float("inf"),
        name: Optional[str] = None,
    ) -> Variable:
        """
        Add a single decision variable to the model.

        Args:
            lb: Lower bound (default: 0)
            ub: Upper bound (default: +inf)
            name: Variable name

        Returns:
            The created Variable object
        """
        var = Variable(
            index=len(self._vars),
            lb=lb,
            ub=ub,
            name=name,
        )
        self._vars.append(var)
        return var

    def add_vars(
        self,
        count: int,
        lb: Union[float, np.ndarray] = 0.0,
        ub: Union[float, np.ndarray] = float("inf"),
        name_prefix: str = "x",
    ) -> List[Variable]:
        """
        Add multiple decision variables to the model.

        Args:
            count: Number of variables to add
            lb: Lower bound(s) - scalar or array
            ub: Upper bound(s) - scalar or array
            name_prefix: Prefix for variable names

        Returns:
            List of created Variable objects
        """
        if isinstance(lb, np.ndarray) and len(lb) != count:
            raise DimensionError(f"lb has length {len(lb)}, expected {count}")
        if isinstance(ub, np.ndarray) and len(ub) != count:
            raise DimensionError(f"ub has length {len(ub)}, expected {count}")

        vars = []
        for i in range(count):
            lb_i = lb[i] if isinstance(lb, np.ndarray) else lb
            ub_i = ub[i] if isinstance(ub, np.ndarray) else ub
            var = self.add_var(lb=lb_i, ub=ub_i, name=f"{name_prefix}_{i}")
            vars.append(var)
        return vars

    def add_constr(
        self,
        constraint: Constraint,
        name: Optional[str] = None,
    ) -> Constraint:
        """
        Add a constraint to the model.

        Args:
            constraint: Constraint object (from comparison operators)
            name: Optional constraint name

        Returns:
            The added Constraint object

        Example:
            >>> model.add_constr(x + 2*y <= 20, name="capacity")
        """
        if name:
            constraint.name = name
        constraint.index = len(self._constrs)
        self._constrs.append(constraint)
        return constraint

    def add_constrs(self, constraints: List[Constraint]) -> List[Constraint]:
        """
        Add multiple constraints to the model.

        Args:
            constraints: List of Constraint objects

        Returns:
            List of added Constraint objects
        """
        for c in constraints:
            self.add_constr(c)
        return constraints

    def minimize(self, expr: Union[LinearExpr, Variable, float]) -> None:
        """
        Set the objective to minimize.

        Args:
            expr: Linear expression to minimize
        """
        self._objective = self._to_expr(expr)
        self._sense = "minimize"

    def maximize(self, expr: Union[LinearExpr, Variable, float]) -> None:
        """
        Set the objective to maximize.

        Args:
            expr: Linear expression to maximize
        """
        self._objective = self._to_expr(expr)
        self._sense = "maximize"

    def _to_expr(self, expr: Union[LinearExpr, Variable, float]) -> LinearExpr:
        """Convert various types to LinearExpr."""
        if isinstance(expr, LinearExpr):
            return expr
        elif isinstance(expr, Variable):
            return LinearExpr.from_var(expr)
        else:
            return LinearExpr(constant=float(expr))

    def solve(
        self,
        params: Optional[Dict[str, Any]] = None,
        warm_start: Optional["SolveResult"] = None,
    ) -> "SolveResult":
        """
        Solve the optimization model.

        Args:
            params: Solver parameters (tolerance, max_iterations, etc.)
            warm_start: Previous solution for warm starting

        Returns:
            SolveResult with status, objective, and solution
        """
        from .solver import solve

        # Convert to matrix form
        A, b, c, lb, ub, senses = self._to_standard_form()

        return solve(
            c=c,
            A=A,
            b=b,
            lb=lb,
            ub=ub,
            constraint_senses=senses,
            params=params,
            warm_start=warm_start,
        )

    def _to_standard_form(
        self,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert model to standard matrix form.

        Returns:
            (A, b, c, lb, ub, senses) tuple
        """
        n = self.num_vars
        m = self.num_constrs

        # Objective
        c = np.zeros(n)
        if self._objective:
            for idx, coef in self._objective.terms.items():
                c[idx] = coef
            # Handle constant term (add to objective value after solve)
        if self._sense == "maximize":
            c = -c  # Convert max to min

        # Bounds
        lb = np.array([v.lb for v in self._vars])
        ub = np.array([v.ub for v in self._vars])

        # Constraints
        if m == 0:
            if HAS_SCIPY:
                A = sparse.csr_matrix((0, n))
            else:
                A = np.zeros((0, n))
            b = np.array([])
            senses = np.array([])
        else:
            rows, cols, data = [], [], []
            b = np.zeros(m)
            senses = []

            for i, constr in enumerate(self._constrs):
                for idx, coef in constr.lhs.terms.items():
                    rows.append(i)
                    cols.append(idx)
                    data.append(coef)
                # Move constant to RHS
                b[i] = -constr.lhs.constant
                senses.append(constr.sense)

            if HAS_SCIPY:
                A = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
            else:
                A = np.zeros((m, n))
                for r, c_idx, d in zip(rows, cols, data):
                    A[r, c_idx] = d

            senses = np.array(senses)

        return A, b, c, lb, ub, senses

    @classmethod
    def from_matrices(
        cls,
        c: np.ndarray,
        A_ub: Optional[Any] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[Any] = None,
        b_eq: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        P: Optional[Any] = None,
        name: str = "",
    ) -> "Model":
        """
        Create model directly from matrices.

        This is the preferred method for large-scale problems.

        Args:
            c: Objective coefficients (n,)
            A_ub: Inequality constraint matrix (m_ub, n)
            b_ub: Inequality RHS (m_ub,)
            A_eq: Equality constraint matrix (m_eq, n)
            b_eq: Equality RHS (m_eq,)
            lb: Variable lower bounds (n,)
            ub: Variable upper bounds (n,)
            P: Quadratic objective matrix (n, n) for QP
            name: Model name

        Returns:
            Model object ready to solve
        """
        model = cls(name=name)

        n = len(c)

        # Add variables
        if lb is None:
            lb = np.zeros(n)
        if ub is None:
            ub = np.full(n, np.inf)
        model._vars = [Variable(i, lb[i], ub[i]) for i in range(n)]

        # Store matrices directly (bypass algebraic interface for efficiency)
        model._matrix_form = {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "lb": lb,
            "ub": ub,
            "P": P,
        }

        return model

    def __repr__(self) -> str:
        return f"Model(vars={self.num_vars}, constrs={self.num_constrs})"


# Import SolveResult for type hints (avoid circular import at runtime)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import SolveResult
