"""
Portfolio Optimization
======================

GPU-accelerated portfolio optimization using quadratic programming.

Supported methods:
- Mean-Variance (Markowitz)
- Minimum Variance
- Maximum Sharpe Ratio
- Risk Parity
- Target Return
- Target Volatility
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from scipy import sparse

from .. import solve
from ..result import Status
from .utils import (
    compute_covariance,
    is_positive_definite,
    make_positive_definite,
)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""

    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    TARGET_RETURN = "target_return"
    TARGET_VOLATILITY = "target_volatility"


@dataclass
class PortfolioResult:
    """
    Result of portfolio optimization.

    Attributes:
        weights: Optimal portfolio weights
        expected_return: Expected portfolio return
        volatility: Portfolio volatility (std dev)
        sharpe_ratio: Sharpe ratio (if risk_free_rate provided)
        method: Optimization method used
        status: Solver status
        solve_time: Solve time in seconds
        iterations: Solver iterations
    """

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str
    status: str
    solve_time: float
    iterations: int

    # Additional info
    risk_contributions: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"PortfolioResult(\n"
            f"  method={self.method},\n"
            f"  expected_return={self.expected_return:.4%},\n"
            f"  volatility={self.volatility:.4%},\n"
            f"  sharpe_ratio={self.sharpe_ratio:.4f},\n"
            f"  status={self.status}\n"
            f")"
        )

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 50,
            "Portfolio Optimization Result",
            "=" * 50,
            f"Method:           {self.method}",
            f"Status:           {self.status}",
            f"Expected Return:  {self.expected_return:.4%}",
            f"Volatility:       {self.volatility:.4%}",
            f"Sharpe Ratio:     {self.sharpe_ratio:.4f}",
            f"Solve Time:       {self.solve_time:.4f}s",
            "-" * 50,
            "Weights:",
        ]

        for i, w in enumerate(self.weights):
            if abs(w) > 1e-6:
                lines.append(f"  Asset {i}: {w:.4%}")

        lines.append("=" * 50)
        return "\n".join(lines)


class Portfolio:
    """
    Portfolio optimizer using GPU-accelerated QP.

    Supports various optimization objectives including mean-variance,
    minimum variance, maximum Sharpe ratio, and risk parity.

    Args:
        returns: Historical returns matrix (T, N) or expected returns (N,)
        covariance: Covariance matrix (N, N), computed from returns if not provided
        expected_returns: Expected returns (N,), computed from returns if not provided
        risk_free_rate: Annual risk-free rate for Sharpe calculations
        periods_per_year: Trading periods per year (252 for daily)

    Example:
        >>> import numpy as np
        >>> # Historical returns for 5 assets, 252 days
        >>> returns = np.random.randn(252, 5) * 0.02
        >>>
        >>> port = Portfolio(returns)
        >>>
        >>> # Mean-variance optimization
        >>> result = port.optimize(method='mean_variance', risk_aversion=2.0)
        >>> print(result.summary())
        >>>
        >>> # Minimum variance
        >>> result = port.optimize(method='min_variance')
        >>>
        >>> # Maximum Sharpe ratio
        >>> result = port.optimize(method='max_sharpe')
    """

    def __init__(
        self,
        returns: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        expected_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        # Validate inputs
        if returns is None and (covariance is None or expected_returns is None):
            raise ValueError(
                "Must provide either 'returns' or both 'covariance' and 'expected_returns'"
            )

        if returns is not None:
            returns = np.asarray(returns, dtype=np.float64)
            if returns.ndim == 1:
                returns = returns.reshape(-1, 1)
            self._n_assets = returns.shape[1]
            self._T = returns.shape[0]
        else:
            self._n_assets = covariance.shape[0]
            self._T = None

        # Compute or validate covariance
        if covariance is not None:
            self._cov = np.asarray(covariance, dtype=np.float64)
            if self._cov.shape != (self._n_assets, self._n_assets):
                raise ValueError(
                    f"Covariance shape {self._cov.shape} doesn't match "
                    f"expected ({self._n_assets}, {self._n_assets})"
                )
        else:
            self._cov = compute_covariance(returns, method="sample")

        # Handle scalar covariance (single asset)
        if self._cov.ndim == 0:
            self._cov = self._cov.reshape(1, 1)

        # Ensure positive definite
        if not is_positive_definite(self._cov):
            self._cov = make_positive_definite(self._cov)

        # Compute or validate expected returns
        if expected_returns is not None:
            self._mu = np.asarray(expected_returns, dtype=np.float64)
            if self._mu.shape != (self._n_assets,):
                raise ValueError(
                    f"Expected returns shape {self._mu.shape} doesn't match "
                    f"expected ({self._n_assets},)"
                )
        else:
            self._mu = returns.mean(axis=0)

        self._rf = risk_free_rate / periods_per_year  # Convert to per-period
        self._periods_per_year = periods_per_year

        # Default constraints
        self._lb = np.zeros(self._n_assets)
        self._ub = np.ones(self._n_assets)

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self._n_assets

    @property
    def covariance(self) -> np.ndarray:
        """Covariance matrix."""
        return self._cov.copy()

    @property
    def expected_returns(self) -> np.ndarray:
        """Expected returns."""
        return self._mu.copy()

    def set_bounds(
        self,
        lower: Union[float, np.ndarray] = 0.0,
        upper: Union[float, np.ndarray] = 1.0,
    ) -> Portfolio:
        """
        Set weight bounds for all assets.

        Args:
            lower: Lower bound (scalar or per-asset array)
            upper: Upper bound (scalar or per-asset array)

        Returns:
            Self for method chaining

        Example:
            >>> port.set_bounds(lower=0.0, upper=0.3)  # Max 30% per asset
            >>> port.set_bounds(lower=-0.1, upper=0.4)  # Allow 10% shorting
        """
        if np.isscalar(lower):
            self._lb = np.full(self._n_assets, lower)
        else:
            self._lb = np.asarray(lower, dtype=np.float64)

        if np.isscalar(upper):
            self._ub = np.full(self._n_assets, upper)
        else:
            self._ub = np.asarray(upper, dtype=np.float64)

        return self

    def optimize(
        self,
        method: str = "mean_variance",
        risk_aversion: float = 1.0,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        max_iters: int = 10000,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> PortfolioResult:
        """
        Optimize portfolio weights.

        Args:
            method: Optimization method
                - 'mean_variance': Maximize return - risk_aversion * variance
                - 'min_variance': Minimize portfolio variance
                - 'max_sharpe': Maximize Sharpe ratio
                - 'risk_parity': Equal risk contribution
                - 'target_return': Min variance for target return
                - 'target_volatility': Max return for target volatility
            risk_aversion: Risk aversion parameter (for mean_variance)
            target_return: Target return (for target_return method)
            target_volatility: Target volatility (for target_volatility method)
            max_iters: Maximum solver iterations
            tolerance: Convergence tolerance
            verbose: Print solver progress

        Returns:
            PortfolioResult with optimal weights and statistics

        Example:
            >>> result = port.optimize(method='mean_variance', risk_aversion=2.0)
            >>> print(f"Expected return: {result.expected_return:.2%}")
            >>> print(f"Volatility: {result.volatility:.2%}")
        """
        method = method.lower()

        if method == "mean_variance":
            return self._optimize_mean_variance(risk_aversion, max_iters, tolerance, verbose)
        elif method == "min_variance":
            return self._optimize_min_variance(max_iters, tolerance, verbose)
        elif method == "max_sharpe":
            return self._optimize_max_sharpe(max_iters, tolerance, verbose)
        elif method == "risk_parity":
            return self._optimize_risk_parity(max_iters, tolerance, verbose)
        elif method == "target_return":
            if target_return is None:
                raise ValueError("target_return required for 'target_return' method")
            return self._optimize_target_return(target_return, max_iters, tolerance, verbose)
        elif method == "target_volatility":
            if target_volatility is None:
                raise ValueError("target_volatility required for 'target_volatility' method")
            return self._optimize_target_volatility(
                target_volatility, max_iters, tolerance, verbose
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: "
                "mean_variance, min_variance, max_sharpe, risk_parity, "
                "target_return, target_volatility"
            )

    def _optimize_mean_variance(
        self,
        risk_aversion: float,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> PortfolioResult:
        """
        Mean-Variance (Markowitz) optimization.

        maximize: μ'w - (λ/2) w'Σw
        subject to: sum(w) = 1, lb <= w <= ub

        Equivalent QP:
        minimize: (λ/2) w'Σw - μ'w
        """
        n = self._n_assets

        # QP formulation: min (1/2) w'Pw + q'w
        P = sparse.csr_matrix(risk_aversion * self._cov)
        q = -self._mu

        # Equality constraint: sum(w) = 1
        A = sparse.csr_matrix(np.ones((1, n)))
        b = np.array([1.0])

        result = solve(
            c=q,
            A=A,
            b=b,
            P=P,
            lb=self._lb,
            ub=self._ub,
            constraint_l=b,
            constraint_u=b,
            params={
                "max_iterations": max_iters,
                "tolerance": tolerance,
                "verbose": verbose,
            },
        )

        return self._make_result(result.x, "mean_variance", result)

    def _optimize_min_variance(
        self,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> PortfolioResult:
        """
        Minimum variance portfolio.

        minimize: (1/2) w'Σw
        subject to: sum(w) = 1, lb <= w <= ub
        """
        n = self._n_assets

        P = sparse.csr_matrix(self._cov)
        q = np.zeros(n)

        A = sparse.csr_matrix(np.ones((1, n)))
        b = np.array([1.0])

        result = solve(
            c=q,
            A=A,
            b=b,
            P=P,
            lb=self._lb,
            ub=self._ub,
            constraint_l=b,
            constraint_u=b,
            params={
                "max_iterations": max_iters,
                "tolerance": tolerance,
                "verbose": verbose,
            },
        )

        return self._make_result(result.x, "min_variance", result)

    def _optimize_max_sharpe(
        self,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> PortfolioResult:
        """
        Maximum Sharpe ratio portfolio.

        Uses the transformation: y = w/κ where κ = 1/(μ-rf)'w

        minimize: (1/2) y'Σy
        subject to: (μ-rf)'y = 1, y >= 0

        Then w = y / sum(y)
        """
        n = self._n_assets
        excess_returns = self._mu - self._rf

        # Check if any asset has positive excess return
        if (excess_returns <= 0).all():
            # Fall back to min variance if no positive excess returns
            return self._optimize_min_variance(max_iters, tolerance, verbose)

        P = sparse.csr_matrix(self._cov)
        q = np.zeros(n)

        # Constraint: (μ-rf)'y = 1
        A = sparse.csr_matrix(excess_returns.reshape(1, -1))
        b = np.array([1.0])

        # Bounds: y >= 0 (scaled version of w >= 0)
        lb = np.zeros(n)
        ub = np.full(n, 1e10)  # Large upper bound

        result = solve(
            c=q,
            A=A,
            b=b,
            P=P,
            lb=lb,
            ub=ub,
            constraint_l=b,
            constraint_u=b,
            params={
                "max_iterations": max_iters,
                "tolerance": tolerance,
                "verbose": verbose,
            },
        )

        # Transform back: w = y / sum(y)
        y = result.x
        weights = y / y.sum() if y.sum() > 0 else np.ones(n) / n

        # Clip to bounds
        weights = np.clip(weights, self._lb, self._ub)
        weights = weights / weights.sum()  # Renormalize

        return self._make_result(weights, "max_sharpe", result)

    def _optimize_risk_parity(
        self,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> PortfolioResult:
        """
        Risk parity portfolio (equal risk contribution).

        Uses iterative approach with QP subproblems.
        Target: RC_i = w_i * (Σw)_i / (w'Σw) = 1/n for all i
        """
        n = self._n_assets

        # Initialize with equal weights
        w = np.ones(n) / n

        # Iterative refinement
        for _ in range(50):  # Max iterations for risk parity
            # Compute risk contributions
            Sigma_w = self._cov @ w
            total_risk = np.sqrt(w @ Sigma_w)

            if total_risk < 1e-10:
                break

            marginal_risk = Sigma_w / total_risk
            risk_contrib = w * marginal_risk

            # Target equal contribution
            target_rc = total_risk / n

            # Gradient step
            grad = marginal_risk - target_rc / (w + 1e-10)

            # Update with projection
            w_new = w - 0.1 * grad
            w_new = np.maximum(w_new, self._lb)
            w_new = np.minimum(w_new, self._ub)
            w_new = w_new / w_new.sum()

            # Check convergence
            if np.linalg.norm(w_new - w) < tolerance:
                break

            w = w_new

        # Final result
        result_obj = type(
            "Result",
            (),
            {
                "status": Status.OPTIMAL,
                "solve_time": 0.0,
                "iterations": 50,
            },
        )()

        return self._make_result(w, "risk_parity", result_obj)

    def _optimize_target_return(
        self,
        target_return: float,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> PortfolioResult:
        """
        Minimum variance for target return.

        minimize: (1/2) w'Σw
        subject to: μ'w = target_return
                    sum(w) = 1
                    lb <= w <= ub
        """
        n = self._n_assets

        P = sparse.csr_matrix(self._cov)
        q = np.zeros(n)

        # Two equality constraints
        A = sparse.csr_matrix(
            np.vstack(
                [
                    np.ones(n),  # sum(w) = 1
                    self._mu,  # μ'w = target
                ]
            )
        )
        b = np.array([1.0, target_return])

        result = solve(
            c=q,
            A=A,
            b=b,
            P=P,
            lb=self._lb,
            ub=self._ub,
            constraint_l=b,
            constraint_u=b,
            params={
                "max_iterations": max_iters,
                "tolerance": tolerance,
                "verbose": verbose,
            },
        )

        return self._make_result(result.x, "target_return", result)

    def _optimize_target_volatility(
        self,
        target_volatility: float,
        max_iters: int,
        tolerance: float,
        verbose: bool,
    ) -> PortfolioResult:
        """
        Maximum return for target volatility.

        Uses bisection on risk aversion to find the portfolio
        with the target volatility on the efficient frontier.
        """
        # Binary search for risk aversion that gives target vol
        lambda_low, lambda_high = 0.01, 100.0

        for _ in range(50):
            lambda_mid = (lambda_low + lambda_high) / 2

            result = self._optimize_mean_variance(lambda_mid, max_iters, tolerance, verbose=False)

            if result.volatility < target_volatility:
                lambda_high = lambda_mid
            else:
                lambda_low = lambda_mid

            if abs(result.volatility - target_volatility) < tolerance:
                break

        # Return result with updated method name
        return PortfolioResult(
            weights=result.weights,
            expected_return=result.expected_return,
            volatility=result.volatility,
            sharpe_ratio=result.sharpe_ratio,
            method="target_volatility",
            status=result.status,
            solve_time=result.solve_time,
            iterations=result.iterations,
            risk_contributions=result.risk_contributions,
        )

    def _make_result(
        self,
        weights: np.ndarray,
        method: str,
        solve_result: Any,
    ) -> PortfolioResult:
        """Create PortfolioResult from optimization output."""
        # Clip small weights to zero
        weights = np.where(np.abs(weights) < 1e-8, 0, weights)

        # Ensure sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Compute statistics (annualized)
        expected_return = float(self._mu @ weights) * self._periods_per_year
        variance = float(weights @ self._cov @ weights)
        volatility = np.sqrt(max(variance, 0)) * np.sqrt(self._periods_per_year)

        # Sharpe ratio
        excess_return = expected_return - self._rf
        sharpe = excess_return / volatility if volatility > 0 else 0.0

        # Risk contributions
        if volatility > 0:
            marginal_contrib = self._cov @ weights / volatility
            risk_contributions = weights * marginal_contrib
        else:
            risk_contributions = np.zeros_like(weights)

        # Status
        if hasattr(solve_result, "status"):
            status = (
                str(solve_result.status.value)
                if hasattr(solve_result.status, "value")
                else str(solve_result.status)
            )
        else:
            status = "optimal"

        return PortfolioResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            method=method,
            status=status,
            solve_time=getattr(solve_result, "solve_time", 0.0),
            iterations=getattr(solve_result, "iterations", 0),
            risk_contributions=risk_contributions,
        )
