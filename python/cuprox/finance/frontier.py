"""
Efficient Frontier
==================

Compute and analyze the efficient frontier of portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .portfolio import Portfolio, PortfolioResult


@dataclass
class FrontierPoint:
    """Single point on the efficient frontier."""

    return_: float
    volatility: float
    sharpe_ratio: float
    weights: np.ndarray


class EfficientFrontier:
    """
    Efficient frontier computation and analysis.

    Computes the set of optimal portfolios that offer the highest
    expected return for a given level of risk.

    Args:
        returns: Historical returns (T, N) or expected returns (N,)
        covariance: Covariance matrix (N, N)
        expected_returns: Expected returns (N,)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Example:
        >>> ef = EfficientFrontier(returns)
        >>> frontier = ef.compute(n_points=50)
        >>>
        >>> # Get portfolio at target return
        >>> weights = ef.portfolio_at_return(target_return=0.10)
        >>>
        >>> # Get portfolio at target volatility
        >>> weights = ef.portfolio_at_volatility(target_vol=0.15)
        >>>
        >>> # Plot frontier
        >>> ef.plot()
    """

    def __init__(
        self,
        returns: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        expected_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        self._portfolio = Portfolio(
            returns=returns,
            covariance=covariance,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

        self._rf = risk_free_rate
        self._periods = periods_per_year
        self._frontier: List[FrontierPoint] = []
        self._min_var_portfolio: Optional[PortfolioResult] = None
        self._max_sharpe_portfolio: Optional[PortfolioResult] = None

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self._portfolio.n_assets

    @property
    def frontier(self) -> List[FrontierPoint]:
        """Computed frontier points."""
        return self._frontier

    @property
    def min_variance_portfolio(self) -> Optional[PortfolioResult]:
        """Minimum variance portfolio."""
        return self._min_var_portfolio

    @property
    def max_sharpe_portfolio(self) -> Optional[PortfolioResult]:
        """Maximum Sharpe ratio portfolio."""
        return self._max_sharpe_portfolio

    def set_bounds(
        self,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> EfficientFrontier:
        """Set weight bounds."""
        self._portfolio.set_bounds(lower, upper)
        return self

    def compute(
        self,
        n_points: int = 50,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
    ) -> List[FrontierPoint]:
        """
        Compute the efficient frontier.

        Args:
            n_points: Number of frontier points
            min_return: Minimum return (default: min variance return)
            max_return: Maximum return (default: max asset return)

        Returns:
            List of FrontierPoint objects
        """
        # Get min variance portfolio
        self._min_var_portfolio = self._portfolio.optimize(method="min_variance")
        min_var_return = self._min_var_portfolio.expected_return

        # Get max sharpe portfolio
        try:
            self._max_sharpe_portfolio = self._portfolio.optimize(method="max_sharpe")
        except Exception:
            self._max_sharpe_portfolio = None

        # Determine return range
        if min_return is None:
            min_return = min_var_return

        if max_return is None:
            max_return = float(self._portfolio.expected_returns.max())

        # Ensure valid range
        if max_return <= min_return:
            max_return = min_return + 0.01

        # Compute frontier points
        target_returns = np.linspace(min_return, max_return, n_points)
        self._frontier = []

        for target in target_returns:
            try:
                result = self._portfolio.optimize(
                    method="target_return",
                    target_return=target,
                )

                if result.volatility > 0:  # Valid point
                    self._frontier.append(
                        FrontierPoint(
                            return_=result.expected_return,
                            volatility=result.volatility,
                            sharpe_ratio=result.sharpe_ratio,
                            weights=result.weights,
                        )
                    )
            except Exception:
                continue

        return self._frontier

    def portfolio_at_return(
        self,
        target_return: float,
    ) -> np.ndarray:
        """
        Get portfolio weights for target return.

        Args:
            target_return: Target expected return

        Returns:
            Optimal weights
        """
        result = self._portfolio.optimize(
            method="target_return",
            target_return=target_return,
        )
        return result.weights

    def portfolio_at_volatility(
        self,
        target_volatility: float,
    ) -> np.ndarray:
        """
        Get portfolio weights for target volatility.

        Args:
            target_volatility: Target volatility

        Returns:
            Optimal weights
        """
        result = self._portfolio.optimize(
            method="target_volatility",
            target_volatility=target_volatility,
        )
        return result.weights

    def get_returns_volatilities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get arrays of returns and volatilities from frontier.

        Returns:
            Tuple of (returns, volatilities) arrays
        """
        if not self._frontier:
            raise ValueError("Frontier not computed. Call compute() first.")

        returns = np.array([p.return_ for p in self._frontier])
        vols = np.array([p.volatility for p in self._frontier])

        return returns, vols

    def plot(
        self,
        show_assets: bool = True,
        show_min_var: bool = True,
        show_max_sharpe: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Efficient Frontier",
    ):
        """
        Plot the efficient frontier.

        Args:
            show_assets: Show individual asset points
            show_min_var: Show minimum variance portfolio
            show_max_sharpe: Show maximum Sharpe portfolio
            figsize: Figure size
            title: Plot title

        Returns:
            matplotlib figure (if matplotlib available)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting. Install with: pip install matplotlib")
            return None

        if not self._frontier:
            self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot frontier
        returns, vols = self.get_returns_volatilities()
        ax.plot(vols, returns, "b-", linewidth=2, label="Efficient Frontier")

        # Plot special points
        if show_min_var and self._min_var_portfolio:
            ax.scatter(
                [self._min_var_portfolio.volatility],
                [self._min_var_portfolio.expected_return],
                marker="o",
                s=100,
                c="green",
                label=f"Min Variance (SR={self._min_var_portfolio.sharpe_ratio:.2f})",
            )

        if show_max_sharpe and self._max_sharpe_portfolio:
            ax.scatter(
                [self._max_sharpe_portfolio.volatility],
                [self._max_sharpe_portfolio.expected_return],
                marker="*",
                s=200,
                c="red",
                label=f"Max Sharpe (SR={self._max_sharpe_portfolio.sharpe_ratio:.2f})",
            )

        # Plot individual assets
        if show_assets:
            asset_returns = self._portfolio.expected_returns
            asset_vols = np.sqrt(np.diag(self._portfolio.covariance))
            ax.scatter(
                asset_vols,
                asset_returns,
                marker="^",
                s=80,
                c="gray",
                alpha=0.6,
                label="Individual Assets",
            )

        # Capital Market Line (if risk-free rate provided)
        if self._rf > 0 and self._max_sharpe_portfolio:
            max_vol = max(vols) * 1.2
            cml_vols = np.linspace(0, max_vol, 100)
            cml_returns = self._rf + self._max_sharpe_portfolio.sharpe_ratio * cml_vols
            ax.plot(cml_vols, cml_returns, "r--", alpha=0.5, label="Capital Market Line")

        ax.set_xlabel("Volatility (Annualized)")
        ax.set_ylabel("Expected Return (Annualized)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

        plt.tight_layout()

        return fig

    def to_dataframe(self):
        """
        Convert frontier to pandas DataFrame.

        Returns:
            DataFrame with columns: return, volatility, sharpe_ratio
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required. Install with: pip install pandas")

        if not self._frontier:
            raise ValueError("Frontier not computed. Call compute() first.")

        return pd.DataFrame(
            {
                "return": [p.return_ for p in self._frontier],
                "volatility": [p.volatility for p in self._frontier],
                "sharpe_ratio": [p.sharpe_ratio for p in self._frontier],
            }
        )
