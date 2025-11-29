"""
Risk Metrics and Analysis
=========================

Comprehensive risk measurement tools for portfolio analysis.

Metrics:
- Volatility (standard deviation)
- Value at Risk (VaR)
- Conditional VaR (CVaR / Expected Shortfall)
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Information Ratio
- Beta
- Tracking Error
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class RiskResult:
    """
    Risk analysis result.
    
    Attributes:
        volatility: Annualized volatility
        var_95: Value at Risk (95% confidence)
        var_99: Value at Risk (99% confidence)
        cvar_95: Conditional VaR (95%)
        cvar_99: Conditional VaR (99%)
        max_drawdown: Maximum drawdown
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        skewness: Return skewness
        kurtosis: Return kurtosis
    """
    volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    skewness: float
    kurtosis: float
    
    def __repr__(self) -> str:
        return (
            f"RiskResult(\n"
            f"  volatility={self.volatility:.4%},\n"
            f"  var_95={self.var_95:.4%},\n"
            f"  cvar_95={self.cvar_95:.4%},\n"
            f"  max_drawdown={self.max_drawdown:.4%},\n"
            f"  sharpe_ratio={self.sharpe_ratio:.4f}\n"
            f")"
        )
    
    def summary(self) -> str:
        """Return formatted summary."""
        return (
            "=" * 50 + "\n"
            "Risk Analysis Summary\n"
            "=" * 50 + "\n"
            f"Volatility (annual):  {self.volatility:.4%}\n"
            f"VaR (95%):           {self.var_95:.4%}\n"
            f"VaR (99%):           {self.var_99:.4%}\n"
            f"CVaR (95%):          {self.cvar_95:.4%}\n"
            f"CVaR (99%):          {self.cvar_99:.4%}\n"
            f"Max Drawdown:        {self.max_drawdown:.4%}\n"
            "-" * 50 + "\n"
            f"Sharpe Ratio:        {self.sharpe_ratio:.4f}\n"
            f"Sortino Ratio:       {self.sortino_ratio:.4f}\n"
            "-" * 50 + "\n"
            f"Skewness:            {self.skewness:.4f}\n"
            f"Kurtosis:            {self.kurtosis:.4f}\n"
            "=" * 50
        )


class RiskMetrics:
    """
    Risk measurement and analysis.
    
    Provides comprehensive risk metrics for portfolio analysis
    including volatility, VaR, CVaR, drawdown, and risk-adjusted
    return measures.
    
    Args:
        returns: Historical returns matrix (T, N) for N assets
        covariance: Covariance matrix (optional, computed if not provided)
        periods_per_year: Trading periods per year (252 for daily)
        risk_free_rate: Annual risk-free rate
    
    Example:
        >>> returns = np.random.randn(252, 5) * 0.02
        >>> risk = RiskMetrics(returns)
        >>> 
        >>> weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        >>> print(f"Volatility: {risk.volatility(weights):.2%}")
        >>> print(f"VaR (95%): {risk.var(weights, alpha=0.05):.2%}")
        >>> print(f"Sharpe: {risk.sharpe_ratio(weights):.2f}")
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        covariance: Optional[np.ndarray] = None,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        self._returns = np.asarray(returns, dtype=np.float64)
        
        if self._returns.ndim == 1:
            self._returns = self._returns.reshape(-1, 1)
        
        self._T, self._N = self._returns.shape
        self._periods = periods_per_year
        self._rf = risk_free_rate / periods_per_year
        
        # Compute covariance if not provided
        if covariance is not None:
            self._cov = np.asarray(covariance, dtype=np.float64)
        else:
            self._cov = np.cov(self._returns, rowvar=False)
            if self._cov.ndim == 0:  # Single asset
                self._cov = self._cov.reshape(1, 1)
        
        self._mu = self._returns.mean(axis=0)
    
    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self._N
    
    @property
    def n_periods(self) -> int:
        """Number of time periods."""
        return self._T
    
    def portfolio_returns(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute portfolio return series.
        
        Args:
            weights: Portfolio weights (N,)
        
        Returns:
            Portfolio returns (T,)
        """
        weights = np.asarray(weights, dtype=np.float64)
        return self._returns @ weights
    
    def volatility(
        self,
        weights: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Compute portfolio volatility.
        
        Args:
            weights: Portfolio weights (N,)
            annualize: Whether to annualize (default: True)
        
        Returns:
            Portfolio volatility
        """
        weights = np.asarray(weights, dtype=np.float64)
        variance = weights @ self._cov @ weights
        vol = np.sqrt(max(variance, 0))
        
        if annualize:
            vol *= np.sqrt(self._periods)
        
        return float(vol)
    
    def var(
        self,
        weights: np.ndarray,
        alpha: float = 0.05,
        method: str = "historical",
    ) -> float:
        """
        Compute Value at Risk.
        
        VaR is the loss threshold that is exceeded with probability alpha.
        A 95% VaR (alpha=0.05) means there is a 5% chance of losses
        exceeding this value.
        
        Args:
            weights: Portfolio weights (N,)
            alpha: Significance level (default: 0.05 for 95% VaR)
            method: 'historical' or 'parametric'
        
        Returns:
            VaR as a positive number (loss)
        
        Example:
            >>> var_95 = risk.var(weights, alpha=0.05)
            >>> print(f"5% chance of losing more than {var_95:.2%}")
        """
        port_returns = self.portfolio_returns(weights)
        
        if method == "historical":
            var = -np.percentile(port_returns, 100 * alpha)
        elif method == "parametric":
            from scipy import stats
            mu = port_returns.mean()
            sigma = port_returns.std()
            var = -(mu + stats.norm.ppf(alpha) * sigma)
        else:
            raise ValueError(f"method must be 'historical' or 'parametric'")
        
        return float(var)
    
    def cvar(
        self,
        weights: np.ndarray,
        alpha: float = 0.05,
        method: str = "historical",
    ) -> float:
        """
        Compute Conditional Value at Risk (Expected Shortfall).
        
        CVaR is the expected loss given that losses exceed VaR.
        It's a more conservative risk measure than VaR.
        
        Args:
            weights: Portfolio weights (N,)
            alpha: Significance level (default: 0.05 for 95% CVaR)
            method: 'historical' or 'parametric'
        
        Returns:
            CVaR as a positive number (expected loss in tail)
        """
        port_returns = self.portfolio_returns(weights)
        
        if method == "historical":
            var_threshold = np.percentile(port_returns, 100 * alpha)
            tail_returns = port_returns[port_returns <= var_threshold]
            cvar = -tail_returns.mean() if len(tail_returns) > 0 else -var_threshold
        elif method == "parametric":
            from scipy import stats
            mu = port_returns.mean()
            sigma = port_returns.std()
            cvar = -(mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
        else:
            raise ValueError(f"method must be 'historical' or 'parametric'")
        
        return float(cvar)
    
    def max_drawdown(
        self,
        weights: np.ndarray,
    ) -> float:
        """
        Compute maximum drawdown.
        
        Maximum drawdown is the largest peak-to-trough decline
        in portfolio value.
        
        Args:
            weights: Portfolio weights (N,)
        
        Returns:
            Maximum drawdown as a positive number
        """
        port_returns = self.portfolio_returns(weights)
        
        # Compute cumulative returns
        cum_returns = np.cumprod(1 + port_returns)
        
        # Running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Drawdown series
        drawdowns = (running_max - cum_returns) / running_max
        
        return float(np.max(drawdowns))
    
    def sharpe_ratio(
        self,
        weights: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Compute Sharpe ratio.
        
        Sharpe = (E[R] - Rf) / σ[R]
        
        Args:
            weights: Portfolio weights (N,)
            annualize: Whether to annualize (default: True)
        
        Returns:
            Sharpe ratio
        """
        port_returns = self.portfolio_returns(weights)
        
        mean_return = port_returns.mean()
        std_return = port_returns.std()
        
        if std_return < 1e-10:
            return 0.0
        
        sharpe = (mean_return - self._rf) / std_return
        
        if annualize:
            sharpe *= np.sqrt(self._periods)
        
        return float(sharpe)
    
    def sortino_ratio(
        self,
        weights: np.ndarray,
        target: float = 0.0,
        annualize: bool = True,
    ) -> float:
        """
        Compute Sortino ratio.
        
        Like Sharpe, but uses downside deviation instead of
        total volatility.
        
        Args:
            weights: Portfolio weights (N,)
            target: Target return (default: 0)
            annualize: Whether to annualize
        
        Returns:
            Sortino ratio
        """
        port_returns = self.portfolio_returns(weights)
        
        mean_return = port_returns.mean()
        
        # Downside deviation
        downside = port_returns[port_returns < target]
        if len(downside) == 0:
            return float('inf')
        
        downside_std = np.sqrt(np.mean((downside - target) ** 2))
        
        if downside_std < 1e-10:
            return float('inf')
        
        sortino = (mean_return - self._rf) / downside_std
        
        if annualize:
            sortino *= np.sqrt(self._periods)
        
        return float(sortino)
    
    def beta(
        self,
        weights: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """
        Compute portfolio beta relative to benchmark.
        
        Beta = Cov(Rp, Rb) / Var(Rb)
        
        Args:
            weights: Portfolio weights (N,)
            benchmark_returns: Benchmark return series (T,)
        
        Returns:
            Portfolio beta
        """
        port_returns = self.portfolio_returns(weights)
        benchmark = np.asarray(benchmark_returns)
        
        cov = np.cov(port_returns, benchmark)[0, 1]
        var = np.var(benchmark)
        
        if var < 1e-10:
            return 0.0
        
        return float(cov / var)
    
    def tracking_error(
        self,
        weights: np.ndarray,
        benchmark_returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Compute tracking error relative to benchmark.
        
        Tracking error is the standard deviation of active returns.
        
        Args:
            weights: Portfolio weights (N,)
            benchmark_returns: Benchmark return series (T,)
            annualize: Whether to annualize
        
        Returns:
            Tracking error
        """
        port_returns = self.portfolio_returns(weights)
        benchmark = np.asarray(benchmark_returns)
        
        active_returns = port_returns - benchmark
        te = active_returns.std()
        
        if annualize:
            te *= np.sqrt(self._periods)
        
        return float(te)
    
    def information_ratio(
        self,
        weights: np.ndarray,
        benchmark_returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Compute information ratio.
        
        IR = Active Return / Tracking Error
        
        Args:
            weights: Portfolio weights (N,)
            benchmark_returns: Benchmark return series (T,)
            annualize: Whether to annualize
        
        Returns:
            Information ratio
        """
        port_returns = self.portfolio_returns(weights)
        benchmark = np.asarray(benchmark_returns)
        
        active_returns = port_returns - benchmark
        active_mean = active_returns.mean()
        active_std = active_returns.std()
        
        if active_std < 1e-10:
            return 0.0
        
        ir = active_mean / active_std
        
        if annualize:
            ir *= np.sqrt(self._periods)
        
        return float(ir)
    
    def full_analysis(
        self,
        weights: np.ndarray,
    ) -> RiskResult:
        """
        Perform comprehensive risk analysis.
        
        Args:
            weights: Portfolio weights (N,)
        
        Returns:
            RiskResult with all metrics
        """
        port_returns = self.portfolio_returns(weights)
        
        return RiskResult(
            volatility=self.volatility(weights),
            var_95=self.var(weights, alpha=0.05),
            var_99=self.var(weights, alpha=0.01),
            cvar_95=self.cvar(weights, alpha=0.05),
            cvar_99=self.cvar(weights, alpha=0.01),
            max_drawdown=self.max_drawdown(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            sortino_ratio=self.sortino_ratio(weights),
            skewness=float(_skewness(port_returns)),
            kurtosis=float(_kurtosis(port_returns)),
        )


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = x.mean()
    std = x.std()
    if std < 1e-10:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    mean = x.mean()
    std = x.std()
    if std < 1e-10:
        return 0.0
    return np.mean(((x - mean) / std) ** 4) - 3

