"""
cuProx Finance Module
=====================

GPU-accelerated portfolio optimization and risk management.

This module provides high-performance solvers for common finance
optimization problems, leveraging cuProx's GPU-accelerated QP solver.

Portfolio Optimization
----------------------
>>> from cuprox.finance import Portfolio
>>> 
>>> # Mean-Variance (Markowitz) optimization
>>> port = Portfolio(returns)
>>> weights = port.optimize(method='mean_variance', risk_aversion=2.0)
>>> 
>>> # Minimum variance portfolio
>>> weights = port.optimize(method='min_variance')
>>> 
>>> # Maximum Sharpe ratio
>>> weights = port.optimize(method='max_sharpe')
>>> 
>>> # Risk parity
>>> weights = port.optimize(method='risk_parity')

Risk Metrics
------------
>>> from cuprox.finance import RiskMetrics
>>> 
>>> risk = RiskMetrics(returns)
>>> print(f"Volatility: {risk.volatility(weights):.2%}")
>>> print(f"VaR (95%): {risk.var(weights, alpha=0.05):.2%}")
>>> print(f"CVaR (95%): {risk.cvar(weights, alpha=0.05):.2%}")
>>> print(f"Sharpe Ratio: {risk.sharpe_ratio(weights):.2f}")

Efficient Frontier
------------------
>>> from cuprox.finance import EfficientFrontier
>>> 
>>> ef = EfficientFrontier(returns)
>>> frontier = ef.compute(n_points=50)
>>> ef.plot()

Classes
-------
Portfolio
    Main portfolio optimization class
RiskMetrics
    Risk measurement and analysis
EfficientFrontier
    Efficient frontier computation and visualization
FactorModel
    Factor-based covariance estimation

See Also
--------
- Markowitz (1952): "Portfolio Selection"
- Black & Litterman (1992): "Global Portfolio Optimization"
"""

from .frontier import EfficientFrontier
from .portfolio import Portfolio, PortfolioResult
from .risk import RiskMetrics, RiskResult
from .utils import (
    annualize_returns,
    annualize_volatility,
    compute_covariance,
    compute_returns,
)

__all__ = [
    # Main classes
    "Portfolio",
    "PortfolioResult",
    "RiskMetrics",
    "RiskResult",
    "EfficientFrontier",
    # Utilities
    "compute_returns",
    "compute_covariance",
    "annualize_returns",
    "annualize_volatility",
]
