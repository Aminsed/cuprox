"""
Finance Utility Functions
=========================

Helper functions for financial data processing and analysis.
"""

from __future__ import annotations

from typing import Union, Optional
import numpy as np

# Type aliases
ArrayLike = Union[np.ndarray, list]


def compute_returns(
    prices: ArrayLike,
    method: str = "simple",
    periods: int = 1,
) -> np.ndarray:
    """
    Compute returns from price data.
    
    Args:
        prices: Price series (T,) or matrix (T, N) for N assets
        method: 'simple' for arithmetic returns, 'log' for log returns
        periods: Number of periods for return calculation (default: 1)
    
    Returns:
        Returns array with shape (T-periods, N)
    
    Example:
        >>> prices = np.array([100, 102, 101, 105])
        >>> returns = compute_returns(prices)
        >>> print(returns)  # [0.02, -0.0098, 0.0396]
    """
    prices = np.asarray(prices, dtype=np.float64)
    
    if method == "simple":
        returns = prices[periods:] / prices[:-periods] - 1
    elif method == "log":
        returns = np.log(prices[periods:] / prices[:-periods])
    else:
        raise ValueError(f"method must be 'simple' or 'log', got '{method}'")
    
    return returns


def compute_covariance(
    returns: ArrayLike,
    method: str = "sample",
    shrinkage: float = 0.0,
) -> np.ndarray:
    """
    Compute covariance matrix from returns.
    
    Args:
        returns: Return matrix (T, N) for N assets
        method: 'sample' for sample covariance, 'ledoit_wolf' for shrinkage
        shrinkage: Shrinkage intensity for manual shrinkage (0 to 1)
    
    Returns:
        Covariance matrix (N, N)
    
    Example:
        >>> returns = np.random.randn(100, 5)
        >>> cov = compute_covariance(returns, method='sample')
        >>> print(cov.shape)  # (5, 5)
    """
    returns = np.asarray(returns, dtype=np.float64)
    
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    
    if method == "sample":
        cov = np.cov(returns, rowvar=False)
    elif method == "ledoit_wolf":
        cov = _ledoit_wolf_shrinkage(returns)
    else:
        raise ValueError(f"method must be 'sample' or 'ledoit_wolf', got '{method}'")
    
    # Apply manual shrinkage if specified
    if shrinkage > 0:
        n = cov.shape[0]
        target = np.trace(cov) / n * np.eye(n)
        cov = (1 - shrinkage) * cov + shrinkage * target
    
    # Ensure symmetric
    cov = (cov + cov.T) / 2
    
    return cov


def _ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage estimator for covariance."""
    T, N = returns.shape
    
    # Sample covariance
    mean = returns.mean(axis=0)
    centered = returns - mean
    sample_cov = centered.T @ centered / T
    
    # Shrinkage target: scaled identity
    mu = np.trace(sample_cov) / N
    target = mu * np.eye(N)
    
    # Optimal shrinkage intensity (simplified)
    delta = sample_cov - target
    shrinkage = min(1.0, (np.sum(delta ** 2) / T) / np.sum(delta ** 2))
    
    return (1 - shrinkage) * sample_cov + shrinkage * target


def annualize_returns(
    returns: float,
    periods_per_year: int = 252,
) -> float:
    """
    Annualize returns.
    
    Args:
        returns: Period returns (e.g., daily)
        periods_per_year: Trading periods per year (252 for daily)
    
    Returns:
        Annualized return
    
    Example:
        >>> daily_return = 0.0005
        >>> annual = annualize_returns(daily_return)
        >>> print(f"{annual:.2%}")  # ~13.4%
    """
    return (1 + returns) ** periods_per_year - 1


def annualize_volatility(
    volatility: float,
    periods_per_year: int = 252,
) -> float:
    """
    Annualize volatility (standard deviation).
    
    Args:
        volatility: Period volatility
        periods_per_year: Trading periods per year (252 for daily)
    
    Returns:
        Annualized volatility
    
    Example:
        >>> daily_vol = 0.01
        >>> annual_vol = annualize_volatility(daily_vol)
        >>> print(f"{annual_vol:.2%}")  # ~15.9%
    """
    return volatility * np.sqrt(periods_per_year)


def validate_weights(
    weights: ArrayLike,
    n_assets: int,
    allow_negative: bool = False,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Validate and normalize portfolio weights.
    
    Args:
        weights: Portfolio weights
        n_assets: Expected number of assets
        allow_negative: Allow short positions
        tolerance: Tolerance for sum-to-one check
    
    Returns:
        Validated weights array
    
    Raises:
        ValueError: If weights are invalid
    """
    weights = np.asarray(weights, dtype=np.float64)
    
    if weights.shape != (n_assets,):
        raise ValueError(f"Expected {n_assets} weights, got {weights.shape}")
    
    if not allow_negative and (weights < -tolerance).any():
        raise ValueError("Negative weights not allowed")
    
    if abs(weights.sum() - 1.0) > tolerance:
        raise ValueError(f"Weights must sum to 1, got {weights.sum():.6f}")
    
    return weights


def is_positive_definite(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check if matrix is positive definite."""
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues > -tolerance)
    except np.linalg.LinAlgError:
        return False


def make_positive_definite(
    matrix: np.ndarray,
    min_eigenvalue: float = 1e-8,
) -> np.ndarray:
    """
    Ensure matrix is positive definite by adjusting eigenvalues.
    
    Args:
        matrix: Input matrix (should be symmetric)
        min_eigenvalue: Minimum eigenvalue to enforce
    
    Returns:
        Positive definite matrix
    """
    # Handle scalar case (single asset)
    if matrix.ndim == 0:
        return np.maximum(matrix.reshape(1, 1), min_eigenvalue)
    
    if matrix.shape == (1, 1):
        return np.maximum(matrix, min_eigenvalue)
    
    # Ensure symmetric
    matrix = (matrix + matrix.T) / 2
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Clip eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

