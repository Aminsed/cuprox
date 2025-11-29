"""
Probability Distributions for Stochastic Programming
=====================================================

Distribution classes for generating random scenarios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np


class Distribution(ABC):
    """Base class for distributions."""
    
    @abstractmethod
    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        pass
    
    @abstractmethod
    def mean(self) -> np.ndarray:
        """Distribution mean."""
        pass


@dataclass
class DiscreteDistribution(Distribution):
    """
    Discrete distribution with finite support.
    
    Args:
        values: Array of possible values (n_outcomes, dim) or (n_outcomes,)
        probabilities: Probability of each value (n_outcomes,)
    
    Example:
        >>> # Demand can be 10, 20, or 30 with probabilities 0.2, 0.5, 0.3
        >>> dist = DiscreteDistribution(
        ...     values=np.array([10, 20, 30]),
        ...     probabilities=np.array([0.2, 0.5, 0.3])
        ... )
        >>> sample = dist.sample()
    """
    values: np.ndarray
    probabilities: np.ndarray
    
    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        self.probabilities = np.asarray(self.probabilities, dtype=np.float64)
        
        # Normalize probabilities
        self.probabilities = self.probabilities / self.probabilities.sum()
        
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
    
    @property
    def n_outcomes(self) -> int:
        return len(self.probabilities)
    
    @property
    def dim(self) -> int:
        return self.values.shape[1]
    
    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Sample from discrete distribution."""
        idx = np.random.choice(
            self.n_outcomes,
            size=size,
            p=self.probabilities
        )
        
        if size is None:
            return self.values[idx].flatten() if self.dim == 1 else self.values[idx]
        else:
            return self.values[idx]
    
    def mean(self) -> np.ndarray:
        """Expected value."""
        return (self.probabilities.reshape(-1, 1) * self.values).sum(axis=0)


@dataclass
class NormalDistribution(Distribution):
    """
    Multivariate normal distribution.
    
    Args:
        mean_: Mean vector
        std: Standard deviation (scalar or vector)
        cov: Covariance matrix (optional, overrides std)
    
    Example:
        >>> dist = NormalDistribution(mean=np.array([10, 20]), std=2.0)
        >>> sample = dist.sample()
    """
    mean_: np.ndarray
    std: Optional[Union[float, np.ndarray]] = None
    cov: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.mean_ = np.asarray(self.mean_, dtype=np.float64)
        
        if self.cov is not None:
            self.cov = np.asarray(self.cov, dtype=np.float64)
        elif self.std is not None:
            if np.isscalar(self.std):
                self.cov = np.eye(len(self.mean_)) * self.std**2
            else:
                self.cov = np.diag(np.asarray(self.std)**2)
        else:
            self.cov = np.eye(len(self.mean_))
    
    @property
    def dim(self) -> int:
        return len(self.mean_)
    
    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Sample from multivariate normal."""
        if size is None:
            return np.random.multivariate_normal(self.mean_, self.cov)
        else:
            return np.random.multivariate_normal(self.mean_, self.cov, size=size)
    
    def mean(self) -> np.ndarray:
        return self.mean_.copy()


@dataclass
class UniformDistribution(Distribution):
    """
    Uniform distribution.
    
    Args:
        low: Lower bound
        high: Upper bound
    
    Example:
        >>> dist = UniformDistribution(low=np.array([5, 10]), high=np.array([15, 30]))
        >>> sample = dist.sample()
    """
    low: np.ndarray
    high: np.ndarray
    
    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=np.float64)
        self.high = np.asarray(self.high, dtype=np.float64)
    
    @property
    def dim(self) -> int:
        return len(self.low)
    
    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """Sample from uniform distribution."""
        if size is None:
            return np.random.uniform(self.low, self.high)
        else:
            samples = np.zeros((size, self.dim))
            for i in range(size):
                samples[i] = np.random.uniform(self.low, self.high)
            return samples
    
    def mean(self) -> np.ndarray:
        return (self.low + self.high) / 2


@dataclass
class LogNormalDistribution(Distribution):
    """
    Log-normal distribution (always positive).
    
    Args:
        mu: Mean of log(X)
        sigma: Std of log(X)
    """
    mu: np.ndarray
    sigma: np.ndarray
    
    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
    
    @property
    def dim(self) -> int:
        return len(self.mu)
    
    def sample(self, size: Optional[int] = None) -> np.ndarray:
        if size is None:
            return np.random.lognormal(self.mu, self.sigma)
        else:
            samples = np.zeros((size, self.dim))
            for i in range(size):
                samples[i] = np.random.lognormal(self.mu, self.sigma)
            return samples
    
    def mean(self) -> np.ndarray:
        return np.exp(self.mu + self.sigma**2 / 2)


def generate_scenarios(
    n_scenarios: int,
    distributions: List[Distribution],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate scenario matrix from multiple distributions.
    
    Args:
        n_scenarios: Number of scenarios
        distributions: List of distributions (one per uncertain parameter)
        seed: Random seed
    
    Returns:
        Scenario matrix (n_scenarios, sum of dims)
    
    Example:
        >>> demand_dist = NormalDistribution(mean=np.array([100]), std=10)
        >>> price_dist = UniformDistribution(low=np.array([5]), high=np.array([15]))
        >>> scenarios = generate_scenarios(1000, [demand_dist, price_dist])
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    for dist in distributions:
        s = dist.sample(size=n_scenarios)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        samples.append(s)
    
    return np.hstack(samples)


def latin_hypercube_sample(
    distributions: List[Distribution],
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Latin Hypercube Sampling for better coverage.
    
    Args:
        distributions: List of distributions
        n_samples: Number of samples
        seed: Random seed
    
    Returns:
        Sample matrix (n_samples, total_dim)
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_dim = sum(d.dim for d in distributions)
    
    # Generate LHS in unit hypercube
    samples = np.zeros((n_samples, total_dim))
    
    col = 0
    for dist in distributions:
        for d in range(dist.dim):
            # Stratified sampling
            perm = np.random.permutation(n_samples)
            samples[:, col] = (perm + np.random.rand(n_samples)) / n_samples
            col += 1
    
    # Transform to actual distributions using inverse CDF
    # For simplicity, we use the marginal approach
    result = np.zeros((n_samples, total_dim))
    col = 0
    
    for dist in distributions:
        if isinstance(dist, UniformDistribution):
            for d in range(dist.dim):
                result[:, col] = dist.low[d] + samples[:, col] * (dist.high[d] - dist.low[d])
                col += 1
        elif isinstance(dist, NormalDistribution):
            from scipy import stats
            std_diag = np.sqrt(np.diag(dist.cov))
            for d in range(dist.dim):
                result[:, col] = stats.norm.ppf(samples[:, col], dist.mean_[d], std_diag[d])
                col += 1
        else:
            # Fall back to regular sampling
            s = dist.sample(n_samples)
            if s.ndim == 1:
                s = s.reshape(-1, 1)
            result[:, col:col + dist.dim] = s
            col += dist.dim
    
    return result

