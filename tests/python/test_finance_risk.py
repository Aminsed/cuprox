"""
Tests for Risk Metrics.

Comprehensive tests covering:
1. Volatility
2. Value at Risk (VaR)
3. Conditional VaR (CVaR)
4. Maximum Drawdown
5. Sharpe Ratio
6. Sortino Ratio
7. Beta and Tracking Error
"""

import numpy as np
import pytest


@pytest.fixture
def risk_returns():
    """Return data for risk testing."""
    np.random.seed(42)
    # 504 trading days (2 years), 5 assets
    returns = np.random.randn(504, 5) * 0.02
    # Add some structure
    returns[:, 0] += 0.0003  # High return, high vol
    returns[:, 1] += 0.0002
    returns[:, 2] += 0.0001
    returns[:, 3] += 0.0000
    returns[:, 4] -= 0.0001  # Low return
    return returns


@pytest.fixture
def equal_weights():
    """Equal weights for 5 assets."""
    return np.array([0.2, 0.2, 0.2, 0.2, 0.2])


@pytest.fixture
def benchmark_returns(risk_returns):
    """Market benchmark returns."""
    # Simple market average
    return risk_returns.mean(axis=1)


class TestRiskMetricsConstruction:
    """Test RiskMetrics initialization."""

    def test_from_returns(self, risk_returns):
        """Create from returns."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)

        assert risk.n_assets == 5
        assert risk.n_periods == 504

    def test_with_covariance(self, risk_returns):
        """Create with explicit covariance."""
        from cuprox.finance import RiskMetrics

        cov = np.cov(risk_returns, rowvar=False)
        risk = RiskMetrics(risk_returns, covariance=cov)

        np.testing.assert_allclose(risk._cov, cov)

    def test_single_asset(self):
        """Single asset returns."""
        from cuprox.finance import RiskMetrics

        returns = np.random.randn(100) * 0.02
        risk = RiskMetrics(returns)

        assert risk.n_assets == 1


class TestVolatility:
    """Test volatility calculation."""

    def test_basic_volatility(self, risk_returns, equal_weights):
        """Basic volatility."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        vol = risk.volatility(equal_weights)

        assert vol > 0
        assert vol < 1.0  # Should be reasonable

    def test_annualized_vs_daily(self, risk_returns, equal_weights):
        """Annualized vs non-annualized."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)

        annual_vol = risk.volatility(equal_weights, annualize=True)
        daily_vol = risk.volatility(equal_weights, annualize=False)

        expected_ratio = np.sqrt(252)
        assert abs(annual_vol / daily_vol - expected_ratio) < 0.1

    def test_concentrated_portfolio(self, risk_returns):
        """Concentrated portfolio has different vol."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)

        equal = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        concentrated = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        vol_equal = risk.volatility(equal)
        vol_conc = risk.volatility(concentrated)

        # Should be different (diversification effect)
        assert vol_equal != vol_conc


class TestVaR:
    """Test Value at Risk."""

    def test_historical_var_95(self, risk_returns, equal_weights):
        """95% VaR (historical)."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        var_95 = risk.var(equal_weights, alpha=0.05, method="historical")

        # VaR should be positive (it's a loss)
        assert var_95 > 0

    def test_historical_var_99(self, risk_returns, equal_weights):
        """99% VaR should be >= 95% VaR."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)

        var_95 = risk.var(equal_weights, alpha=0.05, method="historical")
        var_99 = risk.var(equal_weights, alpha=0.01, method="historical")

        assert var_99 >= var_95

    def test_parametric_var(self, risk_returns, equal_weights):
        """Parametric VaR."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        var_param = risk.var(equal_weights, alpha=0.05, method="parametric")

        assert var_param > 0

    def test_var_invalid_method(self, risk_returns, equal_weights):
        """Invalid method raises error."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)

        with pytest.raises(ValueError, match="method must be"):
            risk.var(equal_weights, method="invalid")


class TestCVaR:
    """Test Conditional Value at Risk."""

    def test_historical_cvar_95(self, risk_returns, equal_weights):
        """95% CVaR (historical)."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        cvar_95 = risk.cvar(equal_weights, alpha=0.05, method="historical")

        assert cvar_95 > 0

    def test_cvar_geq_var(self, risk_returns, equal_weights):
        """CVaR should be >= VaR."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)

        var_95 = risk.var(equal_weights, alpha=0.05)
        cvar_95 = risk.cvar(equal_weights, alpha=0.05)

        assert cvar_95 >= var_95 - 1e-6

    def test_parametric_cvar(self, risk_returns, equal_weights):
        """Parametric CVaR."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        cvar_param = risk.cvar(equal_weights, alpha=0.05, method="parametric")

        assert cvar_param > 0


class TestMaxDrawdown:
    """Test maximum drawdown."""

    def test_basic_max_drawdown(self, risk_returns, equal_weights):
        """Basic max drawdown."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        mdd = risk.max_drawdown(equal_weights)

        assert 0 <= mdd <= 1.0  # Between 0% and 100%

    def test_drawdown_positive_returns(self):
        """Drawdown with consistently positive returns."""
        from cuprox.finance import RiskMetrics

        # Consistently positive returns (small drawdowns)
        np.random.seed(123)
        returns = np.abs(np.random.randn(100, 3)) * 0.01

        risk = RiskMetrics(returns)
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        mdd = risk.max_drawdown(weights)

        # Should be relatively small
        assert mdd < 0.2

    def test_drawdown_negative_returns(self):
        """Drawdown with negative returns."""
        from cuprox.finance import RiskMetrics

        # Consistently negative returns (large drawdowns)
        returns = -np.abs(np.random.randn(100, 3)) * 0.02

        risk = RiskMetrics(returns)
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        mdd = risk.max_drawdown(weights)

        # Should be large
        assert mdd > 0.3


class TestSharpeRatio:
    """Test Sharpe ratio."""

    def test_basic_sharpe(self, risk_returns, equal_weights):
        """Basic Sharpe ratio."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        sharpe = risk.sharpe_ratio(equal_weights)

        # Should be a reasonable value
        assert -5 < sharpe < 5

    def test_sharpe_with_risk_free(self, risk_returns, equal_weights):
        """Sharpe with risk-free rate."""
        from cuprox.finance import RiskMetrics

        risk_zero_rf = RiskMetrics(risk_returns, risk_free_rate=0.0)
        risk_with_rf = RiskMetrics(risk_returns, risk_free_rate=0.05)

        sharpe_zero = risk_zero_rf.sharpe_ratio(equal_weights)
        sharpe_rf = risk_with_rf.sharpe_ratio(equal_weights)

        # Sharpe with rf should be lower
        assert sharpe_rf < sharpe_zero

    def test_annualized_vs_daily_sharpe(self, risk_returns, equal_weights):
        """Annualized vs daily Sharpe."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)

        annual_sharpe = risk.sharpe_ratio(equal_weights, annualize=True)
        daily_sharpe = risk.sharpe_ratio(equal_weights, annualize=False)

        expected_ratio = np.sqrt(252)
        assert abs(annual_sharpe / daily_sharpe - expected_ratio) < 0.5


class TestSortinoRatio:
    """Test Sortino ratio."""

    def test_basic_sortino(self, risk_returns, equal_weights):
        """Basic Sortino ratio."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        sortino = risk.sortino_ratio(equal_weights)

        # Should be a number (could be inf if no downside)
        assert not np.isnan(sortino)

    def test_sortino_with_target(self, risk_returns, equal_weights):
        """Sortino with target return."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        sortino = risk.sortino_ratio(equal_weights, target=0.001)

        assert not np.isnan(sortino)


class TestBetaAndTracking:
    """Test beta and tracking error."""

    def test_beta(self, risk_returns, equal_weights, benchmark_returns):
        """Test beta calculation."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        beta = risk.beta(equal_weights, benchmark_returns)

        # Equal-weighted portfolio vs market average should have beta ~1
        assert 0.5 < beta < 1.5

    def test_tracking_error(self, risk_returns, equal_weights, benchmark_returns):
        """Test tracking error."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        te = risk.tracking_error(equal_weights, benchmark_returns)

        assert te >= 0

    def test_information_ratio(self, risk_returns, equal_weights, benchmark_returns):
        """Test information ratio."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        ir = risk.information_ratio(equal_weights, benchmark_returns)

        # Should be a reasonable value
        assert -10 < ir < 10


class TestFullAnalysis:
    """Test comprehensive risk analysis."""

    def test_full_analysis(self, risk_returns, equal_weights):
        """Full risk analysis."""
        from cuprox.finance import RiskMetrics, RiskResult

        risk = RiskMetrics(risk_returns)
        result = risk.full_analysis(equal_weights)

        assert isinstance(result, RiskResult)
        assert result.volatility > 0
        assert result.var_99 >= result.var_95
        assert result.cvar_99 >= result.cvar_95

    def test_result_summary(self, risk_returns, equal_weights):
        """Test result summary."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        result = risk.full_analysis(equal_weights)

        summary = result.summary()

        assert "Risk Analysis Summary" in summary
        assert "Volatility" in summary
        assert "VaR" in summary
        assert "Sharpe" in summary

    def test_result_repr(self, risk_returns, equal_weights):
        """Test result repr."""
        from cuprox.finance import RiskMetrics

        risk = RiskMetrics(risk_returns)
        result = risk.full_analysis(equal_weights)

        repr_str = repr(result)

        assert "RiskResult" in repr_str
        assert "volatility" in repr_str


class TestUtilityFunctions:
    """Test utility functions."""

    def test_compute_returns(self):
        """Test return computation."""
        from cuprox.finance import compute_returns

        prices = np.array([100, 102, 101, 105])

        simple_returns = compute_returns(prices, method="simple")
        log_returns = compute_returns(prices, method="log")

        # First return: (102-100)/100 = 0.02
        assert abs(simple_returns[0] - 0.02) < 1e-6

        # Log returns should be close to simple for small returns
        np.testing.assert_allclose(simple_returns, log_returns, atol=0.01)

    def test_compute_covariance(self):
        """Test covariance computation."""
        from cuprox.finance import compute_covariance

        np.random.seed(42)
        returns = np.random.randn(100, 3) * 0.02

        cov = compute_covariance(returns, method="sample")

        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)  # Symmetric
        assert (np.linalg.eigvalsh(cov) >= -1e-10).all()  # PSD

    def test_annualize_returns(self):
        """Test return annualization."""
        from cuprox.finance import annualize_returns

        daily_return = 0.0005
        annual = annualize_returns(daily_return)

        # (1.0005)^252 - 1 ≈ 0.134
        assert abs(annual - 0.134) < 0.01

    def test_annualize_volatility(self):
        """Test volatility annualization."""
        from cuprox.finance import annualize_volatility

        daily_vol = 0.01
        annual_vol = annualize_volatility(daily_vol)

        # 0.01 * sqrt(252) ≈ 0.159
        expected = 0.01 * np.sqrt(252)
        assert abs(annual_vol - expected) < 1e-6
