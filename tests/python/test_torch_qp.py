"""
Tests for QPLayer - Differentiable Quadratic Programming.

These tests verify:
1. Forward pass produces correct solutions
2. Backward pass computes correct gradients
3. Gradients match finite differences
"""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dtype():
    return torch.float64


class TestQPLayerForward:
    """Forward pass tests."""

    def test_unconstrained_qp(self, device, dtype):
        """
        Unconstrained QP: min (1/2)x'Px + q'x
        P = 2I, q = [-2, -4] => x* = [1, 2], obj = -5
        """
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2)
        P = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype)
        q = torch.tensor([-2.0, -4.0], device=device, dtype=dtype)

        x = layer(P, q)

        assert x.shape == (2,)
        torch.testing.assert_close(
            x, torch.tensor([1.0, 2.0], device=device, dtype=dtype), atol=1e-3, rtol=1e-3
        )

    def test_box_constrained_qp(self, device, dtype):
        """
        Box-constrained QP: min (1/2)x'Px + q'x s.t. 0 <= x <= 1
        Unconstrained x* = [1.5, 1.5], with bounds x* = [1, 1]
        """
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2)
        P = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype)
        q = torch.tensor([-3.0, -3.0], device=device, dtype=dtype)
        lb = torch.zeros(2, device=device, dtype=dtype)
        ub = torch.ones(2, device=device, dtype=dtype)

        x = layer(P, q, lb=lb, ub=ub)

        torch.testing.assert_close(
            x, torch.tensor([1.0, 1.0], device=device, dtype=dtype), atol=1e-3, rtol=1e-3
        )

    def test_equality_constrained_qp(self, device, dtype):
        """
        Equality-constrained QP: min (1/2)||x||^2 s.t. x + y = 2
        Optimal: x = y = 1
        """
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2, n_eq=1)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.zeros(2, device=device, dtype=dtype)
        A = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        b = torch.tensor([2.0], device=device, dtype=dtype)

        x = layer(P, q, A=A, b=b)

        # Check constraint
        torch.testing.assert_close(A @ x, b, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            x, torch.tensor([1.0, 1.0], device=device, dtype=dtype), atol=1e-2, rtol=1e-2
        )

    def test_inequality_constrained_qp(self, device, dtype):
        """
        Inequality-constrained QP: min (1/2)||x||^2 - 2*sum(x) s.t. sum(x) <= 1
        """
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2, n_ineq=1)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-2.0, -2.0], device=device, dtype=dtype)
        G = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
        h = torch.tensor([1.0], device=device, dtype=dtype)
        lb = torch.zeros(2, device=device, dtype=dtype)

        x = layer(P, q, G=G, h=h, lb=lb)

        # Check constraint satisfied
        assert (G @ x <= h + 1e-3).all()

    def test_single_variable(self, device, dtype):
        """QP with single variable."""
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=1)
        P = torch.tensor([[2.0]], device=device, dtype=dtype)
        q = torch.tensor([-4.0], device=device, dtype=dtype)

        x = layer(P, q)

        torch.testing.assert_close(
            x, torch.tensor([2.0], device=device, dtype=dtype), atol=1e-3, rtol=1e-3
        )

    def test_larger_problem(self, device, dtype):
        """QP with 50 variables."""
        from cuprox.torch import QPLayer

        n = 50
        layer = QPLayer(n_vars=n)

        # Random positive definite P
        torch.manual_seed(42)
        A = torch.randn(n, n, device=device, dtype=dtype)
        P = A @ A.T + torch.eye(n, device=device, dtype=dtype)
        q = torch.randn(n, device=device, dtype=dtype)

        x = layer(P, q)

        assert x.shape == (n,)
        # Check optimality: Px + q ≈ 0
        residual = P @ x + q
        rel_error = residual.norm() / (P.norm() * x.norm() + q.norm())
        assert rel_error < 0.1


class TestQPLayerBackward:
    """Backward pass (gradient) tests."""

    def test_gradient_wrt_q(self, device, dtype):
        """Gradient of loss w.r.t. linear cost q."""
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2)
        P = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype)
        q = torch.tensor([-2.0, -4.0], device=device, dtype=dtype, requires_grad=True)

        x = layer(P, q)
        loss = x.sum()
        loss.backward()

        # dx*/dq = -P^{-1}, so d(sum(x*))/dq = [1,1] @ (-P^{-1}) = [-0.5, -0.5]
        expected = torch.tensor([-0.5, -0.5], device=device, dtype=dtype)

        assert q.grad is not None
        torch.testing.assert_close(q.grad, expected, atol=1e-2, rtol=1e-2)

    def test_gradient_wrt_P(self, device, dtype):
        """Gradient of loss w.r.t. quadratic cost P."""
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2)
        P = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype, requires_grad=True)
        q = torch.tensor([-2.0, -4.0], device=device, dtype=dtype)

        x = layer(P, q)
        loss = (x**2).sum()
        loss.backward()

        assert P.grad is not None
        assert P.grad.abs().sum() > 0

    def test_gradient_finite_difference(self, device, dtype):
        """Verify gradients match finite differences."""
        from cuprox.torch import solve_qp

        P = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device=device, dtype=dtype)
        q = torch.tensor([-2.0, -4.0], device=device, dtype=dtype, requires_grad=True)

        # Analytical gradient
        x = solve_qp(P, q)
        x.sum().backward()
        analytical = q.grad.clone()

        # Finite difference
        eps = 1e-5
        fd_grad = torch.zeros_like(q)
        for i in range(len(q)):
            q_plus = q.detach().clone()
            q_plus[i] += eps
            x_plus = solve_qp(P, q_plus)

            q_minus = q.detach().clone()
            q_minus[i] -= eps
            x_minus = solve_qp(P, q_minus)

            fd_grad[i] = (x_plus.sum() - x_minus.sum()) / (2 * eps)

        torch.testing.assert_close(analytical, fd_grad, atol=1e-3, rtol=1e-3)

    def test_no_grad_mode(self, device, dtype):
        """Test layer with differentiable=False."""
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2, differentiable=False)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1.0, -2.0], device=device, dtype=dtype, requires_grad=True)

        x = layer(P, q)

        assert x.shape == (2,)
        assert not x.requires_grad


class TestQPInputValidation:
    """Input validation tests."""

    def test_wrong_P_shape(self, device, dtype):
        """Error on wrong P shape."""
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2)
        P = torch.eye(3, device=device, dtype=dtype)
        q = torch.zeros(2, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="P must have shape"):
            layer(P, q)

    def test_wrong_q_shape(self, device, dtype):
        """Error on wrong q shape."""
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.zeros(3, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="q must have shape"):
            layer(P, q)

    def test_missing_constraints(self, device, dtype):
        """Error when constraint matrices missing."""
        from cuprox.torch import QPLayer

        layer = QPLayer(n_vars=2, n_eq=1)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.zeros(2, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="A and b required"):
            layer(P, q)

    def test_invalid_n_vars(self):
        """Error on invalid n_vars."""
        from cuprox.torch import QPLayer

        with pytest.raises(ValueError, match="n_vars must be positive"):
            QPLayer(n_vars=0)

        with pytest.raises(ValueError, match="n_vars must be positive"):
            QPLayer(n_vars=-1)


class TestQPFunctionalAPI:
    """Tests for solve_qp function."""

    def test_basic(self, device, dtype):
        """Basic solve_qp usage."""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1.0, -2.0], device=device, dtype=dtype, requires_grad=True)

        x = solve_qp(P, q)

        assert x.shape == (2,)
        torch.testing.assert_close(
            x, torch.tensor([1.0, 2.0], device=device, dtype=dtype), atol=1e-3, rtol=1e-3
        )

        x.sum().backward()
        assert q.grad is not None
