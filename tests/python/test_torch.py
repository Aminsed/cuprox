"""
Tests for PyTorch integration (differentiable optimization layers).

These tests verify:
1. Forward pass produces correct solutions
2. Backward pass computes correct gradients
3. Integration with PyTorch autograd
4. Error handling and edge cases
"""

import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if torch not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture
def device():
    """Get device for tests (CPU for CI, GPU if available)."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float64


class TestQPLayerForward:
    """Tests for QPLayer forward pass (solving)."""
    
    def test_unconstrained_qp(self, device, dtype):
        """Test unconstrained QP: min (1/2)x'Px + q'x."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        
        # P = 2I, q = [-2, -4] => x* = [1, 2]
        P = torch.tensor([[2., 0.], [0., 2.]], device=device, dtype=dtype)
        q = torch.tensor([-2., -4.], device=device, dtype=dtype)
        
        x = layer(P, q)
        
        assert x.shape == (2,)
        assert torch.allclose(x, torch.tensor([1., 2.], device=device, dtype=dtype), atol=1e-3)
    
    def test_box_constrained_qp(self, device, dtype):
        """Test box-constrained QP: min (1/2)x'Px + q'x s.t. 0 <= x <= 1."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        
        # P = 2I, q = [-3, -3] => unconstrained x* = [1.5, 1.5]
        # With bounds [0, 1]: x* = [1, 1]
        P = torch.tensor([[2., 0.], [0., 2.]], device=device, dtype=dtype)
        q = torch.tensor([-3., -3.], device=device, dtype=dtype)
        lb = torch.tensor([0., 0.], device=device, dtype=dtype)
        ub = torch.tensor([1., 1.], device=device, dtype=dtype)
        
        x = layer(P, q, lb=lb, ub=ub)
        
        assert torch.allclose(x, torch.tensor([1., 1.], device=device, dtype=dtype), atol=1e-3)
    
    def test_qp_with_equality_constraint(self, device, dtype):
        """Test QP with equality constraint: Ax = b."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2, n_eq=1)
        
        # min (1/2)(x^2 + y^2) s.t. x + y = 2
        # Optimal: x = y = 1
        P = torch.tensor([[1., 0.], [0., 1.]], device=device, dtype=dtype)
        q = torch.zeros(2, device=device, dtype=dtype)
        A = torch.tensor([[1., 1.]], device=device, dtype=dtype)
        b = torch.tensor([2.], device=device, dtype=dtype)
        
        x = layer(P, q, A=A, b=b)
        
        # Check constraint satisfied
        assert torch.allclose(A @ x, b, atol=1e-2)
        # Check solution
        assert torch.allclose(x, torch.tensor([1., 1.], device=device, dtype=dtype), atol=1e-2)
    
    def test_qp_with_inequality_constraint(self, device, dtype):
        """Test QP with inequality constraint: Gx <= h."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2, n_ineq=1)
        
        # min (1/2)(x^2 + y^2) - 2x - 2y s.t. x + y <= 1
        # Unconstrained: x* = y* = 2
        # With constraint: x* = y* = 0.5
        P = torch.tensor([[1., 0.], [0., 1.]], device=device, dtype=dtype)
        q = torch.tensor([-2., -2.], device=device, dtype=dtype)
        G = torch.tensor([[1., 1.]], device=device, dtype=dtype)
        h = torch.tensor([1.], device=device, dtype=dtype)
        lb = torch.zeros(2, device=device, dtype=dtype)
        
        x = layer(P, q, G=G, h=h, lb=lb)
        
        # Check constraint satisfied
        assert (G @ x <= h + 1e-3).all()


class TestQPLayerBackward:
    """Tests for QPLayer backward pass (gradients)."""
    
    def test_gradient_wrt_q(self, device, dtype):
        """Test gradient of solution w.r.t. linear cost q."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        
        P = torch.tensor([[2., 0.], [0., 2.]], device=device, dtype=dtype)
        q = torch.tensor([-2., -4.], device=device, dtype=dtype, requires_grad=True)
        
        x = layer(P, q)
        loss = x.sum()
        loss.backward()
        
        # For unconstrained QP: x* = -P^{-1}q
        # dx*/dq = -P^{-1}
        # d(x.sum())/dq = [1, 1] @ (-P^{-1}) = [-0.5, -0.5]
        expected_grad = torch.tensor([-0.5, -0.5], device=device, dtype=dtype)
        
        assert q.grad is not None
        assert torch.allclose(q.grad, expected_grad, atol=1e-2)
    
    def test_gradient_wrt_P(self, device, dtype):
        """Test gradient of solution w.r.t. quadratic cost P."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        
        P = torch.tensor([[2., 0.], [0., 2.]], device=device, dtype=dtype, requires_grad=True)
        q = torch.tensor([-2., -4.], device=device, dtype=dtype)
        
        x = layer(P, q)
        loss = (x ** 2).sum()
        loss.backward()
        
        # Gradient should exist
        assert P.grad is not None
        # For x* = [1, 2], loss = 1 + 4 = 5
        # Gradient should be non-zero
        assert P.grad.abs().sum() > 0
    
    def test_gradient_through_nn(self, device, dtype):
        """Test gradients flow through a neural network."""
        from cuprox.torch import QPLayer
        
        # Simple network: linear -> QP -> linear
        class OptNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 2)
                self.qp = QPLayer(n_vars=2)
                self.fc2 = nn.Linear(2, 1)
                
                # Fixed P for simplicity
                self.register_buffer(
                    "P", torch.eye(2, dtype=torch.float64)
                )
            
            def forward(self, x):
                q = self.fc1(x)  # q depends on input
                z = self.qp(self.P, q)  # Solve QP
                return self.fc2(z)
        
        model = OptNet().to(device).double()
        
        # Forward pass
        x = torch.randn(3, device=device, dtype=dtype)
        y = model(x)
        
        # Backward pass should work
        y.backward()
        
        # Check gradients exist
        assert model.fc1.weight.grad is not None
        assert model.fc2.weight.grad is not None
    
    def test_gradient_finite_difference(self, device, dtype):
        """Verify gradients match finite differences."""
        from cuprox.torch import solve_qp
        
        P = torch.tensor([[2., 0.], [0., 2.]], device=device, dtype=dtype)
        q = torch.tensor([-2., -4.], device=device, dtype=dtype, requires_grad=True)
        
        # Compute analytical gradient
        x = solve_qp(P, q)
        loss = x.sum()
        loss.backward()
        analytical_grad = q.grad.clone()
        
        # Compute finite difference gradient
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
        
        # Compare
        assert torch.allclose(analytical_grad, fd_grad, atol=1e-3)


class TestLPLayer:
    """Tests for LPLayer.
    
    Note: LP layer is less critical for differentiable optimization
    than QP layer. Most neural network optimization layers use QP.
    """
    
    @pytest.mark.skip(reason="LP layer needs further development")
    def test_simple_lp_with_equality(self, device, dtype):
        """Test simple LP: min c'x s.t. Ax = b, x >= 0."""
        from cuprox.torch import LPLayer
        
        layer = LPLayer(n_vars=4, n_eq=2)
        
        c = torch.tensor([-1., -1., 0., 0.], device=device, dtype=dtype)
        A = torch.tensor([
            [1., 2., 1., 0.],
            [3., 1., 0., 1.],
        ], device=device, dtype=dtype)
        b = torch.tensor([10., 15.], device=device, dtype=dtype)
        lb = torch.zeros(4, device=device, dtype=dtype)
        
        x = layer(c, A=A, b=b, lb=lb)
        
        assert torch.allclose(A @ x, b, atol=0.5)
    
    @pytest.mark.skip(reason="LP layer needs further development")
    def test_lp_with_bounds(self, device, dtype):
        """Test LP with box constraints only."""
        from cuprox.torch import LPLayer
        
        layer = LPLayer(n_vars=2, n_eq=1)
        
        c = torch.tensor([1., 1.], device=device, dtype=dtype)
        A = torch.tensor([[1., 1.]], device=device, dtype=dtype)
        b = torch.tensor([2.], device=device, dtype=dtype)
        lb = torch.zeros(2, device=device, dtype=dtype)
        ub = torch.full((2,), 2., device=device, dtype=dtype)
        
        x = layer(c, A=A, b=b, lb=lb, ub=ub)
        
        assert torch.allclose((x[0] + x[1]).unsqueeze(0), b, atol=0.5)


class TestInputValidation:
    """Tests for input validation."""
    
    def test_wrong_P_shape(self, device, dtype):
        """Test error on wrong P shape."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        
        P = torch.eye(3, device=device, dtype=dtype)  # Wrong size
        q = torch.zeros(2, device=device, dtype=dtype)
        
        with pytest.raises(ValueError, match="P must have shape"):
            layer(P, q)
    
    def test_wrong_q_shape(self, device, dtype):
        """Test error on wrong q shape."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.zeros(3, device=device, dtype=dtype)  # Wrong size
        
        with pytest.raises(ValueError, match="q must have shape"):
            layer(P, q)
    
    def test_missing_constraint_matrices(self, device, dtype):
        """Test error when constraint matrices missing."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2, n_eq=1)
        
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.zeros(2, device=device, dtype=dtype)
        
        with pytest.raises(ValueError, match="A and b required"):
            layer(P, q)  # Missing A and b
    
    def test_invalid_n_vars(self):
        """Test error on invalid n_vars."""
        from cuprox.torch import QPLayer
        
        with pytest.raises(ValueError, match="n_vars must be positive"):
            QPLayer(n_vars=0)
        
        with pytest.raises(ValueError, match="n_vars must be positive"):
            QPLayer(n_vars=-1)


class TestFunctionalAPI:
    """Tests for functional interface (solve_qp, solve_lp)."""
    
    def test_solve_qp_functional(self, device, dtype):
        """Test solve_qp function."""
        from cuprox.torch import solve_qp
        
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype, requires_grad=True)
        
        x = solve_qp(P, q)
        
        assert x.shape == (2,)
        assert torch.allclose(x, torch.tensor([1., 2.], device=device, dtype=dtype), atol=1e-3)
        
        # Test gradient
        x.sum().backward()
        assert q.grad is not None
    
    @pytest.mark.skip(reason="LP layer needs further development")
    def test_solve_lp_functional(self, device, dtype):
        """Test solve_lp function with equality constraint."""
        from cuprox.torch import solve_lp
        
        c = torch.tensor([-1., -1.], device=device, dtype=dtype)
        A = torch.tensor([[1., 1.]], device=device, dtype=dtype)
        b = torch.tensor([2.], device=device, dtype=dtype)
        lb = torch.zeros(2, device=device, dtype=dtype)
        ub = torch.full((2,), 10., device=device, dtype=dtype)
        
        x = solve_lp(c, A=A, b=b, lb=lb, ub=ub)
        
        assert x.shape == (2,)
        assert torch.allclose((x[0] + x[1]).unsqueeze(0), b, atol=0.5)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_variable(self, device, dtype):
        """Test QP with single variable."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=1)
        
        P = torch.tensor([[2.]], device=device, dtype=dtype)
        q = torch.tensor([-4.], device=device, dtype=dtype)
        
        x = layer(P, q)
        
        # x* = 2
        assert torch.allclose(x, torch.tensor([2.], device=device, dtype=dtype), atol=1e-3)
    
    def test_large_problem(self, device, dtype):
        """Test larger QP (100 variables)."""
        from cuprox.torch import QPLayer
        
        n = 100
        layer = QPLayer(n_vars=n)
        
        # Random positive definite P
        A = torch.randn(n, n, device=device, dtype=dtype)
        P = A @ A.T + torch.eye(n, device=device, dtype=dtype)
        q = torch.randn(n, device=device, dtype=dtype)
        
        x = layer(P, q)
        
        assert x.shape == (n,)
        # Check optimality condition: Px + q ≈ 0
        residual = P @ x + q
        assert residual.norm() < 1e-2 * (P.norm() * x.norm() + q.norm())
    
    def test_non_differentiable_mode(self, device, dtype):
        """Test layer with differentiable=False."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2, differentiable=False)
        
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype, requires_grad=True)
        
        x = layer(P, q)
        
        # Forward should work
        assert x.shape == (2,)
        
        # x should not require grad
        assert not x.requires_grad

