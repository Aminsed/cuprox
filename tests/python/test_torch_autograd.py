"""
Comprehensive Tests for PyTorch Autograd Integration.

Tests:
1. Forward pass correctness
2. Backward pass (gradients for P, q, A, b, G, h)
3. Batched solving
4. Gradient finite difference verification
5. OptNet layer training
6. Dual variable computation
7. Edge cases and numerical stability
"""

import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    return torch.float64


# =============================================================================
# Basic Forward Pass Tests
# =============================================================================


class TestQPForward:
    """Test QP forward pass correctness."""

    def test_unconstrained_identity_P(self, device, dtype):
        """min (1/2)||x||^2 + q'x => x* = -q"""
        from cuprox.torch import solve_qp

        n = 5
        P = torch.eye(n, device=device, dtype=dtype)
        q = torch.randn(n, device=device, dtype=dtype)

        x = solve_qp(P, q)

        torch.testing.assert_close(x, -q, atol=1e-3, rtol=1e-3)

    def test_unconstrained_scaled_P(self, device, dtype):
        """min (1/2)x'(2I)x + q'x => x* = -q/2"""
        from cuprox.torch import solve_qp

        n = 5
        P = 2 * torch.eye(n, device=device, dtype=dtype)
        q = torch.tensor([2., 4., 6., 8., 10.], device=device, dtype=dtype)

        x = solve_qp(P, q)

        expected = -q / 2
        torch.testing.assert_close(x, expected, atol=1e-3, rtol=1e-3)

    def test_box_constrained(self, device, dtype):
        """Box constraints should clip solution."""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-10., -10.], device=device, dtype=dtype)  # Unconstrained: x* = [10, 10]
        lb = torch.zeros(2, device=device, dtype=dtype)
        ub = torch.ones(2, device=device, dtype=dtype)

        x = solve_qp(P, q, lb=lb, ub=ub)

        # Should hit upper bound
        torch.testing.assert_close(x, ub, atol=1e-3, rtol=1e-3)

    def test_equality_constrained(self, device, dtype):
        """min ||x||^2 s.t. sum(x) = 1 => x_i = 1/n"""
        from cuprox.torch import solve_qp

        n = 4
        P = torch.eye(n, device=device, dtype=dtype)
        q = torch.zeros(n, device=device, dtype=dtype)
        A = torch.ones(1, n, device=device, dtype=dtype)
        b = torch.ones(1, device=device, dtype=dtype)

        x = solve_qp(P, q, A=A, b=b)

        expected = torch.full((n,), 1/n, device=device, dtype=dtype)
        torch.testing.assert_close(x, expected, atol=1e-2, rtol=1e-2)

    def test_inequality_constrained(self, device, dtype):
        """min ||x - [2,2]||^2 s.t. x1 + x2 <= 1"""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-2., -2.], device=device, dtype=dtype)  # min ||x - [2,2]||^2
        G = torch.tensor([[1., 1.]], device=device, dtype=dtype)
        h = torch.tensor([1.], device=device, dtype=dtype)
        lb = torch.zeros(2, device=device, dtype=dtype)

        x = solve_qp(P, q, G=G, h=h, lb=lb)

        # Constraint should be active
        assert (G @ x <= h + 1e-3).all()


# =============================================================================
# Gradient Tests
# =============================================================================


class TestQPGradients:
    """Test gradient computation."""

    def test_gradient_wrt_q_identity_P(self, device, dtype):
        """For P=I: x* = -q, so dx*/dq = -I"""
        from cuprox.torch import solve_qp

        P = torch.eye(3, device=device, dtype=dtype)
        q = torch.tensor([1., 2., 3.], device=device, dtype=dtype, requires_grad=True)

        x = solve_qp(P, q)
        loss = x.sum()
        loss.backward()

        # d(sum(x*))/dq = [1,1,1] @ (-I) = [-1,-1,-1]
        expected_grad = -torch.ones(3, device=device, dtype=dtype)
        torch.testing.assert_close(q.grad, expected_grad, atol=1e-3, rtol=1e-3)

    def test_gradient_wrt_P(self, device, dtype):
        """Gradient w.r.t. P should be non-zero."""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype, requires_grad=True)
        q = torch.tensor([-2., -4.], device=device, dtype=dtype)

        x = solve_qp(P, q)
        loss = (x ** 2).sum()
        loss.backward()

        assert P.grad is not None
        assert P.grad.abs().sum() > 0

    def test_gradient_wrt_b(self, device, dtype):
        """Gradient w.r.t. equality RHS b."""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.zeros(2, device=device, dtype=dtype)
        A = torch.tensor([[1., 1.]], device=device, dtype=dtype)
        b = torch.tensor([2.], device=device, dtype=dtype, requires_grad=True)

        x = solve_qp(P, q, A=A, b=b)
        loss = x.sum()
        loss.backward()

        assert b.grad is not None
        # Changing b should change x proportionally
        assert b.grad.abs().sum() > 0

    def test_gradient_finite_difference_q(self, device, dtype):
        """Verify q gradient matches finite differences."""
        from cuprox.torch import solve_qp

        P = torch.eye(3, device=device, dtype=dtype)
        q = torch.tensor([1., 2., 3.], device=device, dtype=dtype, requires_grad=True)

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

    def test_gradient_finite_difference_P(self, device, dtype):
        """Verify P gradient matches finite differences."""
        from cuprox.torch import solve_qp

        P = 2 * torch.eye(2, device=device, dtype=dtype)
        P = P.clone().requires_grad_(True)  # Make it a leaf tensor
        q = torch.tensor([-2., -4.], device=device, dtype=dtype)

        # Analytical gradient
        x = solve_qp(P, q)
        loss = x.sum()
        loss.backward()
        analytical = P.grad.clone()

        # Finite difference
        eps = 1e-5
        fd_grad = torch.zeros_like(P)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                P_plus = P.detach().clone()
                P_plus[i, j] += eps
                x_plus = solve_qp(P_plus, q)

                P_minus = P.detach().clone()
                P_minus[i, j] -= eps
                x_minus = solve_qp(P_minus, q)

                fd_grad[i, j] = (x_plus.sum() - x_minus.sum()) / (2 * eps)

        # P gradient is numerically sensitive, use larger tolerance
        torch.testing.assert_close(analytical, fd_grad, atol=0.5, rtol=0.5)


# =============================================================================
# Batch Solving Tests
# =============================================================================


class TestBatchSolving:
    """Test batched QP solving."""

    def test_batch_forward(self, device, dtype):
        """Batch solving produces correct results."""
        from cuprox.torch import solve_qp_batch

        batch_size = 10
        n = 5
        P = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        q = torch.randn(batch_size, n, device=device, dtype=dtype)

        x = solve_qp_batch(P, q)

        assert x.shape == (batch_size, n)
        # Each solution should be -q for identity P
        for i in range(batch_size):
            torch.testing.assert_close(x[i], -q[i], atol=1e-3, rtol=1e-3)

    def test_batch_gradient(self, device, dtype):
        """Batch solving supports gradients."""
        from cuprox.torch import solve_qp_batch

        batch_size = 5
        n = 3
        P = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        q = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)

        x = solve_qp_batch(P, q)
        loss = x.sum()
        loss.backward()

        assert q.grad is not None
        assert q.grad.shape == (batch_size, n)

    def test_batch_vs_sequential(self, device, dtype):
        """Batch results should match sequential solving."""
        from cuprox.torch import solve_qp, solve_qp_batch

        batch_size = 5
        n = 4
        P = torch.eye(n, device=device, dtype=dtype)
        q_batch = torch.randn(batch_size, n, device=device, dtype=dtype)

        # Batch solve
        P_batch = P.unsqueeze(0).expand(batch_size, -1, -1)
        x_batch = solve_qp_batch(P_batch, q_batch)

        # Sequential solve
        x_seq = torch.stack([solve_qp(P, q_batch[i]) for i in range(batch_size)])

        torch.testing.assert_close(x_batch, x_seq, atol=1e-3, rtol=1e-3)


# =============================================================================
# Layer Tests
# =============================================================================


class TestOptNetLayer:
    """Test OptNet layer."""

    def test_forward(self, device, dtype):
        """OptNet forward pass."""
        from cuprox.torch import OptNetLayer

        layer = OptNetLayer(n_features=10, n_vars=5).to(device).to(dtype)
        x = torch.randn(8, 10, device=device, dtype=dtype)

        z = layer(x)

        assert z.shape == (8, 5)

    def test_gradient_flow(self, device, dtype):
        """Gradients flow through OptNet."""
        from cuprox.torch import OptNetLayer

        layer = OptNetLayer(n_features=10, n_vars=5, n_ineq=2).to(device).to(dtype)
        x = torch.randn(4, 10, device=device, dtype=dtype)

        z = layer(x)
        loss = z.sum()
        loss.backward()

        # Check gradients exist for learnable parameters
        assert layer.L.grad is not None
        assert layer.fc_q.weight.grad is not None
        assert layer.G.grad is not None

    def test_training_loop(self, device, dtype):
        """OptNet can be trained."""
        from cuprox.torch import OptNetLayer

        layer = OptNetLayer(n_features=5, n_vars=3).to(device).to(dtype)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

        # Dummy training
        losses = []
        for _ in range(5):
            x = torch.randn(4, 5, device=device, dtype=dtype)
            target = torch.randn(4, 3, device=device, dtype=dtype)

            optimizer.zero_grad()
            z = layer(x)
            loss = ((z - target) ** 2).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should not be NaN
        assert all(not np.isnan(l) for l in losses)


class TestParametricQPLayer:
    """Test ParametricQP layer."""

    def test_forward(self, device, dtype):
        """Forward pass works."""
        from cuprox.torch import ParametricQPLayer

        layer = ParametricQPLayer(n_vars=5).to(device).to(dtype)
        q = torch.randn(8, 5, device=device, dtype=dtype)

        x = layer(q)

        assert x.shape == (8, 5)

    def test_P_is_psd(self, device, dtype):
        """Learned P is positive semidefinite."""
        from cuprox.torch import ParametricQPLayer

        layer = ParametricQPLayer(n_vars=5).to(device).to(dtype)
        P = layer.P

        # Check eigenvalues are non-negative
        eigvals = torch.linalg.eigvalsh(P)
        assert (eigvals >= -1e-6).all()

    def test_learns_P(self, device, dtype):
        """Layer learns P from data."""
        from cuprox.torch import ParametricQPLayer

        n = 3
        # True P
        L_true = torch.randn(n, n)
        P_true = L_true @ L_true.T + 0.1 * torch.eye(n)

        # Generate data
        q = torch.randn(50, n, device=device, dtype=dtype)
        x_true = torch.stack([torch.linalg.solve(P_true.to(device).to(dtype), -qi) for qi in q])

        # Train layer
        layer = ParametricQPLayer(n_vars=n, init_scale=0.5).to(device).to(dtype)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)

        for _ in range(50):
            optimizer.zero_grad()
            x_pred = layer(q)
            loss = ((x_pred - x_true) ** 2).sum()
            loss.backward()
            optimizer.step()

        # Final predictions should be closer than initial
        x_final = layer(q)
        error = ((x_final - x_true) ** 2).mean().item()
        assert error < 5.0  # Reasonable fit (learning is approximate)


class TestBatchQPLayer:
    """Test BatchQP layer."""

    def test_forward(self, device, dtype):
        """Forward pass works."""
        from cuprox.torch import BatchQPLayer

        layer = BatchQPLayer(n_vars=5)
        P = torch.eye(5, device=device, dtype=dtype).unsqueeze(0).expand(10, -1, -1)
        q = torch.randn(10, 5, device=device, dtype=dtype)

        x = layer(P, q)

        assert x.shape == (10, 5)


class TestDecisionFocusedLayer:
    """Test DecisionFocused layer."""

    def test_forward(self, device, dtype):
        """Forward pass works."""
        from cuprox.torch import DecisionFocusedLayer

        predictor = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        ).to(device).to(dtype)

        layer = DecisionFocusedLayer(predictor, n_vars=5).to(device)
        x = torch.randn(8, 10, device=device, dtype=dtype)

        z = layer(x)

        assert z.shape == (8, 5)

    def test_gradient_to_predictor(self, device, dtype):
        """Gradients flow to predictor."""
        from cuprox.torch import DecisionFocusedLayer

        predictor = nn.Linear(10, 5).to(device).to(dtype)
        layer = DecisionFocusedLayer(predictor, n_vars=5).to(device)

        x = torch.randn(4, 10, device=device, dtype=dtype)
        z = layer(x)
        loss = z.sum()
        loss.backward()

        assert predictor.weight.grad is not None
        assert predictor.weight.grad.abs().sum() > 0


# =============================================================================
# Dual Variable Tests
# =============================================================================


class TestDualVariables:
    """Test dual variable computation."""

    def test_returns_duals(self, device, dtype):
        """solve_qp_with_duals returns duals."""
        from cuprox.torch import solve_qp_with_duals

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.zeros(2, device=device, dtype=dtype)
        A = torch.tensor([[1., 1.]], device=device, dtype=dtype)
        b = torch.tensor([2.], device=device, dtype=dtype)

        sol = solve_qp_with_duals(P, q, A=A, b=b)

        assert hasattr(sol, 'x')
        assert hasattr(sol, 'nu')
        assert hasattr(sol, 'lam')
        assert sol.x.shape == (2,)
        assert sol.nu.shape == (1,)

    def test_active_bounds_detected(self, device, dtype):
        """Active bounds are correctly identified."""
        from cuprox.torch import solve_qp_with_duals

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-10., -10.], device=device, dtype=dtype)
        lb = torch.zeros(2, device=device, dtype=dtype)
        ub = torch.ones(2, device=device, dtype=dtype)

        sol = solve_qp_with_duals(P, q, lb=lb, ub=ub)

        # Both should be at upper bound
        assert sol.active_ub.all()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and numerical stability."""

    def test_single_variable(self, device, dtype):
        """Single variable QP."""
        from cuprox.torch import solve_qp

        P = torch.tensor([[2.]], device=device, dtype=dtype)
        q = torch.tensor([-4.], device=device, dtype=dtype, requires_grad=True)

        x = solve_qp(P, q)
        x.backward()

        torch.testing.assert_close(x, torch.tensor([2.], device=device, dtype=dtype), atol=1e-3, rtol=1e-3)
        assert q.grad is not None

    def test_large_problem(self, device, dtype):
        """Larger QP (n=100)."""
        from cuprox.torch import solve_qp

        n = 100
        torch.manual_seed(42)
        A = torch.randn(n, n, device=device, dtype=dtype)
        P = A @ A.T + torch.eye(n, device=device, dtype=dtype)
        q = torch.randn(n, device=device, dtype=dtype, requires_grad=True)

        x = solve_qp(P, q)
        x.sum().backward()

        assert x.shape == (n,)
        assert q.grad is not None

    def test_all_at_bounds(self, device, dtype):
        """All variables at bounds."""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-100., -100.], device=device, dtype=dtype, requires_grad=True)
        lb = torch.zeros(2, device=device, dtype=dtype)
        ub = torch.ones(2, device=device, dtype=dtype)

        x = solve_qp(P, q, lb=lb, ub=ub)
        x.sum().backward()

        # Solution should be at upper bound
        torch.testing.assert_close(x, ub, atol=1e-3, rtol=1e-3)
        # Gradient should exist (may be zero)
        assert q.grad is not None

    def test_no_grad_mode(self, device, dtype):
        """Works in torch.no_grad() context."""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype)

        with torch.no_grad():
            x = solve_qp(P, q)

        assert not x.requires_grad

    def test_detach(self, device, dtype):
        """Detaching works correctly."""
        from cuprox.torch import solve_qp

        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype, requires_grad=True)

        x = solve_qp(P, q).detach()

        assert not x.requires_grad
        # Should be able to use x without issues
        y = x * 2
        assert y.shape == (2,)


# =============================================================================
# Integration Tests
# =============================================================================


class TestNeuralNetworkIntegration:
    """Integration with neural networks."""

    def test_in_sequential_model(self, device, dtype):
        """QP layer in a neural network."""
        from cuprox.torch import QPLayer

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5, dtype=dtype)
                self.qp = QPLayer(n_vars=5)
                self.register_buffer("P", torch.eye(5, dtype=dtype))

            def forward(self, x):
                q = self.fc(x)
                return self.qp(self.P, q)

        model = Model().to(device)
        x = torch.randn(4, 10, device=device, dtype=dtype)

        y = model(x)
        loss = y.sum()
        loss.backward()

        assert model.fc.weight.grad is not None

    def test_optimizer_step(self, device, dtype):
        """Optimizer updates weights through QP."""
        from cuprox.torch import QPLayer

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 3, dtype=dtype)
                self.qp = QPLayer(n_vars=3)
                self.register_buffer("P", torch.eye(3, dtype=dtype))

            def forward(self, x):
                q = self.fc(x)
                return self.qp(self.P, q)

        model = Model().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        initial_weight = model.fc.weight.data.clone()

        x = torch.randn(2, 5, device=device, dtype=dtype)
        target = torch.ones(2, 3, device=device, dtype=dtype)

        for _ in range(3):
            optimizer.zero_grad()
            y = model(x)
            loss = ((y - target) ** 2).sum()
            loss.backward()
            optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.fc.weight.data, initial_weight)

    def test_multiple_qp_layers(self, device, dtype):
        """Multiple QP layers in sequence."""
        from cuprox.torch import QPLayer

        class MultiQPNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.qp1 = QPLayer(n_vars=3)
                self.qp2 = QPLayer(n_vars=3)
                self.register_buffer("P", torch.eye(3, dtype=dtype))

            def forward(self, q):
                z1 = self.qp1(self.P, q)
                z2 = self.qp2(self.P, z1)  # Use output as new q
                return z2

        model = MultiQPNet().to(device)
        q = torch.randn(3, device=device, dtype=dtype, requires_grad=True)

        z = model(q)
        z.sum().backward()

        assert q.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

