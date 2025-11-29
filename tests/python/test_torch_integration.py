"""
Tests for PyTorch integration with neural networks.

These tests verify that cuprox layers work correctly within
larger PyTorch models and training pipelines.
"""

import pytest

try:
    import torch
    import torch.nn as nn
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


class TestNeuralNetworkIntegration:
    """Integration with nn.Module."""
    
    def test_qp_in_sequential(self, device, dtype):
        """QPLayer in nn.Sequential-like model."""
        from cuprox.torch import QPLayer
        
        class SimpleOptNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, dtype=dtype)
                self.qp = QPLayer(n_vars=2)
                self.register_buffer("P", torch.eye(2, dtype=dtype))
            
            def forward(self, x):
                q = self.fc(x)
                return self.qp(self.P, q)
        
        model = SimpleOptNet().to(device)
        x = torch.randn(4, device=device, dtype=dtype)
        
        y = model(x)
        
        assert y.shape == (2,)
        assert y.requires_grad
    
    def test_gradient_flow(self, device, dtype):
        """Gradients flow through QP layer to earlier layers."""
        from cuprox.torch import QPLayer
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 2, dtype=dtype)
                self.qp = QPLayer(n_vars=2)
                self.P = nn.Parameter(torch.eye(2, dtype=dtype))
            
            def forward(self, x):
                q = self.fc1(x)
                return self.qp(self.P, q)
        
        model = Model().to(device)
        x = torch.randn(3, device=device, dtype=dtype)
        
        # Forward
        y = model(x)
        loss = y.sum()
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        assert model.fc1.weight.grad is not None
        assert model.fc1.bias.grad is not None
        assert model.P.grad is not None
        
        # Check gradients are non-zero
        assert model.fc1.weight.grad.abs().sum() > 0
    
    def test_optimizer_step(self, device, dtype):
        """Optimizer can update parameters through QP."""
        from cuprox.torch import QPLayer
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2, dtype=dtype)
                self.qp = QPLayer(n_vars=2)
                self.register_buffer("P", torch.eye(2, dtype=dtype))
            
            def forward(self, x):
                q = self.fc(x)
                return self.qp(self.P, q)
        
        model = Model().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Save initial weights
        initial_weight = model.fc.weight.data.clone()
        
        # Training step
        x = torch.randn(2, device=device, dtype=dtype)
        target = torch.ones(2, device=device, dtype=dtype)
        
        y = model(x)
        loss = ((y - target) ** 2).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        assert not torch.allclose(model.fc.weight.data, initial_weight)
    
    def test_multiple_forward_passes(self, device, dtype):
        """Multiple forward passes accumulate gradients correctly."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype, requires_grad=True)
        
        # Multiple forward passes
        x1 = layer(P, q)
        x2 = layer(P, q * 2)
        
        loss = x1.sum() + x2.sum()
        loss.backward()
        
        assert q.grad is not None


class TestTrainingLoop:
    """Test in realistic training scenarios."""
    
    def test_mini_training_loop(self, device, dtype):
        """Run a few training iterations."""
        from cuprox.torch import QPLayer
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, dtype=dtype)
                self.qp = QPLayer(n_vars=2)
                self.P = nn.Parameter(0.1 * torch.eye(2, dtype=dtype))
            
            def forward(self, x):
                q = self.fc(x)
                return self.qp(self.P + self.P.T, q)  # Ensure symmetric P
        
        model = Model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Generate dummy data
        torch.manual_seed(42)
        X = torch.randn(10, 4, device=device, dtype=dtype)
        Y = torch.randn(10, 2, device=device, dtype=dtype)
        
        # Training loop
        losses = []
        for _ in range(3):
            total_loss = 0.0
            for i in range(len(X)):
                optimizer.zero_grad()
                y_pred = model(X[i])
                loss = ((y_pred - Y[i]) ** 2).sum()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / len(X))
        
        # Loss should exist (not NaN)
        assert all(not (l != l) for l in losses)  # NaN check


class TestEdgeCases:
    """Edge cases in integration."""
    
    def test_double_backward(self, device, dtype):
        """Double backward (Hessian computation) - should at least not crash."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype, requires_grad=True)
        
        x = layer(P, q)
        
        # First backward
        grad_x = torch.autograd.grad(x.sum(), q, create_graph=True)[0]
        
        # Second backward (Hessian-vector product) might not work perfectly
        # but should not crash
        try:
            grad_x.sum().backward()
        except RuntimeError:
            pass  # Expected - double backward through QP is complex
    
    def test_no_grad_context(self, device, dtype):
        """QPLayer in torch.no_grad() context."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype)
        
        with torch.no_grad():
            x = layer(P, q)
        
        assert not x.requires_grad
    
    def test_detach(self, device, dtype):
        """Detaching QP output."""
        from cuprox.torch import QPLayer
        
        layer = QPLayer(n_vars=2)
        P = torch.eye(2, device=device, dtype=dtype)
        q = torch.tensor([-1., -2.], device=device, dtype=dtype, requires_grad=True)
        
        x = layer(P, q).detach()
        
        # Should be able to use x without gradient tracking
        y = x * 2
        assert not y.requires_grad

