"""
Tests for MPC Dynamics Models.

Tests covering:
1. LinearSystem creation and validation
2. System simulation
3. Discretization from continuous-time
4. System properties (stability, controllability)
"""

import pytest
import numpy as np


class TestLinearSystem:
    """Test LinearSystem class."""
    
    def test_basic_creation(self):
        """Create basic linear system."""
        from cuprox.mpc import LinearSystem
        
        A = np.array([[1, 0.1], [0, 1]])
        B = np.array([[0.005], [0.1]])
        
        system = LinearSystem(A, B)
        
        assert system.n_states == 2
        assert system.n_inputs == 1
        assert system.n_outputs == 2  # Default: outputs = states
    
    def test_with_output(self):
        """System with output equation."""
        from cuprox.mpc import LinearSystem
        
        A = np.array([[1, 0.1], [0, 1]])
        B = np.array([[0.005], [0.1]])
        C = np.array([[1, 0]])  # Only observe position
        
        system = LinearSystem(A, B, C=C)
        
        assert system.n_outputs == 1
    
    def test_step(self):
        """Simulate one step."""
        from cuprox.mpc import LinearSystem
        
        A = np.array([[1, 0.1], [0, 1]])
        B = np.array([[0.005], [0.1]])
        system = LinearSystem(A, B)
        
        x = np.array([0, 1])
        u = np.array([0])
        
        x_next = system.step(x, u)
        
        expected = A @ x + B @ u
        np.testing.assert_allclose(x_next, expected)
    
    def test_simulate(self):
        """Simulate trajectory."""
        from cuprox.mpc import LinearSystem
        
        A = np.array([[1, 0.1], [0, 1]])
        B = np.array([[0.005], [0.1]])
        system = LinearSystem(A, B)
        
        x0 = np.array([0, 1])
        u_seq = np.zeros((10, 1))
        
        traj = system.simulate(x0, u_seq)
        
        assert traj.shape == (11, 2)  # 10 steps + initial
        np.testing.assert_allclose(traj[0], x0)
    
    def test_output(self):
        """Test output computation."""
        from cuprox.mpc import LinearSystem
        
        A = np.eye(2)
        B = np.zeros((2, 1))
        C = np.array([[1, 2]])
        D = np.array([[0.5]])
        
        system = LinearSystem(A, B, C=C, D=D)
        
        x = np.array([1, 2])
        u = np.array([1])
        
        y = system.output(x, u)
        
        expected = C @ x + D @ u  # [1*1 + 2*2 + 0.5*1] = [5.5]
        np.testing.assert_allclose(y, expected)
    
    def test_invalid_dimensions(self):
        """Error on invalid dimensions."""
        from cuprox.mpc import LinearSystem
        
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1], [1], [1]])  # Wrong rows
        
        with pytest.raises(ValueError, match="B rows"):
            LinearSystem(A, B)
    
    def test_is_stable(self):
        """Test stability check."""
        from cuprox.mpc import LinearSystem
        
        # Stable system (eigenvalues inside unit circle)
        A_stable = np.array([[0.9, 0], [0, 0.8]])
        B = np.array([[1], [1]])
        
        stable = LinearSystem(A_stable, B)
        assert stable.is_stable()
        
        # Unstable system
        A_unstable = np.array([[1.1, 0], [0, 1.0]])
        unstable = LinearSystem(A_unstable, B)
        assert not unstable.is_stable()
    
    def test_is_controllable(self):
        """Test controllability check."""
        from cuprox.mpc import LinearSystem
        
        # Controllable system
        A = np.array([[1, 1], [0, 1]])
        B = np.array([[0], [1]])
        
        controllable = LinearSystem(A, B)
        assert controllable.is_controllable()
        
        # Uncontrollable system
        A_unc = np.array([[1, 0], [0, 2]])
        B_unc = np.array([[1], [0]])
        
        uncontrollable = LinearSystem(A_unc, B_unc)
        assert not uncontrollable.is_controllable()


class TestDiscretization:
    """Test continuous-to-discrete conversion."""
    
    def test_euler_discretization(self):
        """Euler discretization."""
        from cuprox.mpc import LinearSystem
        
        # Simple integrator: dx/dt = u
        Ac = np.array([[0]])
        Bc = np.array([[1]])
        dt = 0.1
        
        system = LinearSystem.from_continuous(Ac, Bc, dt, method='euler')
        
        # A = I + Ac*dt = [[1]]
        # B = Bc*dt = [[0.1]]
        np.testing.assert_allclose(system.A, [[1]])
        np.testing.assert_allclose(system.B, [[0.1]])
    
    def test_zoh_discretization(self):
        """Zero-order hold discretization."""
        from cuprox.mpc import LinearSystem
        
        # Double integrator: x'' = u
        Ac = np.array([[0, 1], [0, 0]])
        Bc = np.array([[0], [1]])
        dt = 0.1
        
        system = LinearSystem.from_continuous(Ac, Bc, dt, method='zoh')
        
        # Check approximate values
        assert system.A[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert system.A[0, 1] == pytest.approx(dt, abs=1e-6)
        assert system.A[1, 1] == pytest.approx(1.0, abs=1e-6)
    
    def test_invalid_method(self):
        """Error on invalid discretization method."""
        from cuprox.mpc import LinearSystem
        
        Ac = np.array([[0]])
        Bc = np.array([[1]])
        
        with pytest.raises(ValueError, match="Unknown method"):
            LinearSystem.from_continuous(Ac, Bc, 0.1, method='invalid')


class TestAffineSystem:
    """Test AffineSystem class."""
    
    def test_basic_creation(self):
        """Create affine system."""
        from cuprox.mpc.dynamics import AffineSystem
        
        A = np.eye(2)
        B = np.zeros((2, 1))
        c = np.array([0, -0.1])  # Gravity offset
        
        system = AffineSystem(A, B, c)
        
        assert system.n_states == 2
        assert system.n_inputs == 1
    
    def test_step_with_offset(self):
        """Step includes affine term."""
        from cuprox.mpc.dynamics import AffineSystem
        
        A = np.eye(2)
        B = np.array([[1], [0]])
        c = np.array([0, -0.1])
        
        system = AffineSystem(A, B, c)
        
        x = np.array([1, 2])
        u = np.array([0.5])
        
        x_next = system.step(x, u)
        
        expected = A @ x + B @ u + c
        np.testing.assert_allclose(x_next, expected)


class TestPredefinedSystems:
    """Test predefined system factories."""
    
    def test_double_integrator(self):
        """Test double integrator."""
        from cuprox.mpc.dynamics import double_integrator
        
        system = double_integrator(dt=0.1)
        
        assert system.n_states == 2
        assert system.n_inputs == 1
        
        # From rest, acceleration should increase velocity
        x0 = np.array([0, 0])
        u = np.array([1])  # Unit acceleration
        
        x1 = system.step(x0, u)
        
        assert x1[1] > 0  # Velocity increased
    
    def test_double_integrator_2d(self):
        """Test 2D double integrator."""
        from cuprox.mpc.dynamics import double_integrator_2d
        
        system = double_integrator_2d(dt=0.1)
        
        assert system.n_states == 4
        assert system.n_inputs == 2
    
    def test_cart_pole(self):
        """Test cart-pole system."""
        from cuprox.mpc.dynamics import cart_pole
        
        system = cart_pole(dt=0.02)
        
        assert system.n_states == 4
        assert system.n_inputs == 1
        
        # System should be unstable at upright (without control)
        assert not system.is_stable()

