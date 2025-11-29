"""
Integration Tests for MPC Module.

End-to-end tests covering:
1. Full control loops
2. Trajectory generation and tracking
3. Realistic robotics scenarios
"""

import pytest
import numpy as np


class TestTrajectoryGeneration:
    """Test trajectory generation utilities."""
    
    def test_constant_reference(self):
        """Constant reference trajectory."""
        from cuprox.mpc import constant_reference
        
        x_ref = np.array([1.0, 0.0])
        traj = constant_reference(x_ref, horizon=50)
        
        assert traj.horizon == 50
        assert traj.n_states == 2
        np.testing.assert_allclose(traj.get_state(0), x_ref)
        np.testing.assert_allclose(traj.get_state(49), x_ref)
    
    def test_step_reference(self):
        """Step reference trajectory."""
        from cuprox.mpc import step_reference
        
        traj = step_reference(
            x_initial=np.array([0, 0]),
            x_final=np.array([1, 0]),
            horizon=100,
            step_time=20
        )
        
        np.testing.assert_allclose(traj.get_state(10), [0, 0])
        np.testing.assert_allclose(traj.get_state(50), [1, 0])
    
    def test_sinusoidal_reference(self):
        """Sinusoidal reference trajectory."""
        from cuprox.mpc import sinusoidal_reference
        
        traj = sinusoidal_reference(
            amplitude=np.array([1.0, 0.0]),
            frequency=np.array([0.1, 0.0]),
            phase=np.array([0.0, 0.0]),
            offset=np.array([0.0, 0.0]),
            horizon=100,
            dt=0.1
        )
        
        assert traj.horizon == 100
        # At t=0, sin(0) = 0
        assert abs(traj.get_state(0)[0]) < 0.01
    
    def test_trajectory_window(self):
        """Get trajectory window."""
        from cuprox.mpc import constant_reference
        
        traj = constant_reference(np.array([1, 0]), horizon=100)
        
        window = traj.get_window(start=50, length=20)
        
        assert window.horizon == 20
        np.testing.assert_allclose(window.get_state(0), [1, 0])


class TestConstraints:
    """Test constraint classes."""
    
    def test_box_constraints(self):
        """Box constraints."""
        from cuprox.mpc import BoxConstraints
        
        box = BoxConstraints(-1.0, 1.0, dim=3)
        
        assert box.dim == 3
        assert box.is_satisfied(np.array([0, 0.5, -0.5]))
        assert not box.is_satisfied(np.array([0, 1.5, 0]))
    
    def test_box_project(self):
        """Box projection."""
        from cuprox.mpc import BoxConstraints
        
        box = BoxConstraints(-1.0, 1.0, dim=2)
        
        x = np.array([2.0, -3.0])
        proj = box.project(x)
        
        np.testing.assert_allclose(proj, [1.0, -1.0])
    
    def test_polytope_constraints(self):
        """Polytope constraints."""
        from cuprox.mpc import PolytopeConstraints
        
        # x + y <= 1
        A = np.array([[1, 1]])
        b = np.array([1])
        
        poly = PolytopeConstraints(A, b)
        
        assert poly.is_satisfied(np.array([0.3, 0.3]))
        assert not poly.is_satisfied(np.array([1.0, 1.0]))


class TestDoubleIntegratorControl:
    """End-to-end tests with double integrator."""
    
    def test_position_regulation(self):
        """Regulate to target position."""
        from cuprox.mpc import LinearMPC
        from cuprox.mpc.dynamics import double_integrator
        
        system = double_integrator(dt=0.1)
        
        mpc = LinearMPC(
            system,
            horizon=30,
            Q=np.diag([10, 1]),
            R=np.array([[0.1]]),
            u_min=-2.0,
            u_max=2.0,
        )
        
        # Start at position 5, regulate to origin
        x0 = np.array([5.0, 0.0])
        sim = mpc.simulate(x0, n_steps=100)
        
        # Check simulation ran
        assert sim['x'].shape == (101, 2)
        assert sim['u'].shape == (100, 1)
    
    def test_trajectory_tracking(self):
        """Track moving reference."""
        from cuprox.mpc import TrackingMPC
        from cuprox.mpc.dynamics import double_integrator
        from cuprox.mpc.trajectory import ramp_reference
        
        system = double_integrator(dt=0.1)
        
        mpc = TrackingMPC(
            system,
            horizon=20,
            Q=np.diag([10, 1]),
            R=np.array([[0.1]]),
            u_min=-2.0,
            u_max=2.0,
        )
        
        # Ramp from 0 to 5
        trajectory = ramp_reference(
            x_initial=np.array([0, 0]),
            x_final=np.array([5, 0]),
            horizon=150,
            ramp_duration=100
        )
        
        x0 = np.array([0, 0])
        sim = mpc.simulate_tracking(x0, trajectory, n_steps=50)
        
        # Check simulation completed
        assert 'x' in sim
        assert 'tracking_error' in sim


class TestCartPoleControl:
    """Test cart-pole (inverted pendulum) control."""
    
    def test_balance_control(self):
        """Balance cart-pole at upright."""
        from cuprox.mpc import LinearMPC
        from cuprox.mpc.dynamics import cart_pole
        
        system = cart_pole(dt=0.02)
        
        # LQR-like weights for balancing
        Q = np.diag([1, 1, 10, 1])  # Penalize angle more
        R = np.array([[0.1]])
        
        mpc = LinearMPC(
            system,
            horizon=50,
            Q=Q,
            R=R,
            u_min=-20.0,
            u_max=20.0,
        )
        
        # Small initial angle perturbation
        x0 = np.array([0, 0, 0.1, 0])  # 0.1 rad angle
        
        result = mpc.solve(x0)
        
        assert result.is_optimal
        assert result.x.shape == (51, 4)


class TestRoboticsScenarios:
    """Realistic robotics scenarios."""
    
    def test_2d_point_mass(self):
        """2D point mass trajectory tracking."""
        from cuprox.mpc import TrackingMPC
        from cuprox.mpc.dynamics import double_integrator_2d
        from cuprox.mpc.trajectory import circular_reference
        
        system = double_integrator_2d(dt=0.1)
        
        mpc = TrackingMPC(
            system,
            horizon=20,
            Q=np.diag([10, 10, 1, 1]),
            R=np.diag([0.1, 0.1]),
            u_min=-2.0,
            u_max=2.0,
        )
        
        # Circular trajectory
        trajectory = circular_reference(
            center=np.array([0, 0]),
            radius=1.0,
            angular_velocity=0.5,
            horizon=200,
            dt=0.1
        )
        
        x0 = np.array([1, 0, 0, 0.5])  # Start on circle
        
        result = mpc.solve(x0, trajectory=trajectory)
        
        assert result.is_optimal
    
    def test_waypoint_navigation(self):
        """Navigate through waypoints."""
        from cuprox.mpc import LinearMPC
        from cuprox.mpc.dynamics import double_integrator_2d
        
        system = double_integrator_2d(dt=0.1)
        
        mpc = LinearMPC(
            system,
            horizon=30,
            Q=np.diag([5, 5, 1, 1]),
            R=np.diag([0.1, 0.1]),
            u_min=-1.0,
            u_max=1.0,
        )
        
        waypoints = [
            np.array([1, 0, 0, 0]),
            np.array([1, 1, 0, 0]),
        ]
        
        x = np.array([0, 0, 0, 0])
        
        for wp in waypoints:
            result = mpc.solve(x, x_ref=wp)
            assert result.is_optimal


class TestMPCWithDisturbances:
    """MPC with disturbances."""
    
    def test_disturbance_rejection(self):
        """Reject constant disturbance."""
        from cuprox.mpc import LinearMPC
        from cuprox.mpc.dynamics import double_integrator
        
        system = double_integrator(dt=0.1)
        
        mpc = LinearMPC(
            system,
            horizon=20,
            Q=np.diag([10, 1]),
            R=np.array([[0.1]]),
            u_min=-2.0,
            u_max=2.0,
        )
        
        x0 = np.array([0, 0])
        
        # Constant disturbance pushing in +x direction
        disturbance = np.tile([0.01, 0], (50, 1))
        
        sim = mpc.simulate(x0, n_steps=50, disturbance=disturbance)
        
        # Check simulation ran
        assert sim['x'].shape == (51, 2)


class TestMPCComputationTime:
    """Test solve time characteristics."""
    
    def test_solve_time_exists(self):
        """Solve time is reported."""
        from cuprox.mpc import LinearMPC
        from cuprox.mpc.dynamics import double_integrator
        
        system = double_integrator(dt=0.1)
        
        mpc = LinearMPC(
            system,
            horizon=20,
            Q=np.diag([1, 1]),
            R=np.array([[0.1]]),
        )
        
        result = mpc.solve(np.array([1, 0]))
        
        assert result.solve_time >= 0
        assert result.iterations >= 0

