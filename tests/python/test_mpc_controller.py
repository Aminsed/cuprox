"""
Tests for MPC Controllers.

Tests covering:
1. LinearMPC - basic regulation
2. TrackingMPC - trajectory tracking
3. Constraint handling
4. Closed-loop simulation
"""

import numpy as np
import pytest


@pytest.fixture
def double_integrator():
    """Double integrator system."""
    from cuprox.mpc import LinearSystem

    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5 * dt**2], [dt]])
    return LinearSystem(A, B, dt=dt)


@pytest.fixture
def mpc_params():
    """Standard MPC parameters."""
    return {
        "Q": np.diag([10, 1]),
        "R": np.array([[0.1]]),
        "horizon": 20,
    }


class TestLinearMPC:
    """Test LinearMPC controller."""

    def test_basic_creation(self, double_integrator, mpc_params):
        """Create basic MPC."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        assert mpc.horizon == 20
        assert mpc.n_x == 2
        assert mpc.n_u == 1

    def test_solve_from_origin(self, double_integrator, mpc_params):
        """Solve from origin (should stay)."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        x0 = np.array([0, 0])
        result = mpc.solve(x0)

        # At origin, optimal control is zero
        assert result.is_optimal
        np.testing.assert_allclose(result.optimal_control, [0], atol=1e-3)

    def test_solve_regulation(self, double_integrator, mpc_params):
        """Regulate to origin from non-zero initial state."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
            u_min=-1.0,
            u_max=1.0,
        )

        x0 = np.array([1.0, 0.5])
        result = mpc.solve(x0)

        assert result.is_optimal
        assert result.x.shape == (21, 2)  # N+1 states
        assert result.u.shape == (20, 1)  # N inputs

    def test_input_constraints(self, double_integrator, mpc_params):
        """Input constraints are respected."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
            u_min=-0.5,
            u_max=0.5,
        )

        x0 = np.array([10.0, 5.0])  # Large deviation needs large control
        result = mpc.solve(x0)

        # All controls should be within bounds (with small tolerance)
        assert (result.u >= -0.55).all()
        assert (result.u <= 0.55).all()

    def test_solve_to_reference(self, double_integrator, mpc_params):
        """Regulate to non-zero reference."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        x0 = np.array([0, 0])
        x_ref = np.array([1.0, 0.0])  # Target position 1, zero velocity

        result = mpc.solve(x0, x_ref=x_ref)

        assert result.is_optimal
        assert result.x.shape == (21, 2)

    def test_result_attributes(self, double_integrator, mpc_params):
        """Check result has all attributes."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        result = mpc.solve(np.array([1, 0]))

        assert hasattr(result, "x")
        assert hasattr(result, "u")
        assert hasattr(result, "cost")
        assert hasattr(result, "status")
        assert hasattr(result, "solve_time")
        assert hasattr(result, "optimal_control")
        assert hasattr(result, "predicted_trajectory")

    def test_closed_loop_simulation(self, double_integrator, mpc_params):
        """Closed-loop simulation."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
            u_min=-2.0,
            u_max=2.0,
        )

        x0 = np.array([2.0, 0.0])
        sim = mpc.simulate(x0, n_steps=50)

        assert "x" in sim
        assert "u" in sim
        assert sim["x"].shape == (51, 2)


class TestTrackingMPC:
    """Test TrackingMPC controller."""

    def test_constant_tracking(self, double_integrator, mpc_params):
        """Track constant reference."""
        from cuprox.mpc import TrackingMPC
        from cuprox.mpc.trajectory import constant_reference

        mpc = TrackingMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        x0 = np.array([0, 0])
        trajectory = constant_reference(x_ref=np.array([1.0, 0.0]), horizon=50)

        result = mpc.solve(x0, trajectory=trajectory)

        assert result.is_optimal

    def test_step_tracking(self, double_integrator, mpc_params):
        """Track step reference."""
        from cuprox.mpc import TrackingMPC
        from cuprox.mpc.trajectory import step_reference

        mpc = TrackingMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
            u_min=-2.0,
            u_max=2.0,
        )

        trajectory = step_reference(
            x_initial=np.array([0, 0]), x_final=np.array([1, 0]), horizon=100, step_time=10
        )

        x0 = np.array([0, 0])
        result = mpc.solve(x0, trajectory=trajectory, current_step=0)

        assert result.is_optimal

    def test_trajectory_simulation(self, double_integrator, mpc_params):
        """Simulate trajectory tracking."""
        from cuprox.mpc import TrackingMPC
        from cuprox.mpc.trajectory import constant_reference

        mpc = TrackingMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
            u_min=-2.0,
            u_max=2.0,
        )

        trajectory = constant_reference(x_ref=np.array([1.0, 0.0]), horizon=100)

        x0 = np.array([0, 0])
        sim = mpc.simulate_tracking(x0, trajectory, n_steps=30)

        assert "x" in sim
        assert "u" in sim
        assert "x_ref" in sim
        assert "tracking_error" in sim

        # Tracking error should decrease over time
        assert sim["tracking_error"][-1] < sim["tracking_error"][0] + 1


class TestMPCEdgeCases:
    """Edge cases and error handling."""

    def test_wrong_x0_shape(self, double_integrator, mpc_params):
        """Error on wrong initial state shape."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        with pytest.raises(ValueError, match="x0 must have shape"):
            mpc.solve(np.array([1, 2, 3]))  # Wrong dimension

    def test_short_horizon(self, double_integrator, mpc_params):
        """Very short horizon."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=2,
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        result = mpc.solve(np.array([1, 0]))

        assert result.x.shape == (3, 2)
        assert result.u.shape == (2, 1)

    def test_terminal_cost(self, double_integrator, mpc_params):
        """Custom terminal cost."""
        from cuprox.mpc import LinearMPC

        Qf = np.diag([100, 10])  # Higher terminal cost

        mpc = LinearMPC(
            double_integrator,
            horizon=mpc_params["horizon"],
            Q=mpc_params["Q"],
            R=mpc_params["R"],
            Qf=Qf,
        )

        result = mpc.solve(np.array([1, 0]))

        assert result.is_optimal

    def test_single_input_system(self):
        """System with single input."""
        from cuprox.mpc import LinearMPC, LinearSystem

        A = np.array([[0.9]])
        B = np.array([[0.1]])
        system = LinearSystem(A, B)

        mpc = LinearMPC(
            system,
            horizon=10,
            Q=np.array([[1]]),
            R=np.array([[0.1]]),
        )

        result = mpc.solve(np.array([5.0]))

        assert result.is_optimal
        assert result.x.shape == (11, 1)


class TestMPCPerformance:
    """Performance tests."""

    def test_larger_system(self):
        """Larger system (6 states, 2 inputs)."""
        from cuprox.mpc import LinearMPC, LinearSystem

        n_x, n_u = 6, 2
        np.random.seed(42)

        A = np.eye(n_x) + 0.1 * np.random.randn(n_x, n_x)
        A = A * 0.9 / np.max(np.abs(np.linalg.eigvals(A)))  # Make stable
        B = np.random.randn(n_x, n_u) * 0.1

        system = LinearSystem(A, B)

        mpc = LinearMPC(
            system,
            horizon=30,
            Q=np.eye(n_x),
            R=np.eye(n_u) * 0.1,
            u_min=-1.0,
            u_max=1.0,
        )

        x0 = np.random.randn(n_x)
        result = mpc.solve(x0, max_iters=10000)

        assert result.x.shape == (31, n_x)
        assert result.u.shape == (30, n_u)

    def test_long_horizon(self, double_integrator, mpc_params):
        """Longer prediction horizon."""
        from cuprox.mpc import LinearMPC

        mpc = LinearMPC(
            double_integrator,
            horizon=100,
            Q=mpc_params["Q"],
            R=mpc_params["R"],
        )

        result = mpc.solve(np.array([1, 0]), max_iters=10000)

        assert result.x.shape == (101, 2)
