"""
cuProx Model Predictive Control (MPC)
=====================================

GPU-accelerated Model Predictive Control for robotics and control systems.

MPC solves optimal control problems over a receding horizon, making it
ideal for real-time control of robots, vehicles, and industrial processes.

Quick Start
-----------
>>> from cuprox.mpc import LinearMPC, LinearSystem
>>> 
>>> # Define system dynamics: x_{k+1} = A @ x_k + B @ u_k
>>> A = np.array([[1, 0.1], [0, 1]])  # Double integrator
>>> B = np.array([[0.005], [0.1]])
>>> system = LinearSystem(A, B)
>>> 
>>> # Create MPC controller
>>> mpc = LinearMPC(
...     system,
...     horizon=20,
...     Q=np.diag([10, 1]),    # State cost
...     R=np.array([[0.1]]),   # Input cost
...     u_min=-1.0, u_max=1.0  # Input constraints
... )
>>> 
>>> # Compute control action
>>> x0 = np.array([1.0, 0.5])  # Current state
>>> result = mpc.solve(x0)
>>> u_optimal = result.u[0]    # Apply first control

Reference Tracking
------------------
>>> from cuprox.mpc import TrackingMPC
>>> 
>>> # Track a reference trajectory
>>> mpc = TrackingMPC(system, horizon=20, Q=Q, R=R)
>>> result = mpc.solve(x0, x_ref=target_trajectory)

Classes
-------
LinearMPC
    Standard linear MPC with state and input constraints
TrackingMPC
    MPC for reference trajectory tracking
LinearSystem
    Linear time-invariant dynamics
MPCResult
    Solution containing optimal trajectory and controls

Theory
------
MPC solves at each timestep:

    minimize    Σ_{k=0}^{N-1} [x_k' Q x_k + u_k' R u_k] + x_N' Q_f x_N
    subject to  x_{k+1} = A x_k + B u_k
                x_min <= x_k <= x_max
                u_min <= u_k <= u_max
                x_0 = x_current

This is a Quadratic Program (QP), perfectly suited for cuProx's
GPU-accelerated ADMM solver.

See Also
--------
- Rawlings & Mayne (2009): "Model Predictive Control: Theory and Design"
- Borrelli, Bemporad & Morari (2017): "Predictive Control"
"""

from .dynamics import LinearSystem, AffineSystem
from .controller import LinearMPC, TrackingMPC, MPCResult
from .constraints import BoxConstraints, PolytopeConstraints
from .trajectory import (
    Trajectory,
    constant_reference,
    step_reference,
    sinusoidal_reference,
)

__all__ = [
    # Controllers
    "LinearMPC",
    "TrackingMPC",
    "MPCResult",
    # Dynamics
    "LinearSystem",
    "AffineSystem",
    # Constraints
    "BoxConstraints",
    "PolytopeConstraints",
    # Trajectories
    "Trajectory",
    "constant_reference",
    "step_reference",
    "sinusoidal_reference",
]

