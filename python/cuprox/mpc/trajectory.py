"""
Reference Trajectories
======================

Utilities for generating reference trajectories for MPC tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class Trajectory:
    """
    Reference trajectory for MPC tracking.
    
    Args:
        states: State trajectory (N, n_x)
        inputs: Input trajectory (N, n_u), optional
        time: Time stamps (N,), optional
    
    Example:
        >>> # Create trajectory from arrays
        >>> x_ref = np.zeros((100, 2))
        >>> x_ref[:, 0] = np.linspace(0, 10, 100)  # position ramp
        >>> traj = Trajectory(states=x_ref)
        >>> 
        >>> # Get reference at step k
        >>> x_ref_k = traj.get_state(k)
    """
    states: np.ndarray
    inputs: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate trajectory."""
        self.states = np.asarray(self.states, dtype=np.float64)
        if self.states.ndim == 1:
            self.states = self.states.reshape(-1, 1)
        
        if self.inputs is not None:
            self.inputs = np.asarray(self.inputs, dtype=np.float64)
            if self.inputs.ndim == 1:
                self.inputs = self.inputs.reshape(-1, 1)
    
    @property
    def horizon(self) -> int:
        """Trajectory length."""
        return len(self.states)
    
    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.states.shape[1]
    
    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        if self.inputs is None:
            return 0
        return self.inputs.shape[1]
    
    def get_state(self, k: int) -> np.ndarray:
        """Get state reference at step k (clamped to bounds)."""
        k = min(max(k, 0), len(self.states) - 1)
        return self.states[k]
    
    def get_input(self, k: int) -> Optional[np.ndarray]:
        """Get input reference at step k."""
        if self.inputs is None:
            return None
        k = min(max(k, 0), len(self.inputs) - 1)
        return self.inputs[k]
    
    def get_window(
        self,
        start: int,
        length: int,
    ) -> "Trajectory":
        """
        Get trajectory window starting at 'start' with given length.
        
        If window extends beyond trajectory, the last value is repeated.
        """
        end = start + length
        
        # Pad if needed
        if end > len(self.states):
            states = np.zeros((length, self.n_states))
            available = min(len(self.states) - start, length)
            states[:available] = self.states[start:start + available]
            states[available:] = self.states[-1]  # Repeat last
            
            inputs = None
            if self.inputs is not None:
                inputs = np.zeros((length, self.n_inputs))
                inputs[:available] = self.inputs[start:start + available]
                inputs[available:] = self.inputs[-1]
        else:
            states = self.states[start:end]
            inputs = self.inputs[start:end] if self.inputs is not None else None
        
        return Trajectory(states=states, inputs=inputs)


def constant_reference(
    x_ref: np.ndarray,
    horizon: int,
    u_ref: Optional[np.ndarray] = None,
) -> Trajectory:
    """
    Create constant (setpoint) reference trajectory.
    
    Args:
        x_ref: Target state (n_x,)
        horizon: Trajectory length
        u_ref: Target input (n_u,), optional
    
    Returns:
        Trajectory with constant reference
    
    Example:
        >>> x_target = np.array([1.0, 0.0])  # target position, zero velocity
        >>> traj = constant_reference(x_target, horizon=50)
    """
    x_ref = np.asarray(x_ref)
    states = np.tile(x_ref, (horizon, 1))
    
    inputs = None
    if u_ref is not None:
        u_ref = np.asarray(u_ref)
        inputs = np.tile(u_ref, (horizon, 1))
    
    return Trajectory(states=states, inputs=inputs)


def step_reference(
    x_initial: np.ndarray,
    x_final: np.ndarray,
    horizon: int,
    step_time: int = 0,
) -> Trajectory:
    """
    Create step reference trajectory.
    
    Args:
        x_initial: Initial state
        x_final: Final state (after step)
        horizon: Total trajectory length
        step_time: Time step when step occurs
    
    Returns:
        Trajectory with step change
    
    Example:
        >>> x0 = np.array([0.0, 0.0])
        >>> x1 = np.array([1.0, 0.0])
        >>> traj = step_reference(x0, x1, horizon=100, step_time=20)
    """
    x_initial = np.asarray(x_initial)
    x_final = np.asarray(x_final)
    
    states = np.zeros((horizon, len(x_initial)))
    states[:step_time] = x_initial
    states[step_time:] = x_final
    
    return Trajectory(states=states)


def ramp_reference(
    x_initial: np.ndarray,
    x_final: np.ndarray,
    horizon: int,
    ramp_duration: Optional[int] = None,
) -> Trajectory:
    """
    Create linear ramp reference trajectory.
    
    Args:
        x_initial: Initial state
        x_final: Final state
        horizon: Total trajectory length
        ramp_duration: Duration of ramp (default: full horizon)
    
    Returns:
        Trajectory with linear interpolation
    """
    x_initial = np.asarray(x_initial)
    x_final = np.asarray(x_final)
    
    if ramp_duration is None:
        ramp_duration = horizon
    
    ramp_duration = min(ramp_duration, horizon)
    
    states = np.zeros((horizon, len(x_initial)))
    
    # Ramp portion
    for i in range(ramp_duration):
        alpha = i / max(ramp_duration - 1, 1)
        states[i] = (1 - alpha) * x_initial + alpha * x_final
    
    # Hold at final value
    states[ramp_duration:] = x_final
    
    return Trajectory(states=states)


def sinusoidal_reference(
    amplitude: np.ndarray,
    frequency: np.ndarray,
    phase: np.ndarray,
    offset: np.ndarray,
    horizon: int,
    dt: float = 1.0,
) -> Trajectory:
    """
    Create sinusoidal reference trajectory.
    
    x_ref[k, i] = amplitude[i] * sin(2*pi*frequency[i]*k*dt + phase[i]) + offset[i]
    
    Args:
        amplitude: Amplitude for each state
        frequency: Frequency (Hz) for each state
        phase: Phase offset (radians) for each state
        offset: DC offset for each state
        horizon: Trajectory length
        dt: Time step
    
    Returns:
        Trajectory with sinusoidal reference
    
    Example:
        >>> # Sinusoidal position, zero velocity
        >>> traj = sinusoidal_reference(
        ...     amplitude=np.array([1.0, 0.0]),
        ...     frequency=np.array([0.1, 0.0]),
        ...     phase=np.array([0.0, 0.0]),
        ...     offset=np.array([0.0, 0.0]),
        ...     horizon=100,
        ...     dt=0.1
        ... )
    """
    amplitude = np.asarray(amplitude)
    frequency = np.asarray(frequency)
    phase = np.asarray(phase)
    offset = np.asarray(offset)
    
    n_states = len(amplitude)
    states = np.zeros((horizon, n_states))
    
    for k in range(horizon):
        t = k * dt
        states[k] = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
    
    return Trajectory(states=states)


def circular_reference(
    center: np.ndarray,
    radius: float,
    angular_velocity: float,
    horizon: int,
    dt: float = 1.0,
    include_velocity: bool = True,
) -> Trajectory:
    """
    Create circular reference trajectory in 2D.
    
    Args:
        center: Circle center (2,)
        radius: Circle radius
        angular_velocity: Angular velocity (rad/s)
        horizon: Trajectory length
        dt: Time step
        include_velocity: Include velocity states
    
    Returns:
        Trajectory for circular motion
        States: [x, y] or [x, y, vx, vy] if include_velocity
    
    Example:
        >>> traj = circular_reference(
        ...     center=np.array([0, 0]),
        ...     radius=1.0,
        ...     angular_velocity=0.5,
        ...     horizon=100,
        ...     dt=0.1
        ... )
    """
    center = np.asarray(center)
    
    if include_velocity:
        states = np.zeros((horizon, 4))
    else:
        states = np.zeros((horizon, 2))
    
    for k in range(horizon):
        theta = angular_velocity * k * dt
        
        # Position
        states[k, 0] = center[0] + radius * np.cos(theta)
        states[k, 1] = center[1] + radius * np.sin(theta)
        
        # Velocity
        if include_velocity:
            states[k, 2] = -radius * angular_velocity * np.sin(theta)
            states[k, 3] = radius * angular_velocity * np.cos(theta)
    
    return Trajectory(states=states)


def figure_eight_reference(
    center: np.ndarray,
    size: float,
    angular_velocity: float,
    horizon: int,
    dt: float = 1.0,
) -> Trajectory:
    """
    Create figure-8 reference trajectory.
    
    Args:
        center: Figure-8 center
        size: Size parameter
        angular_velocity: Angular velocity
        horizon: Trajectory length
        dt: Time step
    
    Returns:
        Trajectory for figure-8 motion (states: [x, y, vx, vy])
    """
    center = np.asarray(center)
    states = np.zeros((horizon, 4))
    
    for k in range(horizon):
        t = angular_velocity * k * dt
        
        # Lemniscate of Bernoulli
        denom = 1 + np.sin(t) ** 2
        states[k, 0] = center[0] + size * np.cos(t) / denom
        states[k, 1] = center[1] + size * np.cos(t) * np.sin(t) / denom
        
        # Numerical velocity (simplified)
        if k > 0:
            states[k, 2] = (states[k, 0] - states[k-1, 0]) / dt
            states[k, 3] = (states[k, 1] - states[k-1, 1]) / dt
    
    return Trajectory(states=states)

