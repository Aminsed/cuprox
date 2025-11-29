"""
System Dynamics Models
======================

Classes for defining dynamical systems used in MPC.

Supported models:
- Linear Time-Invariant (LTI): x_{k+1} = A x_k + B u_k
- Affine: x_{k+1} = A x_k + B u_k + c
- Discretized continuous systems
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class LinearSystem:
    """
    Linear Time-Invariant (LTI) discrete-time system.
    
    Dynamics: x_{k+1} = A @ x_k + B @ u_k
    Output:   y_k = C @ x_k + D @ u_k (optional)
    
    Args:
        A: State transition matrix (n_x, n_x)
        B: Input matrix (n_x, n_u)
        C: Output matrix (n_y, n_x), optional
        D: Feedthrough matrix (n_y, n_u), optional
        dt: Sampling time (for reference only)
    
    Example:
        >>> # Double integrator (position, velocity)
        >>> dt = 0.1
        >>> A = np.array([[1, dt], [0, 1]])
        >>> B = np.array([[0.5*dt**2], [dt]])
        >>> system = LinearSystem(A, B, dt=dt)
        >>> 
        >>> # Simulate one step
        >>> x = np.array([0, 1])  # pos=0, vel=1
        >>> u = np.array([0.5])   # acceleration
        >>> x_next = system.step(x, u)
    """
    A: np.ndarray
    B: np.ndarray
    C: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    dt: float = 1.0
    
    def __post_init__(self):
        """Validate dimensions."""
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {self.A.shape}")
        if self.B.ndim != 2:
            raise ValueError(f"B must be 2D, got shape {self.B.shape}")
        
        n_x = self.A.shape[0]
        if self.A.shape != (n_x, n_x):
            raise ValueError(f"A must be square, got shape {self.A.shape}")
        if self.B.shape[0] != n_x:
            raise ValueError(
                f"B rows ({self.B.shape[0]}) must match A ({n_x})"
            )
        
        if self.C is not None:
            self.C = np.asarray(self.C, dtype=np.float64)
            if self.C.shape[1] != n_x:
                raise ValueError(f"C columns must match state dim {n_x}")
        
        if self.D is not None:
            self.D = np.asarray(self.D, dtype=np.float64)
    
    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.A.shape[0]
    
    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        return self.B.shape[1]
    
    @property
    def n_outputs(self) -> int:
        """Number of outputs."""
        if self.C is not None:
            return self.C.shape[0]
        return self.n_states
    
    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Simulate one time step.
        
        Args:
            x: Current state (n_x,)
            u: Control input (n_u,)
        
        Returns:
            Next state (n_x,)
        """
        return self.A @ x + self.B @ u
    
    def output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute system output.
        
        Args:
            x: Current state (n_x,)
            u: Control input (n_u,)
        
        Returns:
            Output (n_y,)
        """
        if self.C is None:
            return x
        
        y = self.C @ x
        if self.D is not None:
            y = y + self.D @ u
        return y
    
    def simulate(
        self,
        x0: np.ndarray,
        u_sequence: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate system over a sequence of inputs.
        
        Args:
            x0: Initial state (n_x,)
            u_sequence: Control sequence (N, n_u)
        
        Returns:
            State trajectory (N+1, n_x) including initial state
        """
        x0 = np.asarray(x0)
        u_sequence = np.asarray(u_sequence)
        
        N = len(u_sequence)
        trajectory = np.zeros((N + 1, self.n_states))
        trajectory[0] = x0
        
        for k in range(N):
            trajectory[k + 1] = self.step(trajectory[k], u_sequence[k])
        
        return trajectory
    
    def is_stable(self) -> bool:
        """Check if system is stable (all eigenvalues inside unit circle)."""
        eigenvalues = np.linalg.eigvals(self.A)
        return np.all(np.abs(eigenvalues) < 1.0)
    
    def is_controllable(self) -> bool:
        """Check if system is controllable."""
        n = self.n_states
        controllability = self.B
        
        for i in range(1, n):
            controllability = np.hstack([
                controllability,
                np.linalg.matrix_power(self.A, i) @ self.B
            ])
        
        return np.linalg.matrix_rank(controllability) == n
    
    @classmethod
    def from_continuous(
        cls,
        Ac: np.ndarray,
        Bc: np.ndarray,
        dt: float,
        method: str = "zoh",
    ) -> "LinearSystem":
        """
        Create discrete system from continuous-time dynamics.
        
        Continuous: dx/dt = Ac @ x + Bc @ u
        Discrete:   x_{k+1} = A @ x_k + B @ u_k
        
        Args:
            Ac: Continuous state matrix
            Bc: Continuous input matrix
            dt: Sampling time
            method: Discretization method ('zoh', 'euler', 'tustin')
        
        Returns:
            Discrete LinearSystem
        """
        Ac = np.asarray(Ac, dtype=np.float64)
        Bc = np.asarray(Bc, dtype=np.float64)
        
        if method == "euler":
            # Forward Euler: A = I + Ac*dt, B = Bc*dt
            A = np.eye(Ac.shape[0]) + Ac * dt
            B = Bc * dt
        
        elif method == "zoh":
            # Zero-Order Hold (exact discretization)
            from scipy.linalg import expm
            
            n = Ac.shape[0]
            m = Bc.shape[1]
            
            # Build augmented matrix [Ac, Bc; 0, 0]
            M = np.zeros((n + m, n + m))
            M[:n, :n] = Ac * dt
            M[:n, n:] = Bc * dt
            
            # Matrix exponential
            eM = expm(M)
            A = eM[:n, :n]
            B = eM[:n, n:]
        
        elif method == "tustin":
            # Bilinear (Tustin) transform
            n = Ac.shape[0]
            I = np.eye(n)
            
            inv_term = np.linalg.inv(I - (dt / 2) * Ac)
            A = inv_term @ (I + (dt / 2) * Ac)
            B = inv_term @ Bc * dt
        
        else:
            raise ValueError(f"Unknown method '{method}'")
        
        return cls(A, B, dt=dt)


@dataclass 
class AffineSystem:
    """
    Affine discrete-time system.
    
    Dynamics: x_{k+1} = A @ x_k + B @ u_k + c
    
    Useful for systems with constant disturbances or operating
    point offsets.
    
    Args:
        A: State transition matrix (n_x, n_x)
        B: Input matrix (n_x, n_u)
        c: Constant offset (n_x,)
        dt: Sampling time
    
    Example:
        >>> # System with gravity offset
        >>> A = np.array([[1, 0.1], [0, 1]])
        >>> B = np.array([[0], [0.1]])
        >>> c = np.array([0, -0.098])  # gravity
        >>> system = AffineSystem(A, B, c)
    """
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    dt: float = 1.0
    
    def __post_init__(self):
        """Validate dimensions."""
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        self.c = np.asarray(self.c, dtype=np.float64)
        
        n_x = self.A.shape[0]
        if self.c.shape != (n_x,):
            raise ValueError(f"c shape {self.c.shape} must match A ({n_x},)")
    
    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.A.shape[0]
    
    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        return self.B.shape[1]
    
    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Simulate one time step."""
        return self.A @ x + self.B @ u + self.c
    
    def simulate(
        self,
        x0: np.ndarray,
        u_sequence: np.ndarray,
    ) -> np.ndarray:
        """Simulate system over a sequence of inputs."""
        N = len(u_sequence)
        trajectory = np.zeros((N + 1, self.n_states))
        trajectory[0] = x0
        
        for k in range(N):
            trajectory[k + 1] = self.step(trajectory[k], u_sequence[k])
        
        return trajectory
    
    def to_linear(self) -> Tuple[LinearSystem, np.ndarray]:
        """
        Convert to augmented linear system.
        
        Returns:
            Tuple of (LinearSystem, augmentation info)
        """
        # Augment state with constant 1
        n_x = self.n_states
        n_u = self.n_inputs
        
        A_aug = np.zeros((n_x + 1, n_x + 1))
        A_aug[:n_x, :n_x] = self.A
        A_aug[:n_x, n_x] = self.c
        A_aug[n_x, n_x] = 1
        
        B_aug = np.zeros((n_x + 1, n_u))
        B_aug[:n_x, :] = self.B
        
        return LinearSystem(A_aug, B_aug, dt=self.dt), self.c


def double_integrator(dt: float = 0.1) -> LinearSystem:
    """
    Create a double integrator (point mass) system.
    
    States: [position, velocity]
    Input: acceleration
    
    Args:
        dt: Sampling time
    
    Returns:
        LinearSystem for double integrator
    """
    A = np.array([
        [1, dt],
        [0, 1]
    ])
    B = np.array([
        [0.5 * dt**2],
        [dt]
    ])
    return LinearSystem(A, B, dt=dt)


def double_integrator_2d(dt: float = 0.1) -> LinearSystem:
    """
    Create a 2D double integrator (point mass in plane).
    
    States: [x, y, vx, vy]
    Inputs: [ax, ay]
    
    Args:
        dt: Sampling time
    
    Returns:
        LinearSystem for 2D double integrator
    """
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    B = np.array([
        [0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt]
    ])
    return LinearSystem(A, B, dt=dt)


def cart_pole(dt: float = 0.02, linearize_at_origin: bool = True) -> LinearSystem:
    """
    Create linearized cart-pole (inverted pendulum) system.
    
    States: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    Input: force on cart
    
    Args:
        dt: Sampling time
        linearize_at_origin: If True, linearize around upright position
    
    Returns:
        LinearSystem for linearized cart-pole
    """
    # Physical parameters
    M = 1.0   # Cart mass
    m = 0.1   # Pole mass
    L = 0.5   # Pole length
    g = 9.81  # Gravity
    
    # Linearized continuous dynamics around upright
    # States: [x, x_dot, theta, theta_dot]
    denom = M + m
    
    Ac = np.array([
        [0, 1, 0, 0],
        [0, 0, -m * g / denom, 0],
        [0, 0, 0, 1],
        [0, 0, (M + m) * g / (L * denom), 0]
    ])
    
    Bc = np.array([
        [0],
        [1 / denom],
        [0],
        [-1 / (L * denom)]
    ])
    
    return LinearSystem.from_continuous(Ac, Bc, dt, method="zoh")

