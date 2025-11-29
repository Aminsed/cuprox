"""
MPC Controllers
===============

GPU-accelerated Model Predictive Control implementations.

Classes:
- LinearMPC: Standard linear MPC with box constraints
- TrackingMPC: Reference trajectory tracking MPC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import numpy as np
from scipy import sparse

from .. import solve
from ..result import Status
from .dynamics import LinearSystem, AffineSystem
from .constraints import BoxConstraints
from .trajectory import Trajectory, constant_reference


@dataclass
class MPCResult:
    """
    MPC solution result.
    
    Attributes:
        x: Predicted state trajectory (N+1, n_x)
        u: Optimal control sequence (N, n_u)
        cost: Optimal cost value
        status: Solver status
        solve_time: Computation time (seconds)
        iterations: Solver iterations
    """
    x: np.ndarray
    u: np.ndarray
    cost: float
    status: str
    solve_time: float
    iterations: int
    
    @property
    def optimal_control(self) -> np.ndarray:
        """First control action to apply (n_u,)."""
        return self.u[0]
    
    @property
    def predicted_trajectory(self) -> np.ndarray:
        """Predicted state trajectory (N+1, n_x)."""
        return self.x
    
    @property
    def is_optimal(self) -> bool:
        """Whether solution is optimal."""
        return self.status in ["optimal", "max_iterations"]
    
    def __repr__(self) -> str:
        return (
            f"MPCResult(\n"
            f"  status={self.status},\n"
            f"  cost={self.cost:.4f},\n"
            f"  solve_time={self.solve_time*1000:.2f}ms,\n"
            f"  horizon={len(self.u)}\n"
            f")"
        )


class LinearMPC:
    """
    Linear Model Predictive Controller.
    
    Solves at each time step:
    
        minimize    Σ_{k=0}^{N-1} [x_k' Q x_k + u_k' R u_k] + x_N' Qf x_N
        subject to  x_{k+1} = A x_k + B u_k
                    x_min <= x_k <= x_max
                    u_min <= u_k <= u_max
                    x_0 = x_current
    
    Args:
        system: LinearSystem dynamics
        horizon: Prediction horizon N
        Q: State cost matrix (n_x, n_x)
        R: Input cost matrix (n_u, n_u)
        Qf: Terminal cost matrix (default: Q)
        x_min: State lower bounds
        x_max: State upper bounds
        u_min: Input lower bounds
        u_max: Input upper bounds
    
    Example:
        >>> # Double integrator
        >>> A = np.array([[1, 0.1], [0, 1]])
        >>> B = np.array([[0.005], [0.1]])
        >>> system = LinearSystem(A, B)
        >>> 
        >>> # Create MPC
        >>> mpc = LinearMPC(
        ...     system,
        ...     horizon=20,
        ...     Q=np.diag([10, 1]),
        ...     R=np.array([[0.1]]),
        ...     u_min=-1.0, u_max=1.0
        ... )
        >>> 
        >>> # Solve from current state
        >>> x0 = np.array([1.0, 0.5])
        >>> result = mpc.solve(x0)
        >>> u_apply = result.optimal_control
    """
    
    def __init__(
        self,
        system: Union[LinearSystem, AffineSystem],
        horizon: int,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: Optional[np.ndarray] = None,
        x_min: Optional[Union[float, np.ndarray]] = None,
        x_max: Optional[Union[float, np.ndarray]] = None,
        u_min: Optional[Union[float, np.ndarray]] = None,
        u_max: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        self.system = system
        self.horizon = horizon
        
        # Dimensions
        self.n_x = system.n_states
        self.n_u = system.n_inputs
        
        # Cost matrices
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.Qf = np.asarray(Qf, dtype=np.float64) if Qf is not None else self.Q
        
        self._validate_costs()
        
        # Constraints
        self._setup_constraints(x_min, x_max, u_min, u_max)
        
        # Build QP matrices (cached for efficiency)
        self._build_qp_matrices()
    
    def _validate_costs(self):
        """Validate cost matrix dimensions."""
        if self.Q.shape != (self.n_x, self.n_x):
            raise ValueError(f"Q must be ({self.n_x}, {self.n_x})")
        if self.R.shape != (self.n_u, self.n_u):
            raise ValueError(f"R must be ({self.n_u}, {self.n_u})")
        if self.Qf.shape != (self.n_x, self.n_x):
            raise ValueError(f"Qf must be ({self.n_x}, {self.n_x})")
    
    def _setup_constraints(self, x_min, x_max, u_min, u_max):
        """Setup state and input constraints."""
        # State constraints
        if x_min is None:
            self.x_min = np.full(self.n_x, -1e20)
        elif np.isscalar(x_min):
            self.x_min = np.full(self.n_x, x_min)
        else:
            self.x_min = np.asarray(x_min, dtype=np.float64)
        
        if x_max is None:
            self.x_max = np.full(self.n_x, 1e20)
        elif np.isscalar(x_max):
            self.x_max = np.full(self.n_x, x_max)
        else:
            self.x_max = np.asarray(x_max, dtype=np.float64)
        
        # Input constraints
        if u_min is None:
            self.u_min = np.full(self.n_u, -1e20)
        elif np.isscalar(u_min):
            self.u_min = np.full(self.n_u, u_min)
        else:
            self.u_min = np.asarray(u_min, dtype=np.float64)
        
        if u_max is None:
            self.u_max = np.full(self.n_u, 1e20)
        elif np.isscalar(u_max):
            self.u_max = np.full(self.n_u, u_max)
        else:
            self.u_max = np.asarray(u_max, dtype=np.float64)
    
    def _build_qp_matrices(self):
        """
        Build sparse QP matrices for the MPC problem.
        
        Decision variables: z = [x_0, x_1, ..., x_N, u_0, u_1, ..., u_{N-1}]
        
        QP: minimize    (1/2) z' P z + q' z
            subject to  A_eq z = b_eq  (dynamics)
                        lb <= z <= ub  (bounds)
        """
        N = self.horizon
        n_x = self.n_x
        n_u = self.n_u
        
        # Total decision variables
        n_vars = (N + 1) * n_x + N * n_u
        
        # Build P (block diagonal)
        P_blocks = []
        
        # State costs: Q for k=0..N-1, Qf for k=N
        for k in range(N):
            P_blocks.append(self.Q)
        P_blocks.append(self.Qf)  # Terminal cost
        
        # Input costs
        for k in range(N):
            P_blocks.append(self.R)
        
        self._P = sparse.block_diag(P_blocks, format='csr')
        
        # Build equality constraints for dynamics
        # x_{k+1} = A x_k + B u_k
        # => -A x_k + I x_{k+1} - B u_k = 0
        
        A_sys = self.system.A
        B_sys = self.system.B
        
        n_eq = N * n_x  # N dynamics constraints
        
        rows = []
        cols = []
        data = []
        
        for k in range(N):
            row_start = k * n_x
            
            # -A x_k
            x_k_start = k * n_x
            for i in range(n_x):
                for j in range(n_x):
                    if abs(A_sys[i, j]) > 1e-12:
                        rows.append(row_start + i)
                        cols.append(x_k_start + j)
                        data.append(-A_sys[i, j])
            
            # +I x_{k+1}
            x_k1_start = (k + 1) * n_x
            for i in range(n_x):
                rows.append(row_start + i)
                cols.append(x_k1_start + i)
                data.append(1.0)
            
            # -B u_k
            u_k_start = (N + 1) * n_x + k * n_u
            for i in range(n_x):
                for j in range(n_u):
                    if abs(B_sys[i, j]) > 1e-12:
                        rows.append(row_start + i)
                        cols.append(u_k_start + j)
                        data.append(-B_sys[i, j])
        
        self._A_eq = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_eq, n_vars)
        )
        
        # Bounds for full decision vector
        # z = [x_0, x_1, ..., x_N, u_0, ..., u_{N-1}]
        lb = []
        ub = []
        
        # State bounds (N+1 states)
        for k in range(N + 1):
            lb.extend(self.x_min)
            ub.extend(self.x_max)
        
        # Input bounds (N inputs)
        for k in range(N):
            lb.extend(self.u_min)
            ub.extend(self.u_max)
        
        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self._n_vars = n_vars
        self._n_eq = n_eq
    
    def solve(
        self,
        x0: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
        max_iters: int = 5000,
        tolerance: float = 1e-4,
        verbose: bool = False,
        warm_start: Optional[np.ndarray] = None,
    ) -> MPCResult:
        """
        Solve MPC problem from current state.
        
        Args:
            x0: Current state (n_x,)
            x_ref: State reference for regulation (default: origin)
            u_ref: Input reference (default: zero)
            max_iters: Maximum solver iterations
            tolerance: Convergence tolerance
            verbose: Print solver progress
            warm_start: Initial guess for decision variables
        
        Returns:
            MPCResult with optimal trajectory and controls
        """
        x0 = np.asarray(x0, dtype=np.float64)
        
        if x0.shape != (self.n_x,):
            raise ValueError(f"x0 must have shape ({self.n_x},), got {x0.shape}")
        
        N = self.horizon
        n_x = self.n_x
        n_u = self.n_u
        
        # Default references (regulation to origin)
        if x_ref is None:
            x_ref = np.zeros(n_x)
        else:
            x_ref = np.asarray(x_ref)
        
        if u_ref is None:
            u_ref = np.zeros(n_u)
        else:
            u_ref = np.asarray(u_ref)
        
        # Build linear cost q (for tracking x_ref, u_ref)
        # Cost: (x - x_ref)' Q (x - x_ref) = x'Qx - 2 x_ref'Q x + const
        q = np.zeros(self._n_vars)
        
        # State cost terms
        for k in range(N):
            q[k * n_x:(k + 1) * n_x] = -self.Q @ x_ref
        q[N * n_x:(N + 1) * n_x] = -self.Qf @ x_ref  # Terminal
        
        # Input cost terms
        u_start = (N + 1) * n_x
        for k in range(N):
            q[u_start + k * n_u:u_start + (k + 1) * n_u] = -self.R @ u_ref
        
        # RHS for equality constraints (dynamics)
        # Includes affine term if system is AffineSystem
        b_eq = np.zeros(self._n_eq)
        
        if isinstance(self.system, AffineSystem):
            for k in range(N):
                b_eq[k * n_x:(k + 1) * n_x] = -self.system.c
        
        # Fix initial state: x_0 = x0
        lb = self._lb.copy()
        ub = self._ub.copy()
        lb[:n_x] = x0
        ub[:n_x] = x0
        
        # Solve QP
        result = solve(
            c=q,
            A=self._A_eq,
            b=b_eq,
            P=self._P,
            lb=lb,
            ub=ub,
            constraint_l=b_eq,
            constraint_u=b_eq,
            params={
                "max_iterations": max_iters,
                "tolerance": tolerance,
                "verbose": verbose,
            }
        )
        
        # Extract solution
        z = result.x
        
        # States: z[0:(N+1)*n_x]
        x_traj = z[:(N + 1) * n_x].reshape(N + 1, n_x)
        
        # Inputs: z[(N+1)*n_x:]
        u_seq = z[(N + 1) * n_x:].reshape(N, n_u)
        
        # Compute actual cost
        cost = self._compute_cost(x_traj, u_seq, x_ref, u_ref)
        
        status = str(result.status.value) if hasattr(result.status, 'value') else str(result.status)
        
        return MPCResult(
            x=x_traj,
            u=u_seq,
            cost=cost,
            status=status,
            solve_time=result.solve_time,
            iterations=result.iterations,
        )
    
    def _compute_cost(
        self,
        x_traj: np.ndarray,
        u_seq: np.ndarray,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
    ) -> float:
        """Compute MPC cost."""
        N = self.horizon
        cost = 0.0
        
        for k in range(N):
            dx = x_traj[k] - x_ref
            du = u_seq[k] - u_ref
            cost += dx @ self.Q @ dx + du @ self.R @ du
        
        # Terminal cost
        dx = x_traj[N] - x_ref
        cost += dx @ self.Qf @ dx
        
        return 0.5 * cost
    
    def simulate(
        self,
        x0: np.ndarray,
        n_steps: int,
        x_ref: Optional[np.ndarray] = None,
        disturbance: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate closed-loop MPC control.
        
        Args:
            x0: Initial state
            n_steps: Number of simulation steps
            x_ref: State reference
            disturbance: Additive disturbance (n_steps, n_x)
        
        Returns:
            Dictionary with 'x' (states), 'u' (inputs), 'cost' (per step)
        """
        x = np.zeros((n_steps + 1, self.n_x))
        u = np.zeros((n_steps, self.n_u))
        costs = np.zeros(n_steps)
        
        x[0] = x0
        
        for k in range(n_steps):
            # Solve MPC
            result = self.solve(x[k], x_ref=x_ref)
            
            # Apply first control
            u[k] = result.optimal_control
            costs[k] = result.cost
            
            # Simulate system
            x[k + 1] = self.system.step(x[k], u[k])
            
            # Add disturbance if provided
            if disturbance is not None:
                x[k + 1] += disturbance[k]
        
        return {'x': x, 'u': u, 'cost': costs}


class TrackingMPC(LinearMPC):
    """
    MPC for reference trajectory tracking.
    
    Extends LinearMPC to track time-varying reference trajectories.
    
    Args:
        system: LinearSystem dynamics
        horizon: Prediction horizon
        Q: State tracking cost
        R: Input cost
        Qf: Terminal cost (default: Q)
        **kwargs: Additional arguments for LinearMPC
    
    Example:
        >>> mpc = TrackingMPC(system, horizon=20, Q=Q, R=R)
        >>> 
        >>> # Create reference trajectory
        >>> x_ref = circular_reference(center=[0,0], radius=1.0, ...)
        >>> 
        >>> # Solve tracking problem
        >>> result = mpc.solve(x0, trajectory=x_ref)
    """
    
    def solve(
        self,
        x0: np.ndarray,
        trajectory: Optional[Trajectory] = None,
        x_ref: Optional[np.ndarray] = None,
        **kwargs,
    ) -> MPCResult:
        """
        Solve tracking MPC.
        
        Args:
            x0: Current state
            trajectory: Reference trajectory to track
            x_ref: Constant reference (if no trajectory)
            **kwargs: Additional solver arguments
        
        Returns:
            MPCResult with optimal tracking controls
        """
        if trajectory is not None:
            # Time-varying reference
            return self._solve_tracking(x0, trajectory, **kwargs)
        else:
            # Constant reference (regulation)
            return super().solve(x0, x_ref=x_ref, **kwargs)
    
    def _solve_tracking(
        self,
        x0: np.ndarray,
        trajectory: Trajectory,
        current_step: int = 0,
        **kwargs,
    ) -> MPCResult:
        """Solve with time-varying reference."""
        N = self.horizon
        n_x = self.n_x
        n_u = self.n_u
        
        # Get reference window
        ref_window = trajectory.get_window(current_step, N + 1)
        
        # Build cost with time-varying reference
        q = np.zeros(self._n_vars)
        
        for k in range(N):
            x_ref_k = ref_window.get_state(k)
            q[k * n_x:(k + 1) * n_x] = -self.Q @ x_ref_k
        
        # Terminal
        x_ref_N = ref_window.get_state(N)
        q[N * n_x:(N + 1) * n_x] = -self.Qf @ x_ref_N
        
        # Input reference
        u_start = (N + 1) * n_x
        for k in range(N):
            u_ref_k = ref_window.get_input(k)
            if u_ref_k is not None:
                q[u_start + k * n_u:u_start + (k + 1) * n_u] = -self.R @ u_ref_k
        
        # Equality constraints RHS
        b_eq = np.zeros(self._n_eq)
        if isinstance(self.system, AffineSystem):
            for k in range(N):
                b_eq[k * n_x:(k + 1) * n_x] = -self.system.c
        
        # Fix initial state
        lb = self._lb.copy()
        ub = self._ub.copy()
        lb[:n_x] = x0
        ub[:n_x] = x0
        
        # Solve
        result = solve(
            c=q,
            A=self._A_eq,
            b=b_eq,
            P=self._P,
            lb=lb,
            ub=ub,
            constraint_l=b_eq,
            constraint_u=b_eq,
            params={
                "max_iterations": kwargs.get('max_iters', 5000),
                "tolerance": kwargs.get('tolerance', 1e-4),
                "verbose": kwargs.get('verbose', False),
            }
        )
        
        # Extract solution
        z = result.x
        x_traj = z[:(N + 1) * n_x].reshape(N + 1, n_x)
        u_seq = z[(N + 1) * n_x:].reshape(N, n_u)
        
        # Compute tracking cost
        cost = 0.0
        for k in range(N):
            dx = x_traj[k] - ref_window.get_state(k)
            cost += dx @ self.Q @ dx + u_seq[k] @ self.R @ u_seq[k]
        dx = x_traj[N] - ref_window.get_state(N)
        cost += dx @ self.Qf @ dx
        
        status = str(result.status.value) if hasattr(result.status, 'value') else str(result.status)
        
        return MPCResult(
            x=x_traj,
            u=u_seq,
            cost=0.5 * cost,
            status=status,
            solve_time=result.solve_time,
            iterations=result.iterations,
        )
    
    def simulate_tracking(
        self,
        x0: np.ndarray,
        trajectory: Trajectory,
        n_steps: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate closed-loop trajectory tracking.
        
        Args:
            x0: Initial state
            trajectory: Reference trajectory
            n_steps: Number of steps (default: trajectory length)
        
        Returns:
            Dictionary with 'x', 'u', 'x_ref', 'tracking_error'
        """
        if n_steps is None:
            n_steps = trajectory.horizon - self.horizon
        
        x = np.zeros((n_steps + 1, self.n_x))
        u = np.zeros((n_steps, self.n_u))
        x_ref = np.zeros((n_steps + 1, self.n_x))
        
        x[0] = x0
        
        for k in range(n_steps):
            x_ref[k] = trajectory.get_state(k)
            
            result = self._solve_tracking(x[k], trajectory, current_step=k)
            u[k] = result.optimal_control
            x[k + 1] = self.system.step(x[k], u[k])
        
        x_ref[n_steps] = trajectory.get_state(n_steps)
        tracking_error = np.linalg.norm(x - x_ref, axis=1)
        
        return {
            'x': x,
            'u': u,
            'x_ref': x_ref,
            'tracking_error': tracking_error,
        }

