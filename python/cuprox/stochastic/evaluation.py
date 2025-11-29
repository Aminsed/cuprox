"""
Solution Evaluation for Stochastic Programming
==============================================

Functions for evaluating solutions and computing important metrics:
- EVPI: Expected Value of Perfect Information
- VSS: Value of the Stochastic Solution
- Out-of-sample evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np

from .problem import TwoStageLP, TwoStageResult
from .scenarios import ScenarioSet


@dataclass
class EvaluationResult:
    """
    Solution evaluation result.
    
    Attributes:
        objective: Estimated objective value
        first_stage_cost: c'x
        expected_recourse: E[Q(x, ξ)]
        std_recourse: Standard deviation of recourse
        worst_case: Maximum recourse cost
        best_case: Minimum recourse cost
        n_scenarios: Number of scenarios evaluated
    """
    objective: float
    first_stage_cost: float
    expected_recourse: float
    std_recourse: float
    worst_case: float
    best_case: float
    n_scenarios: int
    
    def __repr__(self) -> str:
        return (
            f"EvaluationResult(\n"
            f"  objective={self.objective:.4f},\n"
            f"  expected_recourse={self.expected_recourse:.4f},\n"
            f"  std_recourse={self.std_recourse:.4f}\n"
            f")"
        )


def evaluate_solution(
    x: np.ndarray,
    problem: TwoStageLP,
    scenarios: Optional[ScenarioSet] = None,
) -> EvaluationResult:
    """
    Evaluate first-stage decision on scenarios.
    
    Args:
        x: First-stage decision
        problem: Two-stage problem
        scenarios: Scenarios to evaluate on (default: problem's scenarios)
    
    Returns:
        EvaluationResult
    """
    if scenarios is None:
        scenarios = problem.scenarios
    
    first_stage = float(problem.c @ x)
    
    recourse_costs = []
    for s in scenarios:
        Q = s.evaluate_recourse(x)
        recourse_costs.append(s.probability * Q)
    
    recourse_arr = np.array(recourse_costs)
    
    expected_recourse = recourse_arr.sum()
    
    # Unweighted statistics
    raw_costs = np.array([s.evaluate_recourse(x) for s in scenarios])
    
    return EvaluationResult(
        objective=first_stage + expected_recourse,
        first_stage_cost=first_stage,
        expected_recourse=expected_recourse,
        std_recourse=raw_costs.std(),
        worst_case=raw_costs.max(),
        best_case=raw_costs.min(),
        n_scenarios=len(scenarios),
    )


def compute_evpi(problem: TwoStageLP) -> float:
    """
    Compute Expected Value of Perfect Information (EVPI).
    
    EVPI measures the value of knowing the future perfectly.
    
    EVPI = RP - WS
    
    where:
    - RP = optimal value of stochastic program (recourse problem)
    - WS = wait-and-see solution = E[min_{x,y} cost given ξ]
    
    Args:
        problem: Two-stage stochastic problem (solved)
    
    Returns:
        EVPI value (always non-negative)
    
    Example:
        >>> result = problem.solve()
        >>> evpi = compute_evpi(problem)
        >>> print(f"Perfect information worth: {evpi:.2f}")
    """
    # Solve stochastic problem
    rp_result = problem.solve()
    rp = rp_result.total_cost
    
    # Compute wait-and-see
    ws = 0.0
    
    for s in problem.scenarios:
        # For each scenario, solve deterministically
        # This gives the optimal solution if we knew ξ in advance
        
        from scipy.optimize import linprog
        
        n_x = problem.n_x
        n_y = s.n_second_stage_vars
        
        # Combined problem: min c'x + q'y s.t. Ax<=b, Tx+Wy=h, x,y>=0
        c_full = np.concatenate([problem.c, s.q])
        
        # First stage constraints
        m_1 = len(problem.b)
        m_2 = s.n_second_stage_constraints
        
        A_ub = None
        b_ub = None
        
        if m_1 > 0:
            A_ub = np.zeros((m_1, n_x + n_y))
            A_ub[:, :n_x] = problem.A
            b_ub = problem.b
        
        # Second stage constraints (equality)
        A_eq = np.zeros((m_2, n_x + n_y))
        A_eq[:, :n_x] = s.T
        A_eq[:, n_x:] = s.W
        b_eq = s.h
        
        # Bounds
        bounds = [(problem.lb[i], problem.ub[i]) for i in range(n_x)]
        bounds += [(0, None) for _ in range(n_y)]
        
        result = linprog(
            c=c_full,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if result.success:
            ws += s.probability * result.fun
        else:
            ws += s.probability * float('inf')
    
    return max(rp - ws, 0)


def compute_vss(problem: TwoStageLP) -> float:
    """
    Compute Value of the Stochastic Solution (VSS).
    
    VSS measures the value of considering uncertainty in the model.
    
    VSS = EEV - RP
    
    where:
    - RP = optimal value of stochastic program
    - EEV = Expected result of using Expected Value solution
    
    Args:
        problem: Two-stage stochastic problem
    
    Returns:
        VSS value (always non-negative)
    
    Example:
        >>> vss = compute_vss(problem)
        >>> print(f"Ignoring uncertainty costs: {vss:.2f}")
    """
    # Solve stochastic problem
    rp_result = problem.solve()
    rp = rp_result.total_cost
    
    # Compute expected value solution (EV)
    # Replace random parameters with their expectations
    scenarios = list(problem.scenarios)
    
    n_y = scenarios[0].n_second_stage_vars
    m_2 = scenarios[0].n_second_stage_constraints
    n_x = problem.n_x
    
    # Compute expected parameters
    q_exp = np.zeros(n_y)
    h_exp = np.zeros(m_2)
    
    for s in scenarios:
        q_exp += s.probability * s.q
        h_exp += s.probability * s.h
    
    # Assume W and T are the same across scenarios
    W = scenarios[0].W
    T = scenarios[0].T
    
    # Solve EV problem
    from scipy.optimize import linprog
    
    c_full = np.concatenate([problem.c, q_exp])
    
    m_1 = len(problem.b)
    
    A_ub = None
    b_ub = None
    
    if m_1 > 0:
        A_ub = np.zeros((m_1, n_x + n_y))
        A_ub[:, :n_x] = problem.A
        b_ub = problem.b
    
    A_eq = np.zeros((m_2, n_x + n_y))
    A_eq[:, :n_x] = T
    A_eq[:, n_x:] = W
    b_eq = h_exp
    
    bounds = [(problem.lb[i], problem.ub[i]) for i in range(n_x)]
    bounds += [(0, None) for _ in range(n_y)]
    
    ev_result = linprog(
        c=c_full,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )
    
    if not ev_result.success:
        return float('inf')
    
    x_ev = ev_result.x[:n_x]
    
    # Evaluate EV solution on true distribution
    eev = problem.evaluate(x_ev)
    
    return max(eev - rp, 0)


def out_of_sample_evaluation(
    x: np.ndarray,
    problem: TwoStageLP,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate solution on fresh out-of-sample scenarios.
    
    Args:
        x: First-stage decision
        problem: Problem definition
        n_samples: Number of out-of-sample scenarios
        seed: Random seed
    
    Returns:
        Dictionary with statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample from existing scenarios
    sampled = problem.scenarios.sample(n_samples)
    
    result = evaluate_solution(x, problem, sampled)
    
    return {
        'mean': result.objective,
        'std': result.std_recourse,
        'worst_case': result.first_stage_cost + result.worst_case,
        'best_case': result.first_stage_cost + result.best_case,
        'n_samples': n_samples,
    }


def stability_analysis(
    problem: TwoStageLP,
    n_replications: int = 20,
    sample_sizes: List[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Analyze stability of SAA solution w.r.t. sample size.
    
    Args:
        problem: Two-stage problem
        n_replications: Number of replications per sample size
        sample_sizes: List of sample sizes to test
        seed: Random seed
    
    Returns:
        Dictionary with:
        - sample_sizes: Sample sizes tested
        - mean_objectives: Mean objective per sample size
        - std_objectives: Std of objective per sample size
    """
    if sample_sizes is None:
        sample_sizes = [10, 50, 100, 500, 1000]
    
    if seed is not None:
        np.random.seed(seed)
    
    mean_objs = []
    std_objs = []
    
    for n in sample_sizes:
        objs = []
        
        for _ in range(n_replications):
            # Sample and solve
            sampled = problem.scenarios.sample(n)
            
            sub_problem = TwoStageLP(
                c=problem.c,
                A=problem.A,
                b=problem.b,
                sense=problem.sense,
                lb=problem.lb,
                ub=problem.ub,
            )
            sub_problem.add_scenarios_from_set(sampled)
            
            try:
                result = sub_problem.solve()
                objs.append(result.total_cost)
            except Exception:
                pass
        
        if objs:
            mean_objs.append(np.mean(objs))
            std_objs.append(np.std(objs))
        else:
            mean_objs.append(np.nan)
            std_objs.append(np.nan)
    
    return {
        'sample_sizes': np.array(sample_sizes),
        'mean_objectives': np.array(mean_objs),
        'std_objectives': np.array(std_objs),
    }

