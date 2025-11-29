#ifndef CUPROX_SOLVERS_ADMM_CUH
#define CUPROX_SOLVERS_ADMM_CUH

#include "../core/types.hpp"
#include "../core/dense_vector.cuh"
#include "../core/sparse_matrix.cuh"
#include "../preprocess/scaling.cuh"

namespace cuprox {

/**
 * @brief ADMM solver settings
 */
template <typename T>
struct AdmmSettings {
    int max_iters = 4000;
    T eps_abs = T(1e-6);
    T eps_rel = T(1e-6);
    int check_interval = 25;
    bool verbose = false;
    bool scaling = false;  // Disabled for now
    int scaling_iters = 10;
    
    // ADMM-specific parameters
    T rho = T(1.0);              // Penalty parameter (0 = auto)
    bool adaptive_rho = true;    // Adapt rho during solve
    T rho_min = T(1e-6);
    T rho_max = T(1e6);
    
    // CG settings for x-update
    int cg_max_iters = 100;
    T cg_tol = T(1e-10);
};

/**
 * @brief ADMM solver result
 */
template <typename T>
struct AdmmResult {
    DeviceVector<T> x;      // Primal solution
    DeviceVector<T> y;      // Dual solution (Lagrange multipliers)
    DeviceVector<T> z;      // Slack variable
    T primal_obj;           // Objective: (1/2)x'Px + q'x
    T primal_res;           // Primal residual ||Ax - z||
    T dual_res;             // Dual residual ||ρA'(z - z_prev)||
    int iterations;         // Iterations performed
    Status status;          // Solver status
    double solve_time;      // Wall clock time
};

/**
 * @brief QP problem definition for ADMM
 * 
 * Standard form:
 *   minimize    (1/2)x'Px + q'x
 *   subject to  l <= Ax <= u
 *               lb <= x <= ub
 */
template <typename T>
struct QPProblem {
    CsrMatrix<T> P;         // Quadratic cost (n x n, symmetric positive semidefinite)
    CsrMatrix<T> A;         // Constraint matrix (m x n)
    DeviceVector<T> q;      // Linear cost (n,)
    DeviceVector<T> l;      // Lower bound on Ax (m,)
    DeviceVector<T> u;      // Upper bound on Ax (m,)
    DeviceVector<T> lb;     // Variable lower bounds (n,)
    DeviceVector<T> ub;     // Variable upper bounds (n,)
    
    Index num_vars() const { return q.size(); }
    Index num_constraints() const { return A.num_rows(); }
};

/**
 * @brief ADMM Solver for Quadratic Programs
 * 
 * Solves QP using Alternating Direction Method of Multipliers.
 * 
 * Algorithm per iteration:
 *   x_{k+1} = (P + ρA'A)^{-1} (ρA'z_k - A'y_k - q)
 *   z_{k+1} = proj_{[l,u]}(Ax_{k+1} + y_k/ρ)
 *   y_{k+1} = y_k + ρ(Ax_{k+1} - z_{k+1})
 * 
 * The x-update is solved using Conjugate Gradient.
 */
template <typename T>
class AdmmSolver {
public:
    AdmmSolver() = default;
    explicit AdmmSolver(const AdmmSettings<T>& settings) : settings_(settings) {}
    
    AdmmResult<T> solve(QPProblem<T>& problem);
    
    void set_settings(const AdmmSettings<T>& settings) { settings_ = settings; }
    const AdmmSettings<T>& settings() const { return settings_; }

private:
    void initialize(QPProblem<T>& problem);
    void x_update();    // Solve (P + ρA'A)x = rhs using CG
    void z_update();    // z = proj_{[l,u]}(Ax + y/ρ)
    void y_update();    // y = y + ρ(Ax - z)
    bool check_convergence();
    void compute_residuals();
    void update_rho();
    void solve_unconstrained();  // Special case for m=0
    
    AdmmSettings<T> settings_;
    
    // Problem dimensions
    Index n_ = 0;  // Variables
    Index m_ = 0;  // Constraints
    
    // ADMM parameter
    T rho_ = T(1.0);
    
    // Iterates
    DeviceVector<T> x_;
    DeviceVector<T> z_;
    DeviceVector<T> z_prev_;
    DeviceVector<T> y_;
    
    // Workspace
    DeviceVector<T> Ax_;      // A * x
    DeviceVector<T> Px_;      // P * x
    DeviceVector<T> Aty_;     // A^T * y
    DeviceVector<T> Atz_;     // A^T * z
    DeviceVector<T> rhs_;     // RHS for CG
    DeviceVector<T> temp_;
    
    // CG workspace
    DeviceVector<T> cg_r_;    // Residual
    DeviceVector<T> cg_p_;    // Search direction
    DeviceVector<T> cg_Ap_;   // Matrix-vector product
    
    // Problem data
    CsrMatrix<T>* P_ = nullptr;
    CsrMatrix<T>* A_ = nullptr;
    DeviceVector<T>* q_ = nullptr;
    DeviceVector<T>* l_ = nullptr;           // Constraint bounds
    DeviceVector<T>* u_ = nullptr;           // Constraint bounds
    DeviceVector<T>* problem_lb_ = nullptr;  // Variable lower bounds
    DeviceVector<T>* problem_ub_ = nullptr;  // Variable upper bounds
    
    // Residuals
    T primal_res_ = T(0);
    T dual_res_ = T(0);
    
    // Iteration count
    int iter_ = 0;
};

}  // namespace cuprox

#endif  // CUPROX_SOLVERS_ADMM_CUH

