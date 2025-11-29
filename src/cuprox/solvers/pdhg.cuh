#ifndef CUPROX_SOLVERS_PDHG_CUH
#define CUPROX_SOLVERS_PDHG_CUH

#include "../core/types.hpp"
#include "../core/dense_vector.cuh"
#include "../core/sparse_matrix.cuh"
#include "../preprocess/scaling.cuh"

namespace cuprox {

/**
 * @brief PDHG solver settings
 */
template <typename T>
struct PdhgSettings {
    int max_iters = 10000;
    T eps_abs = T(1e-6);
    T eps_rel = T(1e-6);
    int check_interval = 25;
    bool verbose = false;
    bool adaptive_restart = true;
    bool scaling = true;
    int scaling_iters = 10;
    T sigma = T(0);  // 0 = auto-compute
    T tau = T(0);    // 0 = auto-compute
};

/**
 * @brief PDHG solver result
 */
template <typename T>
struct PdhgResult {
    DeviceVector<T> x;      // Primal solution
    DeviceVector<T> y;      // Dual solution
    T primal_obj;           // Primal objective c^T x
    T dual_obj;             // Dual objective b^T y
    T primal_res;           // Primal residual ||Ax - b||
    T dual_res;             // Dual residual ||A^T y + s - c||
    int iterations;         // Iterations performed
    Status status;          // Solver status
    double solve_time;      // Wall clock time
};

/**
 * @brief LP problem definition for PDHG
 * 
 * Standard form:
 *   minimize    c^T x
 *   subject to  l <= Ax <= u
 *               lb <= x <= ub
 */
template <typename T>
struct LPProblem {
    CsrMatrix<T> A;
    DeviceVector<T> c;
    DeviceVector<T> b;      // For equality Ax = b
    DeviceVector<T> l;      // Lower bound on Ax
    DeviceVector<T> u;      // Upper bound on Ax
    DeviceVector<T> lb;     // Lower bound on x
    DeviceVector<T> ub;     // Upper bound on x
    
    Index num_vars() const { return A.num_cols(); }
    Index num_constraints() const { return A.num_rows(); }
};

/**
 * @brief PDHG Solver for Linear Programs
 * 
 * Solves LP using Primal-Dual Hybrid Gradient (Chambolle-Pock) method.
 * 
 * Algorithm per iteration:
 *   x_{k+1} = proj_X(x_k - tau * (c + A^T * y_k))
 *   x_bar   = 2 * x_{k+1} - x_k  (extrapolation)
 *   y_{k+1} = proj_Y(y_k + sigma * (A * x_bar - b))
 */
template <typename T>
class PdhgSolver {
public:
    PdhgSolver() = default;
    explicit PdhgSolver(const PdhgSettings<T>& settings) : settings_(settings) {}
    
    PdhgResult<T> solve(LPProblem<T>& problem);
    
    void set_settings(const PdhgSettings<T>& settings) { settings_ = settings; }
    const PdhgSettings<T>& settings() const { return settings_; }

private:
    void initialize(LPProblem<T>& problem);
    void iterate();
    bool check_convergence();
    void compute_residuals();
    void adaptive_restart();
    
    PdhgSettings<T> settings_;
    
    // Problem dimensions
    Index n_ = 0;  // Variables
    Index m_ = 0;  // Constraints
    
    // Step sizes
    T tau_ = T(0);
    T sigma_ = T(0);
    
    // Iterates
    DeviceVector<T> x_;
    DeviceVector<T> x_prev_;
    DeviceVector<T> x_bar_;
    DeviceVector<T> y_;
    DeviceVector<T> y_prev_;
    
    // Workspace
    DeviceVector<T> Ax_;      // A * x
    DeviceVector<T> Aty_;     // A^T * y
    DeviceVector<T> temp_;
    
    // Problem data (may be scaled)
    CsrMatrix<T>* A_ = nullptr;
    DeviceVector<T>* c_ = nullptr;
    DeviceVector<T>* b_ = nullptr;
    DeviceVector<T>* lb_ = nullptr;
    DeviceVector<T>* ub_ = nullptr;
    DeviceVector<T>* l_ = nullptr;
    DeviceVector<T>* u_ = nullptr;
    
    // Residuals
    T primal_res_ = T(0);
    T dual_res_ = T(0);
    T gap_ = T(0);
    
    // Scaling
    ScalingFactors<T> scaling_;
    
    // Iteration count
    int iter_ = 0;
};

}  // namespace cuprox

#endif  // CUPROX_SOLVERS_PDHG_CUH

