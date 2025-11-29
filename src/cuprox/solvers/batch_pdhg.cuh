#ifndef CUPROX_SOLVERS_BATCH_PDHG_CUH
#define CUPROX_SOLVERS_BATCH_PDHG_CUH

#include "../core/types.hpp"
#include "../core/dense_vector.cuh"
#include "../core/sparse_matrix.cuh"
#include "../core/memory.cuh"

namespace cuprox {

/**
 * @brief Batch PDHG solver settings
 */
template <typename T>
struct BatchPdhgSettings {
    int max_iters = 5000;
    T eps_abs = T(1e-5);
    T eps_rel = T(1e-5);
    int check_interval = 50;
    bool verbose = false;
};

/**
 * @brief Batch PDHG solver result
 */
template <typename T>
struct BatchPdhgResult {
    DevicePtr<T> x;         // Solutions: (batch_size x n)
    DevicePtr<T> y;         // Duals: (batch_size x m)
    DevicePtr<T> objectives; // Objectives: (batch_size)
    DevicePtr<int> statuses; // Status per problem: (batch_size)
    DevicePtr<int> iterations; // Iterations per problem: (batch_size)
    Index batch_size;
    Index n;
    Index m;
    double total_solve_time;
};

/**
 * @brief Batch LP problem definition
 * 
 * Solves batch_size LP problems of the form:
 *   minimize    c[i]'x
 *   subject to  A*x = b[i]   (same A for all problems)
 *               lb <= x <= ub (same bounds for all)
 * 
 * This is the common case for MPC, portfolio, etc.
 */
template <typename T>
struct BatchLPProblem {
    CsrMatrix<T>* A;        // Shared constraint matrix (m x n)
    DevicePtr<T> c_batch;   // Batched objectives: (batch_size x n)
    DevicePtr<T> b_batch;   // Batched RHS: (batch_size x m)
    DeviceVector<T>* lb;    // Shared lower bounds (n)
    DeviceVector<T>* ub;    // Shared upper bounds (n)
    Index batch_size;
    Index n;
    Index m;
};

/**
 * @brief Batch PDHG Solver for Linear Programs
 * 
 * Solves many LPs in parallel on GPU. All problems share the same
 * constraint matrix A and bounds, but have different c and b.
 * 
 * This is extremely efficient for:
 * - Model Predictive Control (many horizons)
 * - Monte Carlo scenario optimization
 * - Batch inference in ML
 */
template <typename T>
class BatchPdhgSolver {
public:
    BatchPdhgSolver() = default;
    explicit BatchPdhgSolver(const BatchPdhgSettings<T>& settings) 
        : settings_(settings) {}
    
    BatchPdhgResult<T> solve(BatchLPProblem<T>& problem);
    
    void set_settings(const BatchPdhgSettings<T>& settings) { 
        settings_ = settings; 
    }

private:
    BatchPdhgSettings<T> settings_;
};

/**
 * @brief Helper to create batched problem from host data
 */
template <typename T>
BatchLPProblem<T> make_batch_lp(
    CsrMatrix<T>& A,
    const T* c_batch_host,  // (batch_size x n) row-major
    const T* b_batch_host,  // (batch_size x m) row-major
    DeviceVector<T>& lb,
    DeviceVector<T>& ub,
    Index batch_size,
    Index n,
    Index m
);

}  // namespace cuprox

#endif  // CUPROX_SOLVERS_BATCH_PDHG_CUH

