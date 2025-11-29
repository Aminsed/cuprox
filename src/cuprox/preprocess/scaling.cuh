#ifndef CUPROX_PREPROCESS_SCALING_CUH
#define CUPROX_PREPROCESS_SCALING_CUH

#include "../core/types.hpp"
#include "../core/dense_vector.cuh"
#include "../core/sparse_matrix.cuh"

namespace cuprox {

/**
 * @brief Scaling factors computed by Ruiz equilibration
 */
template <typename T>
struct ScalingFactors {
    DeviceVector<T> D;      // Row scaling (m x 1)
    DeviceVector<T> E;      // Column scaling (n x 1)
    T c_scale;              // Objective scaling
    T b_scale;              // RHS scaling
    
    ScalingFactors() : c_scale(T(1)), b_scale(T(1)) {}
    
    ScalingFactors(Index m, Index n) 
        : D(m, T(1)), E(n, T(1)), c_scale(T(1)), b_scale(T(1)) {}
};

/**
 * @brief Compute infinity norm of each row of CSR matrix
 */
template <typename T>
void compute_row_inf_norms(const CsrMatrix<T>& A, DeviceVector<T>& row_norms);

/**
 * @brief Compute infinity norm of each column of CSR matrix
 */
template <typename T>
void compute_col_inf_norms(const CsrMatrix<T>& A, DeviceVector<T>& col_norms);

/**
 * @brief Apply row scaling to CSR matrix: A = diag(D) * A
 */
template <typename T>
void scale_rows(CsrMatrix<T>& A, const DeviceVector<T>& D);

/**
 * @brief Apply column scaling to CSR matrix: A = A * diag(E)
 */
template <typename T>
void scale_cols(CsrMatrix<T>& A, const DeviceVector<T>& E);

/**
 * @brief Ruiz equilibration
 * 
 * Scales matrix A so that row and column infinity norms are ~1.
 * Returns scaling factors D, E such that: D * A * E has balanced norms.
 * 
 * @param A Constraint matrix (modified in-place)
 * @param c Objective vector (modified in-place)
 * @param b RHS vector (modified in-place)
 * @param max_iters Maximum equilibration iterations
 * @return ScalingFactors containing D, E, and scalar scalings
 */
template <typename T>
ScalingFactors<T> ruiz_equilibrate(
    CsrMatrix<T>& A,
    DeviceVector<T>& c,
    DeviceVector<T>& b,
    int max_iters = 10
);

/**
 * @brief Unscale solution after solving
 * 
 * Given scaled solution (x_scaled, y_scaled), recover original:
 * x = E * x_scaled
 * y = D * y_scaled * c_scale
 */
template <typename T>
void unscale_solution(
    DeviceVector<T>& x,
    DeviceVector<T>& y,
    const ScalingFactors<T>& scaling
);

/**
 * @brief Estimate spectral norm ||A||_2 using power iteration
 * 
 * Used to compute safe step sizes for PDHG: tau * sigma < 1/||A||_2^2
 */
template <typename T>
T estimate_operator_norm(const CsrMatrix<T>& A, int max_iters = 20);

}  // namespace cuprox

#endif  // CUPROX_PREPROCESS_SCALING_CUH

