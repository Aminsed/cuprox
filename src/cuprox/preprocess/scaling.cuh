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
 * @brief Modified Ruiz equilibration for a QP.
 *
 * Balances the problem
 *     min 1/2 x'Px + q'x   s.t.   l <= Ax <= u,  lb <= x <= ub
 * by finding diagonal scalings E (variables) and D (constraints) and a scalar
 * cost scaling c, then rewriting the data in place:
 *
 *     P <- c E P E,  q <- c E q,  A <- D A E,  l <- D l,  u <- D u,
 *     lb <- E^-1 lb, ub <- E^-1 ub
 *
 * A first-order method's convergence depends on the conditioning of the data it
 * is handed, and nothing inside the iteration can compensate for a badly scaled
 * problem. Portfolio problems make the point: a covariance of order 1e-4 against
 * a return constraint of order 1e-3 left ADMM unable to hit a target return at
 * all, returning +0.20 when asked for -0.0007.
 *
 * Each sweep sets the variable scaling from the largest entry in that variable's
 * column of either P or A, and the constraint scaling from the largest entry in
 * that row of A, so both matrices end up with infinity norms near one. The cost
 * scaling then balances the objective against the constraints.
 *
 * @return factors needed to recover the original solution, see unscale_qp
 */
template <typename T>
ScalingFactors<T> ruiz_equilibrate_qp(
    CsrMatrix<T>& P,
    CsrMatrix<T>& A,
    DeviceVector<T>& q,
    DeviceVector<T>& l,
    DeviceVector<T>& u,
    DeviceVector<T>& lb,
    DeviceVector<T>& ub,
    int max_iters = 10
);

/**
 * @brief Undo ruiz_equilibrate_qp on both the solution and the problem data.
 *
 * The solution maps back as x = E x~ and y = c^-1 D y~. The problem data is
 * restored as well, so a solver that scales internally does not leave the
 * caller's problem altered -- solving the same QPProblem twice must give the
 * same answer.
 */
template <typename T>
void unscale_qp(
    CsrMatrix<T>& P,
    CsrMatrix<T>& A,
    DeviceVector<T>& q,
    DeviceVector<T>& l,
    DeviceVector<T>& u,
    DeviceVector<T>& lb,
    DeviceVector<T>& ub,
    DeviceVector<T>& x,
    DeviceVector<T>& y,
    const ScalingFactors<T>& s
);

/**
 * @brief 2-norm of a vector measured in the caller's (unscaled) units.
 *
 * Equilibration leaves the iterates in scaled space, so a tolerance applied to
 * them would mean something different from what the caller asked for. Dividing
 * by the Ruiz factor for that space puts the quantity back in the original
 * units: primal quantities carry a factor D, dual quantities a factor c*E.
 *
 * @param v       Vector in scaled units
 * @param s       Ruiz factor for v's space (D for m-vectors, E for n-vectors)
 * @param alpha   Extra scalar factor (1 for primal, 1/c for dual)
 * @param scratch Workspace, resized to v.size()
 */
template <typename T>
T unscaled_norm2(const DeviceVector<T>& v, const DeviceVector<T>& s, T alpha,
                 DeviceVector<T>& scratch);

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

