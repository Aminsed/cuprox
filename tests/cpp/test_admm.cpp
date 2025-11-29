#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

#include "cuprox/solvers/admm.cuh"

using namespace cuprox;

class AdmmTest : public ::testing::Test {
protected:
    static constexpr double kTolerance = 1e-3;
    static constexpr double kInf = 1e20;
};

// Helper to create diagonal matrix as CSR
template <typename T>
CsrMatrix<T> make_diagonal_matrix(const std::vector<T>& diag) {
    Index n = static_cast<Index>(diag.size());
    std::vector<Index> row_offsets(n + 1);
    std::vector<Index> col_indices(n);
    
    for (Index i = 0; i <= n; ++i) row_offsets[i] = i;
    for (Index i = 0; i < n; ++i) col_indices[i] = i;
    
    return CsrMatrix<T>::from_csr(n, n, n,
        row_offsets.data(), col_indices.data(), diag.data());
}

// Helper to create QP problem
template <typename T>
QPProblem<T> make_qp(
    const std::vector<T>& P_diag,
    Index m, Index n, Index nnz,
    const std::vector<Index>& A_row_offsets,
    const std::vector<Index>& A_col_indices,
    const std::vector<T>& A_values,
    const std::vector<T>& q,
    const std::vector<T>& l,
    const std::vector<T>& u,
    const std::vector<T>& var_lb = {},
    const std::vector<T>& var_ub = {}
) {
    QPProblem<T> qp;
    qp.P = make_diagonal_matrix(P_diag);
    qp.A = CsrMatrix<T>::from_csr(m, n, nnz,
        A_row_offsets.data(), A_col_indices.data(), A_values.data());
    
    qp.q.resize(n);
    qp.q.copy_from_host(q.data(), n);
    
    qp.l.resize(m);
    qp.l.copy_from_host(l.data(), m);
    
    qp.u.resize(m);
    qp.u.copy_from_host(u.data(), m);
    
    // Variable bounds (optional)
    if (!var_lb.empty()) {
        qp.lb.resize(n);
        qp.lb.copy_from_host(var_lb.data(), n);
    }
    if (!var_ub.empty()) {
        qp.ub.resize(n);
        qp.ub.copy_from_host(var_ub.data(), n);
    }
    
    return qp;
}

TEST_F(AdmmTest, SimpleUnconstrainedQP) {
    /*
     * minimize (1/2) x'Px + q'x where P = 2I, q = [-2, -4]
     * No constraints (l = -inf, u = inf)
     * 
     * Optimal: x = P^{-1} * (-q) = [1, 2]
     * Objective: (1/2)(1*2 + 2*4) - 2*1 - 4*2 = 5 - 10 = -5
     */
    std::vector<double> P_diag = {2.0, 2.0};
    std::vector<Index> A_row_off = {0, 1, 2};
    std::vector<Index> A_col_idx = {0, 1};
    std::vector<double> A_vals = {1.0, 1.0};  // Identity constraint (for ADMM)
    
    std::vector<double> q = {-2.0, -4.0};
    std::vector<double> l = {-kInf, -kInf};
    std::vector<double> u = {kInf, kInf};
    
    auto qp = make_qp<double>(P_diag, 2, 2, 2, A_row_off, A_col_idx, A_vals, q, l, u);
    
    AdmmSettings<double> settings;
    settings.max_iters = 1000;
    settings.eps_abs = 1e-5;
    settings.eps_rel = 1e-5;
    settings.verbose = false;
    
    AdmmSolver<double> solver(settings);
    auto result = solver.solve(qp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 1.0, kTolerance);
    EXPECT_NEAR(x[1], 2.0, kTolerance);
    EXPECT_NEAR(result.primal_obj, -5.0, 0.1);
    
    std::cout << "SimpleUnconstrainedQP: " << result.iterations << " iterations, "
              << result.solve_time * 1000 << " ms" << std::endl;
}

TEST_F(AdmmTest, BoxConstrainedQP) {
    /*
     * minimize (1/2) x'Px + q'x where P = 2I, q = [-10, -10]
     * subject to 0 <= x <= 2
     * 
     * Unconstrained optimal: x = [5, 5]
     * With box constraints: x = [2, 2]
     * Objective: (1/2)(2*4 + 2*4) - 10*2 - 10*2 = 8 - 40 = -32
     */
    std::vector<double> P_diag = {2.0, 2.0};
    std::vector<Index> A_row_off = {0, 1, 2};  // Identity
    std::vector<Index> A_col_idx = {0, 1};
    std::vector<double> A_vals = {1.0, 1.0};
    
    std::vector<double> q = {-10.0, -10.0};
    std::vector<double> l = {0.0, 0.0};
    std::vector<double> u = {2.0, 2.0};
    
    auto qp = make_qp<double>(P_diag, 2, 2, 2, A_row_off, A_col_idx, A_vals, q, l, u);
    
    AdmmSettings<double> settings;
    settings.max_iters = 1000;
    settings.eps_abs = 1e-5;
    settings.verbose = false;
    
    AdmmSolver<double> solver(settings);
    auto result = solver.solve(qp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 2.0, kTolerance);
    EXPECT_NEAR(x[1], 2.0, kTolerance);
    EXPECT_NEAR(result.primal_obj, -32.0, 0.5);
    
    std::cout << "BoxConstrainedQP: " << result.iterations << " iterations, "
              << result.solve_time * 1000 << " ms" << std::endl;
}

TEST_F(AdmmTest, EqualityConstrainedQP) {
    /*
     * minimize (1/2)(x1^2 + x2^2)
     * subject to x1 + x2 = 1
     * 
     * Lagrangian: L = (1/2)(x1^2 + x2^2) + y(x1 + x2 - 1)
     * KKT: x1 + y = 0, x2 + y = 0, x1 + x2 = 1
     * Solution: x1 = x2 = 0.5
     * Objective: (1/2)(0.25 + 0.25) = 0.25
     */
    std::vector<double> P_diag = {1.0, 1.0};
    std::vector<Index> A_row_off = {0, 2};
    std::vector<Index> A_col_idx = {0, 1};
    std::vector<double> A_vals = {1.0, 1.0};
    
    std::vector<double> q = {0.0, 0.0};
    std::vector<double> l = {1.0};  // Equality: l = u = 1
    std::vector<double> u = {1.0};
    
    auto qp = make_qp<double>(P_diag, 1, 2, 2, A_row_off, A_col_idx, A_vals, q, l, u);
    
    AdmmSettings<double> settings;
    settings.max_iters = 1000;
    settings.eps_abs = 1e-5;
    settings.verbose = false;
    
    AdmmSolver<double> solver(settings);
    auto result = solver.solve(qp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 0.5, kTolerance);
    EXPECT_NEAR(x[1], 0.5, kTolerance);
    EXPECT_NEAR(result.primal_obj, 0.25, 0.01);
    
    std::cout << "EqualityConstrainedQP: " << result.iterations << " iterations, "
              << result.solve_time * 1000 << " ms" << std::endl;
}

TEST_F(AdmmTest, InequalityConstrainedQP) {
    /*
     * minimize (1/2)(x1^2 + x2^2) - x1 - x2
     * subject to x1 + x2 <= 1
     * 
     * Unconstrained optimal: x = [1, 1]
     * But x1 + x2 = 2 > 1, so constraint is active
     * Lagrangian analysis: optimal at x1 = x2 = 0.5
     * Objective: (1/2)(0.5) - 1 = -0.75
     */
    std::vector<double> P_diag = {1.0, 1.0};
    std::vector<Index> A_row_off = {0, 2};
    std::vector<Index> A_col_idx = {0, 1};
    std::vector<double> A_vals = {1.0, 1.0};
    
    std::vector<double> q = {-1.0, -1.0};
    std::vector<double> l = {-kInf};
    std::vector<double> u = {1.0};
    
    auto qp = make_qp<double>(P_diag, 1, 2, 2, A_row_off, A_col_idx, A_vals, q, l, u);
    
    AdmmSettings<double> settings;
    settings.max_iters = 2000;
    settings.eps_abs = 1e-5;
    settings.verbose = false;
    
    AdmmSolver<double> solver(settings);
    auto result = solver.solve(qp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 0.5, kTolerance);
    EXPECT_NEAR(x[1], 0.5, kTolerance);
    EXPECT_NEAR(x[0] + x[1], 1.0, 0.01);  // Constraint should be active
    
    std::cout << "InequalityConstrainedQP: " << result.iterations << " iterations, "
              << result.solve_time * 1000 << " ms" << std::endl;
}

TEST_F(AdmmTest, LargerQP) {
    /*
     * Larger QP: minimize (1/2) x'Px + q'x
     *            subject to l <= Ax <= u
     */
    const Index n = 100;
    const Index m = 50;
    
    // P = diagonal
    std::vector<double> P_diag(n);
    for (Index i = 0; i < n; ++i) {
        P_diag[i] = 1.0 + 0.1 * (i % 5);
    }
    
    // A = sparse random-ish
    std::vector<Index> A_row_off(m + 1);
    std::vector<Index> A_col_idx;
    std::vector<double> A_vals;
    
    Index nnz = 0;
    for (Index i = 0; i < m; ++i) {
        A_row_off[i] = nnz;
        for (Index j = 0; j < 3; ++j) {
            Index col = (i * 2 + j * 7) % n;
            A_col_idx.push_back(col);
            A_vals.push_back(1.0 + 0.1 * ((i + j) % 3));
            ++nnz;
        }
    }
    A_row_off[m] = nnz;
    
    std::vector<double> q(n);
    std::vector<double> l(m), u(m);
    
    for (Index i = 0; i < n; ++i) {
        q[i] = -1.0 + 0.1 * (i % 10);
    }
    for (Index i = 0; i < m; ++i) {
        l[i] = -10.0;
        u[i] = 10.0;
    }
    
    auto qp = make_qp<double>(P_diag, m, n, nnz, A_row_off, A_col_idx, A_vals, q, l, u);
    
    AdmmSettings<double> settings;
    settings.max_iters = 2000;
    settings.eps_abs = 1e-4;
    settings.eps_rel = 1e-4;
    settings.verbose = false;
    
    AdmmSolver<double> solver(settings);
    auto result = solver.solve(qp);
    
    EXPECT_TRUE(result.status == Status::Optimal || result.iterations < settings.max_iters);
    
    std::cout << "LargerQP (" << n << "x" << m << "): " 
              << result.iterations << " iterations, "
              << result.solve_time * 1000 << " ms, "
              << "status=" << status_to_string(result.status) << std::endl;
}

TEST_F(AdmmTest, FloatPrecision) {
    /*
     * Same simple QP but with float
     */
    std::vector<float> P_diag = {2.0f, 2.0f};
    std::vector<Index> A_row_off = {0, 1, 2};
    std::vector<Index> A_col_idx = {0, 1};
    std::vector<float> A_vals = {1.0f, 1.0f};
    
    std::vector<float> q = {-2.0f, -4.0f};
    std::vector<float> l = {-1e20f, -1e20f};
    std::vector<float> u = {1e20f, 1e20f};
    
    QPProblem<float> qp;
    qp.P = make_diagonal_matrix(P_diag);
    qp.A = CsrMatrix<float>::from_csr(2, 2, 2,
        A_row_off.data(), A_col_idx.data(), A_vals.data());
    
    qp.q.resize(2);
    qp.q.copy_from_host(q.data(), 2);
    qp.l.resize(2);
    qp.l.copy_from_host(l.data(), 2);
    qp.u.resize(2);
    qp.u.copy_from_host(u.data(), 2);
    
    AdmmSettings<float> settings;
    settings.max_iters = 1000;
    settings.eps_abs = 1e-4f;
    settings.verbose = false;
    
    AdmmSolver<float> solver(settings);
    auto result = solver.solve(qp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 1.0f, 0.01f);
    EXPECT_NEAR(x[1], 2.0f, 0.01f);
}

