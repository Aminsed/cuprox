#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

#include "cuprox/solvers/pdhg.cuh"

using namespace cuprox;

class PdhgTest : public ::testing::Test {
protected:
    static constexpr double kTolerance = 1e-4;
    static constexpr double kInf = std::numeric_limits<double>::infinity();
};

// Helper to create LP problem from host data
template <typename T>
LPProblem<T> make_lp(
    Index m, Index n, Index nnz,
    const std::vector<Index>& row_offsets,
    const std::vector<Index>& col_indices,
    const std::vector<T>& values,
    const std::vector<T>& c,
    const std::vector<T>& b,
    const std::vector<T>& lb,
    const std::vector<T>& ub
) {
    LPProblem<T> lp;
    lp.A = CsrMatrix<T>::from_csr(m, n, nnz,
        row_offsets.data(), col_indices.data(), values.data());
    lp.c.copy_from_host(c);
    lp.b.copy_from_host(b);
    lp.lb.copy_from_host(lb);
    lp.ub.copy_from_host(ub);
    
    // Default: equality constraints
    std::vector<T> l_vec(m, T(0));
    std::vector<T> u_vec(m, T(0));
    for (size_t i = 0; i < m; ++i) {
        l_vec[i] = b[i];
        u_vec[i] = b[i];
    }
    lp.l.copy_from_host(l_vec);
    lp.u.copy_from_host(u_vec);
    
    return lp;
}

TEST_F(PdhgTest, SimpleTwoVarLP) {
    /*
     * minimize   -x - y
     * subject to  x + 2y <= 10
     *             3x + y <= 15
     *             x, y >= 0
     * 
     * Optimal: x=4, y=3, obj=-7
     * 
     * Standard form with slacks:
     * minimize   -x - y + 0*s1 + 0*s2
     * subject to  x + 2y + s1 = 10
     *             3x + y + s2 = 15
     *             x, y, s1, s2 >= 0
     */
    
    // A = [1 2 1 0]
    //     [3 1 0 1]
    std::vector<Index> row_off = {0, 3, 6};
    std::vector<Index> col_idx = {0, 1, 2, 0, 1, 3};
    std::vector<double> vals = {1.0, 2.0, 1.0, 3.0, 1.0, 1.0};
    
    std::vector<double> c = {-1.0, -1.0, 0.0, 0.0};
    std::vector<double> b = {10.0, 15.0};
    std::vector<double> lb = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> ub = {kInf, kInf, kInf, kInf};
    
    auto lp = make_lp<double>(2, 4, 6, row_off, col_idx, vals, c, b, lb, ub);
    
    PdhgSettings<double> settings;
    settings.max_iters = 10000;
    settings.eps_abs = 1e-5;
    settings.eps_rel = 1e-5;
    settings.verbose = false;
    settings.scaling = false;  // Disable scaling for now
    
    PdhgSolver<double> solver(settings);
    auto result = solver.solve(lp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 4.0, 0.1);  // x
    EXPECT_NEAR(x[1], 3.0, 0.1);  // y
    
    // Check objective
    EXPECT_NEAR(result.primal_obj, -7.0, 0.1);
    
    std::cout << "SimpleTwoVarLP: " << result.iterations << " iterations, "
              << result.solve_time * 1000 << " ms" << std::endl;
}

TEST_F(PdhgTest, FeasibilityLP) {
    /*
     * minimize   0
     * subject to x + y = 5
     *            x, y >= 0
     * 
     * Any solution with x + y = 5 and x, y >= 0 is optimal
     */
    std::vector<Index> row_off = {0, 2};
    std::vector<Index> col_idx = {0, 1};
    std::vector<double> vals = {1.0, 1.0};
    
    std::vector<double> c = {0.0, 0.0};
    std::vector<double> b = {5.0};
    std::vector<double> lb = {0.0, 0.0};
    std::vector<double> ub = {kInf, kInf};
    
    auto lp = make_lp<double>(1, 2, 2, row_off, col_idx, vals, c, b, lb, ub);
    
    PdhgSettings<double> settings;
    settings.max_iters = 10000;
    settings.eps_abs = 1e-5;
    settings.verbose = false;
    settings.scaling = false;
    
    PdhgSolver<double> solver(settings);
    auto result = solver.solve(lp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0] + x[1], 5.0, 0.1);
    EXPECT_GE(x[0], -0.01);
    EXPECT_GE(x[1], -0.01);
}

TEST_F(PdhgTest, BoundedVariables) {
    /*
     * minimize   x + 2y
     * subject to x + y = 3
     *            0 <= x <= 2
     *            0 <= y <= 4
     * 
     * Optimal: x=2, y=1, obj=4
     */
    std::vector<Index> row_off = {0, 2};
    std::vector<Index> col_idx = {0, 1};
    std::vector<double> vals = {1.0, 1.0};
    
    std::vector<double> c = {1.0, 2.0};
    std::vector<double> b = {3.0};
    std::vector<double> lb = {0.0, 0.0};
    std::vector<double> ub = {2.0, 4.0};
    
    auto lp = make_lp<double>(1, 2, 2, row_off, col_idx, vals, c, b, lb, ub);
    
    PdhgSettings<double> settings;
    settings.max_iters = 10000;
    settings.eps_abs = 1e-5;
    settings.verbose = false;
    settings.scaling = false;
    
    PdhgSolver<double> solver(settings);
    auto result = solver.solve(lp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 2.0, 0.1);
    EXPECT_NEAR(x[1], 1.0, 0.1);
    EXPECT_NEAR(result.primal_obj, 4.0, 0.2);
}

TEST_F(PdhgTest, LargerLP) {
    /*
     * Random LP with 100 variables and 50 constraints
     */
    const Index n = 100;
    const Index m = 50;
    
    // Create a sparse random-ish matrix (diagonal + some off-diagonal)
    std::vector<Index> row_off(m + 1);
    std::vector<Index> col_idx;
    std::vector<double> vals;
    
    Index nnz = 0;
    for (Index i = 0; i < m; ++i) {
        row_off[i] = nnz;
        
        // Add 3-5 entries per row
        for (Index j = 0; j < 4; ++j) {
            Index col = (i * 2 + j * 13) % n;
            col_idx.push_back(col);
            vals.push_back(1.0 + (i + j) % 3);
            ++nnz;
        }
    }
    row_off[m] = nnz;
    
    std::vector<double> c(n);
    std::vector<double> b(m);
    std::vector<double> lb(n, 0.0);
    std::vector<double> ub(n, 100.0);
    
    for (Index i = 0; i < n; ++i) {
        c[i] = (i % 5 == 0) ? -1.0 : 1.0;
    }
    for (Index i = 0; i < m; ++i) {
        b[i] = 10.0 + i % 10;
    }
    
    auto lp = make_lp<double>(m, n, nnz, row_off, col_idx, vals, c, b, lb, ub);
    
    PdhgSettings<double> settings;
    settings.max_iters = 10000;
    settings.eps_abs = 1e-4;
    settings.eps_rel = 1e-4;
    settings.verbose = false;
    
    PdhgSolver<double> solver(settings);
    auto result = solver.solve(lp);
    
    // Should converge (may be optimal or reach some tolerance)
    EXPECT_TRUE(result.status == Status::Optimal || 
                result.iterations < settings.max_iters);
    
    std::cout << "LargerLP (" << n << "x" << m << "): " 
              << result.iterations << " iterations, "
              << result.solve_time * 1000 << " ms" << std::endl;
}

TEST_F(PdhgTest, FloatPrecision) {
    /*
     * Same simple LP but with float
     */
    std::vector<Index> row_off = {0, 3, 6};
    std::vector<Index> col_idx = {0, 1, 2, 0, 1, 3};
    std::vector<float> vals = {1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 1.0f};
    
    std::vector<float> c = {-1.0f, -1.0f, 0.0f, 0.0f};
    std::vector<float> b = {10.0f, 15.0f};
    std::vector<float> lb = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> ub = {1e30f, 1e30f, 1e30f, 1e30f};
    
    auto lp = make_lp<float>(2, 4, 6, row_off, col_idx, vals, c, b, lb, ub);
    
    PdhgSettings<float> settings;
    settings.max_iters = 10000;
    settings.eps_abs = 1e-4f;
    settings.eps_rel = 1e-4f;
    settings.verbose = false;
    settings.scaling = false;
    
    PdhgSolver<float> solver(settings);
    auto result = solver.solve(lp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    EXPECT_NEAR(x[0], 4.0f, 0.3f);
    EXPECT_NEAR(x[1], 3.0f, 0.3f);
}

TEST_F(PdhgTest, OperatorNormEstimate) {
    // Test that operator norm estimation works
    // For identity matrix, ||I||_2 = 1
    
    std::vector<Index> row_off = {0, 1, 2, 3};
    std::vector<Index> col_idx = {0, 1, 2};
    std::vector<double> vals = {2.0, 2.0, 2.0};  // 2*I
    
    auto A = CsrMatrix<double>::from_csr(3, 3, 3,
        row_off.data(), col_idx.data(), vals.data());
    
    double norm = estimate_operator_norm(A, 30);
    
    // ||2*I||_2 = 2
    EXPECT_NEAR(norm, 2.0, 0.1);
}

