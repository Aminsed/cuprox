#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "cuprox/preprocess/scaling.cuh"
#include "cuprox/solvers/pdhg.cuh"

using namespace cuprox;

class ScalingTest : public ::testing::Test {
protected:
    static constexpr double kTolerance = 1e-4;
};

TEST_F(ScalingTest, OperatorNormIdentity) {
    // ||I||_2 = 1
    std::vector<Index> row_off = {0, 1, 2, 3};
    std::vector<Index> col_idx = {0, 1, 2};
    std::vector<double> vals = {1.0, 1.0, 1.0};
    
    auto A = CsrMatrix<double>::from_csr(3, 3, 3,
        row_off.data(), col_idx.data(), vals.data());
    
    double norm = estimate_operator_norm(A, 30);
    EXPECT_NEAR(norm, 1.0, 0.1);
}

TEST_F(ScalingTest, OperatorNormScaled) {
    // ||2*I||_2 = 2
    std::vector<Index> row_off = {0, 1, 2, 3};
    std::vector<Index> col_idx = {0, 1, 2};
    std::vector<double> vals = {2.0, 2.0, 2.0};
    
    auto A = CsrMatrix<double>::from_csr(3, 3, 3,
        row_off.data(), col_idx.data(), vals.data());
    
    double norm = estimate_operator_norm(A, 30);
    EXPECT_NEAR(norm, 2.0, 0.1);
}

TEST_F(ScalingTest, RuizEquilibration) {
    // Create poorly scaled matrix
    // [1000, 0]
    // [0, 0.001]
    std::vector<Index> row_off = {0, 1, 2};
    std::vector<Index> col_idx = {0, 1};
    std::vector<double> vals = {1000.0, 0.001};
    
    auto A = CsrMatrix<double>::from_csr(2, 2, 2,
        row_off.data(), col_idx.data(), vals.data());
    
    std::vector<double> c_data = {1.0, 1.0};
    std::vector<double> b_data = {1.0, 1.0};
    
    DeviceVector<double> c, b;
    c.copy_from_host(c_data);
    b.copy_from_host(b_data);
    
    // Apply scaling
    auto scaling = ruiz_equilibrate(A, c, b, 10);
    
    // After scaling, row/col norms should be close to 1
    DeviceVector<double> row_norms, col_norms;
    compute_row_inf_norms(A, row_norms);
    compute_col_inf_norms(A, col_norms);
    
    auto rn = row_norms.to_host();
    auto cn = col_norms.to_host();
    
    // Norms should be balanced (close to 1)
    EXPECT_NEAR(rn[0], 1.0, 0.5);
    EXPECT_NEAR(rn[1], 1.0, 0.5);
    EXPECT_NEAR(cn[0], 1.0, 0.5);
    EXPECT_NEAR(cn[1], 1.0, 0.5);
    
    std::cout << "After Ruiz: row_norms = [" << rn[0] << ", " << rn[1] << "]"
              << ", col_norms = [" << cn[0] << ", " << cn[1] << "]" << std::endl;
}

TEST_F(ScalingTest, LPWithScaling) {
    /*
     * Solve LP with scaling enabled
     * minimize -x - y
     * subject to  x + 2y <= 10
     *             3x + y <= 15
     *             x, y >= 0
     * Add slacks and solve
     */
    std::vector<Index> row_off = {0, 3, 6};
    std::vector<Index> col_idx = {0, 1, 2, 0, 1, 3};
    std::vector<double> vals = {1.0, 2.0, 1.0, 3.0, 1.0, 1.0};
    
    LPProblem<double> lp;
    lp.A = CsrMatrix<double>::from_csr(2, 4, 6,
        row_off.data(), col_idx.data(), vals.data());
    
    std::vector<double> c = {-1.0, -1.0, 0.0, 0.0};
    std::vector<double> b = {10.0, 15.0};
    std::vector<double> lb = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> ub = {1e20, 1e20, 1e20, 1e20};
    
    lp.c.copy_from_host(c);
    lp.b.copy_from_host(b);
    lp.lb.copy_from_host(lb);
    lp.ub.copy_from_host(ub);
    lp.l.copy_from_host(b);
    lp.u.copy_from_host(b);
    
    PdhgSettings<double> settings;
    settings.max_iters = 10000;
    settings.eps_abs = 1e-5;
    settings.eps_rel = 1e-5;
    settings.verbose = false;
    settings.scaling = true;  // Enable scaling!
    
    PdhgSolver<double> solver(settings);
    auto result = solver.solve(lp);
    
    EXPECT_EQ(result.status, Status::Optimal);
    
    auto x = result.x.to_host();
    std::cout << "LP with scaling: x = [" << x[0] << ", " << x[1] << "]"
              << ", obj = " << result.primal_obj
              << ", iters = " << result.iterations << std::endl;
    
    EXPECT_NEAR(x[0], 4.0, 0.2);
    EXPECT_NEAR(x[1], 3.0, 0.2);
    EXPECT_NEAR(result.primal_obj, -7.0, 0.2);
}

