#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "cuprox/core/sparse_matrix.cuh"

using namespace cuprox;

class SparseMatrixTest : public ::testing::Test {
protected:
    static constexpr double kTolerance = 1e-10;
    
    // Create a simple 3x3 sparse matrix:
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    void SetUp() override {
        // CSR format
        row_offsets_ = {0, 2, 3, 5};
        col_indices_ = {0, 2, 1, 0, 2};
        values_ = {1.0, 2.0, 3.0, 4.0, 5.0};
    }
    
    std::vector<Index> row_offsets_;
    std::vector<Index> col_indices_;
    std::vector<double> values_;
};

TEST_F(SparseMatrixTest, CreateFromCSR) {
    auto mat = CsrMatrix<double>::from_csr(
        3, 3, 5,
        row_offsets_.data(),
        col_indices_.data(),
        values_.data()
    );
    
    EXPECT_EQ(mat.num_rows(), 3);
    EXPECT_EQ(mat.num_cols(), 3);
    EXPECT_EQ(mat.nnz(), 5);
}

TEST_F(SparseMatrixTest, SpMV) {
    auto mat = CsrMatrix<double>::from_csr(
        3, 3, 5,
        row_offsets_.data(),
        col_indices_.data(),
        values_.data()
    );
    
    // x = [1, 2, 3]
    std::vector<double> x_data = {1.0, 2.0, 3.0};
    DeviceVector<double> x;
    x.copy_from_host(x_data);
    
    // y = A * x
    DeviceVector<double> y(3, 0.0);
    mat.spmv(1.0, x, 0.0, y);
    
    auto result = y.to_host();
    
    // [1 0 2] [1]   [1*1 + 0*2 + 2*3]   [7]
    // [0 3 0] [2] = [0*1 + 3*2 + 0*3] = [6]
    // [4 0 5] [3]   [4*1 + 0*2 + 5*3]   [19]
    EXPECT_NEAR(result[0], 7.0, kTolerance);
    EXPECT_NEAR(result[1], 6.0, kTolerance);
    EXPECT_NEAR(result[2], 19.0, kTolerance);
}

TEST_F(SparseMatrixTest, SpMVWithAlphaBeta) {
    auto mat = CsrMatrix<double>::from_csr(
        3, 3, 5,
        row_offsets_.data(),
        col_indices_.data(),
        values_.data()
    );
    
    std::vector<double> x_data = {1.0, 2.0, 3.0};
    std::vector<double> y_data = {1.0, 1.0, 1.0};
    
    DeviceVector<double> x, y;
    x.copy_from_host(x_data);
    y.copy_from_host(y_data);
    
    // y = 2.0 * A * x + 3.0 * y
    mat.spmv(2.0, x, 3.0, y);
    
    auto result = y.to_host();
    
    // y = 2 * [7, 6, 19] + 3 * [1, 1, 1] = [17, 15, 41]
    EXPECT_NEAR(result[0], 17.0, kTolerance);
    EXPECT_NEAR(result[1], 15.0, kTolerance);
    EXPECT_NEAR(result[2], 41.0, kTolerance);
}

TEST_F(SparseMatrixTest, SpMVTranspose) {
    auto mat = CsrMatrix<double>::from_csr(
        3, 3, 5,
        row_offsets_.data(),
        col_indices_.data(),
        values_.data()
    );
    
    // x = [1, 2, 3]
    std::vector<double> x_data = {1.0, 2.0, 3.0};
    DeviceVector<double> x;
    x.copy_from_host(x_data);
    
    // y = A^T * x
    DeviceVector<double> y(3, 0.0);
    mat.spmv_transpose(1.0, x, 0.0, y);
    
    auto result = y.to_host();
    
    // A^T = [1 0 4]
    //       [0 3 0]
    //       [2 0 5]
    // A^T * [1,2,3] = [1*1+0*2+4*3, 0*1+3*2+0*3, 2*1+0*2+5*3]
    //               = [13, 6, 17]
    EXPECT_NEAR(result[0], 13.0, kTolerance);
    EXPECT_NEAR(result[1], 6.0, kTolerance);
    EXPECT_NEAR(result[2], 17.0, kTolerance);
}

TEST_F(SparseMatrixTest, MoveConstruct) {
    auto mat1 = CsrMatrix<double>::from_csr(
        3, 3, 5,
        row_offsets_.data(),
        col_indices_.data(),
        values_.data()
    );
    
    CsrMatrix<double> mat2(std::move(mat1));
    
    EXPECT_EQ(mat1.num_rows(), 0);
    EXPECT_EQ(mat2.num_rows(), 3);
    EXPECT_EQ(mat2.nnz(), 5);
}

TEST_F(SparseMatrixTest, LargeMatrix) {
    // Create a large diagonal matrix for testing
    const int n = 10000;
    std::vector<Index> row_off(n + 1);
    std::vector<Index> col_idx(n);
    std::vector<double> vals(n);
    
    for (int i = 0; i <= n; ++i) row_off[i] = i;
    for (int i = 0; i < n; ++i) {
        col_idx[i] = i;
        vals[i] = static_cast<double>(i + 1);
    }
    
    auto mat = CsrMatrix<double>::from_csr(n, n, n,
        row_off.data(), col_idx.data(), vals.data());
    
    // x = all ones
    DeviceVector<double> x(n, 1.0);
    DeviceVector<double> y(n, 0.0);
    
    mat.spmv(1.0, x, 0.0, y);
    
    auto result = y.to_host();
    
    // For diagonal matrix, y[i] = D[i,i] * 1 = i+1
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(result[i], static_cast<double>(i + 1), kTolerance);
    }
}

// Float tests
TEST_F(SparseMatrixTest, FloatSpMV) {
    std::vector<float> float_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    auto mat = CsrMatrix<float>::from_csr(
        3, 3, 5,
        row_offsets_.data(),
        col_indices_.data(),
        float_values.data()
    );
    
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    DeviceVector<float> x;
    x.copy_from_host(x_data);
    
    DeviceVector<float> y(3, 0.0f);
    mat.spmv(1.0f, x, 0.0f, y);
    
    auto result = y.to_host();
    EXPECT_NEAR(result[0], 7.0f, 1e-5f);
    EXPECT_NEAR(result[1], 6.0f, 1e-5f);
    EXPECT_NEAR(result[2], 19.0f, 1e-5f);
}

