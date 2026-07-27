// Shape invariants must hold in release builds.
//
// The m > n crash that made every QP with more constraints than variables fail
// got as far as cuSPARSE precisely because the assertion naming its cause was
// compiled out under NDEBUG. These tests exist to keep that class of check
// active.

#include <gtest/gtest.h>
#include <vector>
#include "cuprox/core/sparse_matrix.cuh"
#include "cuprox/core/dense_vector.cuh"
using namespace cuprox;

// The m > n crash reached cuSPARSE because CUPROX_ASSERT is compiled out under
// NDEBUG. These must fail with a named DimensionError in a release build.
TEST(ShapeGuards, TransposeOutputMustBeLengthN) {
    std::vector<Index> ro{0,2,4,6}, ci{0,1,0,1,0,1};
    std::vector<double> va{1,2,3,1,1,1};
    auto A = CsrMatrix<double>::from_csr(3, 2, 6, ro.data(), ci.data(), va.data());
    DeviceVector<double> x(3), wrong(3), right(2);
    EXPECT_THROW(A.spmv_transpose(1.0, x, 0.0, wrong), DimensionError);
    EXPECT_NO_THROW(A.spmv_transpose(1.0, x, 0.0, right));
}

TEST(ShapeGuards, SpmvOperandsMustMatch) {
    std::vector<Index> ro{0,2,4,6}, ci{0,1,0,1,0,1};
    std::vector<double> va{1,2,3,1,1,1};
    auto A = CsrMatrix<double>::from_csr(3, 2, 6, ro.data(), ci.data(), va.data());
    DeviceVector<double> bad_x(3), y(3), good_x(2);
    EXPECT_THROW(A.spmv(1.0, bad_x, 0.0, y), DimensionError);
    EXPECT_NO_THROW(A.spmv(1.0, good_x, 0.0, y));
}
