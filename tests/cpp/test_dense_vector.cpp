#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "cuprox/core/dense_vector.cuh"

using namespace cuprox;

class DenseVectorTest : public ::testing::Test {
protected:
    static constexpr double kTolerance = 1e-10;
};

TEST_F(DenseVectorTest, DefaultConstruct) {
    DeviceVector<double> v;
    EXPECT_EQ(v.size(), 0);
    EXPECT_EQ(v.data(), nullptr);
}

TEST_F(DenseVectorTest, ConstructWithSize) {
    DeviceVector<double> v(100);
    EXPECT_EQ(v.size(), 100);
    EXPECT_NE(v.data(), nullptr);
}

TEST_F(DenseVectorTest, ConstructWithFillValue) {
    DeviceVector<double> v(50, 3.14);
    EXPECT_EQ(v.size(), 50);
    
    auto host = v.to_host();
    for (int i = 0; i < 50; ++i) {
        EXPECT_NEAR(host[i], 3.14, kTolerance);
    }
}

TEST_F(DenseVectorTest, CopyFromHost) {
    std::vector<double> host_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    DeviceVector<double> v;
    v.copy_from_host(host_data);
    
    EXPECT_EQ(v.size(), 5);
    auto result = v.to_host();
    
    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_NEAR(result[i], host_data[i], kTolerance);
    }
}

TEST_F(DenseVectorTest, Fill) {
    DeviceVector<double> v(100);
    v.fill(7.5);
    
    auto host = v.to_host();
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(host[i], 7.5, kTolerance);
    }
}

TEST_F(DenseVectorTest, Axpy) {
    std::vector<double> x_data = {1.0, 2.0, 3.0};
    std::vector<double> y_data = {4.0, 5.0, 6.0};
    
    DeviceVector<double> x, y;
    x.copy_from_host(x_data);
    y.copy_from_host(y_data);
    
    // y = 2.0 * x + y
    y.axpy(2.0, x);
    
    auto result = y.to_host();
    EXPECT_NEAR(result[0], 6.0, kTolerance);   // 2*1 + 4
    EXPECT_NEAR(result[1], 9.0, kTolerance);   // 2*2 + 5
    EXPECT_NEAR(result[2], 12.0, kTolerance);  // 2*3 + 6
}

TEST_F(DenseVectorTest, Scale) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    DeviceVector<double> v;
    v.copy_from_host(data);
    
    v.scale(3.0);
    
    auto result = v.to_host();
    EXPECT_NEAR(result[0], 3.0, kTolerance);
    EXPECT_NEAR(result[1], 6.0, kTolerance);
    EXPECT_NEAR(result[2], 9.0, kTolerance);
}

TEST_F(DenseVectorTest, Dot) {
    std::vector<double> x_data = {1.0, 2.0, 3.0};
    std::vector<double> y_data = {4.0, 5.0, 6.0};
    
    DeviceVector<double> x, y;
    x.copy_from_host(x_data);
    y.copy_from_host(y_data);
    
    double result = x.dot(y);
    
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_NEAR(result, 32.0, kTolerance);
}

TEST_F(DenseVectorTest, Norm2) {
    std::vector<double> data = {3.0, 4.0};
    DeviceVector<double> v;
    v.copy_from_host(data);
    
    double norm = v.norm2();
    EXPECT_NEAR(norm, 5.0, kTolerance);  // sqrt(9 + 16) = 5
}

TEST_F(DenseVectorTest, LargeVector) {
    const int n = 1000000;
    DeviceVector<double> v(n, 1.0);
    
    double norm = v.norm2();
    EXPECT_NEAR(norm, std::sqrt(static_cast<double>(n)), 1e-5);
}

TEST_F(DenseVectorTest, MoveConstruct) {
    DeviceVector<double> v1(100, 2.5);
    DeviceVector<double> v2(std::move(v1));
    
    EXPECT_EQ(v1.size(), 0);
    EXPECT_EQ(v2.size(), 100);
    
    auto host = v2.to_host();
    EXPECT_NEAR(host[0], 2.5, kTolerance);
}

TEST_F(DenseVectorTest, CopyFrom) {
    DeviceVector<double> v1(100, 5.0);
    DeviceVector<double> v2;
    
    v2.copy_from(v1);
    
    EXPECT_EQ(v2.size(), 100);
    auto host = v2.to_host();
    EXPECT_NEAR(host[50], 5.0, kTolerance);
}

// Float precision tests
TEST_F(DenseVectorTest, FloatAxpy) {
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> y_data = {4.0f, 5.0f, 6.0f};
    
    DeviceVector<float> x, y;
    x.copy_from_host(x_data);
    y.copy_from_host(y_data);
    
    y.axpy(2.0f, x);
    
    auto result = y.to_host();
    EXPECT_NEAR(result[0], 6.0f, 1e-5f);
    EXPECT_NEAR(result[1], 9.0f, 1e-5f);
    EXPECT_NEAR(result[2], 12.0f, 1e-5f);
}

