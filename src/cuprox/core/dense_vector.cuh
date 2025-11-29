#ifndef CUPROX_CORE_DENSE_VECTOR_CUH
#define CUPROX_CORE_DENSE_VECTOR_CUH

#include <cublas_v2.h>
#include <vector>
#include <cmath>

#include "types.hpp"
#include "error.hpp"
#include "memory.cuh"
#include "cuda_context.cuh"

namespace cuprox {

template <typename T>
class DeviceVector {
public:
    DeviceVector() = default;

    explicit DeviceVector(Index n) : size_(n), data_(n) {
        if (n > 0) {
            device_memset(data_.get(), 0, n);
        }
    }

    DeviceVector(Index n, T fill_value) : size_(n), data_(n) {
        fill(fill_value);
    }

    // Move constructor
    DeviceVector(DeviceVector&& other) noexcept
        : size_(other.size_), data_(std::move(other.data_)) {
        other.size_ = 0;
    }

    // Move assignment
    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            size_ = other.size_;
            data_ = std::move(other.data_);
            other.size_ = 0;
        }
        return *this;
    }

    // Disable copy
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    Index size() const { return size_; }
    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }

    void resize(Index n) {
        if (n != size_) {
            data_.reset(n);
            size_ = n;
        }
    }

    void fill(T value);

    void copy_from_host(const T* host_data, Index count) {
        CUPROX_ASSERT(count <= size_, "copy_from_host: count exceeds size");
        copy_host_to_device(data_.get(), host_data, count);
    }

    void copy_from_host(const std::vector<T>& host_data) {
        resize(static_cast<Index>(host_data.size()));
        copy_host_to_device(data_.get(), host_data.data(), host_data.size());
    }

    void copy_to_host(T* host_data, Index count) const {
        CUPROX_ASSERT(count <= size_, "copy_to_host: count exceeds size");
        copy_device_to_host(host_data, data_.get(), count);
    }

    std::vector<T> to_host() const {
        std::vector<T> result(size_);
        copy_to_host(result.data(), size_);
        return result;
    }

    void copy_from(const DeviceVector& other) {
        resize(other.size());
        copy_device_to_device(data_.get(), other.data(), size_);
    }

    // y = alpha * x + y
    void axpy(T alpha, const DeviceVector& x);

    // this *= alpha
    void scale(T alpha);

    // dot product
    T dot(const DeviceVector& other) const;

    // L2 norm
    T norm2() const;

    // sum of elements
    T sum() const;

private:
    Index size_ = 0;
    DevicePtr<T> data_;
};

// BLAS specializations for double
template <>
inline void DeviceVector<double>::axpy(double alpha, const DeviceVector<double>& x) {
    CUPROX_ASSERT(size_ == x.size_, "axpy: size mismatch");
    CUPROX_CUBLAS_CHECK(cublasDaxpy(cublas_handle(), size_, &alpha, 
                                     x.data(), 1, data_.get(), 1));
}

template <>
inline void DeviceVector<double>::scale(double alpha) {
    CUPROX_CUBLAS_CHECK(cublasDscal(cublas_handle(), size_, &alpha, data_.get(), 1));
}

template <>
inline double DeviceVector<double>::dot(const DeviceVector<double>& other) const {
    CUPROX_ASSERT(size_ == other.size_, "dot: size mismatch");
    double result = 0.0;
    CUPROX_CUBLAS_CHECK(cublasDdot(cublas_handle(), size_, 
                                    data_.get(), 1, other.data(), 1, &result));
    return result;
}

template <>
inline double DeviceVector<double>::norm2() const {
    double result = 0.0;
    CUPROX_CUBLAS_CHECK(cublasDnrm2(cublas_handle(), size_, data_.get(), 1, &result));
    return result;
}

// BLAS specializations for float
template <>
inline void DeviceVector<float>::axpy(float alpha, const DeviceVector<float>& x) {
    CUPROX_ASSERT(size_ == x.size_, "axpy: size mismatch");
    CUPROX_CUBLAS_CHECK(cublasSaxpy(cublas_handle(), size_, &alpha, 
                                     x.data(), 1, data_.get(), 1));
}

template <>
inline void DeviceVector<float>::scale(float alpha) {
    CUPROX_CUBLAS_CHECK(cublasSscal(cublas_handle(), size_, &alpha, data_.get(), 1));
}

template <>
inline float DeviceVector<float>::dot(const DeviceVector<float>& other) const {
    CUPROX_ASSERT(size_ == other.size_, "dot: size mismatch");
    float result = 0.0f;
    CUPROX_CUBLAS_CHECK(cublasSdot(cublas_handle(), size_, 
                                    data_.get(), 1, other.data(), 1, &result));
    return result;
}

template <>
inline float DeviceVector<float>::norm2() const {
    float result = 0.0f;
    CUPROX_CUBLAS_CHECK(cublasSnrm2(cublas_handle(), size_, data_.get(), 1, &result));
    return result;
}

}  // namespace cuprox

#endif  // CUPROX_CORE_DENSE_VECTOR_CUH

