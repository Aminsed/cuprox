#ifndef CUPROX_CORE_MEMORY_CUH
#define CUPROX_CORE_MEMORY_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

#include "error.hpp"

namespace cuprox {

template <typename T>
T* device_malloc(size_t count) {
    T* ptr = nullptr;
    if (count > 0) {
        CUPROX_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    return ptr;
}

template <typename T>
void device_free(T* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

template <typename T>
void device_memset(T* ptr, int value, size_t count) {
    if (ptr && count > 0) {
        CUPROX_CUDA_CHECK(cudaMemset(ptr, value, count * sizeof(T)));
    }
}

template <typename T>
void copy_host_to_device(T* dst, const T* src, size_t count) {
    if (count > 0) {
        CUPROX_CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), 
                                      cudaMemcpyHostToDevice));
    }
}

template <typename T>
void copy_device_to_host(T* dst, const T* src, size_t count) {
    if (count > 0) {
        CUPROX_CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), 
                                      cudaMemcpyDeviceToHost));
    }
}

template <typename T>
void copy_device_to_device(T* dst, const T* src, size_t count) {
    if (count > 0) {
        CUPROX_CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), 
                                      cudaMemcpyDeviceToDevice));
    }
}

template <typename T>
class DevicePtr {
public:
    DevicePtr() = default;
    
    explicit DevicePtr(size_t count) : size_(count) {
        ptr_ = device_malloc<T>(count);
    }

    ~DevicePtr() {
        device_free(ptr_);
    }

    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;

    DevicePtr(DevicePtr&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DevicePtr& operator=(DevicePtr&& other) noexcept {
        if (this != &other) {
            device_free(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

    T* release() {
        T* p = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return p;
    }

    void reset(size_t count = 0) {
        device_free(ptr_);
        ptr_ = (count > 0) ? device_malloc<T>(count) : nullptr;
        size_ = count;
    }

    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace cuprox

#endif  // CUPROX_CORE_MEMORY_CUH

