#include "dense_vector.cuh"

namespace cuprox {

namespace kernels {

template <typename T>
__global__ void fill_kernel(T* data, T value, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}

// Simple reduction without dynamic shared memory for sum
template <typename T>
__global__ void reduce_sum_kernel(const T* data, T* partial_sums, Index n) {
    __shared__ T sdata[256];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (i < n) ? data[i] : T(0);
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

}  // namespace kernels

template <typename T>
void DeviceVector<T>::fill(T value) {
    if (size_ == 0) return;
    
    constexpr int block_size = 256;
    int num_blocks = (size_ + block_size - 1) / block_size;
    
    kernels::fill_kernel<<<num_blocks, block_size>>>(data_.get(), value, size_);
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
T DeviceVector<T>::sum() const {
    if (size_ == 0) return T(0);
    
    constexpr int block_size = 256;
    int num_blocks = (size_ + block_size - 1) / block_size;
    
    // Allocate partial sums on device
    DevicePtr<T> d_partial(num_blocks);
    
    kernels::reduce_sum_kernel<<<num_blocks, block_size>>>(
        data_.get(), d_partial.get(), size_);
    CUPROX_CUDA_CHECK_LAST();
    
    // Copy partial sums to host and finish reduction
    std::vector<T> h_partial(num_blocks);
    copy_device_to_host(h_partial.data(), d_partial.get(), num_blocks);
    
    T result = T(0);
    for (int i = 0; i < num_blocks; ++i) {
        result += h_partial[i];
    }
    
    return result;
}

// Explicit instantiations
template class DeviceVector<float>;
template class DeviceVector<double>;

}  // namespace cuprox
