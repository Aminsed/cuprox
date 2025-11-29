#include "projections.cuh"
#include "../core/memory.cuh"
#include <cfloat>

namespace cuprox {

namespace kernels {

template <typename T>
__global__ void project_box_kernel(T* x, const T* lb, const T* ub, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T val = x[i];
        val = (val < lb[i]) ? lb[i] : val;
        val = (val > ub[i]) ? ub[i] : val;
        x[i] = val;
    }
}

template <typename T>
__global__ void project_nonneg_kernel(T* x, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = (x[i] < T(0)) ? T(0) : x[i];
    }
}

template <typename T>
__global__ void gradient_step_project_kernel(T* y, const T* x, T alpha, 
                                              const T* grad, const T* lb, 
                                              const T* ub, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T val = x[i] - alpha * grad[i];
        val = (val < lb[i]) ? lb[i] : val;
        val = (val > ub[i]) ? ub[i] : val;
        y[i] = val;
    }
}

template <typename T>
__global__ void reduce_max_abs_kernel(const T* x, T* partial_max, Index n) {
    __shared__ T sdata[256];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + tid;
    
    T val = (i < n) ? fabs(x[i]) : T(0);
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

template <typename T>
__global__ void reduce_box_violation_kernel(const T* x, const T* lb, const T* ub,
                                             T* partial_max, Index n) {
    __shared__ T sdata[256];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + tid;
    
    T violation = T(0);
    if (i < n) {
        T over = x[i] - ub[i];
        T under = lb[i] - x[i];
        violation = fmax(T(0), over) + fmax(T(0), under);
    }
    sdata[tid] = violation;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

}  // namespace kernels

constexpr int kBlockSize = 256;

template <typename T>
void project_box(DeviceVector<T>& x, 
                 const DeviceVector<T>& lb,
                 const DeviceVector<T>& ub) {
    Index n = x.size();
    CUPROX_ASSERT(lb.size() == n && ub.size() == n, "project_box: size mismatch");
    
    if (n == 0) return;
    
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    kernels::project_box_kernel<<<num_blocks, kBlockSize>>>(
        x.data(), lb.data(), ub.data(), n);
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
void project_nonnegative(DeviceVector<T>& x) {
    Index n = x.size();
    if (n == 0) return;
    
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    kernels::project_nonneg_kernel<<<num_blocks, kBlockSize>>>(x.data(), n);
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
void gradient_step_and_project(DeviceVector<T>& y,
                                const DeviceVector<T>& x,
                                T alpha,
                                const DeviceVector<T>& grad,
                                const DeviceVector<T>& lb,
                                const DeviceVector<T>& ub) {
    Index n = x.size();
    CUPROX_ASSERT(y.size() == n && grad.size() == n, 
                  "gradient_step_and_project: size mismatch");
    CUPROX_ASSERT(lb.size() == n && ub.size() == n,
                  "gradient_step_and_project: bounds size mismatch");
    
    if (n == 0) return;
    
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    kernels::gradient_step_project_kernel<<<num_blocks, kBlockSize>>>(
        y.data(), x.data(), alpha, grad.data(), lb.data(), ub.data(), n);
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
T norm_inf(const DeviceVector<T>& x) {
    Index n = x.size();
    if (n == 0) return T(0);
    
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    
    DevicePtr<T> d_partial(num_blocks);
    
    kernels::reduce_max_abs_kernel<<<num_blocks, kBlockSize>>>(
        x.data(), d_partial.get(), n);
    CUPROX_CUDA_CHECK_LAST();
    
    std::vector<T> h_partial(num_blocks);
    copy_device_to_host(h_partial.data(), d_partial.get(), num_blocks);
    
    T result = T(0);
    for (int i = 0; i < num_blocks; ++i) {
        result = std::max(result, h_partial[i]);
    }
    return result;
}

template <typename T>
T box_violation(const DeviceVector<T>& x,
                const DeviceVector<T>& lb,
                const DeviceVector<T>& ub) {
    Index n = x.size();
    CUPROX_ASSERT(lb.size() == n && ub.size() == n, "box_violation: size mismatch");
    
    if (n == 0) return T(0);
    
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    
    DevicePtr<T> d_partial(num_blocks);
    
    kernels::reduce_box_violation_kernel<<<num_blocks, kBlockSize>>>(
        x.data(), lb.data(), ub.data(), d_partial.get(), n);
    CUPROX_CUDA_CHECK_LAST();
    
    std::vector<T> h_partial(num_blocks);
    copy_device_to_host(h_partial.data(), d_partial.get(), num_blocks);
    
    T result = T(0);
    for (int i = 0; i < num_blocks; ++i) {
        result = std::max(result, h_partial[i]);
    }
    return result;
}

// Explicit instantiations
template void project_box<float>(DeviceVector<float>&, 
                                  const DeviceVector<float>&,
                                  const DeviceVector<float>&);
template void project_box<double>(DeviceVector<double>&,
                                   const DeviceVector<double>&,
                                   const DeviceVector<double>&);

template void project_nonnegative<float>(DeviceVector<float>&);
template void project_nonnegative<double>(DeviceVector<double>&);

template void gradient_step_and_project<float>(DeviceVector<float>&,
                                                const DeviceVector<float>&,
                                                float,
                                                const DeviceVector<float>&,
                                                const DeviceVector<float>&,
                                                const DeviceVector<float>&);
template void gradient_step_and_project<double>(DeviceVector<double>&,
                                                 const DeviceVector<double>&,
                                                 double,
                                                 const DeviceVector<double>&,
                                                 const DeviceVector<double>&,
                                                 const DeviceVector<double>&);

template float norm_inf<float>(const DeviceVector<float>&);
template double norm_inf<double>(const DeviceVector<double>&);

template float box_violation<float>(const DeviceVector<float>&,
                                     const DeviceVector<float>&,
                                     const DeviceVector<float>&);
template double box_violation<double>(const DeviceVector<double>&,
                                       const DeviceVector<double>&,
                                       const DeviceVector<double>&);

}  // namespace cuprox
