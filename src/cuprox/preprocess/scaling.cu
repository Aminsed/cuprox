#include "scaling.cuh"
#include "../core/memory.cuh"
#include <cmath>
#include <algorithm>

namespace cuprox {

namespace kernels {

template <typename T>
__global__ void row_inf_norm_kernel(
    const Index* row_offsets,
    const T* values,
    T* row_norms,
    Index num_rows
) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        T max_val = T(0);
        for (Index j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
            max_val = fmax(max_val, fabs(values[j]));
        }
        row_norms[row] = max_val;
    }
}

template <typename T>
__global__ void col_inf_norm_kernel(
    const Index* row_offsets,
    const Index* col_indices,
    const T* values,
    T* col_norms,
    Index num_rows
) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (Index j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
            Index col = col_indices[j];
            T val = fabs(values[j]);
            atomicMax((int*)&col_norms[col], __float_as_int((float)val));
        }
    }
}

// For double precision, we need a different approach
template <>
__global__ void col_inf_norm_kernel<double>(
    const Index* row_offsets,
    const Index* col_indices,
    const double* values,
    double* col_norms,
    Index num_rows
) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (Index j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
            Index col = col_indices[j];
            double val = fabs(values[j]);
            // Use atomicMax with unsigned long long for double
            unsigned long long* addr = (unsigned long long*)&col_norms[col];
            unsigned long long old = *addr, assumed;
            do {
                assumed = old;
                double old_val = __longlong_as_double(assumed);
                if (old_val >= val) break;
                old = atomicCAS(addr, assumed, __double_as_longlong(val));
            } while (assumed != old);
        }
    }
}

template <typename T>
__global__ void scale_rows_kernel(
    const Index* row_offsets,
    T* values,
    const T* D,
    Index num_rows
) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        T scale = D[row];
        for (Index j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
            values[j] *= scale;
        }
    }
}

template <typename T>
__global__ void scale_cols_kernel(
    const Index* row_offsets,
    const Index* col_indices,
    T* values,
    const T* E,
    Index num_rows
) {
    Index row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (Index j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
            values[j] *= E[col_indices[j]];
        }
    }
}

template <typename T>
__global__ void invert_sqrt_kernel(T* data, Index n, T epsilon) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T val = data[i];
        data[i] = (val > epsilon) ? T(1) / sqrt(val) : T(1);
    }
}

template <typename T>
__global__ void multiply_vectors_kernel(T* a, const T* b, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= b[i];
    }
}

template <typename T>
__global__ void scale_vector_kernel(T* data, T scale, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= scale;
    }
}

}  // namespace kernels

constexpr int kBlockSize = 256;

template <typename T>
void compute_row_inf_norms(const CsrMatrix<T>& A, DeviceVector<T>& row_norms) {
    Index m = A.num_rows();
    row_norms.resize(m);
    row_norms.fill(T(0));
    
    int num_blocks = (m + kBlockSize - 1) / kBlockSize;
    kernels::row_inf_norm_kernel<<<num_blocks, kBlockSize>>>(
        A.row_offsets(), A.values(), row_norms.data(), m
    );
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
void compute_col_inf_norms(const CsrMatrix<T>& A, DeviceVector<T>& col_norms) {
    Index m = A.num_rows();
    Index n = A.num_cols();
    col_norms.resize(n);
    col_norms.fill(T(0));
    
    int num_blocks = (m + kBlockSize - 1) / kBlockSize;
    kernels::col_inf_norm_kernel<<<num_blocks, kBlockSize>>>(
        A.row_offsets(), A.col_indices(), A.values(), col_norms.data(), m
    );
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
void scale_rows(CsrMatrix<T>& A, const DeviceVector<T>& D) {
    Index m = A.num_rows();
    int num_blocks = (m + kBlockSize - 1) / kBlockSize;
    kernels::scale_rows_kernel<<<num_blocks, kBlockSize>>>(
        A.row_offsets(), A.values(), D.data(), m
    );
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
void scale_cols(CsrMatrix<T>& A, const DeviceVector<T>& E) {
    Index m = A.num_rows();
    int num_blocks = (m + kBlockSize - 1) / kBlockSize;
    kernels::scale_cols_kernel<<<num_blocks, kBlockSize>>>(
        A.row_offsets(), A.col_indices(), A.values(), E.data(), m
    );
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
ScalingFactors<T> ruiz_equilibrate(
    CsrMatrix<T>& A,
    DeviceVector<T>& c,
    DeviceVector<T>& b,
    int max_iters
) {
    Index m = A.num_rows();
    Index n = A.num_cols();
    
    ScalingFactors<T> scaling(m, n);
    DeviceVector<T> row_norms(m);
    DeviceVector<T> col_norms(n);
    
    constexpr T epsilon = T(1e-10);
    
    for (int iter = 0; iter < max_iters; ++iter) {
        // Compute row norms and scale
        compute_row_inf_norms(A, row_norms);
        
        int num_blocks_m = (m + kBlockSize - 1) / kBlockSize;
        kernels::invert_sqrt_kernel<<<num_blocks_m, kBlockSize>>>(
            row_norms.data(), m, epsilon
        );
        CUPROX_CUDA_CHECK_LAST();
        
        scale_rows(A, row_norms);
        
        // Update D
        kernels::multiply_vectors_kernel<<<num_blocks_m, kBlockSize>>>(
            scaling.D.data(), row_norms.data(), m
        );
        CUPROX_CUDA_CHECK_LAST();
        
        // Scale b
        kernels::multiply_vectors_kernel<<<num_blocks_m, kBlockSize>>>(
            b.data(), row_norms.data(), m
        );
        CUPROX_CUDA_CHECK_LAST();
        
        // Compute column norms and scale
        compute_col_inf_norms(A, col_norms);
        
        int num_blocks_n = (n + kBlockSize - 1) / kBlockSize;
        kernels::invert_sqrt_kernel<<<num_blocks_n, kBlockSize>>>(
            col_norms.data(), n, epsilon
        );
        CUPROX_CUDA_CHECK_LAST();
        
        scale_cols(A, col_norms);
        
        // Update E
        kernels::multiply_vectors_kernel<<<num_blocks_n, kBlockSize>>>(
            scaling.E.data(), col_norms.data(), n
        );
        CUPROX_CUDA_CHECK_LAST();
        
        // Scale c
        kernels::multiply_vectors_kernel<<<num_blocks_n, kBlockSize>>>(
            c.data(), col_norms.data(), n
        );
        CUPROX_CUDA_CHECK_LAST();
    }
    
    // Compute objective and RHS scaling
    T c_norm = c.norm2();
    T b_norm = b.norm2();
    
    if (c_norm > epsilon) {
        scaling.c_scale = T(1) / c_norm;
        int num_blocks = (n + kBlockSize - 1) / kBlockSize;
        kernels::scale_vector_kernel<<<num_blocks, kBlockSize>>>(
            c.data(), scaling.c_scale, n
        );
        CUPROX_CUDA_CHECK_LAST();
    }
    
    if (b_norm > epsilon) {
        scaling.b_scale = T(1) / b_norm;
        int num_blocks = (m + kBlockSize - 1) / kBlockSize;
        kernels::scale_vector_kernel<<<num_blocks, kBlockSize>>>(
            b.data(), scaling.b_scale, m
        );
        CUPROX_CUDA_CHECK_LAST();
    }
    
    return scaling;
}

template <typename T>
void unscale_solution(
    DeviceVector<T>& x,
    DeviceVector<T>& y,
    const ScalingFactors<T>& scaling
) {
    Index n = x.size();
    Index m = y.size();
    
    // x = E * x_scaled * b_scale
    int num_blocks_n = (n + kBlockSize - 1) / kBlockSize;
    kernels::multiply_vectors_kernel<<<num_blocks_n, kBlockSize>>>(
        x.data(), scaling.E.data(), n
    );
    kernels::scale_vector_kernel<<<num_blocks_n, kBlockSize>>>(
        x.data(), scaling.b_scale, n
    );
    CUPROX_CUDA_CHECK_LAST();
    
    // y = D * y_scaled * c_scale
    int num_blocks_m = (m + kBlockSize - 1) / kBlockSize;
    kernels::multiply_vectors_kernel<<<num_blocks_m, kBlockSize>>>(
        y.data(), scaling.D.data(), m
    );
    kernels::scale_vector_kernel<<<num_blocks_m, kBlockSize>>>(
        y.data(), scaling.c_scale, m
    );
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
T estimate_operator_norm(const CsrMatrix<T>& A, int max_iters) {
    Index n = A.num_cols();
    Index m = A.num_rows();
    
    DeviceVector<T> x(n, T(1) / sqrt(static_cast<T>(n)));
    DeviceVector<T> y(m);
    DeviceVector<T> x_new(n);
    
    T sigma = T(1);
    
    for (int iter = 0; iter < max_iters; ++iter) {
        // y = A * x
        y.fill(T(0));
        A.spmv(T(1), x, T(0), y);
        
        // x_new = A^T * y
        x_new.fill(T(0));
        A.spmv_transpose(T(1), y, T(0), x_new);
        
        // sigma = ||x_new|| / ||x||
        T x_norm = x.norm2();
        T x_new_norm = x_new.norm2();
        
        if (x_norm > T(1e-10)) {
            sigma = x_new_norm / x_norm;
        }
        
        // Normalize x_new
        if (x_new_norm > T(1e-10)) {
            x_new.scale(T(1) / x_new_norm);
        }
        
        // x = x_new (swap)
        std::swap(x, x_new);
    }
    
    return sqrt(sigma);  // ||A||_2 = sqrt(sigma)
}

// Explicit instantiations
template void compute_row_inf_norms<float>(const CsrMatrix<float>&, DeviceVector<float>&);
template void compute_row_inf_norms<double>(const CsrMatrix<double>&, DeviceVector<double>&);

template void compute_col_inf_norms<float>(const CsrMatrix<float>&, DeviceVector<float>&);
template void compute_col_inf_norms<double>(const CsrMatrix<double>&, DeviceVector<double>&);

template void scale_rows<float>(CsrMatrix<float>&, const DeviceVector<float>&);
template void scale_rows<double>(CsrMatrix<double>&, const DeviceVector<double>&);

template void scale_cols<float>(CsrMatrix<float>&, const DeviceVector<float>&);
template void scale_cols<double>(CsrMatrix<double>&, const DeviceVector<double>&);

template ScalingFactors<float> ruiz_equilibrate<float>(
    CsrMatrix<float>&, DeviceVector<float>&, DeviceVector<float>&, int);
template ScalingFactors<double> ruiz_equilibrate<double>(
    CsrMatrix<double>&, DeviceVector<double>&, DeviceVector<double>&, int);

template void unscale_solution<float>(DeviceVector<float>&, DeviceVector<float>&, 
                                       const ScalingFactors<float>&);
template void unscale_solution<double>(DeviceVector<double>&, DeviceVector<double>&,
                                        const ScalingFactors<double>&);

template float estimate_operator_norm<float>(const CsrMatrix<float>&, int);
template double estimate_operator_norm<double>(const CsrMatrix<double>&, int);

}  // namespace cuprox

