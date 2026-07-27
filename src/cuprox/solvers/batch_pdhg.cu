#include "batch_pdhg.cuh"
#include "../preprocess/scaling.cuh"
#include "../core/cuda_context.cuh"
#include <chrono>
#include <cmath>

namespace cuprox {

namespace batch_kernels {

// Initialize all x to zero
template <typename T>
__global__ void init_zero_kernel(T* data, Index total_size) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_size) {
        data[i] = T(0);
    }
}

// Batched primal update: x[b] = proj(x[b] - tau * (c[b] + Aty[b]))
template <typename T>
__global__ void batch_primal_update_kernel(
    T* x_new,           // (batch_size x n)
    const T* x,         // (batch_size x n)
    const T* c,         // (batch_size x n)
    const T* Aty,       // (batch_size x n)
    const T* lb,        // (n) shared
    const T* ub,        // (n) shared
    T tau,
    Index batch_size,
    Index n
) {
    Index total = batch_size * n;
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        Index j = idx % n;  // Variable index
        T val = x[idx] - tau * (c[idx] + Aty[idx]);
        val = fmax(val, lb[j]);
        val = fmin(val, ub[j]);
        x_new[idx] = val;
    }
}

// Batched extrapolation: x_bar = 2*x_new - x_old
template <typename T>
__global__ void batch_extrapolation_kernel(
    T* x_bar,
    const T* x_new,
    const T* x_old,
    Index total_size
) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        x_bar[idx] = T(2) * x_new[idx] - x_old[idx];
    }
}

// Batched dual update: y[b] = y[b] + sigma * (Ax[b] - b[b])
template <typename T>
__global__ void batch_dual_update_kernel(
    T* y_new,           // (batch_size x m)
    const T* y,         // (batch_size x m)
    const T* Ax,        // (batch_size x m)
    const T* b,         // (batch_size x m)
    T sigma,
    Index total_size
) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        y_new[idx] = y[idx] + sigma * (Ax[idx] - b[idx]);
    }
}

// Copy kernel
template <typename T>
__global__ void batch_copy_kernel(T* dst, const T* src, Index total_size) {
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        dst[idx] = src[idx];
    }
}

// Compute objectives: obj[b] = c[b]' * x[b]
template <typename T>
__global__ void batch_objective_kernel(
    T* objectives,      // (batch_size)
    const T* c,         // (batch_size x n)
    const T* x,         // (batch_size x n)
    Index batch_size,
    Index n
) {
    Index b = blockIdx.x;  // One block per problem
    if (b >= batch_size) return;
    
    __shared__ T sdata[256];
    
    Index tid = threadIdx.x;
    Index offset = b * n;
    
    // Each thread computes partial dot product
    T sum = T(0);
    for (Index j = tid; j < n; j += blockDim.x) {
        sum += c[offset + j] * x[offset + j];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        objectives[b] = sdata[0];
    }
}

// Compute residual norms per problem
template <typename T>
__global__ void batch_residual_kernel(
    T* residuals,       // (batch_size)
    const T* r,         // (batch_size x m) residual vectors
    Index batch_size,
    Index m
) {
    Index b = blockIdx.x;
    if (b >= batch_size) return;
    
    __shared__ T sdata[256];
    
    Index tid = threadIdx.x;
    Index offset = b * m;
    
    T sum = T(0);
    for (Index j = tid; j < m; j += blockDim.x) {
        T val = r[offset + j];
        sum += val * val;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        residuals[b] = sqrt(sdata[0]);
    }
}

// Fill an int buffer. cudaMemset writes *bytes*, so memset(p, 3, n*sizeof(int))
// produces 0x03030303 per element rather than 3 -- which left every status
// holding a value no branch could ever match.
__global__ void batch_fill_int_kernel(int* p, int value, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = value;
}

// r = Ax - b, written to a distinct buffer.
template <typename T>
__global__ void batch_primal_residual_vec_kernel(
    T* r, const T* Ax, const T* b, Index total_m
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_m) r[i] = Ax[i] - b[i];
}

// Dual residual vector: the reduced cost c + A^T y, restricted to the
// directions the box does not already account for. At a solution this is zero
// on free coordinates and correctly signed on active bounds.
template <typename T>
__global__ void batch_dual_residual_vec_kernel(
    T* r, const T* c_batch, const T* Aty, const T* x,
    const T* lb, const T* ub, Index n, Index total_n
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_n) return;
    const Index j = i % n;                    // shared bounds across the batch
    const T red = c_batch[i] + Aty[i];
    if (x[i] <= lb[j] + T(1e-9)) {
        r[i] = fmin(red, T(0));               // at lower bound: red >= 0
    } else if (x[i] >= ub[j] - T(1e-9)) {
        r[i] = fmax(red, T(0));               // at upper bound: red <= 0
    } else {
        r[i] = red;                           // interior: red == 0
    }
}

// Check convergence and update status
template <typename T>
__global__ void batch_check_convergence_kernel(
    int* statuses,          // (batch_size)
    int* iterations,        // (batch_size)
    const T* primal_res,    // (batch_size)
    const T* dual_res,      // (batch_size)
    const T* scale_primal,  // (batch_size) ||Ax||
    const T* scale_dual,    // (batch_size) ||c||
    T eps_abs,
    T eps_rel,
    int current_iter,
    Index batch_size
) {
    Index b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    
    // Skip if already converged
    if (statuses[b] == 0) return;  // 0 = Optimal
    
    // Scale each tolerance by the quantity it measures, as OSQP does. The old
    // `eps_abs + eps_rel` ignored problem scale entirely, so the same absolute
    // threshold was applied whether the residual was of order 1 or 1e6.
    const T p_tol = eps_abs + eps_rel * fmax(scale_primal[b], T(1));
    const T d_tol = eps_abs + eps_rel * fmax(scale_dual[b], T(1));

    if (primal_res[b] < p_tol && dual_res[b] < d_tol) {
        statuses[b] = 0;  // Optimal
        iterations[b] = current_iter;
    }
}

}  // namespace batch_kernels

constexpr int kBlockSize = 256;

// Batched SpMV: y[b] = A * x[b] for all b
/**
 * Batched sparse mat-vec, as a single sparse-dense product.
 *
 * Every problem in the batch shares A, so applying it to all of them at once is
 * A @ X with X dense (n x batch_size) -- one cusparseSpMM, not batch_size
 * SpMVs. The previous implementation looped over the batch and, inside the
 * loop, allocated two DeviceVectors and made two device-to-device copies per
 * problem per iteration. That is strictly more work than solving the problems
 * one at a time, which is why "batched" solving measured slower than sequential.
 *
 * Layout: x_batch is (batch_size x n) row-major, which read column-major is
 * (n x batch_size) with leading dimension n -- exactly the operand SpMM wants,
 * so no transposition or repacking is needed anywhere.
 */
template <typename T>
static void batched_spmm(
    const CsrMatrix<T>& A,
    cusparseOperation_t op,
    const T* x_batch,
    T* y_batch,
    Index batch_size,
    Index rows_in,     // leading dimension of x_batch
    Index rows_out     // leading dimension of y_batch
) {
    const cudaDataType value_type = (sizeof(T) == 4) ? CUDA_R_32F : CUDA_R_64F;
    const T alpha = T(1);
    const T beta = T(0);

    cusparseDnMatDescr_t x_descr, y_descr;
    CUPROX_CUSPARSE_CHECK(cusparseCreateDnMat(
        &x_descr, rows_in, batch_size, rows_in,
        const_cast<T*>(x_batch), value_type, CUSPARSE_ORDER_COL));
    CUPROX_CUSPARSE_CHECK(cusparseCreateDnMat(
        &y_descr, rows_out, batch_size, rows_out,
        y_batch, value_type, CUSPARSE_ORDER_COL));

    size_t buffer_size = 0;
    CUPROX_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparse_handle(), op, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.descriptor(), x_descr, &beta, y_descr,
        value_type, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

    void* buffer = nullptr;
    if (buffer_size > 0) CUPROX_CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUPROX_CUSPARSE_CHECK(cusparseSpMM(
        cusparse_handle(), op, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.descriptor(), x_descr, &beta, y_descr,
        value_type, CUSPARSE_SPMM_ALG_DEFAULT, buffer));

    if (buffer) cudaFree(buffer);
    CUPROX_CUSPARSE_CHECK(cusparseDestroyDnMat(x_descr));
    CUPROX_CUSPARSE_CHECK(cusparseDestroyDnMat(y_descr));
}

template <typename T>
void batched_spmv(
    const CsrMatrix<T>& A,
    const T* x_batch,   // (batch_size x n)
    T* y_batch,         // (batch_size x m)
    Index batch_size,
    Index n,
    Index m
) {
    batched_spmm(A, CUSPARSE_OPERATION_NON_TRANSPOSE, x_batch, y_batch,
                 batch_size, n, m);
}

template <typename T>
void batched_spmv_transpose(
    const CsrMatrix<T>& A,
    const T* x_batch,   // (batch_size x m)
    T* y_batch,         // (batch_size x n)
    Index batch_size,
    Index n,
    Index m
) {
    batched_spmm(A, CUSPARSE_OPERATION_TRANSPOSE, x_batch, y_batch,
                 batch_size, m, n);
}

template <typename T>
BatchPdhgResult<T> BatchPdhgSolver<T>::solve(BatchLPProblem<T>& problem) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Index batch_size = problem.batch_size;
    Index n = problem.n;
    Index m = problem.m;
    
    // Compute step sizes
    T norm_A = estimate_operator_norm(*problem.A, 20);
    T tau = T(0.9) / norm_A;
    T sigma = T(0.9) / norm_A;
    
    // Allocate batched arrays
    DevicePtr<T> x(batch_size * n);
    DevicePtr<T> x_prev(batch_size * n);
    DevicePtr<T> x_bar(batch_size * n);
    DevicePtr<T> y(batch_size * m);
    DevicePtr<T> y_prev(batch_size * m);
    DevicePtr<T> Ax(batch_size * m);
    DevicePtr<T> Aty(batch_size * n);
    DevicePtr<T> primal_res(batch_size);
    DevicePtr<T> dual_res(batch_size);
    DevicePtr<T> objectives(batch_size);
    DevicePtr<int> statuses(batch_size);
    DevicePtr<int> iterations(batch_size);
    DevicePtr<T> scale_primal(batch_size);
    DevicePtr<T> scale_dual(batch_size);
    
    Index total_n = batch_size * n;
    Index total_m = batch_size * m;
    int blocks_n = (total_n + kBlockSize - 1) / kBlockSize;
    int blocks_m = (total_m + kBlockSize - 1) / kBlockSize;
    int blocks_batch = (batch_size + kBlockSize - 1) / kBlockSize;
    
    // Initialize
    batch_kernels::init_zero_kernel<<<blocks_n, kBlockSize>>>(x.get(), total_n);
    batch_kernels::init_zero_kernel<<<blocks_m, kBlockSize>>>(y.get(), total_m);
    
    // Initialise statuses to MaxIterations. cudaMemset writes bytes, so it
    // cannot be used to store the value 3 in an int.
    batch_kernels::batch_fill_int_kernel<<<blocks_batch, kBlockSize>>>(
        statuses.get(), static_cast<int>(Status::MaxIterations), batch_size);
    
    // Set iterations to 0
    cudaMemset(iterations.get(), 0, batch_size * sizeof(int));
    CUPROX_CUDA_CHECK_LAST();
    
    // Main PDHG loop
    for (int iter = 1; iter <= settings_.max_iters; ++iter) {
        // Save previous iterates
        batch_kernels::batch_copy_kernel<<<blocks_n, kBlockSize>>>(
            x_prev.get(), x.get(), total_n);
        batch_kernels::batch_copy_kernel<<<blocks_m, kBlockSize>>>(
            y_prev.get(), y.get(), total_m);
        
        // Compute A' * y (batched)
        batched_spmv_transpose(*problem.A, y.get(), Aty.get(), batch_size, n, m);
        
        // Primal update: x = proj(x - tau*(c + A'y))
        batch_kernels::batch_primal_update_kernel<<<blocks_n, kBlockSize>>>(
            x.get(), x_prev.get(), problem.c_batch.get(), Aty.get(),
            problem.lb->data(), problem.ub->data(), tau, batch_size, n);
        
        // Extrapolation: x_bar = 2*x - x_prev
        batch_kernels::batch_extrapolation_kernel<<<blocks_n, kBlockSize>>>(
            x_bar.get(), x.get(), x_prev.get(), total_n);
        
        // Compute A * x_bar (batched)
        batched_spmv(*problem.A, x_bar.get(), Ax.get(), batch_size, n, m);
        
        // Dual update: y = y + sigma*(Ax - b)
        batch_kernels::batch_dual_update_kernel<<<blocks_m, kBlockSize>>>(
            y.get(), y_prev.get(), Ax.get(), problem.b_batch.get(), sigma, total_m);
        
        CUPROX_CUDA_CHECK_LAST();
        
        // Check convergence periodically
        if (iter % settings_.check_interval == 0) {
            // Compute A * x for residuals
            batched_spmv(*problem.A, x.get(), Ax.get(), batch_size, n, m);
            
            // Primal residual ||Ax - b|| per problem.
            DevicePtr<T> residual_vec(total_m);
            batch_kernels::batch_primal_residual_vec_kernel<<<blocks_m, kBlockSize>>>(
                residual_vec.get(), Ax.get(), problem.b_batch.get(), total_m);
            batch_kernels::batch_residual_kernel<<<batch_size, kBlockSize,
                kBlockSize * sizeof(T)>>>(primal_res.get(), residual_vec.get(),
                                          batch_size, m);
            batch_kernels::batch_residual_kernel<<<batch_size, kBlockSize,
                kBlockSize * sizeof(T)>>>(scale_primal.get(), Ax.get(),
                                          batch_size, m);

            // Dual residual ||c + A^T y|| restricted by the active bounds. This
            // used to be memset to zero, so convergence rested on the primal
            // residual alone and "optimal" carried no dual information.
            batched_spmv_transpose(*problem.A, y.get(), Aty.get(), batch_size, n, m);
            DevicePtr<T> dual_vec(total_n);
            batch_kernels::batch_dual_residual_vec_kernel<<<blocks_n, kBlockSize>>>(
                dual_vec.get(), problem.c_batch.get(), Aty.get(), x.get(),
                problem.lb->data(), problem.ub->data(), n, total_n);
            batch_kernels::batch_residual_kernel<<<batch_size, kBlockSize,
                kBlockSize * sizeof(T)>>>(dual_res.get(), dual_vec.get(),
                                          batch_size, n);
            batch_kernels::batch_residual_kernel<<<batch_size, kBlockSize,
                kBlockSize * sizeof(T)>>>(scale_dual.get(), problem.c_batch.get(),
                                          batch_size, n);

            // Check convergence
            batch_kernels::batch_check_convergence_kernel<<<blocks_batch, kBlockSize>>>(
                statuses.get(), iterations.get(), primal_res.get(), dual_res.get(),
                scale_primal.get(), scale_dual.get(),
                settings_.eps_abs, settings_.eps_rel, iter, batch_size);
            
            CUPROX_CUDA_CHECK_LAST();
        }
    }
    
    // Final: compute A*x for objective computation
    batched_spmv(*problem.A, x.get(), Ax.get(), batch_size, n, m);
    
    // Compute objectives
    batch_kernels::batch_objective_kernel<<<batch_size, kBlockSize, 
        kBlockSize * sizeof(T)>>>(objectives.get(), problem.c_batch.get(), 
                                   x.get(), batch_size, n);
    CUPROX_CUDA_CHECK_LAST();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    // Prepare result
    BatchPdhgResult<T> result;
    result.x = std::move(x);
    result.y = std::move(y);
    result.objectives = std::move(objectives);
    result.statuses = std::move(statuses);
    result.iterations = std::move(iterations);
    result.batch_size = batch_size;
    result.n = n;
    result.m = m;
    result.total_solve_time = elapsed;
    
    return result;
}

template <typename T>
BatchLPProblem<T> make_batch_lp(
    CsrMatrix<T>& A,
    const T* c_batch_host,
    const T* b_batch_host,
    DeviceVector<T>& lb,
    DeviceVector<T>& ub,
    Index batch_size,
    Index n,
    Index m
) {
    BatchLPProblem<T> problem;
    problem.A = &A;
    problem.lb = &lb;
    problem.ub = &ub;
    problem.batch_size = batch_size;
    problem.n = n;
    problem.m = m;
    
    // Copy batched data to device
    problem.c_batch.reset(batch_size * n);
    problem.b_batch.reset(batch_size * m);
    
    copy_host_to_device(problem.c_batch.get(), c_batch_host, batch_size * n);
    copy_host_to_device(problem.b_batch.get(), b_batch_host, batch_size * m);
    
    return problem;
}

// Explicit instantiations
template class BatchPdhgSolver<float>;
template class BatchPdhgSolver<double>;

template BatchLPProblem<float> make_batch_lp<float>(
    CsrMatrix<float>&, const float*, const float*,
    DeviceVector<float>&, DeviceVector<float>&, Index, Index, Index);
template BatchLPProblem<double> make_batch_lp<double>(
    CsrMatrix<double>&, const double*, const double*,
    DeviceVector<double>&, DeviceVector<double>&, Index, Index, Index);

}  // namespace cuprox

