#include "pdhg.cuh"
#include "../linalg/projections.cuh"
#include "../core/cuda_context.cuh"
#include <chrono>
#include <cmath>
#include <iostream>

namespace cuprox {

namespace kernels {

template <typename T>
__global__ void primal_update_kernel(
    T* x_new,
    const T* x,
    const T* c,
    const T* Aty,
    const T* lb,
    const T* ub,
    T tau,
    Index n
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // x_new = x - tau * (c + A^T * y)
        T val = x[i] - tau * (c[i] + Aty[i]);
        // Project onto box [lb, ub]
        val = fmax(val, lb[i]);
        val = fmin(val, ub[i]);
        x_new[i] = val;
    }
}

template <typename T>
__global__ void extrapolation_kernel(
    T* x_bar,
    const T* x_new,
    const T* x_old,
    Index n
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // x_bar = 2 * x_new - x_old
        x_bar[i] = T(2) * x_new[i] - x_old[i];
    }
}

template <typename T>
__global__ void dual_update_eq_kernel(
    T* y_new,
    const T* y,
    const T* Ax,
    const T* b,
    T sigma,
    Index m
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        // y_new = y + sigma * (A * x_bar - b)
        // For equality constraints, no projection needed
        y_new[i] = y[i] + sigma * (Ax[i] - b[i]);
    }
}

template <typename T>
__global__ void dual_update_ineq_kernel(
    T* y_new,
    const T* y,
    const T* Ax,
    const T* l,
    const T* u,
    T sigma,
    Index m
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        // For l <= Ax <= u constraints
        // y_new = y + sigma * Ax, then project onto [l - Ax, u - Ax] (dual feasibility)
        T ax = Ax[i];
        T y_val = y[i] + sigma * ax;
        
        // Dual variable for l <= Ax: y >= 0
        // Dual variable for Ax <= u: y <= 0
        // Combined: project to enforce complementarity
        if (l[i] > T(-1e20) && u[i] < T(1e20)) {
            // Two-sided constraint
            y_new[i] = y_val;
        } else if (l[i] > T(-1e20)) {
            // ax >= l only: y >= 0
            y_new[i] = fmax(y_val, T(0));
        } else if (u[i] < T(1e20)) {
            // ax <= u only: y <= 0
            y_new[i] = fmin(y_val, T(0));
        } else {
            // Free constraint (shouldn't happen in well-formed LP)
            y_new[i] = T(0);
        }
    }
}

template <typename T>
__global__ void compute_primal_residual_kernel(
    const T* Ax,
    const T* b,
    T* residual,
    Index m
) {
    __shared__ T sdata[256];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + tid;
    
    T val = (i < m) ? (Ax[i] - b[i]) * (Ax[i] - b[i]) : T(0);
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        residual[blockIdx.x] = sdata[0];
    }
}

template <typename T>
__global__ void compute_dual_residual_kernel(
    const T* c,
    const T* Aty,
    const T* x,
    const T* lb,
    const T* ub,
    T* residual,
    Index n
) {
    __shared__ T sdata[256];
    
    Index tid = threadIdx.x;
    Index i = blockIdx.x * blockDim.x + tid;
    
    T val = T(0);
    if (i < n) {
        // Reduced cost: r = c + A^T y
        T r = c[i] + Aty[i];
        
        // For bounded variables, check complementarity
        T xi = x[i];
        if (xi <= lb[i] + T(1e-8)) {
            // At lower bound: r >= 0
            val = fmin(r, T(0));
        } else if (xi >= ub[i] - T(1e-8)) {
            // At upper bound: r <= 0
            val = fmax(r, T(0));
        } else {
            // Interior: r = 0
            val = r;
        }
        val = val * val;
    }
    
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        residual[blockIdx.x] = sdata[0];
    }
}

template <typename T>
__global__ void copy_kernel(T* dst, const T* src, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

}  // namespace kernels

constexpr int kBlockSize = 256;

template <typename T>
void PdhgSolver<T>::initialize(LPProblem<T>& problem) {
    n_ = problem.num_vars();
    m_ = problem.num_constraints();
    
    A_ = &problem.A;
    c_ = &problem.c;
    b_ = &problem.b;
    lb_ = &problem.lb;
    ub_ = &problem.ub;
    l_ = &problem.l;
    u_ = &problem.u;
    
    // Apply Ruiz scaling if enabled
    if (settings_.scaling) {
        scaling_ = ruiz_equilibrate(*A_, *c_, *b_, settings_.scaling_iters);
    }
    
    // Compute step sizes
    if (settings_.tau == T(0) || settings_.sigma == T(0)) {
        T norm_A = estimate_operator_norm(*A_, 20);
        T safe_factor = T(0.9);  // Safety margin
        tau_ = safe_factor / norm_A;
        sigma_ = safe_factor / norm_A;
    } else {
        tau_ = settings_.tau;
        sigma_ = settings_.sigma;
    }
    
    // Initialize iterates
    x_.resize(n_);
    x_.fill(T(0));
    x_prev_.resize(n_);
    x_bar_.resize(n_);
    y_.resize(m_);
    y_.fill(T(0));
    y_prev_.resize(m_);
    
    // Workspace
    Ax_.resize(m_);
    Aty_.resize(n_);
    temp_.resize(std::max(m_, n_));
    
    iter_ = 0;
}

template <typename T>
void PdhgSolver<T>::iterate() {
    int num_blocks_n = (n_ + kBlockSize - 1) / kBlockSize;
    int num_blocks_m = (m_ + kBlockSize - 1) / kBlockSize;
    
    // Save previous iterates
    kernels::copy_kernel<<<num_blocks_n, kBlockSize>>>(
        x_prev_.data(), x_.data(), n_);
    kernels::copy_kernel<<<num_blocks_m, kBlockSize>>>(
        y_prev_.data(), y_.data(), m_);
    CUPROX_CUDA_CHECK_LAST();
    
    // Compute A^T * y
    Aty_.fill(T(0));
    A_->spmv_transpose(T(1), y_, T(0), Aty_);
    
    // Primal update: x = proj_X(x - tau * (c + A^T y))
    kernels::primal_update_kernel<<<num_blocks_n, kBlockSize>>>(
        x_.data(), x_prev_.data(), c_->data(), Aty_.data(),
        lb_->data(), ub_->data(), tau_, n_);
    CUPROX_CUDA_CHECK_LAST();
    
    // Extrapolation: x_bar = 2*x - x_prev
    kernels::extrapolation_kernel<<<num_blocks_n, kBlockSize>>>(
        x_bar_.data(), x_.data(), x_prev_.data(), n_);
    CUPROX_CUDA_CHECK_LAST();
    
    // Compute A * x_bar
    Ax_.fill(T(0));
    A_->spmv(T(1), x_bar_, T(0), Ax_);
    
    // Dual update: y = y + sigma * (A * x_bar - b)
    kernels::dual_update_eq_kernel<<<num_blocks_m, kBlockSize>>>(
        y_.data(), y_prev_.data(), Ax_.data(), b_->data(), sigma_, m_);
    CUPROX_CUDA_CHECK_LAST();
    
    ++iter_;
}

template <typename T>
void PdhgSolver<T>::compute_residuals() {
    int num_blocks_n = (n_ + kBlockSize - 1) / kBlockSize;
    int num_blocks_m = (m_ + kBlockSize - 1) / kBlockSize;
    
    // Compute A * x (for primal residual)
    A_->spmv(T(1), x_, T(0), Ax_);
    
    // Compute A^T * y (for dual residual)
    A_->spmv_transpose(T(1), y_, T(0), Aty_);
    
    // Primal residual: ||Ax - b||
    DevicePtr<T> partial_p(num_blocks_m);
    kernels::compute_primal_residual_kernel<<<num_blocks_m, kBlockSize>>>(
        Ax_.data(), b_->data(), partial_p.get(), m_);
    CUPROX_CUDA_CHECK_LAST();
    
    std::vector<T> h_partial_p(num_blocks_m);
    copy_device_to_host(h_partial_p.data(), partial_p.get(), num_blocks_m);
    
    T sum_p = T(0);
    for (int i = 0; i < num_blocks_m; ++i) {
        sum_p += h_partial_p[i];
    }
    primal_res_ = std::sqrt(sum_p);
    
    // Dual residual: ||c + A^T y||_reduced
    DevicePtr<T> partial_d(num_blocks_n);
    kernels::compute_dual_residual_kernel<<<num_blocks_n, kBlockSize>>>(
        c_->data(), Aty_.data(), x_.data(), lb_->data(), ub_->data(),
        partial_d.get(), n_);
    CUPROX_CUDA_CHECK_LAST();
    
    std::vector<T> h_partial_d(num_blocks_n);
    copy_device_to_host(h_partial_d.data(), partial_d.get(), num_blocks_n);
    
    T sum_d = T(0);
    for (int i = 0; i < num_blocks_n; ++i) {
        sum_d += h_partial_d[i];
    }
    dual_res_ = std::sqrt(sum_d);
}

template <typename T>
bool PdhgSolver<T>::check_convergence() {
    compute_residuals();
    
    // Relative tolerances
    T x_norm = x_.norm2();
    T y_norm = y_.norm2();
    T b_norm = b_->norm2();
    T c_norm = c_->norm2();
    
    T primal_tol = settings_.eps_abs + settings_.eps_rel * std::max(b_norm, T(1));
    T dual_tol = settings_.eps_abs + settings_.eps_rel * std::max(c_norm, T(1));
    
    if (settings_.verbose && (iter_ % 100 == 0 || iter_ == 1)) {
        std::cout << "Iter " << iter_ 
                  << " | p_res: " << primal_res_ 
                  << " | d_res: " << dual_res_
                  << " | tol: " << primal_tol << ", " << dual_tol
                  << std::endl;
    }
    
    return (primal_res_ < primal_tol) && (dual_res_ < dual_tol);
}

template <typename T>
void PdhgSolver<T>::adaptive_restart() {
    // Restart if duality gap increases (simple heuristic)
    // More sophisticated: check if ||z_k - z_{k-1}||^2 < ||z_{k-1} - z_{k-2}||^2
    T x_change = T(0);
    T y_change = T(0);
    
    // Compute ||x - x_prev||^2 + ||y - y_prev||^2
    DeviceVector<T> diff_x(n_);
    diff_x.copy_from(x_);
    diff_x.axpy(T(-1), x_prev_);
    x_change = diff_x.norm2();
    
    DeviceVector<T> diff_y(m_);
    diff_y.copy_from(y_);
    diff_y.axpy(T(-1), y_prev_);
    y_change = diff_y.norm2();
    
    // If change is very small, we may be oscillating - restart
    if (x_change + y_change < T(1e-10)) {
        // Restart by resetting to current iterate (no momentum)
        x_prev_.copy_from(x_);
        y_prev_.copy_from(y_);
    }
}

template <typename T>
PdhgResult<T> PdhgSolver<T>::solve(LPProblem<T>& problem) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Save original c and b for objective computation (before scaling)
    DeviceVector<T> c_orig, b_orig;
    c_orig.copy_from(problem.c);
    b_orig.copy_from(problem.b);
    
    initialize(problem);
    
    PdhgResult<T> result;
    result.status = Status::MaxIterations;
    
    for (iter_ = 1; iter_ <= settings_.max_iters; ++iter_) {
        iterate();
        
        if (iter_ % settings_.check_interval == 0) {
            if (check_convergence()) {
                result.status = Status::Optimal;
                break;
            }
            
            if (settings_.adaptive_restart) {
                adaptive_restart();
            }
        }
    }
    
    // Compute final residuals
    compute_residuals();
    
    // Unscale solution if scaling was applied
    if (settings_.scaling) {
        unscale_solution(x_, y_, scaling_);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    // Compute objectives with ORIGINAL (unscaled) c and b
    T primal_obj = c_orig.dot(x_);
    T dual_obj = b_orig.dot(y_);
    
    // Populate result
    result.x = std::move(x_);
    result.y = std::move(y_);
    result.primal_obj = primal_obj;
    result.dual_obj = dual_obj;
    result.primal_res = primal_res_;
    result.dual_res = dual_res_;
    result.iterations = iter_;
    result.solve_time = elapsed;
    
    return result;
}

// Explicit instantiations
template class PdhgSolver<float>;
template class PdhgSolver<double>;

}  // namespace cuprox

