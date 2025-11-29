#include "admm.cuh"
#include "../linalg/projections.cuh"
#include "../core/cuda_context.cuh"
#include <chrono>
#include <cmath>
#include <iostream>

namespace cuprox {

namespace kernels {

// Project z onto box [l, u]
template <typename T>
__global__ void project_box_kernel(T* z, const T* l, const T* u, Index m) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        T val = z[i];
        val = fmax(val, l[i]);
        val = fmin(val, u[i]);
        z[i] = val;
    }
}

// Compute z = Ax + y/rho, then project
template <typename T>
__global__ void z_update_kernel(T* z, const T* Ax, const T* y, T rho_inv,
                                 const T* l, const T* u, Index m) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        T val = Ax[i] + y[i] * rho_inv;
        val = fmax(val, l[i]);
        val = fmin(val, u[i]);
        z[i] = val;
    }
}

// y = y + rho * (Ax - z)
template <typename T>
__global__ void y_update_kernel(T* y, const T* Ax, const T* z, T rho, Index m) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        y[i] += rho * (Ax[i] - z[i]);
    }
}

// Compute RHS for CG: rhs = rho * A'z - A'y - q
template <typename T>
__global__ void compute_rhs_kernel(T* rhs, const T* Atz, const T* Aty, 
                                    const T* q, T rho, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        rhs[i] = rho * Atz[i] - Aty[i] - q[i];
    }
}

// r = r - alpha * Ap
template <typename T>
__global__ void cg_update_r_kernel(T* r, const T* Ap, T alpha, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] -= alpha * Ap[i];
    }
}

// x = x + alpha * p
template <typename T>
__global__ void cg_update_x_kernel(T* x, const T* p, T alpha, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += alpha * p[i];
    }
}

// p = r + beta * p
template <typename T>
__global__ void cg_update_p_kernel(T* p, const T* r, T beta, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = r[i] + beta * p[i];
    }
}

// Compute primal residual: r = Ax - z
template <typename T>
__global__ void primal_res_kernel(T* r, const T* Ax, const T* z, Index m) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        r[i] = Ax[i] - z[i];
    }
}

}  // namespace kernels

constexpr int kBlockSize = 256;

template <typename T>
void AdmmSolver<T>::initialize(QPProblem<T>& problem) {
    n_ = problem.num_vars();
    m_ = problem.num_constraints();
    
    P_ = &problem.P;
    A_ = &problem.A;
    q_ = &problem.q;
    l_ = &problem.l;
    u_ = &problem.u;
    
    // Initialize rho
    if (settings_.rho == T(0)) {
        // Auto-select rho based on problem data
        T norm_q = q_->norm2();
        rho_ = (norm_q > T(1e-6)) ? norm_q / T(n_) : T(1.0);
        rho_ = std::max(settings_.rho_min, std::min(settings_.rho_max, rho_));
    } else {
        rho_ = settings_.rho;
    }
    
    // Initialize iterates
    x_.resize(n_);
    x_.fill(T(0));
    
    z_.resize(m_);
    z_.fill(T(0));
    
    z_prev_.resize(m_);
    
    y_.resize(m_);
    y_.fill(T(0));
    
    // Workspace
    Ax_.resize(m_);
    Px_.resize(n_);
    Aty_.resize(n_);
    Atz_.resize(n_);
    rhs_.resize(n_);
    temp_.resize(std::max(m_, n_));
    
    // CG workspace
    cg_r_.resize(n_);
    cg_p_.resize(n_);
    cg_Ap_.resize(n_);
    
    iter_ = 0;
}

template <typename T>
void AdmmSolver<T>::x_update() {
    int num_blocks = (n_ + kBlockSize - 1) / kBlockSize;
    
    // Compute RHS: rhs = rho * A'z - A'y - q
    Atz_.fill(T(0));
    A_->spmv_transpose(T(1), z_, T(0), Atz_);
    
    Aty_.fill(T(0));
    A_->spmv_transpose(T(1), y_, T(0), Aty_);
    
    kernels::compute_rhs_kernel<<<num_blocks, kBlockSize>>>(
        rhs_.data(), Atz_.data(), Aty_.data(), q_->data(), rho_, n_);
    CUPROX_CUDA_CHECK_LAST();
    
    // Solve (P + rho * A'A) x = rhs using Conjugate Gradient
    // Initial guess: x (warm start from previous iteration)
    
    // Compute initial residual: r = rhs - (P + rho*A'A) * x
    // First compute (P + rho*A'A) * x
    Px_.fill(T(0));
    P_->spmv(T(1), x_, T(0), Px_);  // Px = P * x
    
    Ax_.fill(T(0));
    A_->spmv(T(1), x_, T(0), Ax_);   // Ax = A * x
    
    temp_.fill(T(0));
    A_->spmv_transpose(T(1), Ax_, T(0), temp_);  // temp = A' * Ax
    
    // cg_Ap = P*x + rho * A'*A*x
    cg_Ap_.copy_from(Px_);
    cg_Ap_.axpy(rho_, temp_);
    
    // r = rhs - (P + rho*A'A) * x
    cg_r_.copy_from(rhs_);
    cg_r_.axpy(T(-1), cg_Ap_);
    
    // p = r
    cg_p_.copy_from(cg_r_);
    
    T r_norm_sq = cg_r_.dot(cg_r_);
    T rhs_norm = rhs_.norm2();
    T tol = settings_.cg_tolerance * std::max(rhs_norm, T(1));
    
    for (int cg_iter = 0; cg_iter < settings_.cg_max_iters; ++cg_iter) {
        if (std::sqrt(r_norm_sq) < tol) {
            break;
        }
        
        // Compute Ap = (P + rho * A'A) * p
        Px_.fill(T(0));
        P_->spmv(T(1), cg_p_, T(0), Px_);
        
        Ax_.fill(T(0));
        A_->spmv(T(1), cg_p_, T(0), Ax_);
        
        temp_.fill(T(0));
        A_->spmv_transpose(T(1), Ax_, T(0), temp_);
        
        cg_Ap_.copy_from(Px_);
        cg_Ap_.axpy(rho_, temp_);
        
        T pAp = cg_p_.dot(cg_Ap_);
        if (std::abs(pAp) < T(1e-14)) {
            break;  // Converged or breakdown
        }
        
        T alpha = r_norm_sq / pAp;
        
        // x = x + alpha * p
        kernels::cg_update_x_kernel<<<num_blocks, kBlockSize>>>(
            x_.data(), cg_p_.data(), alpha, n_);
        
        // r = r - alpha * Ap
        kernels::cg_update_r_kernel<<<num_blocks, kBlockSize>>>(
            cg_r_.data(), cg_Ap_.data(), alpha, n_);
        CUPROX_CUDA_CHECK_LAST();
        
        T r_norm_sq_new = cg_r_.dot(cg_r_);
        T beta = r_norm_sq_new / r_norm_sq;
        r_norm_sq = r_norm_sq_new;
        
        // p = r + beta * p
        kernels::cg_update_p_kernel<<<num_blocks, kBlockSize>>>(
            cg_p_.data(), cg_r_.data(), beta, n_);
        CUPROX_CUDA_CHECK_LAST();
    }
}

template <typename T>
void AdmmSolver<T>::z_update() {
    int num_blocks = (m_ + kBlockSize - 1) / kBlockSize;
    
    // Save previous z for dual residual computation
    z_prev_.copy_from(z_);
    
    // Compute Ax
    Ax_.fill(T(0));
    A_->spmv(T(1), x_, T(0), Ax_);
    
    // z = proj_{[l,u]}(Ax + y/rho)
    T rho_inv = T(1) / rho_;
    kernels::z_update_kernel<<<num_blocks, kBlockSize>>>(
        z_.data(), Ax_.data(), y_.data(), rho_inv, l_->data(), u_->data(), m_);
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
void AdmmSolver<T>::y_update() {
    int num_blocks = (m_ + kBlockSize - 1) / kBlockSize;
    
    // y = y + rho * (Ax - z)
    kernels::y_update_kernel<<<num_blocks, kBlockSize>>>(
        y_.data(), Ax_.data(), z_.data(), rho_, m_);
    CUPROX_CUDA_CHECK_LAST();
}

template <typename T>
void AdmmSolver<T>::compute_residuals() {
    int num_blocks_m = (m_ + kBlockSize - 1) / kBlockSize;
    
    // Primal residual: ||Ax - z||
    DeviceVector<T> pres(m_);
    kernels::primal_res_kernel<<<num_blocks_m, kBlockSize>>>(
        pres.data(), Ax_.data(), z_.data(), m_);
    CUPROX_CUDA_CHECK_LAST();
    
    primal_res_ = pres.norm2();
    
    // Dual residual: ||rho * A'(z - z_prev)||
    DeviceVector<T> dz(m_);
    dz.copy_from(z_);
    dz.axpy(T(-1), z_prev_);
    
    DeviceVector<T> Atdz(n_);
    Atdz.fill(T(0));
    A_->spmv_transpose(rho_, dz, T(0), Atdz);
    
    dual_res_ = Atdz.norm2();
}

template <typename T>
bool AdmmSolver<T>::check_convergence() {
    compute_residuals();
    
    // Compute tolerances (OSQP-style)
    T Ax_norm = Ax_.norm2();
    T z_norm = z_.norm2();
    T Aty_norm = Aty_.norm2();
    T Px_norm = Px_.norm2();
    T q_norm = q_->norm2();
    
    T eps_primal = settings_.eps_abs * std::sqrt(T(m_)) + 
                   settings_.eps_rel * std::max(Ax_norm, z_norm);
    T eps_dual = settings_.eps_abs * std::sqrt(T(n_)) + 
                 settings_.eps_rel * std::max(Aty_norm, std::max(Px_norm, q_norm));
    
    if (settings_.verbose && (iter_ % 100 == 0 || iter_ == 1)) {
        std::cout << "Iter " << iter_ 
                  << " | p_res: " << primal_res_ 
                  << " | d_res: " << dual_res_
                  << " | rho: " << rho_
                  << std::endl;
    }
    
    return (primal_res_ < eps_primal) && (dual_res_ < eps_dual);
}

template <typename T>
void AdmmSolver<T>::update_rho() {
    if (!settings_.adaptive_rho) return;
    
    // Simple adaptive rho (OSQP-style)
    const T ratio = primal_res_ / (dual_res_ + T(1e-10));
    const T factor = T(5);
    
    if (ratio > T(10)) {
        rho_ = std::min(rho_ * factor, settings_.rho_max);
    } else if (ratio < T(0.1)) {
        rho_ = std::max(rho_ / factor, settings_.rho_min);
    }
}

template <typename T>
AdmmResult<T> AdmmSolver<T>::solve(QPProblem<T>& problem) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    initialize(problem);
    
    AdmmResult<T> result;
    result.status = Status::MaxIterations;
    
    for (iter_ = 1; iter_ <= settings_.max_iters; ++iter_) {
        x_update();
        z_update();
        y_update();
        
        if (iter_ % settings_.check_interval == 0) {
            if (check_convergence()) {
                result.status = Status::Optimal;
                break;
            }
            
            update_rho();
        }
    }
    
    // Compute final residuals
    compute_residuals();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    // Compute objective: (1/2)x'Px + q'x
    Px_.fill(T(0));
    P_->spmv(T(1), x_, T(0), Px_);
    T quad_term = T(0.5) * x_.dot(Px_);
    T lin_term = q_->dot(x_);
    T primal_obj = quad_term + lin_term;
    
    // Populate result
    result.x = std::move(x_);
    result.y = std::move(y_);
    result.z = std::move(z_);
    result.primal_obj = primal_obj;
    result.primal_res = primal_res_;
    result.dual_res = dual_res_;
    result.iterations = iter_;
    result.solve_time = elapsed;
    
    return result;
}

// Explicit instantiations
template class AdmmSolver<float>;
template class AdmmSolver<double>;

}  // namespace cuprox

