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

template <typename T>
__global__ void elementwise_max_kernel(T* a, const T* b, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (a[i] > b[i]) ? a[i] : b[i];
}

template <typename T>
__global__ void reciprocal_kernel(T* a, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (a[i] != T(0)) ? T(1) / a[i] : T(1);
}

template <typename T>
__global__ void divide_vectors_kernel(T* a, const T* b, Index n) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (b[i] != T(0)) ? a[i] / b[i] : a[i];
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


namespace {

// Largest finite magnitude in a host-side copy. Used for the two scalars the
// cost-scaling step needs; the vectors involved are length n, and this runs
// scaling_iters times, so the transfer is negligible next to the solve.
template <typename T>
T inf_norm_host(const DeviceVector<T>& v) {
    if (v.size() == 0) return T(0);
    const auto h = v.to_host();
    T best = T(0);
    for (T e : h) {
        const T a = std::abs(e);
        if (std::isfinite(a) && a > best) best = a;
    }
    return best;
}

template <typename T>
T mean_host(const DeviceVector<T>& v) {
    if (v.size() == 0) return T(0);
    const auto h = v.to_host();
    double acc = 0.0;
    for (T e : h) acc += static_cast<double>(e);
    return static_cast<T>(acc / static_cast<double>(h.size()));
}

}  // namespace

template <typename T>
ScalingFactors<T> ruiz_equilibrate_qp(
    CsrMatrix<T>& P, CsrMatrix<T>& A,
    DeviceVector<T>& q, DeviceVector<T>& l, DeviceVector<T>& u,
    DeviceVector<T>& lb, DeviceVector<T>& ub,
    int max_iters
) {
    const Index n = A.num_cols();
    const Index m = A.num_rows();
    ScalingFactors<T> s(m, n);
    s.D.fill(T(1));
    s.E.fill(T(1));
    s.c_scale = T(1);

    DeviceVector<T> col_p(n), col_a(n), row_a(m), delta_x(n), delta_z(m);
    const int gn = (n + kBlockSize - 1) / kBlockSize;
    const int gm = (m + kBlockSize - 1) / kBlockSize;

    for (int it = 0; it < max_iters; ++it) {
        // Variable scaling: largest entry of this column in either P or A.
        compute_col_inf_norms(P, col_p);
        compute_col_inf_norms(A, col_a);
        kernels::elementwise_max_kernel<<<gn, kBlockSize>>>(col_p.data(), col_a.data(), n);
        kernels::invert_sqrt_kernel<<<gn, kBlockSize>>>(col_p.data(), n, T(1e-12));
        CUPROX_CUDA_CHECK_LAST();
        delta_x.copy_from(col_p);

        // Constraint scaling: largest entry of this row of A.
        compute_row_inf_norms(A, row_a);
        kernels::invert_sqrt_kernel<<<gm, kBlockSize>>>(row_a.data(), m, T(1e-12));
        CUPROX_CUDA_CHECK_LAST();
        delta_z.copy_from(row_a);

        // P <- Dx P Dx,  A <- Dz A Dx,  q <- Dx q
        scale_rows(P, delta_x);
        scale_cols(P, delta_x);
        scale_cols(A, delta_x);
        scale_rows(A, delta_z);
        kernels::multiply_vectors_kernel<<<gn, kBlockSize>>>(q.data(), delta_x.data(), n);
        CUPROX_CUDA_CHECK_LAST();

        // Accumulate.
        kernels::multiply_vectors_kernel<<<gn, kBlockSize>>>(s.E.data(), delta_x.data(), n);
        kernels::multiply_vectors_kernel<<<gm, kBlockSize>>>(s.D.data(), delta_z.data(), m);
        CUPROX_CUDA_CHECK_LAST();

        // Cost scaling: balance the objective against the constraints.
        compute_col_inf_norms(P, col_p);
        const T mean_p = mean_host(col_p);
        const T norm_q = inf_norm_host(q);
        const T denom = std::max(mean_p, norm_q);
        if (denom > T(1e-12)) {
            const T gamma = T(1) / denom;
            kernels::scale_vector_kernel<<<gn, kBlockSize>>>(q.data(), gamma, n);
            CUPROX_CUDA_CHECK_LAST();
            DeviceVector<T> gvec(n);
            gvec.fill(gamma);
            scale_rows(P, gvec);
            s.c_scale *= gamma;
        }
    }

    // Row bounds move with the constraint scaling; variable bounds move
    // inversely with the variable scaling, since x = E x~.
    kernels::multiply_vectors_kernel<<<gm, kBlockSize>>>(l.data(), s.D.data(), m);
    kernels::multiply_vectors_kernel<<<gm, kBlockSize>>>(u.data(), s.D.data(), m);
    if (lb.size() == n) {
        kernels::divide_vectors_kernel<<<gn, kBlockSize>>>(lb.data(), s.E.data(), n);
        kernels::divide_vectors_kernel<<<gn, kBlockSize>>>(ub.data(), s.E.data(), n);
    }
    CUPROX_CUDA_CHECK_LAST();
    return s;
}

template <typename T>
void unscale_qp(
    CsrMatrix<T>& P, CsrMatrix<T>& A,
    DeviceVector<T>& q, DeviceVector<T>& l, DeviceVector<T>& u,
    DeviceVector<T>& lb, DeviceVector<T>& ub,
    DeviceVector<T>& x, DeviceVector<T>& y,
    const ScalingFactors<T>& s
) {
    const Index n = A.num_cols();
    const Index m = A.num_rows();
    const int gn = (n + kBlockSize - 1) / kBlockSize;
    const int gm = (m + kBlockSize - 1) / kBlockSize;

    // Solution: x = E x~,  y = c^-1 D y~.
    kernels::multiply_vectors_kernel<<<gn, kBlockSize>>>(x.data(), s.E.data(), n);
    kernels::multiply_vectors_kernel<<<gm, kBlockSize>>>(y.data(), s.D.data(), m);
    kernels::scale_vector_kernel<<<gm, kBlockSize>>>(y.data(), T(1) / s.c_scale, m);
    CUPROX_CUDA_CHECK_LAST();

    // Problem data, so the caller's QPProblem is left as it was found.
    DeviceVector<T> inv_e(n), inv_d(m);
    inv_e.copy_from(s.E);
    inv_d.copy_from(s.D);
    kernels::reciprocal_kernel<<<gn, kBlockSize>>>(inv_e.data(), n);
    kernels::reciprocal_kernel<<<gm, kBlockSize>>>(inv_d.data(), m);
    CUPROX_CUDA_CHECK_LAST();

    scale_rows(P, inv_e);
    scale_cols(P, inv_e);
    scale_rows(A, inv_d);
    scale_cols(A, inv_e);
    kernels::multiply_vectors_kernel<<<gn, kBlockSize>>>(q.data(), inv_e.data(), n);
    kernels::scale_vector_kernel<<<gn, kBlockSize>>>(q.data(), T(1) / s.c_scale, n);
    {
        DeviceVector<T> gvec(n);
        gvec.fill(T(1) / s.c_scale);
        scale_rows(P, gvec);
    }
    kernels::multiply_vectors_kernel<<<gm, kBlockSize>>>(l.data(), inv_d.data(), m);
    kernels::multiply_vectors_kernel<<<gm, kBlockSize>>>(u.data(), inv_d.data(), m);
    if (lb.size() == n) {
        kernels::multiply_vectors_kernel<<<gn, kBlockSize>>>(lb.data(), s.E.data(), n);
        kernels::multiply_vectors_kernel<<<gn, kBlockSize>>>(ub.data(), s.E.data(), n);
    }
    CUPROX_CUDA_CHECK_LAST();
}

template ScalingFactors<float> ruiz_equilibrate_qp<float>(
    CsrMatrix<float>&, CsrMatrix<float>&, DeviceVector<float>&, DeviceVector<float>&,
    DeviceVector<float>&, DeviceVector<float>&, DeviceVector<float>&, int);
template ScalingFactors<double> ruiz_equilibrate_qp<double>(
    CsrMatrix<double>&, CsrMatrix<double>&, DeviceVector<double>&, DeviceVector<double>&,
    DeviceVector<double>&, DeviceVector<double>&, DeviceVector<double>&, int);
template void unscale_qp<float>(
    CsrMatrix<float>&, CsrMatrix<float>&, DeviceVector<float>&, DeviceVector<float>&,
    DeviceVector<float>&, DeviceVector<float>&, DeviceVector<float>&,
    DeviceVector<float>&, DeviceVector<float>&, const ScalingFactors<float>&);
template void unscale_qp<double>(
    CsrMatrix<double>&, CsrMatrix<double>&, DeviceVector<double>&, DeviceVector<double>&,
    DeviceVector<double>&, DeviceVector<double>&, DeviceVector<double>&,
    DeviceVector<double>&, DeviceVector<double>&, const ScalingFactors<double>&);

template <typename T>
T unscaled_norm2(const DeviceVector<T>& v, const DeviceVector<T>& s, T alpha,
                 DeviceVector<T>& scratch) {
    const Index n = v.size();
    if (scratch.size() != n) scratch.resize(n);
    scratch.copy_from(v);
    kernels::divide_vectors_kernel<<<(n + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        scratch.data(), s.data(), n);
    CUPROX_CUDA_CHECK_LAST();
    return alpha * scratch.norm2();
}

template float unscaled_norm2<float>(const DeviceVector<float>&,
    const DeviceVector<float>&, float, DeviceVector<float>&);
template double unscaled_norm2<double>(const DeviceVector<double>&,
    const DeviceVector<double>&, double, DeviceVector<double>&);

template <typename T>
void unscale_solution(
    DeviceVector<T>& x,
    DeviceVector<T>& y,
    const ScalingFactors<T>& scaling
) {
    Index n = x.size();
    Index m = y.size();

    // Scaled problem: min c̃'x̃ s.t. Ãx̃ = b̃
    // where: Ã = D*A*E, c̃ = c_scale*E*c, b̃ = b_scale*D*b
    // Variable transform: x̃ = E⁻¹*x, so x = E*x̃
    // But we also normalized b, so we need: x = E * x̃ / b_scale

    // x_orig = E * x_scaled / b_scale
    T b_scale_inv = (scaling.b_scale > T(1e-10)) ? T(1) / scaling.b_scale : T(1);
    int num_blocks_n = (n + kBlockSize - 1) / kBlockSize;
    kernels::multiply_vectors_kernel<<<num_blocks_n, kBlockSize>>>(
        x.data(), scaling.E.data(), n
    );
    kernels::scale_vector_kernel<<<num_blocks_n, kBlockSize>>>(
        x.data(), b_scale_inv, n
    );
    CUPROX_CUDA_CHECK_LAST();

    // y_orig = D * y_scaled / c_scale
    // (dual scaling: y transforms inversely to primal objective scaling)
    T c_scale_inv = (scaling.c_scale > T(1e-10)) ? T(1) / scaling.c_scale : T(1);
    int num_blocks_m = (m + kBlockSize - 1) / kBlockSize;
    kernels::multiply_vectors_kernel<<<num_blocks_m, kBlockSize>>>(
        y.data(), scaling.D.data(), m
    );
    kernels::scale_vector_kernel<<<num_blocks_m, kBlockSize>>>(
        y.data(), c_scale_inv, m
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

