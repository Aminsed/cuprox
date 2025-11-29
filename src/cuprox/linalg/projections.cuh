#ifndef CUPROX_LINALG_PROJECTIONS_CUH
#define CUPROX_LINALG_PROJECTIONS_CUH

#include "../core/types.hpp"
#include "../core/error.hpp"
#include "../core/dense_vector.cuh"

namespace cuprox {

// Project x onto box constraints: result = clip(x, lb, ub)
template <typename T>
void project_box(DeviceVector<T>& x, 
                 const DeviceVector<T>& lb,
                 const DeviceVector<T>& ub);

// Project x onto non-negative orthant: result = max(x, 0)
template <typename T>
void project_nonnegative(DeviceVector<T>& x);

// Fused operation: y = clip(x - alpha * grad, lb, ub)
template <typename T>
void gradient_step_and_project(DeviceVector<T>& y,
                                const DeviceVector<T>& x,
                                T alpha,
                                const DeviceVector<T>& grad,
                                const DeviceVector<T>& lb,
                                const DeviceVector<T>& ub);

// Compute infinity norm
template <typename T>
T norm_inf(const DeviceVector<T>& x);

// Compute max(x - ub, 0) + max(lb - x, 0) for constraint violations
template <typename T>
T box_violation(const DeviceVector<T>& x,
                const DeviceVector<T>& lb,
                const DeviceVector<T>& ub);

}  // namespace cuprox

#endif  // CUPROX_LINALG_PROJECTIONS_CUH

