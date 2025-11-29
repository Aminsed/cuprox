/**
 * @file types.hpp
 * @brief Common type definitions for cuProx
 * 
 * This file defines the fundamental types used throughout the cuProx library.
 * All types are designed for both CPU and GPU compatibility.
 */

#ifndef CUPROX_CORE_TYPES_HPP
#define CUPROX_CORE_TYPES_HPP

#include <cstdint>
#include <limits>

namespace cuprox {

// ============================================================================
// Scalar Types
// ============================================================================

/// Default floating-point type (configurable at compile time)
#ifdef CUPROX_USE_FLOAT
using Scalar = float;
#else
using Scalar = double;
#endif

/// Integer type for indices
using Index = std::int32_t;

/// Size type
using Size = std::size_t;

// ============================================================================
// Constants
// ============================================================================

/// Infinity value for bounds
constexpr Scalar kInfinity = std::numeric_limits<Scalar>::infinity();

/// Machine epsilon
constexpr Scalar kEpsilon = std::numeric_limits<Scalar>::epsilon();

/// Default tolerance for convergence
constexpr Scalar kDefaultTolerance = 1e-6;

/// Default maximum iterations
constexpr Index kDefaultMaxIterations = 100000;

// ============================================================================
// Solver Status
// ============================================================================

/**
 * @brief Status codes returned by solvers
 */
enum class Status : int {
    /// Solution found within tolerance
    Optimal = 0,
    
    /// Problem is primal infeasible
    PrimalInfeasible = 1,
    
    /// Problem is dual infeasible (unbounded)
    DualInfeasible = 2,
    
    /// Maximum iterations reached
    MaxIterations = 3,
    
    /// Time limit reached
    TimeLimit = 4,
    
    /// Numerical issues encountered
    NumericalError = 5,
    
    /// Problem not yet solved
    Unsolved = -1,
    
    /// Invalid problem data
    InvalidInput = -2,
};

/**
 * @brief Convert status to string representation
 */
inline const char* status_to_string(Status status) {
    switch (status) {
        case Status::Optimal: return "optimal";
        case Status::PrimalInfeasible: return "primal_infeasible";
        case Status::DualInfeasible: return "dual_infeasible";
        case Status::MaxIterations: return "max_iterations";
        case Status::TimeLimit: return "time_limit";
        case Status::NumericalError: return "numerical_error";
        case Status::Unsolved: return "unsolved";
        case Status::InvalidInput: return "invalid_input";
        default: return "unknown";
    }
}

// ============================================================================
// Solver Settings
// ============================================================================

/**
 * @brief Solver configuration parameters
 */
struct Settings {
    /// Convergence tolerance (primal/dual residual)
    Scalar tolerance = kDefaultTolerance;
    
    /// Maximum number of iterations
    Index max_iterations = kDefaultMaxIterations;
    
    /// Time limit in seconds (0 = no limit)
    Scalar time_limit = 0.0;
    
    /// Enable Ruiz scaling
    bool scaling = true;
    
    /// Number of scaling iterations
    Index scaling_iterations = 10;
    
    /// Enable adaptive restart
    bool adaptive_restart = true;
    
    /// Interval for convergence checks
    Index check_interval = 50;
    
    /// Enable verbose output
    bool verbose = false;
    
    /// Device ID to use (-1 = auto-select)
    int device_id = -1;
};

// ============================================================================
// Solution Info
// ============================================================================

/**
 * @brief Statistics about the solve process
 */
struct SolveInfo {
    /// Final status
    Status status = Status::Unsolved;
    
    /// Number of iterations performed
    Index iterations = 0;
    
    /// Total solve time in seconds
    Scalar solve_time = 0.0;
    
    /// Setup time (preprocessing) in seconds
    Scalar setup_time = 0.0;
    
    /// Final primal residual
    Scalar primal_residual = kInfinity;
    
    /// Final dual residual
    Scalar dual_residual = kInfinity;
    
    /// Duality gap
    Scalar gap = kInfinity;
    
    /// Objective value
    Scalar objective = 0.0;
};

}  // namespace cuprox

#endif  // CUPROX_CORE_TYPES_HPP

