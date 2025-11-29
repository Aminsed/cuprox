/**
 * @file error.hpp
 * @brief Error handling utilities for cuProx
 * 
 * Provides exception classes and CUDA error checking macros.
 */

#ifndef CUPROX_CORE_ERROR_HPP
#define CUPROX_CORE_ERROR_HPP

#include <stdexcept>
#include <string>
#include <sstream>

#ifdef CUPROX_HAS_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#endif

namespace cuprox {

// ============================================================================
// Exception Classes
// ============================================================================

/**
 * @brief Base exception for all cuProx errors
 */
class CuproxError : public std::runtime_error {
public:
    explicit CuproxError(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Exception for invalid input data
 */
class InvalidInputError : public CuproxError {
public:
    explicit InvalidInputError(const std::string& message)
        : CuproxError("Invalid input: " + message) {}
};

/**
 * @brief Exception for dimension mismatches
 */
class DimensionError : public CuproxError {
public:
    explicit DimensionError(const std::string& message)
        : CuproxError("Dimension error: " + message) {}
};

/**
 * @brief Exception for numerical issues
 */
class NumericalError : public CuproxError {
public:
    explicit NumericalError(const std::string& message)
        : CuproxError("Numerical error: " + message) {}
};

#ifdef CUPROX_HAS_CUDA

/**
 * @brief Exception for CUDA runtime errors
 */
class CudaError : public CuproxError {
public:
    CudaError(cudaError_t error, const char* file, int line)
        : CuproxError(format_message(error, file, line))
        , error_(error) {}
    
    cudaError_t error() const { return error_; }
    
private:
    cudaError_t error_;
    
    static std::string format_message(cudaError_t error, const char* file, int line) {
        std::ostringstream ss;
        ss << "CUDA error at " << file << ":" << line << ": "
           << cudaGetErrorString(error) << " (code " << static_cast<int>(error) << ")";
        return ss.str();
    }
};

/**
 * @brief Exception for cuSPARSE errors
 */
class CusparseError : public CuproxError {
public:
    CusparseError(cusparseStatus_t status, const char* file, int line)
        : CuproxError(format_message(status, file, line))
        , status_(status) {}
    
    cusparseStatus_t status() const { return status_; }
    
private:
    cusparseStatus_t status_;
    
    static std::string format_message(cusparseStatus_t status, const char* file, int line) {
        std::ostringstream ss;
        ss << "cuSPARSE error at " << file << ":" << line << ": "
           << "status code " << static_cast<int>(status);
        return ss.str();
    }
};

/**
 * @brief Exception for cuBLAS errors
 */
class CublasError : public CuproxError {
public:
    CublasError(cublasStatus_t status, const char* file, int line)
        : CuproxError(format_message(status, file, line))
        , status_(status) {}
    
    cublasStatus_t status() const { return status_; }
    
private:
    cublasStatus_t status_;
    
    static std::string format_message(cublasStatus_t status, const char* file, int line) {
        std::ostringstream ss;
        ss << "cuBLAS error at " << file << ":" << line << ": "
           << "status code " << static_cast<int>(status);
        return ss.str();
    }
};

// ============================================================================
// Error Checking Macros
// ============================================================================

/**
 * @brief Check CUDA runtime call and throw on error
 */
#define CUPROX_CUDA_CHECK(call)                                          \
    do {                                                                  \
        cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                       \
            throw ::cuprox::CudaError(error, __FILE__, __LINE__);        \
        }                                                                 \
    } while (0)

/**
 * @brief Check cuSPARSE call and throw on error
 */
#define CUPROX_CUSPARSE_CHECK(call)                                      \
    do {                                                                  \
        cusparseStatus_t status = (call);                                 \
        if (status != CUSPARSE_STATUS_SUCCESS) {                          \
            throw ::cuprox::CusparseError(status, __FILE__, __LINE__);   \
        }                                                                 \
    } while (0)

/**
 * @brief Check cuBLAS call and throw on error
 */
#define CUPROX_CUBLAS_CHECK(call)                                        \
    do {                                                                  \
        cublasStatus_t status = (call);                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                            \
            throw ::cuprox::CublasError(status, __FILE__, __LINE__);     \
        }                                                                 \
    } while (0)

/**
 * @brief Check for CUDA errors after kernel launch
 */
#define CUPROX_CUDA_CHECK_LAST()                                         \
    do {                                                                  \
        cudaError_t error = cudaGetLastError();                           \
        if (error != cudaSuccess) {                                       \
            throw ::cuprox::CudaError(error, __FILE__, __LINE__);        \
        }                                                                 \
    } while (0)

#else  // !CUPROX_HAS_CUDA

// Stub macros for CPU-only builds
#define CUPROX_CUDA_CHECK(call) ((void)0)
#define CUPROX_CUSPARSE_CHECK(call) ((void)0)
#define CUPROX_CUBLAS_CHECK(call) ((void)0)
#define CUPROX_CUDA_CHECK_LAST() ((void)0)

#endif  // CUPROX_HAS_CUDA

// ============================================================================
// Assertion Macros (Active in debug builds)
// ============================================================================

#ifndef NDEBUG
#define CUPROX_ASSERT(condition, message)                                \
    do {                                                                  \
        if (!(condition)) {                                               \
            throw ::cuprox::CuproxError(                                 \
                std::string("Assertion failed: ") + (message));          \
        }                                                                 \
    } while (0)
#else
#define CUPROX_ASSERT(condition, message) ((void)0)
#endif

/**
 * @brief Always-active check (for user-facing validation)
 */
#define CUPROX_CHECK(condition, exception_type, message)                 \
    do {                                                                  \
        if (!(condition)) {                                               \
            throw exception_type(message);                                \
        }                                                                 \
    } while (0)

}  // namespace cuprox

#endif  // CUPROX_CORE_ERROR_HPP

