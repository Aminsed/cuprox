#ifndef CUPROX_CORE_CUDA_CONTEXT_CUH
#define CUPROX_CORE_CUDA_CONTEXT_CUH

#include <cublas_v2.h>
#include <cusparse.h>
#include <memory>
#include <mutex>

#include "error.hpp"

namespace cuprox {

class CudaContext {
public:
    static CudaContext& instance() {
        static CudaContext ctx;
        return ctx;
    }

    cublasHandle_t cublas() const { return cublas_handle_; }
    cusparseHandle_t cusparse() const { return cusparse_handle_; }

    int device_id() const { return device_id_; }
    
    void synchronize() const {
        CUPROX_CUDA_CHECK(cudaDeviceSynchronize());
    }

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

private:
    CudaContext() {
        CUPROX_CUDA_CHECK(cudaGetDevice(&device_id_));
        CUPROX_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUPROX_CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
    }

    ~CudaContext() {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
    }

    int device_id_ = 0;
    cublasHandle_t cublas_handle_ = nullptr;
    cusparseHandle_t cusparse_handle_ = nullptr;
};

inline cublasHandle_t cublas_handle() {
    return CudaContext::instance().cublas();
}

inline cusparseHandle_t cusparse_handle() {
    return CudaContext::instance().cusparse();
}

}  // namespace cuprox

#endif  // CUPROX_CORE_CUDA_CONTEXT_CUH

