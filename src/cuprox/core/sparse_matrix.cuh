#ifndef CUPROX_CORE_SPARSE_MATRIX_CUH
#define CUPROX_CORE_SPARSE_MATRIX_CUH

#include <cusparse.h>
#include <vector>

#include "types.hpp"
#include "error.hpp"
#include "memory.cuh"
#include "dense_vector.cuh"
#include "cuda_context.cuh"

namespace cuprox {

template <typename T>
class CsrMatrix {
public:
    CsrMatrix() = default;

    CsrMatrix(Index rows, Index cols, Index nnz)
        : num_rows_(rows), num_cols_(cols), nnz_(nnz),
          row_offsets_(rows + 1), col_indices_(nnz), values_(nnz) {
        create_descriptors();
    }

    ~CsrMatrix() {
        destroy_descriptors();
    }

    CsrMatrix(const CsrMatrix&) = delete;
    CsrMatrix& operator=(const CsrMatrix&) = delete;

    CsrMatrix(CsrMatrix&& other) noexcept { swap(other); }
    CsrMatrix& operator=(CsrMatrix&& other) noexcept {
        if (this != &other) {
            destroy_descriptors();
            swap(other);
        }
        return *this;
    }

    // Dimensions
    Index num_rows() const { return num_rows_; }
    Index num_cols() const { return num_cols_; }
    Index nnz() const { return nnz_; }

    // Raw data access
    Index* row_offsets() { return row_offsets_.get(); }
    const Index* row_offsets() const { return row_offsets_.get(); }
    Index* col_indices() { return col_indices_.get(); }
    const Index* col_indices() const { return col_indices_.get(); }
    T* values() { return values_.get(); }
    const T* values() const { return values_.get(); }

    // cuSPARSE descriptor
    cusparseSpMatDescr_t descriptor() const { return mat_descr_; }

    // Build from COO format
    static CsrMatrix from_coo(Index rows, Index cols,
                               const std::vector<Index>& row_indices,
                               const std::vector<Index>& col_indices,
                               const std::vector<T>& values);

    // Build from CSR arrays (host)
    static CsrMatrix from_csr(Index rows, Index cols, Index nnz,
                               const Index* row_offsets,
                               const Index* col_indices,
                               const T* values);

    // SpMV: y = alpha * A * x + beta * y
    void spmv(T alpha, const DeviceVector<T>& x, 
              T beta, DeviceVector<T>& y) const;

    // SpMV transpose: y = alpha * A^T * x + beta * y
    void spmv_transpose(T alpha, const DeviceVector<T>& x,
                        T beta, DeviceVector<T>& y) const;

private:
    void create_descriptors();
    void destroy_descriptors();
    void swap(CsrMatrix& other) noexcept;

    Index num_rows_ = 0;
    Index num_cols_ = 0;
    Index nnz_ = 0;

    DevicePtr<Index> row_offsets_;
    DevicePtr<Index> col_indices_;
    DevicePtr<T> values_;

    cusparseSpMatDescr_t mat_descr_ = nullptr;
    mutable void* spmv_buffer_ = nullptr;
    mutable size_t spmv_buffer_size_ = 0;
};

template <typename T>
void CsrMatrix<T>::create_descriptors() {
    if (num_rows_ == 0 || num_cols_ == 0) return;

    cudaDataType value_type = (sizeof(T) == 4) ? CUDA_R_32F : CUDA_R_64F;
    
    CUPROX_CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_descr_,
        num_rows_, num_cols_, nnz_,
        row_offsets_.get(),
        col_indices_.get(),
        values_.get(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        value_type
    ));
}

template <typename T>
void CsrMatrix<T>::destroy_descriptors() {
    if (mat_descr_) {
        cusparseDestroySpMat(mat_descr_);
        mat_descr_ = nullptr;
    }
    if (spmv_buffer_) {
        cudaFree(spmv_buffer_);
        spmv_buffer_ = nullptr;
        spmv_buffer_size_ = 0;
    }
}

template <typename T>
void CsrMatrix<T>::swap(CsrMatrix& other) noexcept {
    std::swap(num_rows_, other.num_rows_);
    std::swap(num_cols_, other.num_cols_);
    std::swap(nnz_, other.nnz_);
    std::swap(row_offsets_, other.row_offsets_);
    std::swap(col_indices_, other.col_indices_);
    std::swap(values_, other.values_);
    std::swap(mat_descr_, other.mat_descr_);
    std::swap(spmv_buffer_, other.spmv_buffer_);
    std::swap(spmv_buffer_size_, other.spmv_buffer_size_);
}

template <typename T>
CsrMatrix<T> CsrMatrix<T>::from_csr(Index rows, Index cols, Index nnz,
                                     const Index* h_row_offsets,
                                     const Index* h_col_indices,
                                     const T* h_values) {
    CsrMatrix<T> mat(rows, cols, nnz);
    copy_host_to_device(mat.row_offsets_.get(), h_row_offsets, rows + 1);
    copy_host_to_device(mat.col_indices_.get(), h_col_indices, nnz);
    copy_host_to_device(mat.values_.get(), h_values, nnz);
    return mat;
}

template <typename T>
void CsrMatrix<T>::spmv(T alpha, const DeviceVector<T>& x,
                        T beta, DeviceVector<T>& y) const {
    CUPROX_ASSERT(x.size() == num_cols_, "spmv: x size mismatch");
    CUPROX_ASSERT(y.size() == num_rows_, "spmv: y size mismatch");

    cudaDataType value_type = (sizeof(T) == 4) ? CUDA_R_32F : CUDA_R_64F;

    cusparseDnVecDescr_t x_descr, y_descr;
    CUPROX_CUSPARSE_CHECK(cusparseCreateDnVec(&x_descr, x.size(), 
        const_cast<T*>(x.data()), value_type));
    CUPROX_CUSPARSE_CHECK(cusparseCreateDnVec(&y_descr, y.size(), 
        y.data(), value_type));

    size_t buffer_size = 0;
    CUPROX_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_descr_, x_descr, &beta, y_descr,
        value_type, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size
    ));

    if (buffer_size > spmv_buffer_size_) {
        if (spmv_buffer_) cudaFree(spmv_buffer_);
        CUPROX_CUDA_CHECK(cudaMalloc(&spmv_buffer_, buffer_size));
        spmv_buffer_size_ = buffer_size;
    }

    CUPROX_CUSPARSE_CHECK(cusparseSpMV(
        cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_descr_, x_descr, &beta, y_descr,
        value_type, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_
    ));

    CUPROX_CUSPARSE_CHECK(cusparseDestroyDnVec(x_descr));
    CUPROX_CUSPARSE_CHECK(cusparseDestroyDnVec(y_descr));
}

template <typename T>
void CsrMatrix<T>::spmv_transpose(T alpha, const DeviceVector<T>& x,
                                   T beta, DeviceVector<T>& y) const {
    CUPROX_ASSERT(x.size() == num_rows_, "spmv_transpose: x size mismatch");
    CUPROX_ASSERT(y.size() == num_cols_, "spmv_transpose: y size mismatch");

    cudaDataType value_type = (sizeof(T) == 4) ? CUDA_R_32F : CUDA_R_64F;

    cusparseDnVecDescr_t x_descr, y_descr;
    CUPROX_CUSPARSE_CHECK(cusparseCreateDnVec(&x_descr, x.size(), 
        const_cast<T*>(x.data()), value_type));
    CUPROX_CUSPARSE_CHECK(cusparseCreateDnVec(&y_descr, y.size(), 
        y.data(), value_type));

    size_t buffer_size = 0;
    CUPROX_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, mat_descr_, x_descr, &beta, y_descr,
        value_type, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size
    ));

    if (buffer_size > spmv_buffer_size_) {
        if (spmv_buffer_) cudaFree(spmv_buffer_);
        CUPROX_CUDA_CHECK(cudaMalloc(&spmv_buffer_, buffer_size));
        spmv_buffer_size_ = buffer_size;
    }

    CUPROX_CUSPARSE_CHECK(cusparseSpMV(
        cusparse_handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, mat_descr_, x_descr, &beta, y_descr,
        value_type, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_
    ));

    CUPROX_CUSPARSE_CHECK(cusparseDestroyDnVec(x_descr));
    CUPROX_CUSPARSE_CHECK(cusparseDestroyDnVec(y_descr));
}

// Explicit instantiations
template class CsrMatrix<float>;
template class CsrMatrix<double>;

}  // namespace cuprox

#endif  // CUPROX_CORE_SPARSE_MATRIX_CUH

