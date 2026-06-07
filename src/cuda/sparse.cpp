#include "ms/cuda/sparse.hpp"
#include "ms/cuda/buffer.hpp"
#include <vector>

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

namespace ms::cuda {

namespace {

Matrix<double> spmv_coo_cpu(
    size_t rows,
    const std::vector<size_t>& row,
    const std::vector<size_t>& col,
    const std::vector<double>& values,
    const Matrix<double, StorageOrder::ColMajor>& x) {
    Matrix<double> y(rows, 1, 0.0);
    for (size_t k = 0; k < values.size(); ++k) {
        y(row[k], 0) += values[k] * x(col[k], 0);
    }
    return y;
}

} // namespace

Result<Matrix<double>> spmv_coo(
    size_t rows,
    size_t cols,
    const std::vector<size_t>& row,
    const std::vector<size_t>& col,
    const std::vector<double>& values,
    const Matrix<double, StorageOrder::ColMajor>& x,
    int device) {
    if (x.cols() != 1) {
        return std::unexpected(DimensionMismatch{x.rows(), x.cols()});
    }
    if (x.rows() != cols) {
        return std::unexpected(DimensionMismatch{x.rows(), cols});
    }

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!available() || values.empty()) {
        return spmv_coo_cpu(rows, row, col, values, x);
    }

    const int nnz = static_cast<int>(values.size());
    const int n_rows = static_cast<int>(rows);
    const int n_cols = static_cast<int>(cols);

    std::vector<int> h_row(nnz);
    std::vector<int> h_col(nnz);
    for (int k = 0; k < nnz; ++k) {
        h_row[static_cast<size_t>(k)] = static_cast<int>(row[static_cast<size_t>(k)]);
        h_col[static_cast<size_t>(k)] = static_cast<int>(col[static_cast<size_t>(k)]);
    }

    std::vector<double> h_x(static_cast<size_t>(n_cols));
    for (int i = 0; i < n_cols; ++i) {
        h_x[static_cast<size_t>(i)] = x(static_cast<size_t>(i), 0);
    }

    auto d_row = make_device_buffer(static_cast<size_t>(nnz) * sizeof(int), device);
    auto d_col = make_device_buffer(static_cast<size_t>(nnz) * sizeof(int), device);
    auto d_vals = make_device_buffer(static_cast<size_t>(nnz) * sizeof(double), device);
    auto d_x = make_device_buffer(static_cast<size_t>(n_cols) * sizeof(double), device);
    auto d_y = make_device_buffer(static_cast<size_t>(n_rows) * sizeof(double), device);
    if (!d_row.data || !d_col.data || !d_vals.data || !d_x.data || !d_y.data) {
        return spmv_coo_cpu(rows, row, col, values, x);
    }

    copy_host_to_device(h_row.data(), d_row, h_row.size() * sizeof(int));
    copy_host_to_device(h_col.data(), d_col, h_col.size() * sizeof(int));
    copy_host_to_device(values.data(), d_vals, values.size() * sizeof(double));
    copy_host_to_device(h_x.data(), d_x, h_x.size() * sizeof(double));

    cusparseHandle_t handle = nullptr;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS) {
        return spmv_coo_cpu(rows, row, col, values, x);
    }

    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    const cusparseStatus_t mat_status = cusparseCreateCoo(
        &mat,
        n_rows,
        n_cols,
        nnz,
        d_row.as<int>(),
        d_col.as<int>(),
        d_vals.as<double>(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F);
    if (mat_status != CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateDnVec(&vec_x, n_cols, d_x.as<double>(), CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateDnVec(&vec_y, n_rows, d_y.as<double>(), CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroy(handle);
        return spmv_coo_cpu(rows, row, col, values, x);
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat,
        vec_x,
        &beta,
        vec_y,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size);

    DeviceBuffer spmv_buffer;
    void* buffer = nullptr;
    if (buffer_size > 0) {
        spmv_buffer = make_device_buffer(buffer_size, device);
        buffer = spmv_buffer.data;
    }

    const cusparseStatus_t spmv_status = cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat,
        vec_x,
        &beta,
        vec_y,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        buffer);

    Matrix<double> y(static_cast<size_t>(n_rows), 1, 0.0);
    if (spmv_status == CUSPARSE_STATUS_SUCCESS) {
        std::vector<double> h_y(static_cast<size_t>(n_rows));
        copy_device_to_host(d_y, h_y.data(), h_y.size() * sizeof(double));
        for (int i = 0; i < n_rows; ++i) {
            y(static_cast<size_t>(i), 0) = h_y[static_cast<size_t>(i)];
        }
    } else {
        y = spmv_coo_cpu(rows, row, col, values, x);
    }

    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
    cusparseDestroy(handle);
    return y;
#else
    (void)device;
    return spmv_coo_cpu(rows, row, col, values, x);
#endif
}

} // namespace ms::cuda
