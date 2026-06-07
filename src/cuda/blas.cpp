#include "ms/cuda/blas.hpp"
#include "ms/cuda/buffer.hpp"
#include <vector>

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include <cublas_v2.h>
#endif

namespace ms::cuda {

namespace {

Matrix<double> matmul_cpu(
    const Matrix<double, StorageOrder::ColMajor>& A,
    const Matrix<double, StorageOrder::ColMajor>& B) {
    Matrix<double> C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

} // namespace

Result<Matrix<double>> matmul(
    const Matrix<double, StorageOrder::ColMajor>& A,
    const Matrix<double, StorageOrder::ColMajor>& B,
    int device) {
    if (A.cols() != B.rows()) {
        return std::unexpected(DimensionMismatch{A.cols(), B.rows()});
    }

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!available()) {
        return matmul_cpu(A, B);
    }

    const int m = static_cast<int>(A.rows());
    const int k = static_cast<int>(A.cols());
    const int n = static_cast<int>(B.cols());
    const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(double);
    const size_t b_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(double);
    const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(double);

    std::vector<double> a_host(static_cast<size_t>(m) * static_cast<size_t>(k));
    std::vector<double> b_host(static_cast<size_t>(k) * static_cast<size_t>(n));
    for (size_t j = 0; j < A.cols(); ++j) {
        for (size_t i = 0; i < A.rows(); ++i) {
            a_host[j * A.rows() + i] = A(i, j);
        }
    }
    for (size_t j = 0; j < B.cols(); ++j) {
        for (size_t i = 0; i < B.rows(); ++i) {
            b_host[j * B.rows() + i] = B(i, j);
        }
    }

    auto d_a = make_device_buffer(a_bytes, device);
    auto d_b = make_device_buffer(b_bytes, device);
    auto d_c = make_device_buffer(c_bytes, device);
    if (d_a.data == nullptr || d_b.data == nullptr || d_c.data == nullptr) {
        return matmul_cpu(A, B);
    }

    copy_host_to_device(a_host.data(), d_a, a_bytes);
    copy_host_to_device(b_host.data(), d_b, b_bytes);

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return matmul_cpu(A, B);
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    const cublasStatus_t status = cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        d_a.as<double>(),
        m,
        d_b.as<double>(),
        k,
        &beta,
        d_c.as<double>(),
        m);

    cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return matmul_cpu(A, B);
    }

    std::vector<double> c_host(static_cast<size_t>(m) * static_cast<size_t>(n));
    copy_device_to_host(d_c, c_host.data(), c_bytes);

    Matrix<double> C(static_cast<size_t>(m), static_cast<size_t>(n));
    for (size_t j = 0; j < C.cols(); ++j) {
        for (size_t i = 0; i < C.rows(); ++i) {
            C(i, j) = c_host[j * C.rows() + i];
        }
    }
    return C;
#else
    (void)device;
    return matmul_cpu(A, B);
#endif
}

} // namespace ms::cuda
