#include "ms/cuda/solver.hpp"
#include "ms/cuda/buffer.hpp"
#include <vector>

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

namespace ms::cuda {

Result<std::tuple<Matrix<double>, Matrix<double>, Matrix<double>>> lu(
    const Matrix<double, StorageOrder::ColMajor>&,
    int) {
    return std::unexpected(DeviceError{10, "cuda lu not implemented; use ms::lu"});
}

Result<Matrix<double>> solve(
    const Matrix<double, StorageOrder::ColMajor>& A,
    const Matrix<double, StorageOrder::ColMajor>& b,
    int device) {
    if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
        return std::unexpected(DimensionMismatch{A.rows(), b.cols()});
    }

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!available()) {
        return std::unexpected(DeviceError{11, "cuda unavailable"});
    }

    const int n = static_cast<int>(A.rows());
    const size_t a_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(double);
    const size_t b_bytes = static_cast<size_t>(n) * sizeof(double);

    std::vector<double> h_A(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<double> h_B(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            h_A[static_cast<size_t>(j * n + i)] = A(static_cast<size_t>(i), static_cast<size_t>(j));
        }
    }
    for (int i = 0; i < n; ++i) {
        h_B[static_cast<size_t>(i)] = b(static_cast<size_t>(i), 0);
    }

    auto d_A = make_device_buffer(a_bytes, device);
    auto d_B = make_device_buffer(b_bytes, device);
    auto d_ipiv = make_device_buffer(static_cast<size_t>(n) * sizeof(int), device);
    if (d_A.data == nullptr || d_B.data == nullptr || d_ipiv.data == nullptr) {
        return std::unexpected(DeviceError{12, "cuda solve allocation failed"});
    }

    copy_host_to_device(h_A.data(), d_A, a_bytes);
    copy_host_to_device(h_B.data(), d_B, b_bytes);

    cusolverDnHandle_t handle = nullptr;
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
        return std::unexpected(DeviceError{13, "cusolver create failed"});
    }

    int lwork = 0;
    if (cusolverDnDgetrf_bufferSize(handle, n, n, d_A.as<double>(), n, &lwork) != CUSOLVER_STATUS_SUCCESS) {
        cusolverDnDestroy(handle);
        return std::unexpected(DeviceError{14, "getrf buffer query failed"});
    }

    auto d_work = make_device_buffer(static_cast<size_t>(lwork) * sizeof(double), device);
    auto d_info = make_device_buffer(sizeof(int), device);
    if (d_work.data == nullptr || d_info.data == nullptr) {
        cusolverDnDestroy(handle);
        return std::unexpected(DeviceError{15, "workspace allocation failed"});
    }

    if (cusolverDnDgetrf(
            handle,
            n,
            n,
            d_A.as<double>(),
            n,
            d_work.as<double>(),
            d_ipiv.as<int>(),
            d_info.as<int>()) != CUSOLVER_STATUS_SUCCESS) {
        cusolverDnDestroy(handle);
        return std::unexpected(DeviceError{16, "getrf failed"});
    }

    if (cusolverDnDgetrs(
            handle,
            CUBLAS_OP_N,
            n,
            1,
            d_A.as<double>(),
            n,
            d_ipiv.as<int>(),
            d_B.as<double>(),
            n,
            d_info.as<int>()) != CUSOLVER_STATUS_SUCCESS) {
        cusolverDnDestroy(handle);
        return std::unexpected(DeviceError{17, "getrs failed"});
    }
    cusolverDnDestroy(handle);

    copy_device_to_host(d_B, h_B.data(), b_bytes);
    Matrix<double> x(static_cast<size_t>(n), 1);
    for (int i = 0; i < n; ++i) {
        x(static_cast<size_t>(i), 0) = h_B[static_cast<size_t>(i)];
    }
    return x;
#else
    (void)device;
    return std::unexpected(DeviceError{18, "cuda disabled at build time"});
#endif
}

} // namespace ms::cuda
