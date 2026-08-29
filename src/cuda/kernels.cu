#include "kernels.hpp"

#include <cuda_runtime.h>

namespace ms::cuda {

namespace {

constexpr int kBlock = 256;

__global__ void add_inplace_k(double* a, const double* b, double alpha, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i < n) {
        a[i] += alpha * b[i];
    }
}

__global__ void fill_k(double* out, double value, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i < n) {
        out[i] = value;
    }
}

__global__ void mul_inplace_k(double* a, const double* b, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i < n) {
        a[i] *= b[i];
    }
}

__global__ void scale_k(double* a, double alpha, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i < n) {
        a[i] *= alpha;
    }
}

int grid(std::size_t n) {
    return static_cast<int>(
        (n + static_cast<std::size_t>(kBlock) - 1) / static_cast<std::size_t>(kBlock));
}

bool launch_ok() {
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

} // namespace

bool try_device_add_inplace(double* a, const double* b, double alpha, std::size_t n) {
    if (n == 0) {
        return true;
    }
    double* da = nullptr;
    double* db = nullptr;
    const std::size_t bytes = n * sizeof(double);
    if (cudaMalloc(&da, bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&db, bytes) != cudaSuccess) {
        cudaFree(da);
        return false;
    }
    bool ok = cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        add_inplace_k<<<grid(n), kBlock>>>(da, db, alpha, n);
        ok = launch_ok() && cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(da);
    cudaFree(db);
    return ok;
}

bool try_device_fill(double* out, double value, std::size_t n) {
    if (n == 0) {
        return true;
    }
    double* d = nullptr;
    const std::size_t bytes = n * sizeof(double);
    if (cudaMalloc(&d, bytes) != cudaSuccess) {
        return false;
    }
    fill_k<<<grid(n), kBlock>>>(d, value, n);
    const bool ok = launch_ok() && cudaMemcpy(out, d, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    return ok;
}

bool try_device_mul_inplace(double* a, const double* b, std::size_t n) {
    if (n == 0) {
        return true;
    }
    double* da = nullptr;
    double* db = nullptr;
    const std::size_t bytes = n * sizeof(double);
    if (cudaMalloc(&da, bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&db, bytes) != cudaSuccess) {
        cudaFree(da);
        return false;
    }
    bool ok = cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        mul_inplace_k<<<grid(n), kBlock>>>(da, db, n);
        ok = launch_ok() && cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(da);
    cudaFree(db);
    return ok;
}

bool try_device_scale(double* a, double alpha, std::size_t n) {
    if (n == 0) {
        return true;
    }
    double* da = nullptr;
    const std::size_t bytes = n * sizeof(double);
    if (cudaMalloc(&da, bytes) != cudaSuccess) {
        return false;
    }
    bool ok = cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        scale_k<<<grid(n), kBlock>>>(da, alpha, n);
        ok = launch_ok() && cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(da);
    return ok;
}

} // namespace ms::cuda
