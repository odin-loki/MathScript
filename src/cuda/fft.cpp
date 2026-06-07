#include "ms/cuda/fft.hpp"
#include "ms/cuda/buffer.hpp"
#include <cmath>

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include <cufft.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms::cuda {

namespace {

struct ComplexPair {
    double x;
    double y;
};

size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

Result<std::vector<std::complex<double>>> fft_cpu(const std::vector<double>& x) {
    const size_t n = next_power_of_two(x.size());
    std::vector<std::complex<double>> input(n);
    for (size_t i = 0; i < x.size(); ++i) {
        input[i] = {x[i], 0.0};
    }

    std::vector<std::complex<double>> out(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t t = 0; t < n; ++t) {
            const double angle = -2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(n);
            sum += input[t] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        out[k] = sum;
    }
    return out;
}

} // namespace

Result<std::vector<std::complex<double>>> fft(const std::vector<double>& x) {
    if (x.empty()) {
        return std::vector<std::complex<double>>{};
    }

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!available()) {
        return fft_cpu(x);
    }

    const size_t n = next_power_of_two(x.size());
    const size_t bytes = n * sizeof(ComplexPair);

    std::vector<ComplexPair> host_in(n);
    for (size_t i = 0; i < x.size(); ++i) {
        host_in[i].x = x[i];
        host_in[i].y = 0.0;
    }

    auto d_in = make_device_buffer(bytes, 0);
    auto d_out = make_device_buffer(bytes, 0);
    if (d_in.data == nullptr || d_out.data == nullptr) {
        return fft_cpu(x);
    }

    copy_host_to_device(host_in.data(), d_in, bytes);

    cufftHandle plan = 0;
    if (cufftPlan1d(&plan, static_cast<int>(n), CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
        return fft_cpu(x);
    }

    const cufftResult exec = cufftExecZ2Z(
        plan,
        reinterpret_cast<cufftDoubleComplex*>(d_in.data),
        reinterpret_cast<cufftDoubleComplex*>(d_out.data),
        CUFFT_FORWARD);
    cufftDestroy(plan);

    if (exec != CUFFT_SUCCESS) {
        return fft_cpu(x);
    }

    std::vector<ComplexPair> host_out(n);
    copy_device_to_host(d_out, host_out.data(), bytes);

    std::vector<std::complex<double>> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = {host_out[i].x, host_out[i].y};
    }
    return out;
#else
    return fft_cpu(x);
#endif
}

Result<std::vector<double>> ifft(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return std::vector<double>{};
    }

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!available()) {
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            out[i] = x[i].real();
        }
        return out;
    }

    const size_t n = x.size();
    const size_t bytes = n * sizeof(ComplexPair);

    std::vector<ComplexPair> host_in(n);
    for (size_t i = 0; i < n; ++i) {
        host_in[i].x = x[i].real();
        host_in[i].y = x[i].imag();
    }

    auto d_in = make_device_buffer(bytes, 0);
    auto d_out = make_device_buffer(bytes, 0);
    if (d_in.data == nullptr || d_out.data == nullptr) {
        return std::unexpected(DeviceError{1, "cuda ifft allocation failed"});
    }

    copy_host_to_device(host_in.data(), d_in, bytes);

    cufftHandle plan = 0;
    if (cufftPlan1d(&plan, static_cast<int>(n), CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
        return std::unexpected(DeviceError{2, "cufft plan failed"});
    }

    const cufftResult exec = cufftExecZ2Z(
        plan,
        reinterpret_cast<cufftDoubleComplex*>(d_in.data),
        reinterpret_cast<cufftDoubleComplex*>(d_out.data),
        CUFFT_INVERSE);
    cufftDestroy(plan);
    if (exec != CUFFT_SUCCESS) {
        return std::unexpected(DeviceError{3, "cufft exec failed"});
    }

    std::vector<ComplexPair> host_out(n);
    copy_device_to_host(d_out, host_out.data(), bytes);

    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = host_out[i].x / static_cast<double>(n);
    }
    return out;
#else
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i].real();
    }
    return out;
#endif
}

} // namespace ms::cuda
