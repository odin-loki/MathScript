#include "ms/cuda/elementwise.hpp"
#include "ms/cuda/buffer.hpp"
#include "ms/simd/simd.hpp"
#include <algorithm>

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include "kernels.hpp"
#endif

namespace ms::cuda {

void add_inplace(std::span<double> a, std::span<const double> b, double alpha) {
    const size_t n = (std::min)(a.size(), b.size());
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (available() && try_device_add_inplace(a.data(), b.data(), alpha, n)) {
        return;
    }
#endif
    simd::axpy(alpha, b.first(n), a.first(n));
}

void fill(std::span<double> out, double value) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (available() && try_device_fill(out.data(), value, out.size())) {
        return;
    }
#endif
    simd::fill(out, value);
}

void mul_inplace(std::span<double> a, std::span<const double> b) {
    const size_t n = (std::min)(a.size(), b.size());
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (available() && try_device_mul_inplace(a.data(), b.data(), n)) {
        return;
    }
#endif
    simd::mul(a.first(n), b.first(n), a.first(n));
}

void scale(std::span<double> a, double alpha) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (available() && try_device_scale(a.data(), alpha, a.size())) {
        return;
    }
#endif
    simd::scale(alpha, a, a);
}

} // namespace ms::cuda
