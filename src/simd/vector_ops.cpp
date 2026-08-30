#include "ms/simd/simd.hpp"
#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstdlib>

namespace ms::simd {

namespace {

using batch_t = xsimd::batch<double>;
constexpr size_t kWidth = batch_t::size;

const IsaFeatures& cached_isa() {
    static IsaFeatures features = detect_isa();
    return features;
}

bool force_scalar() {
    const char* force = std::getenv("MS_SIMD_FORCE_SCALAR");
    return force != nullptr && force[0] != '\0' && force[0] != '0';
}

Kernel active_kernel() {
    if (force_scalar()) {
        return Kernel::Scalar;
    }
#if defined(MS_ENABLE_AVX512) && MS_ENABLE_AVX512
    if (cached_isa().avx512f) {
        return Kernel::Avx512;
    }
#endif
    if (cached_isa().avx2) {
        return Kernel::Avx2;
    }
    return Kernel::Scalar;
}

bool use_vector_kernels() {
    return active_kernel() != Kernel::Scalar;
}

void add_scalar(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] + b[i];
    }
}

void add_avx2(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    size_t i = 0;
    const size_t n = out.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t va = batch_t::load_unaligned(a.data() + i);
        const batch_t vb = batch_t::load_unaligned(b.data() + i);
        (va + vb).store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub_scalar(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] - b[i];
    }
}

void sub_avx2(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    size_t i = 0;
    const size_t n = out.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t va = batch_t::load_unaligned(a.data() + i);
        const batch_t vb = batch_t::load_unaligned(b.data() + i);
        (va - vb).store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul_scalar(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] * b[i];
    }
}

void mul_avx2(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    size_t i = 0;
    const size_t n = out.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t va = batch_t::load_unaligned(a.data() + i);
        const batch_t vb = batch_t::load_unaligned(b.data() + i);
        (va * vb).store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void scale_scalar(double alpha, std::span<const double> x, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = alpha * x[i];
    }
}

void scale_avx2(double alpha, std::span<const double> x, std::span<double> out) {
    const batch_t va(alpha);
    size_t i = 0;
    const size_t n = out.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t vx = batch_t::load_unaligned(x.data() + i);
        (va * vx).store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] = alpha * x[i];
    }
}

void axpy_scalar(double alpha, std::span<const double> x, std::span<double> y) {
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

void axpy_avx2(double alpha, std::span<const double> x, std::span<double> y) {
    const batch_t va(alpha);
    size_t i = 0;
    const size_t n = y.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t vx = batch_t::load_unaligned(x.data() + i);
        const batch_t vy = batch_t::load_unaligned(y.data() + i);
        xsimd::fma(va, vx, vy).store_unaligned(y.data() + i);
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

double dot_scalar(std::span<const double> a, std::span<const double> b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double dot_avx2(std::span<const double> a, std::span<const double> b) {
    batch_t acc(0.0);
    size_t i = 0;
    const size_t n = a.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t va = batch_t::load_unaligned(a.data() + i);
        const batch_t vb = batch_t::load_unaligned(b.data() + i);
        acc = xsimd::fma(va, vb, acc);
    }
    double sum = xsimd::reduce_add(acc);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double sum_scalar(std::span<const double> x) {
    double total = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        total += x[i];
    }
    return total;
}

double sum_avx2(std::span<const double> x) {
    batch_t acc(0.0);
    size_t i = 0;
    const size_t n = x.size();
    for (; i + kWidth <= n; i += kWidth) {
        acc = acc + batch_t::load_unaligned(x.data() + i);
    }
    double total = xsimd::reduce_add(acc);
    for (; i < n; ++i) {
        total += x[i];
    }
    return total;
}

double sum_squares_scalar(std::span<const double> x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

double sum_squares_avx2(std::span<const double> x) {
    batch_t acc(0.0);
    size_t i = 0;
    const size_t n = x.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t vx = batch_t::load_unaligned(x.data() + i);
        acc = xsimd::fma(vx, vx, acc);
    }
    double sum = xsimd::reduce_add(acc);
    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

double norm_l2_scalar(std::span<const double> x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

double norm_l2_avx2(std::span<const double> x) {
    batch_t acc(0.0);
    size_t i = 0;
    const size_t n = x.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t vx = batch_t::load_unaligned(x.data() + i);
        acc = xsimd::fma(vx, vx, acc);
    }
    double sum = xsimd::reduce_add(acc);
    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

void abs_scalar(std::span<const double> x, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::abs(x[i]);
    }
}

void abs_avx2(std::span<const double> x, std::span<double> out) {
    size_t i = 0;
    const size_t n = out.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t vx = batch_t::load_unaligned(x.data() + i);
        xsimd::abs(vx).store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] = std::abs(x[i]);
    }
}

void exp_map_scalar(std::span<const double> x, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::exp(x[i]);
    }
}

void exp_map_avx2(std::span<const double> x, std::span<double> out) {
    size_t i = 0;
    const size_t n = out.size();
    for (; i + kWidth <= n; i += kWidth) {
        const batch_t vx = batch_t::load_unaligned(x.data() + i);
        xsimd::exp(vx).store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] = std::exp(x[i]);
    }
}

void fill_scalar(std::span<double> out, double value) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = value;
    }
}

void fill_avx2(std::span<double> out, double value) {
    const batch_t bv(value);
    size_t i = 0;
    const size_t n = out.size();
    for (; i + kWidth <= n; i += kWidth) {
        bv.store_unaligned(out.data() + i);
    }
    for (; i < n; ++i) {
        out[i] = value;
    }
}

} // namespace

DispatchInfo dispatch_info() {
    DispatchInfo info;
    info.isa = cached_isa();
    info.active = active_kernel();
    info.batch_width = batch_width();
    return info;
}

std::size_t batch_width() {
    if (active_kernel() == Kernel::Scalar) {
        return 1;
    }
    return kWidth;
}

void add(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    if (use_vector_kernels()) {
        add_avx2(a, b, out);
    } else {
        add_scalar(a, b, out);
    }
}

void sub(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    if (use_vector_kernels()) {
        sub_avx2(a, b, out);
    } else {
        sub_scalar(a, b, out);
    }
}

void mul(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    if (use_vector_kernels()) {
        mul_avx2(a, b, out);
    } else {
        mul_scalar(a, b, out);
    }
}

void scale(double alpha, std::span<const double> x, std::span<double> out) {
    if (use_vector_kernels()) {
        scale_avx2(alpha, x, out);
    } else {
        scale_scalar(alpha, x, out);
    }
}

void axpy(double alpha, std::span<const double> x, std::span<double> y) {
    if (use_vector_kernels()) {
        axpy_avx2(alpha, x, y);
    } else {
        axpy_scalar(alpha, x, y);
    }
}

double dot(std::span<const double> a, std::span<const double> b) {
    if (use_vector_kernels()) {
        return dot_avx2(a, b);
    }
    return dot_scalar(a, b);
}

double sum(std::span<const double> x) {
    if (x.empty()) {
        return 0.0;
    }
    if (use_vector_kernels()) {
        return sum_avx2(x);
    }
    return sum_scalar(x);
}

double sum_squares(std::span<const double> x) {
    if (x.empty()) {
        return 0.0;
    }
    if (use_vector_kernels()) {
        return sum_squares_avx2(x);
    }
    return sum_squares_scalar(x);
}

double norm_l2(std::span<const double> x) {
    if (x.empty()) {
        return 0.0;
    }
    if (use_vector_kernels()) {
        return norm_l2_avx2(x);
    }
    return norm_l2_scalar(x);
}

void abs(std::span<const double> x, std::span<double> out) {
    if (use_vector_kernels()) {
        abs_avx2(x, out);
    } else {
        abs_scalar(x, out);
    }
}

void exp_map(std::span<const double> x, std::span<double> out) {
    if (use_vector_kernels()) {
        exp_map_avx2(x, out);
    } else {
        exp_map_scalar(x, out);
    }
}

void fill(std::span<double> out, double value) {
    if (use_vector_kernels()) {
        fill_avx2(out, value);
    } else {
        fill_scalar(out, value);
    }
}

std::vector<double> gemv(
    std::span<const double> A,
    size_t rows,
    size_t cols,
    std::span<const double> x) {
    std::vector<double> y(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        y[i] = dot(A.subspan(i * cols, cols), x);
    }
    return y;
}

} // namespace ms::simd
