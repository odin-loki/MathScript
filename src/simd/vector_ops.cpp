#include "ms/simd/simd.hpp"
#include <cmath>
#include <cstdlib>
#include <immintrin.h>

namespace ms::simd {

namespace {

const IsaFeatures& cached_isa() {
    static IsaFeatures features = detect_isa();
    return features;
}

void add_scalar(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] + b[i];
    }
}

void add_avx2(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    size_t i = 0;
    const size_t n = out.size();
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a.data() + i);
        const __m256d vb = _mm256_loadu_pd(b.data() + i);
        _mm256_storeu_pd(out.data() + i, _mm256_add_pd(va, vb));
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
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a.data() + i);
        const __m256d vb = _mm256_loadu_pd(b.data() + i);
        _mm256_storeu_pd(out.data() + i, _mm256_sub_pd(va, vb));
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
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a.data() + i);
        const __m256d vb = _mm256_loadu_pd(b.data() + i);
        _mm256_storeu_pd(out.data() + i, _mm256_mul_pd(va, vb));
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
    const __m256d va = _mm256_set1_pd(alpha);
    size_t i = 0;
    const size_t n = out.size();
    for (; i + 4 <= n; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x.data() + i);
        _mm256_storeu_pd(out.data() + i, _mm256_mul_pd(va, vx));
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
    const __m256d va = _mm256_set1_pd(alpha);
    size_t i = 0;
    const size_t n = y.size();
    for (; i + 4 <= n; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x.data() + i);
        const __m256d vy = _mm256_loadu_pd(y.data() + i);
        _mm256_storeu_pd(y.data() + i, _mm256_fmadd_pd(va, vx, vy));
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
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;
    const size_t n = a.size();
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a.data() + i);
        const __m256d vb = _mm256_loadu_pd(b.data() + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_pd(va, vb, acc);
#else
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
#endif
    }
    alignas(32) double parts[4];
    _mm256_store_pd(parts, acc);
    double sum = parts[0] + parts[1] + parts[2] + parts[3];
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

double norm_l2_scalar(std::span<const double> x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

double norm_l2_avx2(std::span<const double> x) {
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;
    const size_t n = x.size();
    for (; i + 4 <= n; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x.data() + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_pd(vx, vx, acc);
#else
        acc = _mm256_add_pd(acc, _mm256_mul_pd(vx, vx));
#endif
    }
    alignas(32) double parts[4];
    _mm256_store_pd(parts, acc);
    double sum = parts[0] + parts[1] + parts[2] + parts[3];
    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

double sum_avx2(std::span<const double> x) {
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;
    const size_t n = x.size();
    for (; i + 4 <= n; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x.data() + i);
        acc = _mm256_add_pd(acc, vx);
    }
    alignas(32) double parts[4];
    _mm256_store_pd(parts, acc);
    double total = parts[0] + parts[1] + parts[2] + parts[3];
    for (; i < n; ++i) {
        total += x[i];
    }
    return total;
}

void abs_scalar(std::span<const double> x, std::span<double> out) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::abs(x[i]);
    }
}

void abs_avx2(std::span<const double> x, std::span<double> out) {
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    size_t i = 0;
    const size_t n = out.size();
    for (; i + 4 <= n; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x.data() + i);
        _mm256_storeu_pd(out.data() + i, _mm256_andnot_pd(sign_mask, vx));
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
    for (; i + 4 <= n; i += 4) {
        alignas(32) double in[4];
        alignas(32) double tmp[4];
        _mm256_store_pd(in, _mm256_loadu_pd(x.data() + i));
        for (int k = 0; k < 4; ++k) {
            tmp[k] = std::exp(in[k]);
        }
        _mm256_storeu_pd(out.data() + i, _mm256_load_pd(tmp));
    }
    for (; i < n; ++i) {
        out[i] = std::exp(x[i]);
    }
}

Kernel active_kernel() {
    if (const char* force = std::getenv("MS_SIMD_FORCE_SCALAR"); force != nullptr && force[0] != '\0' && force[0] != '0') {
        return Kernel::Scalar;
    }
    if (cached_isa().avx2) {
        return Kernel::Avx2;
    }
    return Kernel::Scalar;
}

} // namespace

DispatchInfo dispatch_info() {
    DispatchInfo info;
    info.isa = cached_isa();
    info.active = active_kernel();
    return info;
}

void add(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    if (active_kernel() == Kernel::Avx2) {
        add_avx2(a, b, out);
    } else {
        add_scalar(a, b, out);
    }
}

void sub(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    if (active_kernel() == Kernel::Avx2) {
        sub_avx2(a, b, out);
    } else {
        sub_scalar(a, b, out);
    }
}

void mul(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    if (active_kernel() == Kernel::Avx2) {
        mul_avx2(a, b, out);
    } else {
        mul_scalar(a, b, out);
    }
}

void scale(double alpha, std::span<const double> x, std::span<double> out) {
    if (active_kernel() == Kernel::Avx2) {
        scale_avx2(alpha, x, out);
    } else {
        scale_scalar(alpha, x, out);
    }
}

void axpy(double alpha, std::span<const double> x, std::span<double> y) {
    if (active_kernel() == Kernel::Avx2) {
        axpy_avx2(alpha, x, y);
    } else {
        axpy_scalar(alpha, x, y);
    }
}

double dot(std::span<const double> a, std::span<const double> b) {
    if (active_kernel() == Kernel::Avx2) {
        return dot_avx2(a, b);
    }
    return dot_scalar(a, b);
}

double sum(std::span<const double> x) {
    if (x.empty()) {
        return 0.0;
    }
    if (active_kernel() == Kernel::Avx2) {
        return sum_avx2(x);
    }
    return sum_scalar(x);
}

double sum_squares(std::span<const double> x) {
    if (x.empty()) {
        return 0.0;
    }
    return dot(x, x);
}

double norm_l2(std::span<const double> x) {
    if (x.empty()) {
        return 0.0;
    }
    if (active_kernel() == Kernel::Avx2) {
        return norm_l2_avx2(x);
    }
    return norm_l2_scalar(x);
}

void abs(std::span<const double> x, std::span<double> out) {
    if (active_kernel() == Kernel::Avx2) {
        abs_avx2(x, out);
    } else {
        abs_scalar(x, out);
    }
}

void exp_map(std::span<const double> x, std::span<double> out) {
    if (active_kernel() == Kernel::Avx2) {
        exp_map_avx2(x, out);
    } else {
        exp_map_scalar(x, out);
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
