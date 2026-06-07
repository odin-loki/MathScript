#include "ms/cpu/blas.hpp"
#include "ms/simd/simd.hpp"

namespace ms::cpu::blas {

void dger(
    int m,
    int n,
    double alpha,
    const double* x,
    int incx,
    const double* y,
    int incy,
    double* A,
    int lda) {
    if (m <= 0 || n <= 0 || alpha == 0.0 || A == nullptr || x == nullptr || y == nullptr) {
        return;
    }

    if (incx == 1 && incy == 1) {
        for (int j = 0; j < n; ++j) {
            const double yj = y[j];
            if (yj == 0.0) {
                continue;
            }
            double* col = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
            ms::simd::axpy(
                alpha * yj,
                {x, static_cast<std::size_t>(m)},
                {col, static_cast<std::size_t>(m)});
        }
        return;
    }

    for (int j = 0; j < n; ++j) {
        const double yj = y[static_cast<std::size_t>(j) * static_cast<std::size_t>(incy)];
        if (yj == 0.0) {
            continue;
        }
        for (int i = 0; i < m; ++i) {
            A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] +=
                alpha * x[static_cast<std::size_t>(i) * static_cast<std::size_t>(incx)] * yj;
        }
    }
}

} // namespace ms::cpu::blas
