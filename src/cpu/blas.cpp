#include "ms/cpu/blas.hpp"
#include "ms/simd/simd.hpp"

#include <algorithm>
#include <cstddef>

namespace ms::cpu::blas {

namespace {

bool is_no_transpose(char trans) {
    return trans == 'N' || trans == 'n';
}

void scale_vector(int n, double beta, double* y, int incy) {
    if (beta == 1.0) {
        return;
    }
    if (beta == 0.0) {
        if (incy == 1) {
            std::fill(y, y + n, 0.0);
        } else {
            for (int i = 0; i < n; ++i) {
                y[static_cast<std::size_t>(i) * static_cast<std::size_t>(incy)] = 0.0;
            }
        }
        return;
    }
    if (incy == 1) {
        ms::simd::scale(beta, {y, static_cast<std::size_t>(n)}, {y, static_cast<std::size_t>(n)});
        return;
    }
    for (int i = 0; i < n; ++i) {
        y[static_cast<std::size_t>(i) * static_cast<std::size_t>(incy)] *= beta;
    }
}

void dgemv_nn(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    int incx,
    double* y,
    int incy) {
    if (alpha == 0.0) {
        return;
    }

    if (incx == 1 && incy == 1) {
        for (int j = 0; j < n; ++j) {
            const double xj = x[j];
            if (xj == 0.0) {
                continue;
            }
            const double* col = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
            ms::simd::axpy(
                alpha * xj,
                {col, static_cast<std::size_t>(m)},
                {y, static_cast<std::size_t>(m)});
        }
        return;
    }

    for (int j = 0; j < n; ++j) {
        const double xj = x[static_cast<std::size_t>(j) * static_cast<std::size_t>(incx)];
        if (xj == 0.0) {
            continue;
        }
        for (int i = 0; i < m; ++i) {
            y[static_cast<std::size_t>(i) * static_cast<std::size_t>(incy)] +=
                alpha * A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) +
                          static_cast<std::size_t>(i)] *
                xj;
        }
    }
}

void dgemv_t(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    int incx,
    double* y,
    int incy) {
    if (alpha == 0.0) {
        return;
    }

    if (incx == 1 && incy == 1) {
        for (int j = 0; j < n; ++j) {
            const double* col = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
            y[j] += alpha * ms::simd::dot({col, static_cast<std::size_t>(m)}, {x, static_cast<std::size_t>(m)});
        }
        return;
    }

    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < m; ++i) {
            sum += A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                   x[static_cast<std::size_t>(i) * static_cast<std::size_t>(incx)];
        }
        y[static_cast<std::size_t>(j) * static_cast<std::size_t>(incy)] += alpha * sum;
    }
}

} // namespace

double ddot(int n, const double* x, int incx, const double* y, int incy) {
    if (n <= 0 || x == nullptr || y == nullptr) {
        return 0.0;
    }

    if (incx == 1 && incy == 1) {
        return ms::simd::dot({x, static_cast<std::size_t>(n)}, {y, static_cast<std::size_t>(n)});
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[static_cast<std::size_t>(i) * static_cast<std::size_t>(incx)] *
               y[static_cast<std::size_t>(i) * static_cast<std::size_t>(incy)];
    }
    return sum;
}

void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (n <= 0 || alpha == 0.0 || x == nullptr || y == nullptr) {
        return;
    }

    if (incx == 1 && incy == 1) {
        ms::simd::axpy(alpha, {x, static_cast<std::size_t>(n)}, {y, static_cast<std::size_t>(n)});
        return;
    }

    for (int i = 0; i < n; ++i) {
        y[static_cast<std::size_t>(i) * static_cast<std::size_t>(incy)] +=
            alpha * x[static_cast<std::size_t>(i) * static_cast<std::size_t>(incx)];
    }
}

void dgemv(
    char trans,
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    int incx,
    double beta,
    double* y,
    int incy) {
    if (m <= 0 || n <= 0 || A == nullptr || x == nullptr || y == nullptr) {
        return;
    }

    const int y_len = is_no_transpose(trans) ? m : n;
    scale_vector(y_len, beta, y, incy);

    if (is_no_transpose(trans)) {
        dgemv_nn(m, n, alpha, A, lda, x, incx, y, incy);
        return;
    }

    dgemv_t(m, n, alpha, A, lda, x, incx, y, incy);
}

} // namespace ms::cpu::blas
