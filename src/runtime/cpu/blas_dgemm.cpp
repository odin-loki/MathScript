#include "ms/cpu/blas.hpp"
#include "ms/cpu/blas_kernel.hpp"
#include "ms/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ms::cpu::blas {

namespace {

__forceinline void scale_matrix(int m, int n, double beta, double* C, int ldc) {
    if (beta == 1.0) {
        return;
    }
    if (beta == 0.0) {
        for (int j = 0; j < n; ++j) {
            double* col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc);
            std::fill(col, col + m, 0.0);
        }
        return;
    }
    for (int j = 0; j < n; ++j) {
        double* col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc);
        ms::simd::scale(beta, {col, static_cast<std::size_t>(m)}, {col, static_cast<std::size_t>(m)});
    }
}

__forceinline void dgemm_nn_rank1(
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc) {
    scale_matrix(m, n, beta, C, ldc);
    if (alpha == 0.0 || k == 0) {
        return;
    }

    for (int j = 0; j < n; ++j) {
        double* c_col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc);
        for (int p = 0; p < k; ++p) {
            const double bpj = B[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(p)];
            if (bpj == 0.0) {
                continue;
            }
            const double* a_col = A + static_cast<std::size_t>(p) * static_cast<std::size_t>(lda);
            ms::simd::axpy(
                alpha * bpj,
                {a_col, static_cast<std::size_t>(m)},
                {c_col, static_cast<std::size_t>(m)});
        }
    }
}

void dgemm_nn(
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc) {
#if defined(MS_ENABLE_AVX512) && MS_ENABLE_AVX512
    if (m >= 4 && n >= 8 && k >= 4 && avx512::available()) {
        avx512::dgemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    dgemm_nn_rank1(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

double get_a(int transa, const double* A, int lda, int i, int p) {
    if (transa == 'N' || transa == 'n') {
        return A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
    }
    return A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(p)];
}

double get_b(int transb, const double* B, int ldb, int p, int j) {
    if (transb == 'N' || transb == 'n') {
        return B[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(p)];
    }
    return B[static_cast<std::size_t>(p) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(j)];
}

void dgemm_generic(
    char transa,
    char transb,
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc) {
    scale_matrix(m, n, beta, C, ldc);
    if (alpha == 0.0 || k == 0) {
        return;
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += get_a(transa, A, lda, i, p) * get_b(transb, B, ldb, p, j);
            }
            C[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i)] +=
                alpha * sum;
        }
    }
}

bool is_no_transpose(char trans) {
    return trans == 'N' || trans == 'n';
}

} // namespace

void dgemm(
    char transa,
    char transb,
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (k < 0 || A == nullptr || B == nullptr || C == nullptr) {
        return;
    }

    if (is_no_transpose(transa) && is_no_transpose(transb)) {
        dgemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    dgemm_generic(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // namespace ms::cpu::blas
