#include "ms/cpu/blas_kernel.hpp"
#include "ms/simd/isa.hpp"

#include <immintrin.h>

namespace ms::cpu::blas::avx512 {

namespace {

constexpr int kRowBlock = 4;
constexpr int kColBlock = 8;

bool runtime_available() {
    static const bool enabled = ms::simd::detect_isa().avx512f;
    return enabled;
}

void micro_kernel_4x8(
    int k,
    int i0,
    int j0,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double* C,
    int ldc) {
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();
    alignas(64) double bpack[kColBlock] = {};

    for (int p = 0; p < k; ++p) {
        for (int jj = 0; jj < kColBlock; ++jj) {
            bpack[jj] = B[static_cast<std::size_t>(j0 + jj) * static_cast<std::size_t>(ldb) +
                          static_cast<std::size_t>(p)];
        }
        const __m512d bv = _mm512_load_pd(bpack);
        acc0 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) +
                                                             static_cast<std::size_t>(i0 + 0)]),
                               bv,
                               acc0);
        acc1 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) +
                                                             static_cast<std::size_t>(i0 + 1)]),
                               bv,
                               acc1);
        acc2 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) +
                                                             static_cast<std::size_t>(i0 + 2)]),
                               bv,
                               acc2);
        acc3 = _mm512_fmadd_pd(_mm512_set1_pd(alpha * A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) +
                                                             static_cast<std::size_t>(i0 + 3)]),
                               bv,
                               acc3);
    }

    alignas(64) double tmp[kColBlock] = {};
    _mm512_store_pd(tmp, acc0);
    for (int jj = 0; jj < kColBlock; ++jj) {
        C[static_cast<std::size_t>(j0 + jj) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i0 + 0)] +=
            tmp[jj];
    }
    _mm512_store_pd(tmp, acc1);
    for (int jj = 0; jj < kColBlock; ++jj) {
        C[static_cast<std::size_t>(j0 + jj) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i0 + 1)] +=
            tmp[jj];
    }
    _mm512_store_pd(tmp, acc2);
    for (int jj = 0; jj < kColBlock; ++jj) {
        C[static_cast<std::size_t>(j0 + jj) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i0 + 2)] +=
            tmp[jj];
    }
    _mm512_store_pd(tmp, acc3);
    for (int jj = 0; jj < kColBlock; ++jj) {
        C[static_cast<std::size_t>(j0 + jj) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i0 + 3)] +=
            tmp[jj];
    }
}

void rank1_tail(
    int i0,
    int m_count,
    int j0,
    int n_count,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double* C,
    int ldc) {
    for (int j = 0; j < n_count; ++j) {
        for (int p = 0; p < k; ++p) {
            const double bpj = B[static_cast<std::size_t>(j0 + j) * static_cast<std::size_t>(ldb) +
                                  static_cast<std::size_t>(p)];
            if (bpj == 0.0) {
                continue;
            }
            const double scale = alpha * bpj;
            for (int i = 0; i < m_count; ++i) {
                C[static_cast<std::size_t>(j0 + j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i0 + i)] +=
                    scale * A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i0 + i)];
            }
        }
    }
}

void scale_matrix(int m, int n, double beta, double* C, int ldc) {
    if (beta == 1.0) {
        return;
    }
    if (beta == 0.0) {
        for (int j = 0; j < n; ++j) {
            double* col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc);
            for (int i = 0; i < m; ++i) {
                col[i] = 0.0;
            }
        }
        return;
    }
    for (int j = 0; j < n; ++j) {
        double* col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc);
        for (int i = 0; i < m; ++i) {
            col[i] *= beta;
        }
    }
}

} // namespace

bool available() {
    return runtime_available();
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
    scale_matrix(m, n, beta, C, ldc);
    if (alpha == 0.0 || k == 0) {
        return;
    }

    const int m_block = (m / kRowBlock) * kRowBlock;
    const int n_block = (n / kColBlock) * kColBlock;

    for (int j = 0; j < n_block; j += kColBlock) {
        for (int i = 0; i < m_block; i += kRowBlock) {
            micro_kernel_4x8(k, i, j, alpha, A, lda, B, ldb, C, ldc);
        }
    }

    if (m_block < m && n_block > 0) {
        rank1_tail(m_block, m - m_block, 0, n_block, k, alpha, A, lda, B, ldb, C, ldc);
    }
    if (n_block < n) {
        rank1_tail(0, m, n_block, n - n_block, k, alpha, A, lda, B, ldb, C, ldc);
    }
}

} // namespace ms::cpu::blas::avx512
