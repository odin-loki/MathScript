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

__forceinline __m512d load_b_panel(
    int j0,
    int p,
    const double* B,
    std::size_t ldb_u) {
    const std::size_t j0_u = static_cast<std::size_t>(j0);
    const std::size_t p_u = static_cast<std::size_t>(p);
    return _mm512_set_pd(
        B[(j0_u + 7) * ldb_u + p_u],
        B[(j0_u + 6) * ldb_u + p_u],
        B[(j0_u + 5) * ldb_u + p_u],
        B[(j0_u + 4) * ldb_u + p_u],
        B[(j0_u + 3) * ldb_u + p_u],
        B[(j0_u + 2) * ldb_u + p_u],
        B[(j0_u + 1) * ldb_u + p_u],
        B[j0_u * ldb_u + p_u]);
}

__forceinline void store_acc_row(
    __m512d acc,
    int i0,
    int j0,
    double* C,
    std::size_t ldc_u) {
    alignas(64) double tmp[kColBlock] = {};
    _mm512_store_pd(tmp, acc);
    const std::size_t i_u = static_cast<std::size_t>(i0);
    const std::size_t j0_u = static_cast<std::size_t>(j0);
    C[(j0_u + 0) * ldc_u + i_u] += tmp[0];
    C[(j0_u + 1) * ldc_u + i_u] += tmp[1];
    C[(j0_u + 2) * ldc_u + i_u] += tmp[2];
    C[(j0_u + 3) * ldc_u + i_u] += tmp[3];
    C[(j0_u + 4) * ldc_u + i_u] += tmp[4];
    C[(j0_u + 5) * ldc_u + i_u] += tmp[5];
    C[(j0_u + 6) * ldc_u + i_u] += tmp[6];
    C[(j0_u + 7) * ldc_u + i_u] += tmp[7];
}

__forceinline void micro_kernel_4x8(
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

    const std::size_t lda_u = static_cast<std::size_t>(lda);
    const std::size_t ldb_u = static_cast<std::size_t>(ldb);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    const std::size_t i0_u = static_cast<std::size_t>(i0);

    int p = 0;
    for (; p + 1 < k; p += 2) {
        const __m512d bv0 = load_b_panel(j0, p, B, ldb_u);
        const __m512d bv1 = load_b_panel(j0, p + 1, B, ldb_u);

        const std::size_t p0_u = static_cast<std::size_t>(p);
        const std::size_t p1_u = static_cast<std::size_t>(p + 1);

        const double a00 = alpha * A[p0_u * lda_u + i0_u + 0];
        const double a01 = alpha * A[p1_u * lda_u + i0_u + 0];
        const double a10 = alpha * A[p0_u * lda_u + i0_u + 1];
        const double a11 = alpha * A[p1_u * lda_u + i0_u + 1];
        const double a20 = alpha * A[p0_u * lda_u + i0_u + 2];
        const double a21 = alpha * A[p1_u * lda_u + i0_u + 2];
        const double a30 = alpha * A[p0_u * lda_u + i0_u + 3];
        const double a31 = alpha * A[p1_u * lda_u + i0_u + 3];

        acc0 = _mm512_fmadd_pd(_mm512_set1_pd(a00), bv0, acc0);
        acc0 = _mm512_fmadd_pd(_mm512_set1_pd(a01), bv1, acc0);
        acc1 = _mm512_fmadd_pd(_mm512_set1_pd(a10), bv0, acc1);
        acc1 = _mm512_fmadd_pd(_mm512_set1_pd(a11), bv1, acc1);
        acc2 = _mm512_fmadd_pd(_mm512_set1_pd(a20), bv0, acc2);
        acc2 = _mm512_fmadd_pd(_mm512_set1_pd(a21), bv1, acc2);
        acc3 = _mm512_fmadd_pd(_mm512_set1_pd(a30), bv0, acc3);
        acc3 = _mm512_fmadd_pd(_mm512_set1_pd(a31), bv1, acc3);
    }

    for (; p < k; ++p) {
        const __m512d bv = load_b_panel(j0, p, B, ldb_u);
        const std::size_t p_u = static_cast<std::size_t>(p);

        acc0 = _mm512_fmadd_pd(
            _mm512_set1_pd(alpha * A[p_u * lda_u + i0_u + 0]),
            bv,
            acc0);
        acc1 = _mm512_fmadd_pd(
            _mm512_set1_pd(alpha * A[p_u * lda_u + i0_u + 1]),
            bv,
            acc1);
        acc2 = _mm512_fmadd_pd(
            _mm512_set1_pd(alpha * A[p_u * lda_u + i0_u + 2]),
            bv,
            acc2);
        acc3 = _mm512_fmadd_pd(
            _mm512_set1_pd(alpha * A[p_u * lda_u + i0_u + 3]),
            bv,
            acc3);
    }

    store_acc_row(acc0, i0, j0, C, ldc_u);
    store_acc_row(acc1, i0 + 1, j0, C, ldc_u);
    store_acc_row(acc2, i0 + 2, j0, C, ldc_u);
    store_acc_row(acc3, i0 + 3, j0, C, ldc_u);
}

__forceinline void rank1_tail(
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
    const std::size_t lda_u = static_cast<std::size_t>(lda);
    const std::size_t ldb_u = static_cast<std::size_t>(ldb);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    const std::size_t i0_u = static_cast<std::size_t>(i0);
    const std::size_t j0_u = static_cast<std::size_t>(j0);

    for (int j = 0; j < n_count; ++j) {
        const std::size_t col_u = j0_u + static_cast<std::size_t>(j);
        for (int p = 0; p < k; ++p) {
            const std::size_t p_u = static_cast<std::size_t>(p);
            const double bpj = B[col_u * ldb_u + p_u];
            if (bpj == 0.0) {
                continue;
            }
            const double scale = alpha * bpj;
            const double* a_col = A + p_u * lda_u + i0_u;
            double* c_col = C + col_u * ldc_u + i0_u;
            switch (m_count) {
            case 1:
                c_col[0] += scale * a_col[0];
                break;
            case 2:
                c_col[0] += scale * a_col[0];
                c_col[1] += scale * a_col[1];
                break;
            case 3:
                c_col[0] += scale * a_col[0];
                c_col[1] += scale * a_col[1];
                c_col[2] += scale * a_col[2];
                break;
            default:
                for (int i = 0; i < m_count; ++i) {
                    c_col[static_cast<std::size_t>(i)] +=
                        scale * a_col[static_cast<std::size_t>(i)];
                }
                break;
            }
        }
    }
}

__forceinline void scale_matrix(int m, int n, double beta, double* C, int ldc) {
    if (beta == 1.0) {
        return;
    }
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    if (beta == 0.0) {
        for (int j = 0; j < n; ++j) {
            double* col = C + static_cast<std::size_t>(j) * ldc_u;
            for (int i = 0; i < m; ++i) {
                col[i] = 0.0;
            }
        }
        return;
    }
    for (int j = 0; j < n; ++j) {
        double* col = C + static_cast<std::size_t>(j) * ldc_u;
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

    for (int i = 0; i < m_block; i += kRowBlock) {
        for (int j = 0; j < n_block; j += kColBlock) {
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
