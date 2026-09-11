// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The sgemm dispatcher, and the scalar fallback under it.
//
// Same shape as blas_dgemm.cpp: ask the widest kernel whether it is available at
// RUNTIME and whether the problem is big enough to repay packing, and otherwise
// take a loop that is at least ordered to write C contiguously. The float path had
// neither before this: `matmul` fell to a triple loop written i, k, j, which on a
// column-major C strides the innermost index by ldc.

#include "ms/core/attributes.hpp"
#include "ms/cpu/blas.hpp"
#include "ms/cpu/blas_kernel.hpp"

#include <algorithm>
#include <cstddef>

namespace ms::cpu::blas {

namespace {

MS_FORCEINLINE void scale_matrix_f(int m, int n, float beta, float* C, int ldc) {
    if (beta == 1.0F) {
        return;
    }
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    for (int j = 0; j < n; ++j) {
        float* col = C + static_cast<std::size_t>(j) * ldc_u;
        if (beta == 0.0F) {
            std::fill(col, col + m, 0.0F);
        } else {
            for (int i = 0; i < m; ++i) {
                col[i] *= beta;
            }
        }
    }
}

/// C = alpha * A * B + beta * C by rank-1 updates, column-major throughout.
///
/// The accumulation order is the same one the packed kernel uses -- over p, into a
/// C column -- so the two agree to the last bit on any input where they agree at
/// all, which is what makes the kernel's correctness testable against this.
MS_FORCEINLINE void sgemm_nn_rank1(
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    float* C,
    int ldc) {
    scale_matrix_f(m, n, beta, C, ldc);
    if (alpha == 0.0F || k == 0) {
        return;
    }
    const std::size_t lda_u = static_cast<std::size_t>(lda);
    const std::size_t ldb_u = static_cast<std::size_t>(ldb);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    for (int j = 0; j < n; ++j) {
        float* c_col = C + static_cast<std::size_t>(j) * ldc_u;
        for (int p = 0; p < k; ++p) {
            const float bpj = B[static_cast<std::size_t>(j) * ldb_u +
                                static_cast<std::size_t>(p)];
            if (bpj == 0.0F) {
                continue;
            }
            const float scale = alpha * bpj;
            const float* a_col = A + static_cast<std::size_t>(p) * lda_u;
            for (int i = 0; i < m; ++i) {
                c_col[i] += scale * a_col[i];
            }
        }
    }
}

void sgemm_nn_dispatch(
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    float* C,
    int ldc) {
    // Widest first, and available() is a runtime question: a binary built with
    // AVX-512 kernels still has to run on hosts whose OS never enabled ZMM state,
    // and the kernels answer through ms::simd::detect_isa(), which checks OSXSAVE
    // and XCR0 rather than trusting CPUID alone.
#if defined(MS_ENABLE_AVX512) && MS_ENABLE_AVX512
    if (avx512::sgemm_available() && avx512::sgemm_worthwhile(m, n, k)) {
        avx512::sgemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    if (avx2::sgemm_available() && avx2::sgemm_worthwhile(m, n, k)) {
        avx2::sgemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    sgemm_nn_rank1(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

float get_a_f(int transa, const float* A, int lda, int i, int p) {
    if (transa == 'N' || transa == 'n') {
        return A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) +
                 static_cast<std::size_t>(i)];
    }
    return A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) +
             static_cast<std::size_t>(p)];
}

float get_b_f(int transb, const float* B, int ldb, int p, int j) {
    if (transb == 'N' || transb == 'n') {
        return B[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb) +
                 static_cast<std::size_t>(p)];
    }
    return B[static_cast<std::size_t>(p) * static_cast<std::size_t>(ldb) +
             static_cast<std::size_t>(j)];
}

void sgemm_generic(
    char transa,
    char transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    float* C,
    int ldc) {
    scale_matrix_f(m, n, beta, C, ldc);
    if (alpha == 0.0F || k == 0) {
        return;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float sum = 0.0F;
            for (int p = 0; p < k; ++p) {
                sum += get_a_f(transa, A, lda, i, p) * get_b_f(transb, B, ldb, p, j);
            }
            C[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) +
              static_cast<std::size_t>(i)] += alpha * sum;
        }
    }
}

} // namespace

void sgemm(
    char transa,
    char transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    float* C,
    int ldc) {
    if (m <= 0 || n <= 0) {
        return;
    }
    const bool a_plain = (transa == 'N' || transa == 'n');
    const bool b_plain = (transb == 'N' || transb == 'n');
    if (a_plain && b_plain) {
        sgemm_nn_dispatch(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    sgemm_generic(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // namespace ms::cpu::blas
