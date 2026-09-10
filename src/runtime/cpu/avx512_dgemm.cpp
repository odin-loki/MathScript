// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// AVX-512 dgemm.
//
// The previous version of this file had two problems that would have been copied
// into every kernel modelled on it.
//
// It read B through a gather written as a set:
//
//     _mm512_set_pd(B[(j0+7)*ldb + p], B[(j0+6)*ldb + p], ... )
//
// which is eight strided scalar loads per vector, executed on every iteration of
// the innermost loop. Those loads still happen -- B is column-major and the kernel
// needs a row of it -- but they now happen once per panel during packing rather
// than k times per tile. Storing C had the mirror-image problem: the accumulator
// held eight *columns* for one row, so writing it back meant spilling to a stack
// buffer and scattering eight scalars.
//
// And it had a 4x8 micro-kernel with no blocking above it, which is fine while the
// operands fit in cache and stops scaling the moment they do not.
//
// Both are fixed by orienting the vectors along rows -- contiguous in a
// column-major matrix, for both A and C -- and putting the Goto/BLIS loop nest in
// gemm_blocking.hpp underneath. The micro-kernel is 16x8: 16 rows is two 8-wide
// vectors, and 8 columns gives 16 accumulators, 2 A vectors and 1 broadcast, 19 of
// the 32 available zmm registers.
//
// Note that CI builds with -DMS_ENABLE_AVX512=OFF, so this translation unit is not
// exercised there. The differential tests in tests/unit/linalg/test_dgemm_kernels
// cover it on a host that has the ISA.

#include "ms/core/attributes.hpp"
#include "ms/cpu/blas_kernel.hpp"
#include "ms/simd/isa.hpp"

#include "gemm_blocking.hpp"

#include <immintrin.h>

namespace ms::cpu::blas::avx512 {

namespace {

constexpr int kMr = 16;
constexpr int kNr = 8;

// A block of A is 384 * 256 * 8 = 768 KiB, sized for a 1-2 MiB L2; the B block is
// 256 * 2048 * 8 = 4 MiB and streams from L3. Defaults rather than measurements.
constexpr detail::BlockSizes kBlocks{384, 256, 2048};

constexpr long long kMinWork = 64LL * 64LL * 64LL;

/// One 16x8 tile: C[0..15][0..7] += alpha * Apanel(16 x kc) * Bpanel(kc x 8).
MS_FORCEINLINE void micro_kernel_16x8(
    int kc,
    double alpha,
    const double* __restrict Apanel,
    const double* __restrict Bpanel,
    double* __restrict C,
    int ldc) {
    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
    __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
    __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
    __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();

    for (int p = 0; p < kc; ++p) {
        const __m512d a0 = _mm512_load_pd(Apanel + 0);
        const __m512d a1 = _mm512_load_pd(Apanel + 8);
        Apanel += kMr;

        __m512d b = _mm512_set1_pd(Bpanel[0]);
        c00 = _mm512_fmadd_pd(a0, b, c00);
        c01 = _mm512_fmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bpanel[1]);
        c10 = _mm512_fmadd_pd(a0, b, c10);
        c11 = _mm512_fmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bpanel[2]);
        c20 = _mm512_fmadd_pd(a0, b, c20);
        c21 = _mm512_fmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bpanel[3]);
        c30 = _mm512_fmadd_pd(a0, b, c30);
        c31 = _mm512_fmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bpanel[4]);
        c40 = _mm512_fmadd_pd(a0, b, c40);
        c41 = _mm512_fmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bpanel[5]);
        c50 = _mm512_fmadd_pd(a0, b, c50);
        c51 = _mm512_fmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bpanel[6]);
        c60 = _mm512_fmadd_pd(a0, b, c60);
        c61 = _mm512_fmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bpanel[7]);
        c70 = _mm512_fmadd_pd(a0, b, c70);
        c71 = _mm512_fmadd_pd(a1, b, c71);
        Bpanel += kNr;
    }

    const __m512d va = _mm512_set1_pd(alpha);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);

#define MS_STORE_COL(idx, lo, hi)                                                 \
    do {                                                                          \
        double* col = C + static_cast<std::size_t>(idx) * ldc_u;                  \
        _mm512_storeu_pd(col, _mm512_fmadd_pd(va, (lo), _mm512_loadu_pd(col)));   \
        _mm512_storeu_pd(col + 8,                                                 \
                         _mm512_fmadd_pd(va, (hi), _mm512_loadu_pd(col + 8)));    \
    } while (0)

    MS_STORE_COL(0, c00, c01);
    MS_STORE_COL(1, c10, c11);
    MS_STORE_COL(2, c20, c21);
    MS_STORE_COL(3, c30, c31);
    MS_STORE_COL(4, c40, c41);
    MS_STORE_COL(5, c50, c51);
    MS_STORE_COL(6, c60, c61);
    MS_STORE_COL(7, c70, c71);
#undef MS_STORE_COL
}

bool runtime_available() {
    static const bool enabled = ms::simd::detect_isa().avx512f;
    return enabled;
}

} // namespace

bool available() {
    return runtime_available();
}

bool worthwhile(int m, int n, int k) {
    const long long work = static_cast<long long>(m) * static_cast<long long>(n) *
                           static_cast<long long>(k);
    return work >= kMinWork;
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
    detail::scale_c(m, n, beta, C, ldc);
    if (alpha == 0.0 || k == 0) {
        return;
    }
    detail::gemm_blocked<kMr, kNr>(m, n, k, alpha, A, lda, B, ldb, C, ldc, kBlocks,
                                   micro_kernel_16x8);
}

} // namespace ms::cpu::blas::avx512
