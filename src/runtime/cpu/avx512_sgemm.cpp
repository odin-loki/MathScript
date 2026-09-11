// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// AVX-512 sgemm.
//
// The same kernel as avx2_sgemm.cpp with the vector twice as wide, and the same
// shape as the AVX-512 dgemm next door with twice the rows: a zmm holds sixteen
// floats where it holds eight doubles, so a 32x8 micro-kernel is the 16x8 double
// one at single precision.
//
//     16  accumulators   (32 rows / 16 lanes) x 8 columns
//      2  A vectors
//      1  broadcast B scalar
//     --
//     19  of 32
//
// Accumulation is in float, as it is in the AVX2 kernel and as every BLAS sgemm
// does it: the two have to agree, and widening one of them would make the answer
// depend on which ISA the host happens to have.
//
// CI builds with -DMS_ENABLE_AVX512=OFF, so this translation unit is not exercised
// there. `tests/unit/linalg/test_dgemm_kernels.cpp` covers it differentially on a
// host that has the ISA, and skips where it does not.

#include "ms/core/attributes.hpp"
#include "ms/cpu/blas_kernel.hpp"
#include "ms/simd/isa.hpp"

#include "gemm_blocking.hpp"

#include <immintrin.h>

namespace ms::cpu::blas::avx512 {

namespace {

constexpr int kSMr = 32;
constexpr int kSNr = 8;

// The A block is 384 x 512 floats = 768 KiB, the same L2 footprint the double
// kernel takes at twice the depth; the B block is 512 x 2048 x 4 = 4 MiB and
// streams from L3. Defaults rather than measurements, as next door.
constexpr detail::BlockSizes kSBlocks{384, 512, 2048};

constexpr long long kSMinWork = 64LL * 64LL * 64LL;

/// One 32x8 tile: C[0..31][0..7] += alpha * Apanel(32 x kc) * Bpanel(kc x 8).
MS_FORCEINLINE void micro_kernel_32x8(
    int kc,
    float alpha,
    const float* __restrict Apanel,
    const float* __restrict Bpanel,
    float* __restrict C,
    int ldc) {
    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();

    for (int p = 0; p < kc; ++p) {
        const __m512 a0 = _mm512_load_ps(Apanel + 0);
        const __m512 a1 = _mm512_load_ps(Apanel + 16);
        Apanel += kSMr;

        __m512 b = _mm512_set1_ps(Bpanel[0]);
        c00 = _mm512_fmadd_ps(a0, b, c00);
        c01 = _mm512_fmadd_ps(a1, b, c01);
        b = _mm512_set1_ps(Bpanel[1]);
        c10 = _mm512_fmadd_ps(a0, b, c10);
        c11 = _mm512_fmadd_ps(a1, b, c11);
        b = _mm512_set1_ps(Bpanel[2]);
        c20 = _mm512_fmadd_ps(a0, b, c20);
        c21 = _mm512_fmadd_ps(a1, b, c21);
        b = _mm512_set1_ps(Bpanel[3]);
        c30 = _mm512_fmadd_ps(a0, b, c30);
        c31 = _mm512_fmadd_ps(a1, b, c31);
        b = _mm512_set1_ps(Bpanel[4]);
        c40 = _mm512_fmadd_ps(a0, b, c40);
        c41 = _mm512_fmadd_ps(a1, b, c41);
        b = _mm512_set1_ps(Bpanel[5]);
        c50 = _mm512_fmadd_ps(a0, b, c50);
        c51 = _mm512_fmadd_ps(a1, b, c51);
        b = _mm512_set1_ps(Bpanel[6]);
        c60 = _mm512_fmadd_ps(a0, b, c60);
        c61 = _mm512_fmadd_ps(a1, b, c61);
        b = _mm512_set1_ps(Bpanel[7]);
        c70 = _mm512_fmadd_ps(a0, b, c70);
        c71 = _mm512_fmadd_ps(a1, b, c71);
        Bpanel += kSNr;
    }

    // C columns are contiguous (column-major) and the driver guarantees 32 valid
    // rows here, so these are plain unaligned vector read-modify-writes rather
    // than the scatter a column-oriented accumulator would need.
    const __m512 va = _mm512_set1_ps(alpha);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);

#define MS_STORE_ZCOL(idx, lo, hi)                                                 \
    do {                                                                           \
        float* col = C + static_cast<std::size_t>(idx) * ldc_u;                    \
        _mm512_storeu_ps(col, _mm512_fmadd_ps(va, (lo), _mm512_loadu_ps(col)));    \
        _mm512_storeu_ps(col + 16,                                                 \
                         _mm512_fmadd_ps(va, (hi), _mm512_loadu_ps(col + 16)));    \
    } while (0)

    MS_STORE_ZCOL(0, c00, c01);
    MS_STORE_ZCOL(1, c10, c11);
    MS_STORE_ZCOL(2, c20, c21);
    MS_STORE_ZCOL(3, c30, c31);
    MS_STORE_ZCOL(4, c40, c41);
    MS_STORE_ZCOL(5, c50, c51);
    MS_STORE_ZCOL(6, c60, c61);
    MS_STORE_ZCOL(7, c70, c71);
#undef MS_STORE_ZCOL
}

bool runtime_available_s() {
    // Gated on the OS having agreed to preserve ZMM state, not on CPUID alone --
    // see src/simd/isa.cpp.
    static const bool enabled = [] { return ms::simd::detect_isa().avx512f; }();
    return enabled;
}

} // namespace

bool sgemm_available() {
    return runtime_available_s();
}

bool sgemm_worthwhile(int m, int n, int k) {
    const long long work = static_cast<long long>(m) * static_cast<long long>(n) *
                           static_cast<long long>(k);
    return work >= kSMinWork;
}

void sgemm_nn(
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
    detail::scale_c<float>(m, n, beta, C, ldc);
    if (alpha == 0.0F || k == 0) {
        return;
    }
    detail::gemm_blocked<float, kSMr, kSNr>(m, n, k, alpha, A, lda, B, ldb, C, ldc,
                                            kSBlocks, micro_kernel_32x8);
}

} // namespace ms::cpu::blas::avx512
