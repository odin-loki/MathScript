// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// AVX2/FMA sgemm.
//
// `matmul` had a fast path for exactly one scalar type. Every other one -- float
// included -- fell to the triple loop at the bottom of src/linalg/matmul.cpp, and
// that loop is written i, k, j, so with column-major C it strides `C(i, j)` by ldc
// on the innermost index. A single-precision multiply was therefore not merely
// unvectorised; it was the slowest arrangement of the three.
//
// The kernel is a 16x6 micro-kernel over the same packing and cache blocking the
// dgemm kernels use, now templated on the scalar type. 16 rows is two 8-wide ymm
// vectors; 6 columns keeps the register budget inside the 16 that AVX2 provides:
//
//     12  accumulators   (16 rows / 8 lanes) x 6 columns
//      2  A vectors
//      1  broadcast B scalar
//     --
//     15  of 16
//
// Which is the same arithmetic as the 8x6 double kernel next door, at twice the
// rows because a float vector holds twice as many.
//
// ACCUMULATION IS IN FLOAT, not in double. That is the choice a user of a float
// matrix has already made, and it is what every BLAS sgemm does; widening the
// accumulator here would make this kernel disagree with the unpacked fallback
// below it and with the naive loop the tests compare against.

#include "ms/core/attributes.hpp"
#include "ms/cpu/blas_kernel.hpp"
#include "ms/simd/isa.hpp"

#include "gemm_blocking.hpp"

#include <immintrin.h>

namespace ms::cpu::blas::avx2 {

namespace {

constexpr int kSMr = 16;
constexpr int kSNr = 6;

// A block of A is kMc x kKc floats = 256 * 512 * 4 = 512 KiB, the same L2 footprint
// the double kernel takes at half the depth -- a float is half the size, so twice
// the depth fits the same cache. The B block is kKc x kNc = 512 * 1024 * 4 = 2 MiB
// and is expected to live in L3. Defaults rather than measurements, as next door.
constexpr detail::BlockSizes kSBlocks{256, 512, 1024};

// Below this the packing and the two aligned allocations cost more than the
// blocking saves. Cubed because that is how the arithmetic grows against the
// linear packing cost.
constexpr long long kSMinWork = 64LL * 64LL * 64LL;

/// One 16x6 tile: C[0..15][0..5] += alpha * Apanel(16 x kc) * Bpanel(kc x 6).
MS_FORCEINLINE void micro_kernel_16x6(
    int kc,
    float alpha,
    const float* __restrict Apanel,
    const float* __restrict Bpanel,
    float* __restrict C,
    int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int p = 0; p < kc; ++p) {
        const __m256 a0 = _mm256_load_ps(Apanel + 0);
        const __m256 a1 = _mm256_load_ps(Apanel + 8);
        Apanel += kSMr;

        __m256 b = _mm256_broadcast_ss(Bpanel + 0);
        c00 = _mm256_fmadd_ps(a0, b, c00);
        c01 = _mm256_fmadd_ps(a1, b, c01);
        b = _mm256_broadcast_ss(Bpanel + 1);
        c10 = _mm256_fmadd_ps(a0, b, c10);
        c11 = _mm256_fmadd_ps(a1, b, c11);
        b = _mm256_broadcast_ss(Bpanel + 2);
        c20 = _mm256_fmadd_ps(a0, b, c20);
        c21 = _mm256_fmadd_ps(a1, b, c21);
        b = _mm256_broadcast_ss(Bpanel + 3);
        c30 = _mm256_fmadd_ps(a0, b, c30);
        c31 = _mm256_fmadd_ps(a1, b, c31);
        b = _mm256_broadcast_ss(Bpanel + 4);
        c40 = _mm256_fmadd_ps(a0, b, c40);
        c41 = _mm256_fmadd_ps(a1, b, c41);
        b = _mm256_broadcast_ss(Bpanel + 5);
        c50 = _mm256_fmadd_ps(a0, b, c50);
        c51 = _mm256_fmadd_ps(a1, b, c51);
        Bpanel += kSNr;
    }

    // C columns are contiguous (column-major) and the driver guarantees sixteen
    // valid rows here, so these are plain unaligned vector read-modify-writes.
    const __m256 va = _mm256_set1_ps(alpha);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);

#define MS_STORE_SCOL(idx, lo, hi)                                                \
    do {                                                                          \
        float* col = C + static_cast<std::size_t>(idx) * ldc_u;                   \
        _mm256_storeu_ps(col, _mm256_fmadd_ps(va, (lo), _mm256_loadu_ps(col)));   \
        _mm256_storeu_ps(col + 8,                                                 \
                         _mm256_fmadd_ps(va, (hi), _mm256_loadu_ps(col + 8)));    \
    } while (0)

    MS_STORE_SCOL(0, c00, c01);
    MS_STORE_SCOL(1, c10, c11);
    MS_STORE_SCOL(2, c20, c21);
    MS_STORE_SCOL(3, c30, c31);
    MS_STORE_SCOL(4, c40, c41);
    MS_STORE_SCOL(5, c50, c51);
#undef MS_STORE_SCOL
}

bool runtime_available_s() {
    // Both bits, and both already gated on the OS having agreed to preserve YMM
    // state -- see src/simd/isa.cpp.
    static const bool enabled = [] {
        const auto isa = ms::simd::detect_isa();
        return isa.avx2 && isa.fma;
    }();
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
                                            kSBlocks, micro_kernel_16x6);
}

} // namespace ms::cpu::blas::avx2
