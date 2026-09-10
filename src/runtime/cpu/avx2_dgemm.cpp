// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// AVX2/FMA dgemm.
//
// Until this existed, every machine without AVX-512 -- Zen 1 through 3, every
// Intel client part since Alder Lake, and anything running the CI configuration,
// which passes -DMS_ENABLE_AVX512=OFF -- fell all the way to a rank-1 update loop
// for matrix multiply. That is most of the hardware the library runs on.
//
// The kernel is an 8x6 micro-kernel over the packing and cache blocking in
// gemm_blocking.hpp. 8 rows is two 4-wide vectors; 6 columns keeps the register
// budget inside the 16 ymm registers AVX2 provides:
//
//     12  accumulators   (8 rows / 4 lanes) x 6 columns
//      2  A vectors
//      1  broadcast B scalar
//     --
//     15  of 16
//
// A thirteenth accumulator column would spill to the stack every iteration of the
// inner loop and cost more than the extra column earns.

#include "ms/core/attributes.hpp"
#include "ms/cpu/blas_kernel.hpp"
#include "ms/simd/isa.hpp"

#include "gemm_blocking.hpp"

#include <immintrin.h>

namespace ms::cpu::blas::avx2 {

namespace {

constexpr int kMr = 8;
constexpr int kNr = 6;

// A block of A is kMc x kKc doubles = 256 * 256 * 8 = 512 KiB, sized to sit in a
// typical 1-2 MiB L2 alongside the streaming B panel. The B block is kKc x kNc =
// 256 * 1024 * 8 = 2 MiB and is expected to live in L3. These are defaults, not
// measurements: the right numbers depend on the machine, which is why they are a
// struct rather than constants baked into the loop.
constexpr detail::BlockSizes kBlocks{256, 256, 1024};

// Below this the packing and the two aligned allocations cost more than the
// blocking saves, and the caller's simpler path wins. Cubed because that is how
// the arithmetic grows against the linear packing cost.
constexpr long long kMinWork = 64LL * 64LL * 64LL;

/// One 8x6 tile: C[0..7][0..5] += alpha * Apanel(8 x kc) * Bpanel(kc x 6).
///
/// Both panels are read straight through -- Apanel 8 contiguous doubles per depth
/// index, Bpanel 6 -- which is the whole point of having packed them.
MS_FORCEINLINE void micro_kernel_8x6(
    int kc,
    double alpha,
    const double* __restrict Apanel,
    const double* __restrict Bpanel,
    double* __restrict C,
    int ldc) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    for (int p = 0; p < kc; ++p) {
        const __m256d a0 = _mm256_load_pd(Apanel + 0);
        const __m256d a1 = _mm256_load_pd(Apanel + 4);
        Apanel += kMr;

        __m256d b = _mm256_broadcast_sd(Bpanel + 0);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c01 = _mm256_fmadd_pd(a1, b, c01);
        b = _mm256_broadcast_sd(Bpanel + 1);
        c10 = _mm256_fmadd_pd(a0, b, c10);
        c11 = _mm256_fmadd_pd(a1, b, c11);
        b = _mm256_broadcast_sd(Bpanel + 2);
        c20 = _mm256_fmadd_pd(a0, b, c20);
        c21 = _mm256_fmadd_pd(a1, b, c21);
        b = _mm256_broadcast_sd(Bpanel + 3);
        c30 = _mm256_fmadd_pd(a0, b, c30);
        c31 = _mm256_fmadd_pd(a1, b, c31);
        b = _mm256_broadcast_sd(Bpanel + 4);
        c40 = _mm256_fmadd_pd(a0, b, c40);
        c41 = _mm256_fmadd_pd(a1, b, c41);
        b = _mm256_broadcast_sd(Bpanel + 5);
        c50 = _mm256_fmadd_pd(a0, b, c50);
        c51 = _mm256_fmadd_pd(a1, b, c51);
        Bpanel += kNr;
    }

    // C columns are contiguous in memory (column-major), and the caller
    // guarantees eight valid rows here, so these are plain unaligned vector
    // read-modify-writes rather than the scatter the previous kernel needed.
    const __m256d va = _mm256_set1_pd(alpha);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);

#define MS_STORE_COL(idx, lo, hi)                                                 \
    do {                                                                          \
        double* col = C + static_cast<std::size_t>(idx) * ldc_u;                  \
        _mm256_storeu_pd(col, _mm256_fmadd_pd(va, (lo), _mm256_loadu_pd(col)));   \
        _mm256_storeu_pd(col + 4,                                                 \
                         _mm256_fmadd_pd(va, (hi), _mm256_loadu_pd(col + 4)));    \
    } while (0)

    MS_STORE_COL(0, c00, c01);
    MS_STORE_COL(1, c10, c11);
    MS_STORE_COL(2, c20, c21);
    MS_STORE_COL(3, c30, c31);
    MS_STORE_COL(4, c40, c41);
    MS_STORE_COL(5, c50, c51);
#undef MS_STORE_COL
}

bool runtime_available() {
    // Both bits, and both are already gated on the OS having agreed to preserve
    // YMM state -- see src/simd/isa.cpp. FMA without that agreement is the same
    // SIGILL as AVX without it.
    static const bool enabled = [] {
        const auto isa = ms::simd::detect_isa();
        return isa.avx2 && isa.fma;
    }();
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
                                   micro_kernel_8x6);
}

} // namespace ms::cpu::blas::avx2
