// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Differential tests for the per-ISA dgemm kernels.
//
// A blocked, packed gemm has four places to get an index wrong -- the packing of
// A, the packing of B, the micro-kernel, and the fold-back of a partial tile --
// and all four are silent. A kernel that drops the last two columns of every
// block still returns a plausible matrix, and a benchmark still reports a
// speed-up. So each kernel is checked against an independent reference at every
// size that straddles a boundary in the decomposition.
//
// The reference is a straightforward triple loop accumulated in long double. It
// is not the library's scalar path: comparing a kernel against another kernel in
// the same file only proves they were written by the same person on the same day.
//
// TOLERANCE. The kernels reassociate the sum over k and use fused multiply-add,
// so they do not reproduce the reference bit for bit, and should not be expected
// to. Both are legitimate; both change the result. Each C entry is a sum of k
// products, so the standard worst-case bound on the accumulated rounding error is
// about k * eps * max|A| * max|B|. The tests allow 16x that, which is loose
// enough to cover any summation order the compiler chooses and far tighter than
// any indexing mistake could hide behind: dropping a single term of a k-term sum
// of unit-scale values is an error of order 1, and the tolerance at k = 256 is
// about 1e-12.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "ms/cpu/blas.hpp"
#include "ms/cpu/blas_kernel.hpp"
#include "ms/simd/isa.hpp"

namespace {

// Column-major, matching the library: A[p*lda + i], B[j*ldb + p], C[j*ldc + i].
struct Problem {
    int m, n, k;
    std::vector<double> a, b, c0;
    double amax = 0.0, bmax = 0.0;
};

Problem make_problem(int m, int n, int k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Problem p{m, n, k, {}, {}, {}, 0.0, 0.0};
    p.a.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
    p.b.resize(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    p.c0.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (auto& v : p.a) {
        v = dist(rng);
        p.amax = std::max(p.amax, std::abs(v));
    }
    for (auto& v : p.b) {
        v = dist(rng);
        p.bmax = std::max(p.bmax, std::abs(v));
    }
    for (auto& v : p.c0) {
        v = dist(rng);
    }
    return p;
}

/// C = alpha * A * B + beta * C, accumulated in long double.
std::vector<double> reference(const Problem& p, double alpha, double beta) {
    std::vector<double> c = p.c0;
    for (int j = 0; j < p.n; ++j) {
        for (int i = 0; i < p.m; ++i) {
            long double sum = 0.0L;
            for (int q = 0; q < p.k; ++q) {
                sum += static_cast<long double>(
                           p.a[static_cast<std::size_t>(q) * static_cast<std::size_t>(p.m) +
                               static_cast<std::size_t>(i)]) *
                       static_cast<long double>(
                           p.b[static_cast<std::size_t>(j) * static_cast<std::size_t>(p.k) +
                               static_cast<std::size_t>(q)]);
            }
            const std::size_t idx =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(p.m) +
                static_cast<std::size_t>(i);
            c[idx] = static_cast<double>(static_cast<long double>(alpha) * sum +
                                         static_cast<long double>(beta) *
                                             static_cast<long double>(p.c0[idx]));
        }
    }
    return c;
}

double tolerance(const Problem& p, double alpha) {
    const double eps = std::numeric_limits<double>::epsilon();
    const double bound = static_cast<double>(p.k) * eps * p.amax * p.bmax * std::abs(alpha);
    // A floor for the k = 0 and tiny-k cases, where the bound above is zero or
    // near it but the beta * C term still carries its own rounding.
    return 16.0 * bound + 64.0 * eps;
}

using KernelFn = void (*)(int, int, int, double, const double*, int, const double*,
                          int, double, double*, int);

void expect_matches_reference(KernelFn kernel, const char* name, int m, int n, int k,
                              double alpha, double beta, std::uint32_t seed) {
    const Problem p = make_problem(m, n, k, seed);
    const std::vector<double> want = reference(p, alpha, beta);

    std::vector<double> got = p.c0;
    kernel(m, n, k, alpha, p.a.data(), m, p.b.data(), k, beta, got.data(), m);

    const double tol = tolerance(p, alpha);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(m) +
                static_cast<std::size_t>(i);
            ASSERT_NEAR(got[idx], want[idx], tol)
                << name << " m=" << m << " n=" << n << " k=" << k
                << " alpha=" << alpha << " beta=" << beta
                << " at (" << i << "," << j << ")";
        }
    }
}

// Sizes that straddle every boundary in the decomposition: the AVX2 micro-kernel
// is 8x6 and the AVX-512 one is 16x8, so one short of, exactly at, and one past
// each of those; and around the k-blocking factor of 256. A kernel that mishandles
// a partial tile passes at the multiples and fails here.
const std::vector<int> kEdgeSizes = {
    1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 23, 24, 25, 31, 32, 33,
    47, 48, 49, 63, 64, 65,
};

const std::vector<int> kDepths = {1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 255, 256, 257};

} // namespace

TEST(DgemmKernels, Avx2MatchesReferenceAtEveryEdgeSize) {
    if (!ms::cpu::blas::avx2::available()) {
        GTEST_SKIP() << "AVX2/FMA not available on this host";
    }
    std::uint32_t seed = 1;
    for (const int m : kEdgeSizes) {
        for (const int n : kEdgeSizes) {
            expect_matches_reference(&ms::cpu::blas::avx2::dgemm_nn, "avx2", m, n, 17,
                                     1.0, 0.0, seed++);
        }
    }
}

TEST(DgemmKernels, Avx2MatchesReferenceAcrossDepthBlocking) {
    if (!ms::cpu::blas::avx2::available()) {
        GTEST_SKIP() << "AVX2/FMA not available on this host";
    }
    std::uint32_t seed = 5000;
    for (const int k : kDepths) {
        expect_matches_reference(&ms::cpu::blas::avx2::dgemm_nn, "avx2", 37, 29, k,
                                 1.0, 0.0, seed++);
    }
}

TEST(DgemmKernels, Avx2HonoursAlphaAndBeta) {
    if (!ms::cpu::blas::avx2::available()) {
        GTEST_SKIP() << "AVX2/FMA not available on this host";
    }
    std::uint32_t seed = 9000;
    for (const double alpha : {0.0, 1.0, -1.0, 2.5}) {
        for (const double beta : {0.0, 1.0, -1.0, 0.5}) {
            expect_matches_reference(&ms::cpu::blas::avx2::dgemm_nn, "avx2", 19, 23, 33,
                                     alpha, beta, seed++);
        }
    }
}

TEST(DgemmKernels, Avx512MatchesReferenceAtEveryEdgeSize) {
    if (!ms::cpu::blas::avx512::available()) {
        GTEST_SKIP() << "AVX-512F not available on this host or not built in";
    }
    std::uint32_t seed = 20000;
    for (const int m : kEdgeSizes) {
        for (const int n : kEdgeSizes) {
            expect_matches_reference(&ms::cpu::blas::avx512::dgemm_nn, "avx512", m, n, 17,
                                     1.0, 0.0, seed++);
        }
    }
}

TEST(DgemmKernels, Avx512MatchesReferenceAcrossDepthBlocking) {
    if (!ms::cpu::blas::avx512::available()) {
        GTEST_SKIP() << "AVX-512F not available on this host or not built in";
    }
    std::uint32_t seed = 25000;
    for (const int k : kDepths) {
        expect_matches_reference(&ms::cpu::blas::avx512::dgemm_nn, "avx512", 37, 29, k,
                                 1.0, 0.0, seed++);
    }
}

TEST(DgemmKernels, Avx512HonoursAlphaAndBeta) {
    if (!ms::cpu::blas::avx512::available()) {
        GTEST_SKIP() << "AVX-512F not available on this host or not built in";
    }
    std::uint32_t seed = 29000;
    for (const double alpha : {0.0, 1.0, -1.0, 2.5}) {
        for (const double beta : {0.0, 1.0, -1.0, 0.5}) {
            expect_matches_reference(&ms::cpu::blas::avx512::dgemm_nn, "avx512", 19, 23, 33,
                                     alpha, beta, seed++);
        }
    }
}

// Sizes past the point where the dispatcher hands work to a blocked kernel, so
// this covers the selection logic as well as the kernels: whichever path the host
// takes has to agree with the reference.
TEST(DgemmKernels, PublicDgemmAgreesWithReferenceOnLargeProblems) {
    std::uint32_t seed = 40000;
    for (const int size : {65, 96, 128, 129}) {
        const Problem p = make_problem(size, size, size, seed++);
        const std::vector<double> want = reference(p, 1.0, 0.0);
        std::vector<double> got = p.c0;
        ms::cpu::blas::dgemm('N', 'N', size, size, size, 1.0, p.a.data(), size,
                             p.b.data(), size, 0.0, got.data(), size);
        const double tol = tolerance(p, 1.0);
        for (std::size_t idx = 0; idx < got.size(); ++idx) {
            ASSERT_NEAR(got[idx], want[idx], tol) << "size=" << size << " idx=" << idx;
        }
    }
}

// The point of having several kernels is that they compute the same function.
// This is the check that says so directly, rather than inferring it from two
// separate comparisons against the reference.
TEST(DgemmKernels, EveryAvailableIsaPathAgreesWithEveryOther) {
    const bool have_avx2 = ms::cpu::blas::avx2::available();
    const bool have_avx512 = ms::cpu::blas::avx512::available();
    if (!have_avx2 && !have_avx512) {
        GTEST_SKIP() << "no vector dgemm kernel available on this host";
    }

    std::uint32_t seed = 60000;
    for (const int size : {17, 64, 100}) {
        const Problem p = make_problem(size, size, size, seed++);
        const double tol = tolerance(p, 1.0);

        std::vector<double> a2 = p.c0;
        std::vector<double> a5 = p.c0;
        if (have_avx2) {
            ms::cpu::blas::avx2::dgemm_nn(size, size, size, 1.0, p.a.data(), size,
                                          p.b.data(), size, 0.0, a2.data(), size);
        }
        if (have_avx512) {
            ms::cpu::blas::avx512::dgemm_nn(size, size, size, 1.0, p.a.data(), size,
                                            p.b.data(), size, 0.0, a5.data(), size);
        }
        if (have_avx2 && have_avx512) {
            for (std::size_t idx = 0; idx < a2.size(); ++idx) {
                // Doubled: each path carries its own bound against the exact
                // result, so their difference can be up to the sum of the two.
                ASSERT_NEAR(a2[idx], a5[idx], 2.0 * tol)
                    << "AVX2 and AVX-512 disagree at size=" << size << " idx=" << idx;
            }
        }
    }
}

// The blocked kernels decline small problems so the caller keeps the cheaper
// path. That decision must be about cost only: a size the kernel declines still
// has to compute the right answer when called directly.
TEST(DgemmKernels, DeclinedSizesStillComputeCorrectlyWhenCalledDirectly) {
    if (!ms::cpu::blas::avx2::available()) {
        GTEST_SKIP() << "AVX2/FMA not available on this host";
    }
    ASSERT_FALSE(ms::cpu::blas::avx2::worthwhile(4, 4, 4))
        << "expected a 4x4x4 problem to be below the packing threshold";
    expect_matches_reference(&ms::cpu::blas::avx2::dgemm_nn, "avx2", 4, 4, 4, 1.0, 0.0,
                             77000);
}

TEST(DgemmKernels, ZeroDepthLeavesOnlyTheBetaScaling) {
    if (!ms::cpu::blas::avx2::available()) {
        GTEST_SKIP() << "AVX2/FMA not available on this host";
    }
    expect_matches_reference(&ms::cpu::blas::avx2::dgemm_nn, "avx2", 12, 10, 0, 1.0, 0.5,
                             88000);
}
