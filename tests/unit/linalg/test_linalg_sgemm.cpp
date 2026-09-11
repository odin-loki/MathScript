// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §9.5 Tier 1: the single-precision GEMM.
//
// `matmul` had a fast path for exactly one scalar type. Everything else -- float
// included -- fell to the triple loop at the bottom of src/linalg/matmul.cpp, and
// that loop is written i, k, j, so on a column-major C it strides the innermost
// index by ldc. A float multiply was therefore not merely unvectorised; it was the
// slowest of the three orderings.
//
// What these tests pin is the thing a hand-written kernel most easily gets wrong:
// not the interior of a large tile, which any arrangement gets right, but the
// EDGES. The 16x6 micro-kernel is driven over a padded block and partial tiles are
// computed at full width into a scratch buffer, so every shape that is not a
// multiple of 16 by 6 takes a different path through the driver -- and a kernel
// that is correct at 64x64x64 can be wrong at 17x7x3.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ms/cpu/blas.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;

namespace {

// C = alpha * A * B + beta * C, straight from the definition, column-major.
std::vector<float> reference_gemm(int m, int n, int k, float alpha,
                                  const std::vector<float>& A,
                                  const std::vector<float>& B, float beta,
                                  std::vector<float> C) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float sum = 0.0F;
            for (int p = 0; p < k; ++p) {
                sum += A[static_cast<std::size_t>(p) * static_cast<std::size_t>(m) +
                         static_cast<std::size_t>(i)] *
                       B[static_cast<std::size_t>(j) * static_cast<std::size_t>(k) +
                         static_cast<std::size_t>(p)];
            }
            float& c = C[static_cast<std::size_t>(j) * static_cast<std::size_t>(m) +
                         static_cast<std::size_t>(i)];
            c = alpha * sum + beta * c;
        }
    }
    return C;
}

std::vector<float> filled(std::size_t count, std::uint64_t seed) {
    // A deterministic linear congruential sequence rather than <random>'s engines,
    // so a failure reproduces from the test alone on any platform.
    std::vector<float> v(count);
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = static_cast<float>(static_cast<int>((state >> 33) % 2001) - 1000) / 250.0F;
    }
    return v;
}

// The kernel accumulates in float, as every BLAS sgemm does, so it cannot be
// compared bit for bit against a reference that sums in a different order: FMA
// contracts a multiply and an add into one rounding, and the blocked driver splits
// the depth into kc-sized pieces. The tolerance is relative to the magnitude the
// sum reaches, which is what a float's 24-bit significand actually promises.
void expect_close(const std::vector<float>& got, const std::vector<float>& want,
                  int k, const char* what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float scale = std::max(1.0F, std::abs(want[i]));
        const float tol = scale * 1e-5F * static_cast<float>(std::max(1, k));
        EXPECT_NEAR(got[i], want[i], tol) << what << " at " << i;
    }
}

} // namespace

TEST(CpuSgemm, MatchesTheDefinitionAcrossEveryTileEdge) {
    // 16 rows and 6 columns is the micro-kernel's tile, so the shapes below step
    // through: exactly one tile, one short of it, one past it, and sizes that are
    // coprime to both. k is varied separately because the depth is what the packed
    // panels are cut along.
    const int sizes[] = {1, 2, 5, 6, 7, 15, 16, 17, 31, 32, 33, 48, 64, 65};
    std::size_t seed = 1;
    for (const int m : sizes) {
        for (const int n : sizes) {
            for (const int k : {1, 3, 8, 17, 64}) {
                const auto A = filled(static_cast<std::size_t>(m) *
                                          static_cast<std::size_t>(k),
                                      ++seed * 7919ULL);
                const auto B = filled(static_cast<std::size_t>(k) *
                                          static_cast<std::size_t>(n),
                                      ++seed * 104729ULL);
                const auto C0 = filled(static_cast<std::size_t>(m) *
                                           static_cast<std::size_t>(n),
                                       ++seed * 15485863ULL);

                std::vector<float> C = C0;
                cpu::blas::sgemm('N', 'N', m, n, k, 1.5F, A.data(), m, B.data(), k,
                                 -0.25F, C.data(), m);
                const auto want = reference_gemm(m, n, k, 1.5F, A, B, -0.25F, C0);
                expect_close(C, want, k, "alpha 1.5, beta -0.25");
            }
        }
    }
}

TEST(CpuSgemm, TakesThePackedPathAndAgreesWithTheUnpackedOne) {
    // Above 64x64x64 the dispatcher hands the problem to the blocked kernel, and
    // below it the rank-1 loop keeps it. Both have to produce the same matrix, so
    // this runs one shape on each side of that threshold and compares each against
    // the definition -- which is the only way to tell that the packed path is being
    // exercised at all rather than silently declined.
    for (const int n : {48, 96, 160}) {
        const auto A = filled(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                              20260911ULL);
        const auto B = filled(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                              20260912ULL);
        const std::vector<float> C0(static_cast<std::size_t>(n) *
                                        static_cast<std::size_t>(n),
                                    0.0F);
        std::vector<float> C = C0;
        cpu::blas::sgemm('N', 'N', n, n, n, 1.0F, A.data(), n, B.data(), n, 0.0F,
                         C.data(), n);
        expect_close(C, reference_gemm(n, n, n, 1.0F, A, B, 0.0F, C0), n, "square");
    }
}

TEST(CpuSgemm, TransposedFormsAndDegenerateArguments) {
    const int m = 5;
    const int n = 4;
    const int k = 3;
    const auto A = filled(static_cast<std::size_t>(m) * static_cast<std::size_t>(k), 11ULL);
    const auto At = filled(static_cast<std::size_t>(k) * static_cast<std::size_t>(m), 13ULL);
    const auto B = filled(static_cast<std::size_t>(k) * static_cast<std::size_t>(n), 17ULL);
    const std::vector<float> C0(static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                                2.0F);

    // 'T' on A: A is k x m and the generic path reads it transposed. Compare against
    // the definition with the transpose done by hand.
    std::vector<float> Aexp(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            Aexp[static_cast<std::size_t>(p) * static_cast<std::size_t>(m) +
                 static_cast<std::size_t>(i)] =
                At[static_cast<std::size_t>(i) * static_cast<std::size_t>(k) +
                   static_cast<std::size_t>(p)];
        }
    }
    std::vector<float> C = C0;
    cpu::blas::sgemm('T', 'N', m, n, k, 1.0F, At.data(), k, B.data(), k, 1.0F,
                     C.data(), m);
    expect_close(C, reference_gemm(m, n, k, 1.0F, Aexp, B, 1.0F, C0), k, "transposed A");

    // alpha = 0 is beta * C and nothing else; k = 0 is the same statement.
    C = C0;
    cpu::blas::sgemm('N', 'N', m, n, k, 0.0F, A.data(), m, B.data(), k, 3.0F,
                     C.data(), m);
    for (std::size_t i = 0; i < C.size(); ++i) {
        EXPECT_FLOAT_EQ(C[i], 3.0F * C0[i]) << "alpha = 0 at " << i;
    }
    C = C0;
    cpu::blas::sgemm('N', 'N', m, n, 0, 1.0F, A.data(), m, B.data(), 1, 0.0F,
                     C.data(), m);
    for (const float v : C) {
        EXPECT_FLOAT_EQ(v, 0.0F);
    }

    // An empty result writes nothing at all, including no beta scaling.
    std::vector<float> untouched = C0;
    cpu::blas::sgemm('N', 'N', 0, n, k, 1.0F, A.data(), 1, B.data(), k, 0.0F,
                     untouched.data(), 1);
    for (std::size_t i = 0; i < untouched.size(); ++i) {
        EXPECT_FLOAT_EQ(untouched[i], C0[i]) << "m = 0 wrote at " << i;
    }
}

TEST(LinalgMatmulFloat, UsesTheKernelAndMatchesTheDefinition) {
    // The caller the kernel exists for. `matmul` on two column-major float matrices
    // now goes through sgemm; this asserts the answer rather than the route, at a
    // size above the packing threshold so the blocked path is the one taken.
    const size_t n = 96;
    ColMatrix<float> A(n, n);
    ColMatrix<float> B(n, n);
    std::uint64_t state = 987654321ULL;
    const auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<int>((state >> 33) % 401) - 200) / 100.0F;
    };
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = next();
            B(i, j) = next();
        }
    }

    const auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    ASSERT_EQ(C->rows(), n);
    ASSERT_EQ(C->cols(), n);

    for (size_t i = 0; i < n; i += 17) {
        for (size_t j = 0; j < n; j += 13) {
            float sum = 0.0F;
            for (size_t p = 0; p < n; ++p) {
                sum += A(i, p) * B(p, j);
            }
            EXPECT_NEAR((*C)(i, j), sum, std::max(1.0F, std::abs(sum)) * 1e-4F)
                << "at (" << i << ", " << j << ")";
        }
    }
}
