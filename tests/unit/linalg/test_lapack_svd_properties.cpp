// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4/§8.6: the SVD's singular VECTORS, over every shape that reaches a
// different path through the bidiagonal QR.
//
// `src/runtime/cpu/lapack_dbdsqr.cpp` scored 62.5% over viable mutants, and the
// survivors were all in the rotation appliers -- the `dlasr_*` routines that
// accumulate the Givens rotations into U and V^T. One of them turned
// `sn * temp + cs * A[...]` into a subtraction, which is not a rotation at all;
// another read one row further down; a third passed a NEGATIVE offset into a
// working vector. All four SVD suites passed under each.
//
// The reason is what the existing tests ask. They check the singular VALUES,
// which the implicit QR computes from the bidiagonal d and e alone and which are
// therefore insensitive to anything the rotation accumulation does; and they
// check reconstruction at three fixed small shapes, which do not reach the paths
// the survivors sit on.
//
// So this file asks the two questions that cannot be satisfied by accident:
//
//     A = U * Sigma * V^T          the factorisation is of the matrix given
//     U^T U = I and V^T V = I      the vectors are orthonormal
//
// over shapes from 1x1 up past the point where the block update path is taken.
// A rotation applied wrongly still produces a matrix, and still produces the right
// singular values; it does not produce an orthogonal one.
//
// Everything is computed here with plain loops rather than through the library's
// own matrix operations. A test that uses the thing it is testing to check the
// thing it is testing has one fewer independent opinion in it than it appears to.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ms/cpu/lapack.hpp"

namespace {

// Column-major, lda = rows, as dgesvd expects.
std::vector<double> patterned(int rows, int cols, std::uint64_t seed) {
    std::vector<double> a(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    std::uint64_t state = seed;
    for (double& v : a) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<double>(static_cast<int>((state >> 33) % 2001) - 1000) / 250.0;
    }
    return a;
}

struct Svd {
    int m = 0;
    int n = 0;
    int k = 0;
    std::vector<double> s;
    std::vector<double> u;   // m x k, column-major, ldu = m
    std::vector<double> vt;  // k x n, column-major, ldvt = k
};

// Runs the decomposition on a copy, since dgesvd overwrites its input.
Svd decompose(int m, int n, const std::vector<double>& a) {
    Svd out;
    out.m = m;
    out.n = n;
    out.k = std::min(m, n);
    std::vector<double> work = a;
    out.s.assign(static_cast<std::size_t>(out.k), 0.0);
    out.u.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(out.k), 0.0);
    out.vt.assign(static_cast<std::size_t>(out.k) * static_cast<std::size_t>(n), 0.0);
    const int info = ms::cpu::lapack::dgesvd(m, n, work.data(), m, out.s.data(),
                                             out.u.data(), m, out.vt.data(), out.k);
    EXPECT_EQ(info, 0) << "dgesvd failed at " << m << "x" << n;
    return out;
}

// The largest magnitude in A, for a tolerance that scales with the problem rather
// than with taste.
double max_abs(const std::vector<double>& a) {
    double worst = 0.0;
    for (const double v : a) {
        worst = std::max(worst, std::abs(v));
    }
    return worst;
}

void expect_factorisation_of(const std::vector<double>& a, const Svd& f,
                             const char* what) {
    const double scale = std::max(1.0, max_abs(a));
    const double tol = scale * 1e-10 * static_cast<double>(std::max(1, f.k));

    // A = U Sigma V^T, entry by entry.
    for (int j = 0; j < f.n; ++j) {
        for (int i = 0; i < f.m; ++i) {
            double acc = 0.0;
            for (int p = 0; p < f.k; ++p) {
                acc += f.u[static_cast<std::size_t>(p) * static_cast<std::size_t>(f.m) +
                           static_cast<std::size_t>(i)] *
                       f.s[static_cast<std::size_t>(p)] *
                       f.vt[static_cast<std::size_t>(j) * static_cast<std::size_t>(f.k) +
                            static_cast<std::size_t>(p)];
            }
            ASSERT_NEAR(acc,
                        a[static_cast<std::size_t>(j) * static_cast<std::size_t>(f.m) +
                          static_cast<std::size_t>(i)],
                        tol)
                << what << " " << f.m << "x" << f.n << " at (" << i << ", " << j << ")";
        }
    }

    // U has orthonormal columns.
    for (int p = 0; p < f.k; ++p) {
        for (int q = p; q < f.k; ++q) {
            double acc = 0.0;
            for (int i = 0; i < f.m; ++i) {
                acc += f.u[static_cast<std::size_t>(p) * static_cast<std::size_t>(f.m) +
                           static_cast<std::size_t>(i)] *
                       f.u[static_cast<std::size_t>(q) * static_cast<std::size_t>(f.m) +
                           static_cast<std::size_t>(i)];
            }
            ASSERT_NEAR(acc, (p == q) ? 1.0 : 0.0, 1e-10)
                << what << " U^T U " << f.m << "x" << f.n << " at (" << p << ", " << q << ")";
        }
    }

    // V^T has orthonormal rows.
    for (int p = 0; p < f.k; ++p) {
        for (int q = p; q < f.k; ++q) {
            double acc = 0.0;
            for (int j = 0; j < f.n; ++j) {
                acc += f.vt[static_cast<std::size_t>(j) * static_cast<std::size_t>(f.k) +
                            static_cast<std::size_t>(p)] *
                       f.vt[static_cast<std::size_t>(j) * static_cast<std::size_t>(f.k) +
                            static_cast<std::size_t>(q)];
            }
            ASSERT_NEAR(acc, (p == q) ? 1.0 : 0.0, 1e-10)
                << what << " V V^T " << f.m << "x" << f.n << " at (" << p << ", " << q << ")";
        }
    }

    // Singular values are non-negative and ordered.
    for (int p = 0; p < f.k; ++p) {
        EXPECT_GE(f.s[static_cast<std::size_t>(p)], -1e-14) << what << " sigma " << p;
        if (p + 1 < f.k) {
            EXPECT_GE(f.s[static_cast<std::size_t>(p)] + 1e-12,
                      f.s[static_cast<std::size_t>(p) + 1])
                << what << " sigma order at " << p;
        }
    }
}

} // namespace

TEST(LapackSvdProperties, EveryShapeFactorisesTheMatrixItWasGiven) {
    // Square, tall and wide, from the degenerate 1x1 up past 33 -- far enough that
    // the bidiagonal QR takes its block update path rather than staying in the
    // small-matrix one the existing 4x3 and 3x4 tests reach.
    const std::vector<int> sizes = {1, 2, 3, 4, 5, 7, 8, 9, 12, 17, 24, 33};
    std::uint64_t seed = 11;
    for (const int m : sizes) {
        for (const int n : sizes) {
            const auto a = patterned(m, n, ++seed * 7919ULL);
            expect_factorisation_of(a, decompose(m, n, a), "dense");
        }
    }
}

TEST(LapackSvdProperties, RankDeficientAndStructuredInputs) {
    // The cases where the iteration splits and deflates rather than running to the
    // end: repeated singular values, exact zeros, and a matrix whose rows are all
    // multiples of one another.
    std::uint64_t seed = 4242;
    for (const int n : {3, 5, 8, 13, 20}) {
        const int m = n + 2;

        // Rank one: every row a multiple of the first.
        std::vector<double> rank1(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        const auto row = patterned(1, n, ++seed);
        const auto col = patterned(m, 1, ++seed);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                rank1[static_cast<std::size_t>(j) * static_cast<std::size_t>(m) +
                      static_cast<std::size_t>(i)] =
                    col[static_cast<std::size_t>(i)] * row[static_cast<std::size_t>(j)];
            }
        }
        expect_factorisation_of(rank1, decompose(m, n, rank1), "rank one");

        // The identity padded with zero rows: every singular value equal.
        std::vector<double> eye(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
        for (int d = 0; d < n; ++d) {
            eye[static_cast<std::size_t>(d) * static_cast<std::size_t>(m) +
                static_cast<std::size_t>(d)] = 1.0;
        }
        expect_factorisation_of(eye, decompose(m, n, eye), "identity");

        // All zeros: the fully deflated case.
        const std::vector<double> zero(
            static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
        expect_factorisation_of(zero, decompose(m, n, zero), "zero");

        // Wildly different scales, so the iteration has to shift rather than coast.
        auto scaled = patterned(m, n, ++seed);
        for (int j = 0; j < n; ++j) {
            const double factor = std::pow(10.0, static_cast<double>(j % 5) - 2.0);
            for (int i = 0; i < m; ++i) {
                scaled[static_cast<std::size_t>(j) * static_cast<std::size_t>(m) +
                       static_cast<std::size_t>(i)] *= factor;
            }
        }
        expect_factorisation_of(scaled, decompose(m, n, scaled), "scaled columns");
    }
}

// `dbdsqr` accumulates the sweep's rotations into V**T, and then `dbdsqr`
// recomputes V**T from U and B and throws that accumulation away. Measured, not
// assumed: zeroing VT immediately before the recompute leaves every test in the
// tree passing, so for every caller that asks for both -- which is every caller
// there is -- the accumulated V**T is unobservable. That is why the rotation
// appliers survive mutation: not untested, unreachable.
//
// The API still admits U == nullptr with V**T asked for, and on that path the
// accumulation IS the answer. Nothing exercised it. Without U there is no
// factorisation to check, but there is still a property that pins the vectors:
// V**T diagonalises B**T B, with the squared singular values on the diagonal.
TEST(LapackSvdProperties, RightVectorsAccumulatedWithoutU) {
    for (const int n : {2, 3, 4, 6, 9, 14}) {
        for (const char uplo : {'U', 'L'}) {
            const auto d0 = patterned(n, 1, static_cast<std::uint64_t>(n) * 31 + 5);
            const auto e0 = patterned(n - 1, 1, static_cast<std::uint64_t>(n) * 97 + 3);

            std::vector<double> d = d0;
            std::vector<double> e = e0;
            std::vector<double> vt(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
            ASSERT_EQ(ms::cpu::lapack::dbdsqr(uplo, n, d.data(), e.data(), nullptr, 0, vt.data(), n),
                      0)
                << uplo << " " << n;

            // B from the ORIGINAL bidiagonal, since dbdsqr overwrites d and e.
            std::vector<double> b(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
            for (int i = 0; i < n; ++i) {
                b[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(i)] = d0[static_cast<std::size_t>(i)];
            }
            for (int i = 0; i + 1 < n; ++i) {
                const int r = (uplo == 'U') ? i : i + 1;
                const int c = (uplo == 'U') ? i + 1 : i;
                b[static_cast<std::size_t>(c) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(r)] = e0[static_cast<std::size_t>(i)];
            }

            // G = B**T B, symmetric positive semi-definite.
            std::vector<double> g(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
            for (int p = 0; p < n; ++p) {
                for (int q = 0; q < n; ++q) {
                    double acc = 0.0;
                    for (int i = 0; i < n; ++i) {
                        acc += b[static_cast<std::size_t>(p) * static_cast<std::size_t>(n) +
                                 static_cast<std::size_t>(i)] *
                               b[static_cast<std::size_t>(q) * static_cast<std::size_t>(n) +
                                 static_cast<std::size_t>(i)];
                    }
                    g[static_cast<std::size_t>(q) * static_cast<std::size_t>(n) +
                      static_cast<std::size_t>(p)] = acc;
                }
            }

            const double scale = std::max(1.0, max_abs(g));

            // V**T rows are orthonormal.
            for (int p = 0; p < n; ++p) {
                for (int q = p; q < n; ++q) {
                    double acc = 0.0;
                    for (int j = 0; j < n; ++j) {
                        acc += vt[static_cast<std::size_t>(p) * static_cast<std::size_t>(n) +
                                  static_cast<std::size_t>(j)] *
                               vt[static_cast<std::size_t>(q) * static_cast<std::size_t>(n) +
                                  static_cast<std::size_t>(j)];
                    }
                    ASSERT_NEAR(acc, (p == q) ? 1.0 : 0.0, 1e-10)
                        << uplo << " " << n << " orthonormality at (" << p << ", " << q << ")";
                }
            }

            // (V**T G V)[p][q] is sigma_p^2 on the diagonal and zero off it.
            for (int p = 0; p < n; ++p) {
                for (int q = 0; q < n; ++q) {
                    double acc = 0.0;
                    for (int i = 0; i < n; ++i) {
                        double gv = 0.0;
                        for (int j = 0; j < n; ++j) {
                            gv += g[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(i)] *
                                  vt[static_cast<std::size_t>(q) * static_cast<std::size_t>(n) +
                                     static_cast<std::size_t>(j)];
                        }
                        acc += vt[static_cast<std::size_t>(p) * static_cast<std::size_t>(n) +
                                  static_cast<std::size_t>(i)] *
                               gv;
                    }
                    const double want = (p == q) ? d[static_cast<std::size_t>(p)] *
                                                       d[static_cast<std::size_t>(p)]
                                                 : 0.0;
                    ASSERT_NEAR(acc, want, scale * 1e-9)
                        << uplo << " " << n << " V^T B^T B V at (" << p << ", " << q << ")";
                }
            }
        }
    }
}
