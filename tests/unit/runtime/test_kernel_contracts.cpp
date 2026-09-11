// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for the audited ms::cpu::blas / ms::cpu::lapack kernel defects.
//
// Every assertion here is checked against something independent of the kernel
// under test: a defining equation (op(A) * X == alpha * B, A x == b), an identity
// the result must satisfy (Q^T A Q == diag(w), Q^T Q == I), a closed-form spectrum,
// or a naive computation written out in the test itself.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "ms/cpu/blas.hpp"
#include "ms/cpu/lapack.hpp"

namespace {

using std::size_t;

// Column-major accessors for a plain vector<double>.
double get(const std::vector<double>& m, int ld, int r, int c) {
    return m[static_cast<size_t>(c) * static_cast<size_t>(ld) + static_cast<size_t>(r)];
}

void set(std::vector<double>& m, int ld, int r, int c, double v) {
    m[static_cast<size_t>(c) * static_cast<size_t>(ld) + static_cast<size_t>(r)] = v;
}

bool lower_uplo(char uplo) {
    return uplo == 'L' || uplo == 'l';
}

bool no_trans(char t) {
    return t == 'N' || t == 'n';
}

bool unit_diag(char d) {
    return d == 'U' || d == 'u';
}

// Dense k-by-k op(A) exactly as dtrsm is documented to reference it: the named
// triangle only, diagonal forced to 1 for diag='U', transposed for transa='T'.
std::vector<double> dense_op(const std::vector<double>& A, int k, char uplo, char transa, char diag) {
    std::vector<double> T(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    for (int r = 0; r < k; ++r) {
        for (int c = 0; c < k; ++c) {
            const bool in_triangle = lower_uplo(uplo) ? (r >= c) : (r <= c);
            if (!in_triangle) {
                continue;
            }
            const double v = (r == c && unit_diag(diag)) ? 1.0 : get(A, k, r, c);
            if (no_trans(transa)) {
                set(T, k, r, c, v);
            } else {
                set(T, k, c, r, v);
            }
        }
    }
    return T;
}

// Jacobi eigenvalue solver written out here so dsyev is compared against an
// algorithm that shares no code with it.
std::vector<double> jacobi_eigenvalues(std::vector<double> A, int n) {
    for (int sweep = 0; sweep < 200; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                off += get(A, n, p, q) * get(A, n, p, q);
            }
        }
        if (off < 1e-30) {
            break;
        }
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = get(A, n, p, q);
                if (std::abs(apq) < 1e-300) {
                    continue;
                }
                const double theta = (get(A, n, q, q) - get(A, n, p, p)) / (2.0 * apq);
                const double t =
                    (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = get(A, n, k, p);
                    const double akq = get(A, n, k, q);
                    set(A, n, k, p, c * akp - s * akq);
                    set(A, n, k, q, s * akp + c * akq);
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = get(A, n, p, k);
                    const double aqk = get(A, n, q, k);
                    set(A, n, p, k, c * apk - s * aqk);
                    set(A, n, q, k, s * apk + c * aqk);
                }
            }
        }
    }
    std::vector<double> w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        w[static_cast<size_t>(i)] = get(A, n, i, i);
    }
    std::sort(w.begin(), w.end());
    return w;
}

} // namespace

// ---------------------------------------------------------------------------
// dsyev / dsytrd  (audit finding 1, CRITICAL)
// ---------------------------------------------------------------------------

// Before the fix this returned {-208.711, -55.147, 1.231}.
TEST(LapackDsyevContract, DenseThreeByThreeMatchesClosedFormSpectrum) {
    std::vector<double> A{1, 2, 3, 2, 4, 5, 3, 5, 6};
    std::vector<double> w(3);
    ASSERT_EQ(ms::cpu::lapack::dsyev('N', 3, A.data(), 3, w.data()), 0);

    // Roots of det(A - x I) = -x^3 + 11 x^2 + 4 x - 1, computed to 12 digits.
    EXPECT_NEAR(w[0], -0.515729471589, 1e-11);
    EXPECT_NEAR(w[1], 0.170915188827, 1e-11);
    EXPECT_NEAR(w[2], 11.344814282762, 1e-11);

    // trace(A) = 11 and the characteristic polynomial must vanish at each root.
    EXPECT_NEAR(w[0] + w[1] + w[2], 11.0, 1e-12);
    for (double x : w) {
        EXPECT_NEAR(-x * x * x + 11.0 * x * x + 4.0 * x - 1.0, 0.0, 1e-10);
    }
}

// Before the fix the largest eigenvalue of this 4x4 came out around 3.15e6.
TEST(LapackDsyevContract, DenseFourByFourMatchesTraceAndDeterminant) {
    std::vector<double> A{4, 1, 2, 3, 1, 5, 6, 7, 2, 6, 8, 9, 3, 7, 9, 10};
    std::vector<double> w(4);
    ASSERT_EQ(ms::cpu::lapack::dsyev('N', 4, A.data(), 4, w.data()), 0);

    double sum = 0.0;
    double product = 1.0;
    for (double x : w) {
        sum += x;
        product *= x;
    }
    EXPECT_NEAR(sum, 27.0, 1e-11);      // trace
    EXPECT_NEAR(product, -7.0, 1e-9);   // determinant, computed exactly over the rationals
}

TEST(LapackDsyevContract, RandomDenseSimilarityAndOrthonormality) {
    std::mt19937 rng(20240917u);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    for (int trial = 0; trial < 25; ++trial) {
        const int n = 2 + static_cast<int>(rng() % 7u);
        std::vector<double> A0(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                const double v = dist(rng);
                set(A0, n, i, j, v);
                set(A0, n, j, i, v);
            }
        }

        std::vector<double> Q = A0;
        std::vector<double> w(static_cast<size_t>(n));
        ASSERT_EQ(ms::cpu::lapack::dsyev('V', n, Q.data(), n, w.data()), 0) << "n=" << n;

        // Q^T Q == I
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) {
                    s += get(Q, n, k, i) * get(Q, n, k, j);
                }
                EXPECT_NEAR(s, (i == j) ? 1.0 : 0.0, 1e-11) << "Q^T Q at (" << i << "," << j << ")";
            }
        }

        // Q^T A Q == diag(w)
        std::vector<double> AQ(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) {
                    s += get(A0, n, i, k) * get(Q, n, k, j);
                }
                set(AQ, n, i, j, s);
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) {
                    s += get(Q, n, k, i) * get(AQ, n, k, j);
                }
                const double expected = (i == j) ? w[static_cast<size_t>(i)] : 0.0;
                EXPECT_NEAR(s, expected, 1e-10) << "Q^T A Q at (" << i << "," << j << ")";
            }
        }

        // Eigenvalues agree with an independent Jacobi solver, and sum to trace(A).
        const std::vector<double> reference = jacobi_eigenvalues(A0, n);
        double trace = 0.0;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(w[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], 1e-10);
            trace += get(A0, n, i, i);
            sum += w[static_cast<size_t>(i)];
        }
        EXPECT_NEAR(sum, trace, 1e-10);
    }
}

TEST(LapackDsyevContract, TridiagonalControlStillExact) {
    // Already-tridiagonal input was the one case the broken reduction got right;
    // it must keep working.
    std::vector<double> A{2, -1, 0, -1, 2, -1, 0, -1, 2};
    std::vector<double> w(3);
    ASSERT_EQ(ms::cpu::lapack::dsyev('N', 3, A.data(), 3, w.data()), 0);
    EXPECT_NEAR(w[0], 2.0 - std::sqrt(2.0), 1e-13);
    EXPECT_NEAR(w[1], 2.0, 1e-13);
    EXPECT_NEAR(w[2], 2.0 + std::sqrt(2.0), 1e-13);
}

// ---------------------------------------------------------------------------
// dtrsm  (audit finding 2, CRITICAL)
// ---------------------------------------------------------------------------

TEST(BlasDtrsmContract, AllSixteenArgumentCombinationsSolveTheSystem) {
    std::mt19937 rng(987654321u);
    std::uniform_real_distribution<double> dist(-1.5, 1.5);

    const char sides[2] = {'L', 'R'};
    const char uplos[2] = {'L', 'U'};
    const char transes[2] = {'N', 'T'};
    const char diags[2] = {'N', 'U'};

    for (int m : {1, 2, 3, 5}) {
        for (int n : {1, 2, 4}) {
            for (char side : sides) {
                for (char uplo : uplos) {
                    for (char transa : transes) {
                        for (char diag : diags) {
                            const int k = (side == 'L') ? m : n;
                            std::vector<double> A(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
                            for (int r = 0; r < k; ++r) {
                                for (int c = 0; c < k; ++c) {
                                    const double v = dist(rng);
                                    set(A, k, r, c, (r == c) ? (2.0 + std::abs(v)) : v);
                                }
                            }
                            std::vector<double> B(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0);
                            for (int r = 0; r < m; ++r) {
                                for (int c = 0; c < n; ++c) {
                                    set(B, m, r, c, dist(rng));
                                }
                            }
                            const double alpha = (diag == 'N') ? 1.0 : 2.5;

                            std::vector<double> X = B;
                            ms::cpu::blas::dtrsm(
                                side, uplo, transa, diag, m, n, alpha, A.data(), k, X.data(), m);

                            const std::vector<double> T = dense_op(A, k, uplo, transa, diag);
                            for (int r = 0; r < m; ++r) {
                                for (int c = 0; c < n; ++c) {
                                    double acc = 0.0;
                                    if (side == 'L') {
                                        for (int p = 0; p < m; ++p) {
                                            acc += get(T, m, r, p) * get(X, m, p, c);
                                        }
                                    } else {
                                        for (int p = 0; p < n; ++p) {
                                            acc += get(X, m, r, p) * get(T, n, p, c);
                                        }
                                    }
                                    EXPECT_NEAR(acc, alpha * get(B, m, r, c), 1e-11)
                                        << side << uplo << transa << diag << " m=" << m << " n=" << n
                                        << " at (" << r << "," << c << ")";
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Each of these returned a specific wrong number before the fix.
TEST(BlasDtrsmContract, ConcreteCasesFromTheAudit) {
    {
        // was -2
        double A[1] = {2.0};
        double B[1] = {4.0};
        ms::cpu::blas::dtrsm('L', 'L', 'T', 'N', 1, 1, 1.0, A, 1, B, 1);
        EXPECT_NEAR(B[0], 2.0, 1e-14);
    }
    {
        // was [-5.5, -6.75]
        double A[4] = {2.0, 0.0, 1.0, 4.0}; // [[2,1],[0,4]] column-major
        double B[2] = {2.0, 9.0};
        ms::cpu::blas::dtrsm('L', 'U', 'T', 'N', 2, 1, 1.0, A, 2, B, 2);
        EXPECT_NEAR(B[0], 1.0, 1e-14);
        EXPECT_NEAR(B[1], 2.0, 1e-14);
    }
    {
        // was [-1, -2.5]
        double A[4] = {1.0, 0.0, 1.0, 1.0}; // [[1,1],[0,1]] column-major
        double B[2] = {1.0, 2.0};
        ms::cpu::blas::dtrsm('R', 'U', 'N', 'N', 1, 2, 1.0, A, 2, B, 1);
        EXPECT_NEAR(B[0], 1.0, 1e-14);
        EXPECT_NEAR(B[1], 1.0, 1e-14);
    }
    {
        // was [[-7,-9],[-6.67,-8]]
        double A[4] = {1.0, 3.0, 0.0, 1.0}; // [[1,0],[3,1]] column-major
        double B[4] = {7.0, 15.0, 2.0, 4.0}; // = [[1,2],[3,4]] * A
        ms::cpu::blas::dtrsm('R', 'L', 'N', 'N', 2, 2, 1.0, A, 2, B, 2);
        EXPECT_NEAR(B[0], 1.0, 1e-13);
        EXPECT_NEAR(B[1], 3.0, 1e-13);
        EXPECT_NEAR(B[2], 2.0, 1e-13);
        EXPECT_NEAR(B[3], 4.0, 1e-13);
    }
}

TEST(BlasDtrsmContract, UnrecognisedSideOrUploLeavesBUntouched) {
    double A[4] = {2.0, 0.0, 1.0, 3.0};
    double B[4] = {1.0, 2.0, 3.0, 4.0};
    ms::cpu::blas::dtrsm('X', 'U', 'N', 'N', 2, 2, 1.0, A, 2, B, 2);
    EXPECT_DOUBLE_EQ(B[0], 1.0);
    ms::cpu::blas::dtrsm('L', 'Z', 'N', 'N', 2, 2, 1.0, A, 2, B, 2);
    EXPECT_DOUBLE_EQ(B[0], 1.0);
}

// ---------------------------------------------------------------------------
// dgetrs  (audit finding 3, HIGH) and dgetrf singularity (finding 9, MEDIUM)
// ---------------------------------------------------------------------------

// Before the fix dgetrs('T', ...) returned the caller's right-hand side unchanged.
TEST(LapackDgetrsContract, TransposedSolveIsNotANoOp) {
    // A = [[1,2],[3,4]]; A^T x = [5, 11] has the exact solution [6.5, -0.5].
    std::vector<double> LU{1.0, 3.0, 2.0, 4.0};
    std::vector<int> ipiv(2);
    ASSERT_EQ(ms::cpu::lapack::dgetrf(2, 2, LU.data(), 2, ipiv.data()), 0);

    std::vector<double> b{5.0, 11.0};
    ms::cpu::lapack::dgetrs('T', 2, 1, LU.data(), 2, ipiv.data(), b.data(), 2);
    EXPECT_NEAR(b[0], 6.5, 1e-12);
    EXPECT_NEAR(b[1], -0.5, 1e-12);
}

TEST(LapackDgetrsContract, RandomSystemsSatisfyTheirDefiningEquation) {
    std::mt19937 rng(13579u);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    for (int trial = 0; trial < 40; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 8u);
        const int nrhs = 1 + static_cast<int>(rng() % 3u);

        std::vector<double> A0(static_cast<size_t>(n) * static_cast<size_t>(n));
        for (double& v : A0) {
            v = dist(rng);
        }
        for (int i = 0; i < n; ++i) {
            set(A0, n, i, i, get(A0, n, i, i) + 4.0); // diagonally dominant
        }
        std::vector<double> B0(static_cast<size_t>(n) * static_cast<size_t>(nrhs));
        for (double& v : B0) {
            v = dist(rng);
        }

        std::vector<double> LU = A0;
        std::vector<int> ipiv(static_cast<size_t>(n));
        ASSERT_EQ(ms::cpu::lapack::dgetrf(n, n, LU.data(), n, ipiv.data()), 0);

        for (int mode = 0; mode < 2; ++mode) {
            const char trans = (mode == 0) ? 'N' : 'T';
            std::vector<double> X = B0;
            ms::cpu::lapack::dgetrs(trans, n, nrhs, LU.data(), n, ipiv.data(), X.data(), n);
            for (int c = 0; c < nrhs; ++c) {
                for (int r = 0; r < n; ++r) {
                    double acc = 0.0;
                    for (int p = 0; p < n; ++p) {
                        const double a = (mode == 0) ? get(A0, n, r, p) : get(A0, n, p, r);
                        acc += a * get(X, n, p, c);
                    }
                    EXPECT_NEAR(acc, get(B0, n, r, c), 1e-10)
                        << "trans=" << trans << " n=" << n << " at (" << r << "," << c << ")";
                }
            }
        }
    }
}

TEST(LapackDgetrsContract, UnrecognisedTransLeavesBUntouched) {
    std::vector<double> LU{2.0, 1.0, 1.0, 3.0};
    std::vector<int> ipiv(2);
    ASSERT_EQ(ms::cpu::lapack::dgetrf(2, 2, LU.data(), 2, ipiv.data()), 0);
    std::vector<double> b{5.0, 10.0};
    ms::cpu::lapack::dgetrs('X', 2, 1, LU.data(), 2, ipiv.data(), b.data(), 2);
    EXPECT_DOUBLE_EQ(b[0], 5.0);
    EXPECT_DOUBLE_EQ(b[1], 10.0);
}

// Before the fix the absolute epsilon test reported c * I singular for c < 2.2e-16.
TEST(LapackDgetrfContract, SingularityTestIsScaleInvariant) {
    for (double c : {1e-30, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e30}) {
        std::vector<double> A(9, 0.0);
        for (int i = 0; i < 3; ++i) {
            set(A, 3, i, i, c);
        }
        std::vector<int> ipiv(3);
        EXPECT_EQ(ms::cpu::lapack::dgetrf(3, 3, A.data(), 3, ipiv.data()), 0)
            << "c = " << c << " has condition number 1 and must factorize";
        for (int i = 0; i < 3; ++i) {
            EXPECT_DOUBLE_EQ(get(A, 3, i, i), c);
        }
    }
}

TEST(LapackDgetrfContract, GenuinelySingularMatricesStillReported) {
    {
        std::vector<double> A{1.0, 2.0, 2.0, 4.0}; // rank 1
        std::vector<int> ipiv(2);
        EXPECT_EQ(ms::cpu::lapack::dgetrf(2, 2, A.data(), 2, ipiv.data()), 2);
    }
    {
        // Same matrix scaled to 1e-20: still rank 1, still reported.
        std::vector<double> A{1e-20, 2e-20, 2e-20, 4e-20};
        std::vector<int> ipiv(2);
        EXPECT_EQ(ms::cpu::lapack::dgetrf(2, 2, A.data(), 2, ipiv.data()), 2);
    }
    {
        std::vector<double> A(4, 0.0);
        std::vector<int> ipiv(2);
        EXPECT_EQ(ms::cpu::lapack::dgetrf(2, 2, A.data(), 2, ipiv.data()), 1);
    }
}

// ---------------------------------------------------------------------------
// dormqr side='R'  (audit finding 10, MEDIUM)
// ---------------------------------------------------------------------------

TEST(LapackDormqrContract, RightSideAppliesQInsteadOfNoOp) {
    std::mt19937 rng(24680u);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    for (int trial = 0; trial < 20; ++trial) {
        const int nq = 2 + static_cast<int>(rng() % 5u);
        const int k = 1 + static_cast<int>(rng() % static_cast<unsigned>(nq));
        const int mc = 1 + static_cast<int>(rng() % 4u);

        std::vector<double> A(static_cast<size_t>(nq) * static_cast<size_t>(k));
        for (double& v : A) {
            v = dist(rng);
        }
        std::vector<double> tau(static_cast<size_t>(k));
        ASSERT_EQ(ms::cpu::lapack::dgeqrf(nq, k, A.data(), nq, tau.data()), 0);

        // Q built explicitly through the (already correct) side='L' path.
        std::vector<double> Q(static_cast<size_t>(nq) * static_cast<size_t>(nq), 0.0);
        for (int i = 0; i < nq; ++i) {
            set(Q, nq, i, i, 1.0);
        }
        ms::cpu::lapack::dormqr('L', 'N', nq, nq, k, A.data(), nq, tau.data(), Q.data(), nq);

        std::vector<double> C0(static_cast<size_t>(mc) * static_cast<size_t>(nq));
        for (double& v : C0) {
            v = dist(rng);
        }

        std::vector<double> C = C0;
        ms::cpu::lapack::dormqr('R', 'N', mc, nq, k, A.data(), nq, tau.data(), C.data(), mc);
        for (int r = 0; r < mc; ++r) {
            for (int c = 0; c < nq; ++c) {
                double acc = 0.0;
                for (int p = 0; p < nq; ++p) {
                    acc += get(C0, mc, r, p) * get(Q, nq, p, c);
                }
                EXPECT_NEAR(get(C, mc, r, c), acc, 1e-12) << "C*Q at (" << r << "," << c << ")";
            }
        }

        std::vector<double> Ct = C0;
        ms::cpu::lapack::dormqr('R', 'T', mc, nq, k, A.data(), nq, tau.data(), Ct.data(), mc);
        for (int r = 0; r < mc; ++r) {
            for (int c = 0; c < nq; ++c) {
                double acc = 0.0;
                for (int p = 0; p < nq; ++p) {
                    acc += get(C0, mc, r, p) * get(Q, nq, c, p); // Q^T
                }
                EXPECT_NEAR(get(Ct, mc, r, c), acc, 1e-12) << "C*Q^T at (" << r << "," << c << ")";
            }
        }

        // Round trip.
        ms::cpu::lapack::dormqr('R', 'T', mc, nq, k, A.data(), nq, tau.data(), C.data(), mc);
        for (size_t i = 0; i < C.size(); ++i) {
            EXPECT_NEAR(C[i], C0[i], 1e-12);
        }
    }
}

TEST(LapackDormqrContract, UnrecognisedSideLeavesCUntouched) {
    std::vector<double> A{1.0, 2.0, 3.0, 4.0};
    std::vector<double> tau(2, 0.0);
    ASSERT_EQ(ms::cpu::lapack::dgeqrf(2, 2, A.data(), 2, tau.data()), 0);
    std::vector<double> C{1.0, 2.0, 3.0, 4.0};
    ms::cpu::lapack::dormqr('X', 'N', 2, 2, 2, A.data(), 2, tau.data(), C.data(), 2);
    EXPECT_DOUBLE_EQ(C[0], 1.0);
    EXPECT_DOUBLE_EQ(C[3], 4.0);
}
