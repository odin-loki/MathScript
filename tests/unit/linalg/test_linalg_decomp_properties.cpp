// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4/§8.6: what the Schur, Hessenberg and LDL decompositions actually promise.
//
// `src/linalg/decompositions.cpp` scored 50.0% over viable mutants -- twelve of
// twenty-four survived, and none of them failed to compile. They fell into two
// groups, and both groups say the same thing about the tests in front of them.
//
// The unit tests assert SHAPES. In full:
//
//     schur_factorization    T.rows() == 3 and Q.cols() == 3
//     bidiagonal_reduction   B.rows() == 3 and B.cols() == 2
//     hessenberg_form        H.rows() == 3, and H(2, 0) is about zero
//     ldl_3x3                L.rows() == 3
//
// A decomposition that returned the right-sized matrices of zeros passes all four.
// The numerical reference suite does better -- `SchurDecomp.T_Is_Upper_Triangular_
// Or_Quasi` asserts A = Q T Q**T and Q**T Q = I -- but only for one 3x3 SYMMETRIC
// matrix, whose eigenvalues are all real. The Francis double shift exists for the
// case that matrix does not have: a complex conjugate pair, which leaves a real
// 2x2 block on T's diagonal and is the whole reason the shift is a quadratic in H
// rather than a scalar.
//
// So this file asserts the defining identity of each, on inputs that reach the
// paths the existing ones do not:
//
//     A = Q T Q**T,  Q**T Q = I,  T quasi-upper-triangular   (schur)
//     H(i, j) = 0 for i > j + 1, and H similar to A          (hess)
//     P**T A P = L D L**T, L unit lower triangular           (ldl)
//
// Everything is computed with plain loops. `hess` returns H alone -- there is no Q
// to check it against -- so similarity is asserted through the power sums
// tr(A), tr(A^2), tr(A^3), which determine the characteristic polynomial and which
// a wrong reduction does not preserve.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ms/error/error_types.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

DMatrix patterned_square(size_t n, std::uint64_t seed) {
    DMatrix A(n, n, 0.0);
    std::uint64_t state = seed | 1ULL;
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            A(i, j) = static_cast<double>(static_cast<int>((state >> 33) % 1201) - 600) / 100.0;
        }
    }
    return A;
}

DMatrix symmetrise(DMatrix A) {
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = i + 1; j < A.cols(); ++j) {
            const double v = 0.5 * (A(i, j) + A(j, i));
            A(i, j) = v;
            A(j, i) = v;
        }
    }
    return A;
}

double largest_magnitude(const DMatrix& A) {
    double worst = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            worst = std::max(worst, std::abs(A(i, j)));
        }
    }
    return worst;
}

// tr(A^k) for k = 1, 2, 3. Newton's identities turn these into the first three
// coefficients of the characteristic polynomial, so two matrices that agree on all
// three agree on the part of the spectrum any similarity transform must preserve.
void power_sums(const DMatrix& A, double out[3]) {
    const size_t n = A.rows();
    DMatrix p = A;
    for (int k = 0; k < 3; ++k) {
        double trace = 0.0;
        for (size_t i = 0; i < n; ++i) {
            trace += p(i, i);
        }
        out[k] = trace;
        if (k == 2) {
            break;
        }
        DMatrix next(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (size_t t = 0; t < n; ++t) {
                    acc += p(i, t) * A(t, j);
                }
                next(i, j) = acc;
            }
        }
        p = next;
    }
}

void expect_schur_of(const DMatrix& A, const char* what) {
    const size_t n = A.rows();
    auto result = schur(A);
    ASSERT_TRUE(result.has_value()) << what << " " << n;
    const DMatrix& T = result->T;
    const DMatrix& Q = result->Q;
    ASSERT_EQ(T.rows(), n);
    ASSERT_EQ(Q.rows(), n);

    const double scale = std::max(1.0, largest_magnitude(A));
    const double tol = scale * 1e-9 * static_cast<double>(n);

    // Q**T Q = I.
    for (size_t p = 0; p < n; ++p) {
        for (size_t q = p; q < n; ++q) {
            double acc = 0.0;
            for (size_t i = 0; i < n; ++i) {
                acc += Q(i, p) * Q(i, q);
            }
            ASSERT_NEAR(acc, (p == q) ? 1.0 : 0.0, 1e-10)
                << what << " " << n << " Q^T Q at (" << p << ", " << q << ")";
        }
    }

    // A = Q T Q**T.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t p = 0; p < n; ++p) {
                for (size_t q = 0; q < n; ++q) {
                    acc += Q(i, p) * T(p, q) * Q(j, q);
                }
            }
            ASSERT_NEAR(acc, A(i, j), tol)
                << what << " " << n << " A != Q T Q^T at (" << i << ", " << j << ")";
        }
    }

    // T is quasi-upper-triangular: everything below the sub-diagonal is zero, and no
    // two consecutive sub-diagonal entries are both non-zero -- a 2x2 block may sit
    // on the diagonal, but three rows may not be coupled.
    for (size_t i = 2; i < n; ++i) {
        for (size_t j = 0; j + 2 <= i; ++j) {
            ASSERT_NEAR(T(i, j), 0.0, tol)
                << what << " " << n << " T not quasi-triangular at (" << i << ", " << j << ")";
        }
    }
    for (size_t i = 1; i + 1 < n; ++i) {
        const bool a = std::abs(T(i, i - 1)) > tol;
        const bool b = std::abs(T(i + 1, i)) > tol;
        ASSERT_FALSE(a && b)
            << what << " " << n << " three coupled rows at " << i;
    }
}

void expect_ldl_of(const DMatrix& A, const char* what) {
    const size_t n = A.rows();
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value()) << what << " " << n;
    const DMatrix& L = result->L;
    const DMatrix& D = result->D;
    const DMatrix& P = result->P;

    const double scale = std::max(1.0, largest_magnitude(A));
    const double tol = scale * 1e-8 * static_cast<double>(n);

    // L is unit lower triangular.
    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(L(i, i), 1.0, 1e-12) << what << " L diagonal at " << i;
        for (size_t j = i + 1; j < n; ++j) {
            ASSERT_NEAR(L(i, j), 0.0, 1e-12) << what << " L upper at (" << i << ", " << j << ")";
        }
    }

    // P is a permutation: one 1 per row and per column, nothing else.
    for (size_t i = 0; i < n; ++i) {
        double row = 0.0;
        double col = 0.0;
        for (size_t j = 0; j < n; ++j) {
            ASSERT_TRUE(P(i, j) == 0.0 || P(i, j) == 1.0) << what << " P entry";
            row += P(i, j);
            col += P(j, i);
        }
        ASSERT_DOUBLE_EQ(row, 1.0) << what << " P row " << i;
        ASSERT_DOUBLE_EQ(col, 1.0) << what << " P column " << i;
    }

    // P**T A P = L D L**T, entry by entry.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double lhs = 0.0;
            for (size_t p = 0; p < n; ++p) {
                for (size_t q = 0; q < n; ++q) {
                    lhs += P(p, i) * A(p, q) * P(q, j);
                }
            }
            double rhs = 0.0;
            for (size_t t = 0; t < n; ++t) {
                rhs += L(i, t) * D(t, 0) * L(j, t);
            }
            ASSERT_NEAR(lhs, rhs, tol)
                << what << " " << n << " P^T A P != L D L^T at (" << i << ", " << j << ")";
        }
    }
}

} // namespace

TEST(LinalgDecompProperties, SchurFactorisesEveryMatrixItIsGiven) {
    // Symmetric first, which is what the reference suite covers, and then the cases
    // it does not: general non-symmetric matrices, whose spectra include complex
    // conjugate pairs and therefore 2x2 blocks on T's diagonal.
    for (const size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5},
                           size_t{7}, size_t{9}, size_t{12}}) {
        expect_schur_of(symmetrise(patterned_square(n, n * 7919 + 11)), "symmetric");
        expect_schur_of(patterned_square(n, n * 104729 + 3), "general");
    }

    // A rotation block: eigenvalues exp(+-i*theta), purely complex, no real
    // eigenvalue anywhere. The double shift is the only thing that converges here.
    DMatrix rot(4, 4, 0.0);
    const double c = std::cos(0.7);
    const double s = std::sin(0.7);
    rot(0, 0) = c;  rot(0, 1) = -s;
    rot(1, 0) = s;  rot(1, 1) = c;
    rot(2, 2) = 2.0 * c;  rot(2, 3) = -2.0 * s;
    rot(3, 2) = 2.0 * s;  rot(3, 3) = 2.0 * c;
    expect_schur_of(rot, "rotation blocks");

    // A companion matrix of (x^2 + 1)(x^2 + 4)(x - 3): two complex pairs and one
    // real eigenvalue, so T must end up with two 2x2 blocks and one 1x1.
    const double coeffs[5] = {-12.0, 4.0, -5.0, 4.0, -4.0};
    DMatrix comp(5, 5, 0.0);
    for (size_t i = 1; i < 5; ++i) {
        comp(i, i - 1) = 1.0;
    }
    for (size_t i = 0; i < 5; ++i) {
        comp(i, 4) = coeffs[i];
    }
    expect_schur_of(comp, "companion");

    // Already triangular, already diagonal, and zero: the deflation path with
    // nothing to do.
    DMatrix tri{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
    expect_schur_of(tri, "upper triangular");
    expect_schur_of(eye<double>(6), "identity");
    expect_schur_of(DMatrix(4, 4, 0.0), "zero");

    // Repeated eigenvalues, where the shift strategy has no gap to exploit.
    DMatrix jordanish(5, 5, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        jordanish(i, i) = 2.0;
    }
    jordanish(0, 1) = 1.0;
    jordanish(1, 2) = 1.0;
    jordanish(3, 4) = 1.0;
    expect_schur_of(jordanish, "repeated eigenvalues");
}

TEST(LinalgDecompProperties, HessenbergIsSimilarAndActuallyHessenberg) {
    for (const size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{6},
                           size_t{8}, size_t{11}}) {
        const DMatrix A = patterned_square(n, n * 31337 + 7);
        auto H = hess(A);
        ASSERT_TRUE(H.has_value()) << n;
        ASSERT_EQ(H->rows(), n);

        const double scale = std::max(1.0, largest_magnitude(A));
        const double tol = scale * scale * scale * 1e-9 * static_cast<double>(n);

        // Everything below the first sub-diagonal is zero.
        for (size_t i = 2; i < n; ++i) {
            for (size_t j = 0; j + 2 <= i; ++j) {
                ASSERT_NEAR((*H)(i, j), 0.0, 1e-10)
                    << n << " not Hessenberg at (" << i << ", " << j << ")";
            }
        }

        // And it is the same matrix in a different basis. Shapes cannot see this:
        // a reduction that zeroed the lower triangle and stopped would pass the
        // check above and fail every one of these.
        double want[3];
        double got[3];
        power_sums(A, want);
        power_sums(*H, got);
        for (int k = 0; k < 3; ++k) {
            ASSERT_NEAR(got[k], want[k], tol) << n << " tr(H^" << (k + 1) << ") != tr(A^"
                                              << (k + 1) << ")";
        }
    }
}

TEST(LinalgDecompProperties, LdlFactorisesEverySymmetricMatrixItAccepts) {
    for (const size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{6},
                           size_t{9}}) {
        // Positive definite: A**T A + n*I, which always has a usable pivot.
        DMatrix g = patterned_square(n, n * 6151 + 13);
        DMatrix spd(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double acc = (i == j) ? static_cast<double>(n) : 0.0;
                for (size_t t = 0; t < n; ++t) {
                    acc += g(t, i) * g(t, j);
                }
                spd(i, j) = acc;
            }
        }
        expect_ldl_of(spd, "positive definite");

        // Indefinite, which is the case a Cholesky cannot take and this can.
        DMatrix indef = symmetrise(patterned_square(n, n * 2749 + 5));
        for (size_t i = 0; i < n; ++i) {
            indef(i, i) += (i % 2 == 0) ? 4.0 : -4.0;
        }
        expect_ldl_of(indef, "indefinite");

        expect_ldl_of(eye<double>(n), "identity");
    }

    // A tiny leading diagonal beside a large one: the case threshold pivoting is
    // for. Without the interchange, dividing by 1e-14 loses the answer.
    DMatrix needs_pivot{{1e-14, 1.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 1.0, 3.0}};
    expect_ldl_of(needs_pivot, "needs pivoting");

    // The identity holds whether or not the interchange happens -- a factorisation
    // of the permuted matrix is still a factorisation -- so reconstruction cannot
    // see the pivoting decision at all. Both halves of the rule need asserting
    // directly, and they need different matrices to separate them.

    // It fires when the natural pivot has lost its precision: 1e-14 against a best
    // of 4 is below the 1e-8 threshold, so row and column 0 must move.
    auto pivoted = ldl(needs_pivot);
    ASSERT_TRUE(pivoted.has_value());
    EXPECT_DOUBLE_EQ(pivoted->P(0, 0), 0.0)
        << "an unusable pivot was kept: P is the identity";

    // It does NOT fire merely because a larger diagonal exists elsewhere. Here the
    // natural pivots are 1, 9, 5: the first is nine times smaller than the best and
    // perfectly healthy, so the threshold rule must leave it alone. Only a matrix
    // shaped like this can tell `<threshold AND elsewhere` from `<threshold OR
    // elsewhere` -- in a diagonally dominant one the best pivot is already the
    // natural one, and the second half of the condition never differs.
    DMatrix healthy_but_not_largest{{1.0, 0.0, 0.0}, {0.0, 9.0, 0.0}, {0.0, 0.0, 5.0}};
    auto plain = ldl(healthy_but_not_largest);
    ASSERT_TRUE(plain.has_value());
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(plain->P(i, j), (i == j) ? 1.0 : 0.0)
                << "a healthy pivot was interchanged at (" << i << ", " << j << ")";
        }
    }
    expect_ldl_of(healthy_but_not_largest, "healthy but not largest");

    DMatrix diagonally_dominant{{9.0, 1.0, 0.5}, {1.0, 8.0, 1.5}, {0.5, 1.5, 7.0}};
    expect_ldl_of(diagonally_dominant, "diagonally dominant");
}

TEST(LinalgDecompProperties, LdlDistinguishesSingularFromUnsupported) {
    // The no-usable-pivot branch has two outcomes and nothing asserted either. The
    // zero matrix has no pivot because there is nothing there at all.
    auto singular = ldl(DMatrix(3, 3, 0.0));
    ASSERT_FALSE(singular.has_value());
    EXPECT_NE(format_error(singular.error()).find("ingular"), std::string::npos)
        << format_error(singular.error());

    // This one has no 1x1 pivot and is not singular: its off-diagonal is what
    // carries it, which is exactly the 2x2 block pivot that is not implemented.
    DMatrix needs_2x2{{0.0, 1.0}, {1.0, 0.0}};
    auto unsupported = ldl(needs_2x2);
    ASSERT_FALSE(unsupported.has_value());
    EXPECT_NE(format_error(unsupported.error()).find("2x2"), std::string::npos)
        << format_error(unsupported.error());

    // And the same shape appearing in a TRAILING block rather than at the top, so
    // the branch is reached after some columns have already been eliminated.
    DMatrix trailing{{5.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, {0.0, 2.0, 0.0}};
    auto trailing_result = ldl(trailing);
    ASSERT_FALSE(trailing_result.has_value());
    EXPECT_NE(format_error(trailing_result.error()).find("2x2"), std::string::npos)
        << format_error(trailing_result.error());

    // The same, but where the eliminated column actually contributed: the branch
    // that decides singular-versus-unsupported subtracts L(i,t)*L(k,t)*D(t) from
    // each trailing off-diagonal, and in `trailing` above every one of those L
    // entries is zero, so the subtraction is a no-op whatever it subtracts. Here
    // the first column eliminates to L(1,0) = L(2,0) = 1, the Schur complement is
    // [[0, 2], [2, 0]], and the correction is the difference between reporting an
    // unsupported 2x2 block and reporting a singular matrix.
    DMatrix trailing_coupled{{1.0, 1.0, 1.0}, {1.0, 1.0, 3.0}, {1.0, 3.0, 1.0}};
    auto coupled = ldl(trailing_coupled);
    ASSERT_FALSE(coupled.has_value());
    EXPECT_NE(format_error(coupled.error()).find("2x2"), std::string::npos)
        << format_error(coupled.error());
}
