// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for the linalg decomposition contracts that the audit
// found violated:
//   * hess() applied its Householder reflectors only from the LEFT, so it was
//     not a similarity (trace/determinant/spectrum were not preserved).
//   * schur() seeded Q from the identity and ran unshifted QR, so
//     A == Q*T*Q^T held for no non-trivial input.
//   * bidiag() used the Hessenberg reflector, leaving B tridiagonal.
//   * ldl() returned a hard-coded identity P and reported SingularMatrix for
//     nonsingular indefinite input.
// Every assertion below fails against those old behaviours.

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

DMatrix mat(size_t r, size_t c, std::initializer_list<double> vals) {
    DMatrix A(r, c, 0.0);
    size_t k = 0;
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            A(i, j) = *(vals.begin() + k++);
        }
    }
    return A;
}

DMatrix mul(const DMatrix& A, const DMatrix& B) {
    DMatrix C(A.rows(), B.cols(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

DMatrix tp(const DMatrix& A) {
    DMatrix T(A.cols(), A.rows(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            T(j, i) = A(i, j);
        }
    }
    return T;
}

DMatrix identity(size_t n) {
    DMatrix I(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        I(i, i) = 1.0;
    }
    return I;
}

double max_abs_diff(const DMatrix& A, const DMatrix& B) {
    double d = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            d = std::max(d, std::abs(A(i, j) - B(i, j)));
        }
    }
    return d;
}

double trace_of(const DMatrix& A) {
    double t = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        t += A(i, i);
    }
    return t;
}

// Characteristic polynomial by Faddeev-LeVerrier, computed here so the test
// does not depend on any eigen routine from the library under test.
// Returns c with det(lambda I - A) = lambda^n + c[0] lambda^(n-1) + ... + c[n-1].
std::vector<double> char_poly(const DMatrix& A) {
    const size_t n = A.rows();
    std::vector<double> c(n, 0.0);
    DMatrix Mk = identity(n);
    for (size_t k = 1; k <= n; ++k) {
        const DMatrix AM = mul(A, Mk);
        const double ck = -trace_of(AM) / static_cast<double>(k);
        c[k - 1] = ck;
        if (k < n) {
            Mk = AM;
            for (size_t i = 0; i < n; ++i) {
                Mk(i, i) += ck;
            }
        }
    }
    return c;
}

// det by Gaussian elimination with partial pivoting (independent of ms::det).
double det_of(DMatrix A) {
    const size_t n = A.rows();
    double d = 1.0;
    for (size_t k = 0; k < n; ++k) {
        size_t piv = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(A(i, k)) > std::abs(A(piv, k))) {
                piv = i;
            }
        }
        if (std::abs(A(piv, k)) < 1e-300) {
            return 0.0;
        }
        if (piv != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(A(k, j), A(piv, j));
            }
            d = -d;
        }
        d *= A(k, k);
        for (size_t i = k + 1; i < n; ++i) {
            const double f = A(i, k) / A(k, k);
            for (size_t j = k; j < n; ++j) {
                A(i, j) -= f * A(k, j);
            }
        }
    }
    return d;
}

const DMatrix kGen4 = mat(4, 4, {1, 2, 3, 4,
                                 5, 6, 7, 8,
                                 9, 10, 11, 13,
                                 2, 4, 1, 3});

const DMatrix kSym3 = mat(3, 3, {4, 1, -2,
                                 1, 2, 0,
                                 -2, 0, 3});

} // namespace

// ---------------------------------------------------------------------------
// hess: must be a similarity transform, not a one-sided QR-style factor
// ---------------------------------------------------------------------------

TEST(HessSimilarity, PreservesTrace) {
    // Old behaviour: trace(hess(kGen4)) == -13.15 against trace(A) == 21.
    const auto H = hess(kGen4);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR(trace_of(*H), trace_of(kGen4), 1e-12);
    EXPECT_NEAR(trace_of(*H), 21.0, 1e-12);
}

TEST(HessSimilarity, PreservesDeterminant) {
    const auto H = hess(kGen4);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR(det_of(*H), det_of(kGen4), 1e-9 * std::abs(det_of(kGen4)));
}

TEST(HessSimilarity, PreservesCharacteristicPolynomial) {
    // Equality of every characteristic-polynomial coefficient is exactly the
    // statement that H and A are similar (same spectrum with multiplicity).
    const auto H = hess(kGen4);
    ASSERT_TRUE(H.has_value());
    const std::vector<double> ca = char_poly(kGen4);
    const std::vector<double> ch = char_poly(*H);
    ASSERT_EQ(ca.size(), ch.size());
    double scale = 1.0;
    for (double v : ca) {
        scale = std::max(scale, std::abs(v));
    }
    for (size_t i = 0; i < ca.size(); ++i) {
        EXPECT_NEAR(ch[i], ca[i], 1e-9 * scale) << "coefficient " << i;
    }
}

TEST(HessSimilarity, SymmetricInputBecomesTridiagonal) {
    // A similarity keeps symmetry, so a symmetric A must reduce to symmetric
    // TRIDIAGONAL form. The one-sided version returned a non-symmetric matrix
    // of trace 4.447 for this input (true trace 9).
    const auto H = hess(kSym3);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR(trace_of(*H), 9.0, 1e-12);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR((*H)(i, j), (*H)(j, i), 1e-12)
                << "symmetry lost at (" << i << "," << j << ")";
        }
    }
    EXPECT_NEAR((*H)(2, 0), 0.0, 1e-12);
    EXPECT_NEAR((*H)(0, 2), 0.0, 1e-12);
}

TEST(HessSimilarity, ShapeIsUpperHessenberg) {
    const auto H = hess(kGen4);
    ASSERT_TRUE(H.has_value());
    for (size_t j = 0; j + 2 < 4; ++j) {
        for (size_t i = j + 2; i < 4; ++i) {
            EXPECT_NEAR((*H)(i, j), 0.0, 1e-13);
        }
    }
}

// ---------------------------------------------------------------------------
// schur: A == Q T Q^T, Q orthogonal, T quasi-upper-triangular
// ---------------------------------------------------------------------------

TEST(SchurIdentity, ReconstructsGeneralMatrix) {
    const auto sr = schur(kGen4);
    ASSERT_TRUE(sr.has_value());
    const DMatrix rec = mul(sr->Q, mul(sr->T, tp(sr->Q)));
    EXPECT_LT(max_abs_diff(rec, kGen4), 1e-11);
}

TEST(SchurIdentity, QIsOrthogonal) {
    const auto sr = schur(kGen4);
    ASSERT_TRUE(sr.has_value());
    EXPECT_LT(max_abs_diff(mul(tp(sr->Q), sr->Q), identity(4)), 1e-13);
}

TEST(SchurIdentity, SymmetricEigenvaluesMatchClosedForm) {
    // A = [4,1,0; 1,3,1; 0,1,2] has characteristic polynomial
    // lambda^3 - 9 lambda^2 + 24 lambda - 18 = (lambda - 3)(lambda^2 - 6 lambda + 6),
    // i.e. the exact spectrum {3, 3 - sqrt(3), 3 + sqrt(3)}.
    // The old code returned diag(T) = {3.865, -2.633, 1.769}, trace 3.
    const DMatrix A = mat(3, 3, {4, 1, 0, 1, 3, 1, 0, 1, 2});
    const auto sr = schur(A);
    ASSERT_TRUE(sr.has_value());
    std::vector<double> d{sr->T(0, 0), sr->T(1, 1), sr->T(2, 2)};
    std::sort(d.begin(), d.end());
    EXPECT_NEAR(d[0], 3.0 - std::sqrt(3.0), 1e-10);
    EXPECT_NEAR(d[1], 3.0, 1e-10);
    EXPECT_NEAR(d[2], 3.0 + std::sqrt(3.0), 1e-10);
    EXPECT_NEAR(trace_of(sr->T), 9.0, 1e-12);
    EXPECT_LT(max_abs_diff(mul(sr->Q, mul(sr->T, tp(sr->Q))), A), 1e-12);
}

TEST(SchurIdentity, ComplexPairStaysAs2x2Block) {
    // Rotation-like block: eigenvalues 1 +- 2i. Real Schur form keeps the 2x2.
    const DMatrix A = mat(2, 2, {1, -2, 2, 1});
    const auto sr = schur(A);
    ASSERT_TRUE(sr.has_value());
    EXPECT_LT(max_abs_diff(mul(sr->Q, mul(sr->T, tp(sr->Q))), A), 1e-13);
    const double re = 0.5 * (sr->T(0, 0) + sr->T(1, 1));
    const double half = 0.5 * (sr->T(0, 0) - sr->T(1, 1));
    const double disc = half * half + sr->T(0, 1) * sr->T(1, 0);
    ASSERT_LT(disc, 0.0) << "block should carry a complex pair";
    EXPECT_NEAR(re, 1.0, 1e-12);
    EXPECT_NEAR(std::sqrt(-disc), 2.0, 1e-12);
}

TEST(SchurIdentity, NeverTwoAdjacentNonzeroSubdiagonals) {
    // A well-formed real Schur factor has only isolated 2x2 blocks.
    const auto sr = schur(kGen4);
    ASSERT_TRUE(sr.has_value());
    for (size_t i = 1; i + 1 < 4; ++i) {
        EXPECT_FALSE(sr->T(i, i - 1) != 0.0 && sr->T(i + 1, i) != 0.0)
            << "T is not quasi-triangular at row " << i;
    }
    for (size_t j = 0; j + 2 < 4; ++j) {
        for (size_t i = j + 2; i < 4; ++i) {
            EXPECT_EQ(sr->T(i, j), 0.0);
        }
    }
}

TEST(SchurIdentity, CompanionMatrixRecoversChosenRoots) {
    // Companion of (x-1)(x-2)(x-3)(x-4)(x-5)
    //            = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120.
    DMatrix C(5, 5, 0.0);
    for (size_t i = 1; i < 5; ++i) {
        C(i, i - 1) = 1.0;
    }
    const double coeff[5] = {120.0, -274.0, 225.0, -85.0, 15.0};
    for (size_t i = 0; i < 5; ++i) {
        C(i, 4) = coeff[i];
    }
    const auto sr = schur(C);
    ASSERT_TRUE(sr.has_value());
    std::vector<double> d;
    for (size_t i = 0; i < 5; ++i) {
        d.push_back(sr->T(i, i));
    }
    std::sort(d.begin(), d.end());
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(d[i], static_cast<double>(i + 1), 1e-8);
    }
    EXPECT_LT(max_abs_diff(mul(sr->Q, mul(sr->T, tp(sr->Q))), C), 1e-10);
}

// ---------------------------------------------------------------------------
// bidiag: B must be BIdiagonal, not tridiagonal
// ---------------------------------------------------------------------------

TEST(BidiagShape, OnlyDiagonalAndSuperdiagonalAreNonzero) {
    // Old behaviour on this very matrix: B(1,0) = -10.488, B(2,1) = -0.914.
    const DMatrix A = mat(4, 3, {1, 2, 3,
                                 5, 6, 7,
                                 9, 10, 11,
                                 2, 4, 1});
    const auto r = bidiag(A);
    ASSERT_TRUE(r.has_value());
    const DMatrix& B = r->B;
    for (size_t i = 0; i < B.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            if (j == i || j == i + 1) {
                continue;
            }
            EXPECT_NEAR(B(i, j), 0.0, 1e-12)
                << "B(" << i << "," << j << ") must be zero";
        }
    }
}

TEST(BidiagShape, ReconstructionAndOrthogonalityStillHold) {
    const DMatrix A = mat(4, 3, {1, 2, 3,
                                 5, 6, 7,
                                 9, 10, 11,
                                 2, 4, 1});
    const auto r = bidiag(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(max_abs_diff(mul(r->U, mul(r->B, tp(r->V))), A), 1e-12);
    EXPECT_LT(max_abs_diff(mul(tp(r->U), r->U), identity(4)), 1e-13);
    EXPECT_LT(max_abs_diff(mul(tp(r->V), r->V), identity(3)), 1e-13);
}

TEST(BidiagShape, WideMatrixIsAlsoBidiagonal) {
    const DMatrix A = mat(3, 5, {1, 2, 3, 4, 5,
                                 6, 7, 8, 9, 10,
                                 11, 12, 13, 14, 16});
    const auto r = bidiag(A);
    ASSERT_TRUE(r.has_value());
    for (size_t i = 0; i < r->B.rows(); ++i) {
        for (size_t j = 0; j < r->B.cols(); ++j) {
            if (j == i || j == i + 1) {
                continue;
            }
            EXPECT_NEAR(r->B(i, j), 0.0, 1e-12);
        }
    }
    EXPECT_LT(max_abs_diff(mul(r->U, mul(r->B, tp(r->V))), A), 1e-12);
}

// ---------------------------------------------------------------------------
// ldl: P must be the pivot permutation actually used, and a nonsingular
//      indefinite matrix must not be reported as singular
// ---------------------------------------------------------------------------

TEST(LdlPivoting, ZeroLeadingDiagonalIsPivotedNotRejected) {
    // A = [0,2; 2,3] is nonsingular (det = -4) but has a zero leading pivot;
    // the unpivoted recurrence returned SingularMatrix.
    const DMatrix A = mat(2, 2, {0, 2, 2, 3});
    const auto r = ldl(A);
    ASSERT_TRUE(r.has_value()) << "nonsingular indefinite input must factor";

    DMatrix D(2, 2, 0.0);
    D(0, 0) = r->D(0, 0);
    D(1, 1) = r->D(1, 0);
    const DMatrix ldlt = mul(r->L, mul(D, tp(r->L)));
    const DMatrix ptap = mul(tp(r->P), mul(A, r->P));
    EXPECT_LT(max_abs_diff(ldlt, ptap), 1e-12) << "P^T A P == L D L^T";

    // P must carry information: this input requires a swap.
    EXPECT_NE(r->P(0, 0), 1.0) << "P is still a hard-coded identity";
    EXPECT_NEAR(r->D(0, 0) * r->D(1, 0), -4.0, 1e-12) << "det(D) == det(A)";
}

TEST(LdlPivoting, TinyPivotIsExchanged) {
    // det = 1e-20 - 1 ~ -1, so this matrix is nowhere near singular, yet the
    // unpivoted code bailed out on the 1e-20 leading diagonal.
    const DMatrix A = mat(2, 2, {1e-20, 1, 1, 2});
    const auto r = ldl(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->D(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(r->D(1, 0), -0.5, 1e-12);
    DMatrix D(2, 2, 0.0);
    D(0, 0) = r->D(0, 0);
    D(1, 1) = r->D(1, 0);
    EXPECT_LT(max_abs_diff(mul(r->L, mul(D, tp(r->L))),
                           mul(tp(r->P), mul(A, r->P))), 1e-12);
}

TEST(LdlPivoting, PIsIdentityForWellConditionedInput) {
    // Threshold pivoting must not disturb the ordinary case: P == I and the
    // plain identity A == L D L^T still holds.
    const DMatrix A = mat(3, 3, {4, 2, 2, 2, 3, 1, 2, 1, 3});
    const auto r = ldl(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(max_abs_diff(r->P, identity(3)), 0.0 + 1e-15);
    DMatrix D(3, 3, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        D(i, i) = r->D(i, 0);
    }
    EXPECT_LT(max_abs_diff(mul(r->L, mul(D, tp(r->L))), A), 1e-12);
}

TEST(LdlPivoting, TwoByTwoPivotCaseIsReportedHonestly) {
    // [[0,1],[1,0]] has det -1 but no usable 1x1 pivot anywhere; it needs a
    // 2x2 Bunch-Kaufman block. That is a DomainError, not "singular matrix".
    const DMatrix A = mat(2, 2, {0, 1, 1, 0});
    const auto r = ldl(A);
    ASSERT_FALSE(r.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(r.error()))
        << "a nonsingular matrix must not be reported as SingularMatrix";
}
