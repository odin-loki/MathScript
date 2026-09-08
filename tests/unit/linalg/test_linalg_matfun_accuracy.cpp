// Regression tests for the matrix functions.
//
//   * expm() was a bare 12-term Taylor truncation with no scaling and
//     squaring: expm(diag(10,10)) returned 17435.19 against e^10 = 22026.47
//     (21% error), expm(diag(30,30)) was wrong by a factor of ~6000, and
//     expm(diag(-10,-10)) returned 925.05 instead of 4.54e-5.
//   * apply_spectral_func() reduced f(A) to V diag(f(lambda)) V^T for every
//     input; for non-symmetric A that drops the entire off-diagonal part of
//     f(T), so sqrtm([2,1;0,3]) returned diag(sqrt 2, sqrt 3) whose square is
//     diag(2,3), not the input.
//
// The ground truth here is independent of the implementation: closed-form
// scalar values, the defining identities (S^2 == A, expm(logm(A)) == A,
// sin^2 + cos^2 == I, expm(A)expm(-A) == I), and a scaling-and-squaring
// TAYLOR sum written out in this file (a different algorithm family from the
// Pade approximant under test).

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "ms/core/operations.hpp"
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

double max_abs(const DMatrix& A) {
    double d = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            d = std::max(d, std::abs(A(i, j)));
        }
    }
    return d;
}

// Independent reference exponential: scale by 2^-s until the 1-norm is below
// 0.03, sum 60 Taylor terms (far beyond double precision at that norm), then
// square s times.
DMatrix taylor_expm(const DMatrix& A) {
    const size_t n = A.rows();
    double nrm = 0.0;
    for (size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) {
            s += std::abs(A(i, j));
        }
        nrm = std::max(nrm, s);
    }
    int s = 0;
    if (nrm > 0.03) {
        s = static_cast<int>(std::ceil(std::log2(nrm / 0.03)));
    }
    if (s < 0) {
        s = 0;
    }
    const double scale = std::ldexp(1.0, -s);
    DMatrix X(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            X(i, j) = A(i, j) * scale;
        }
    }
    DMatrix R = identity(n);
    DMatrix term = identity(n);
    for (int k = 1; k <= 60; ++k) {
        term = mul(term, X);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                term(i, j) /= static_cast<double>(k);
                R(i, j) += term(i, j);
            }
        }
    }
    for (int k = 0; k < s; ++k) {
        R = mul(R, R);
    }
    return R;
}

} // namespace

// ---------------------------------------------------------------------------
// expm
// ---------------------------------------------------------------------------

TEST(ExpmAccuracy, DiagonalMatchesScalarExponentialAtLargeNorm) {
    // Old results: c=10 -> 17435.19 (e^10 = 22026.47); c=30 -> 1.79e9
    // (e^30 = 1.0686e13); c=-10 -> 925.05 (e^-10 = 4.54e-5).
    for (double c : {1.0, 10.0, 30.0, -10.0, -30.0}) {
        DMatrix D(2, 2, 0.0);
        D(0, 0) = c;
        D(1, 1) = c;
        const auto E = expm(D);
        ASSERT_TRUE(E.has_value()) << "c = " << c;
        const double want = std::exp(c);
        EXPECT_NEAR((*E)(0, 0) / want, 1.0, 1e-13) << "c = " << c;
        EXPECT_NEAR((*E)(1, 1) / want, 1.0, 1e-13) << "c = " << c;
        EXPECT_NEAR((*E)(0, 1), 0.0, 1e-13 * want) << "c = " << c;
    }
}

TEST(ExpmAccuracy, MatchesIndependentTaylorReference) {
    const std::vector<DMatrix> cases{
        mat(2, 2, {0, 1, -1, 0}),
        mat(2, 2, {1, 2, 3, 4}),
        mat(3, 3, {4, 1, -2, 1, 2, 0, -2, 0, 3}),
        mat(2, 2, {30, 1, 0, 30}),
        mat(3, 3, {-10, 1, 2, 0, -10, 3, 0, 0, -10}),
        mat(3, 3, {6, -12, 3, 4, 8, -5, -2, 7, 9}),
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        const auto E = expm(cases[c]);
        ASSERT_TRUE(E.has_value()) << "case " << c;
        const DMatrix ref = taylor_expm(cases[c]);
        EXPECT_LT(max_abs_diff(*E, ref), 1e-12 * std::max(max_abs(ref), 1.0))
            << "case " << c;
    }
}

TEST(ExpmAccuracy, ExpmTimesExpmOfNegativeIsIdentity) {
    const DMatrix A = mat(3, 3, {3, -4, 2, 1, 5, -3, -2, 1, 4});
    DMatrix nA(3, 3, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            nA(i, j) = -A(i, j);
        }
    }
    const auto E1 = expm(A);
    const auto E2 = expm(nA);
    ASSERT_TRUE(E1.has_value());
    ASSERT_TRUE(E2.has_value());
    EXPECT_LT(max_abs_diff(mul(*E1, *E2), identity(3)), 1e-11);
}

TEST(ExpmAccuracy, ZeroMatrixIsIdentity) {
    const auto E = expm(zeros<double>(3, 3));
    ASSERT_TRUE(E.has_value());
    EXPECT_LT(max_abs_diff(*E, identity(3)), 1e-15);
}

// ---------------------------------------------------------------------------
// sqrtm / logm on NON-symmetric input
// ---------------------------------------------------------------------------

TEST(MatrixFunctions, SqrtmOfNonSymmetricSquaresBackToInput) {
    // The true root of [2,1; 0,3] is [sqrt2, 1/(sqrt2+sqrt3); 0, sqrt3];
    // the old code returned diag(sqrt2, sqrt3), whose square is diag(2,3).
    const DMatrix A = mat(2, 2, {2, 1, 0, 3});
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_NEAR((*S)(0, 0), std::sqrt(2.0), 1e-13);
    EXPECT_NEAR((*S)(1, 1), std::sqrt(3.0), 1e-13);
    EXPECT_NEAR((*S)(0, 1), 1.0 / (std::sqrt(2.0) + std::sqrt(3.0)), 1e-13);
    EXPECT_NEAR((*S)(1, 0), 0.0, 1e-15);
    EXPECT_LT(max_abs_diff(mul(*S, *S), A), 1e-13);
}

TEST(MatrixFunctions, SqrtmSquaredEqualsInputOnLargerNonSymmetric) {
    const DMatrix A = mat(4, 4, {4, 1, -2, 3,
                                 0, 9, 1, -1,
                                 0, 0, 16, 2,
                                 0, 0, 0, 25});
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_LT(max_abs_diff(mul(*S, *S), A), 1e-11);
}

TEST(MatrixFunctions, LogmOfExpmRecoversNonSymmetricMatrix) {
    const DMatrix A = mat(3, 3, {-0.4, 0.7, 1.2,
                                 0.0, 0.1, -0.9,
                                 0.0, 0.0, 0.6});
    const auto E = expm(A);
    ASSERT_TRUE(E.has_value());
    const auto L = logm(*E);
    ASSERT_TRUE(L.has_value());
    EXPECT_LT(max_abs_diff(*L, A), 1e-10);
}

TEST(MatrixFunctions, SinmCosmSatisfyPythagoreanIdentityNonSymmetric) {
    const DMatrix A = mat(3, 3, {1, 2, 3, 0, 2, 1, 0, 0, 4});
    const auto S = sinm(A);
    const auto C = cosm(A);
    ASSERT_TRUE(S.has_value());
    ASSERT_TRUE(C.has_value());
    DMatrix sum = mul(*S, *S);
    const DMatrix c2 = mul(*C, *C);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            sum(i, j) += c2(i, j);
        }
    }
    EXPECT_LT(max_abs_diff(sum, identity(3)), 1e-12);
}

TEST(MatrixFunctions, FunmSquareRootMatchesSqrtmOnNonSymmetric) {
    const DMatrix A = mat(3, 3, {4, 1, 2, 0, 9, 3, 0, 0, 16});
    std::function<double(double)> f = [](double x) { return std::sqrt(x); };
    const auto F = funm(A, f);
    ASSERT_TRUE(F.has_value());
    EXPECT_LT(max_abs_diff(mul(*F, *F), A), 1e-11);
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_LT(max_abs_diff(*F, *S), 1e-11);
}

TEST(MatrixFunctions, ComplexSpectrumIsReportedNotSilentlyWrong) {
    // A real matrix with a complex-conjugate pair has no real square root or
    // logarithm reachable by this real Schur-Parlett path. The old code
    // returned a plausible-looking but wrong matrix instead.
    const DMatrix R = mat(2, 2, {0, 1, -1, 0});
    const auto S = sqrtm(R);
    EXPECT_FALSE(S.has_value());
    if (!S.has_value()) {
        EXPECT_TRUE(std::holds_alternative<DomainError>(S.error()));
    }
    const auto L = logm(R);
    EXPECT_FALSE(L.has_value());
}

TEST(MatrixFunctions, SymmetricPathIsUnchangedAndExact) {
    const DMatrix A = mat(2, 2, {4, 2, 2, 3});
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_LT(max_abs_diff(mul(*S, *S), A), 1e-12);
    const auto L = logm(A);
    ASSERT_TRUE(L.has_value());
    const auto E = expm(*L);
    ASSERT_TRUE(E.has_value());
    EXPECT_LT(max_abs_diff(*E, A), 1e-11);
}

TEST(MatrixFunctions, SqrtmRejectsNegativeSymmetricEigenvalue) {
    // Previously the negative eigenvalue was silently clamped to zero, so the
    // result did not square back to A and nothing said so.
    const DMatrix A = mat(2, 2, {1, 0, 0, -4});
    const auto S = sqrtm(A);
    EXPECT_FALSE(S.has_value());
}
