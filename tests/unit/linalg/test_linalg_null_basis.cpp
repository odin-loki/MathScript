// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for the BASIS null() returns, as opposed to its nullity.
//
// The tolerance fix for null() moved the tall/square (m >= n) case onto the
// SVD and handed back the trailing columns of V verbatim. But svd() leaves the
// right singular vector of an exactly-zero singular value as a ZERO COLUMN:
// the LAPACK path never fills it, and the Gram fallback skips it at its
// `sigma < 1e-14` guard. So for every m >= n input whose nullity came from an
// exactly-zero singular value, null() returned zero vectors as the "basis".
//
// A zero column satisfies A*v == 0 vacuously, which is why the checks that
// existed at the time (all of the form "||A*N|| is small") passed against it.
// Every test below therefore asserts the two properties a zero column CANNOT
// satisfy -- unit length and N^T N == I -- alongside A*N == 0 and the nullity.
//
// Measured old behaviour on these inputs (min column norm of the returned N):
//   null(zeros(3,3))               -> 3 columns, all exactly zero
//   null(diag(1,0,0))              -> 2 columns, both exactly zero
//   null([[0,0],[0,1]])            -> 1 column,  zero
//   null([[1,2,3],[2,4,6],[3,6,9]])-> 2 columns, one zero
//   null(rank-2 4x4)               -> 2 columns, both zero

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ms/core/operations.hpp"
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

double max_abs(const DMatrix& A) {
    double d = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            d = std::max(d, std::abs(A(i, j)));
        }
    }
    return d;
}

// The three properties that together define a null-space BASIS. The second and
// third are what a zero column fails; the first alone is not enough.
void expect_null_basis(const DMatrix& A, const DMatrix& N, size_t nullity,
                       const std::string& what) {
    ASSERT_EQ(N.rows(), A.cols()) << what;
    ASSERT_EQ(N.cols(), nullity) << what;
    if (nullity == 0) {
        return;
    }

    const double scale = std::max(1.0, max_abs(A));
    EXPECT_LT(max_abs(mul(A, N)), 1e-8 * scale) << what << ": A*N != 0";

    for (size_t j = 0; j < N.cols(); ++j) {
        double nrm = 0.0;
        for (size_t i = 0; i < N.rows(); ++i) {
            nrm += N(i, j) * N(i, j);
        }
        EXPECT_NEAR(std::sqrt(nrm), 1.0, 1e-10)
            << what << ": column " << j << " is not a unit vector"
            << " (a zero column would trivially satisfy A*N == 0)";
    }

    const DMatrix NtN = mul([&] {
        DMatrix T(N.cols(), N.rows(), 0.0);
        for (size_t i = 0; i < N.rows(); ++i) {
            for (size_t j = 0; j < N.cols(); ++j) {
                T(j, i) = N(i, j);
            }
        }
        return T;
    }(), N);
    for (size_t i = 0; i < N.cols(); ++i) {
        for (size_t j = 0; j < N.cols(); ++j) {
            EXPECT_NEAR(NtN(i, j), (i == j) ? 1.0 : 0.0, 1e-10)
                << what << ": N^T N not I at (" << i << "," << j << ")";
        }
    }
}

} // namespace

TEST(NullBasis, ZeroMatrixReturnsAnOrthonormalFullBasis) {
    // The null space of the 3x3 zero matrix is all of R^3.
    const DMatrix A(3, 3, 0.0);
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 3, "zeros(3,3)");
}

TEST(NullBasis, DiagonalWithExactZerosSpansTheMissingAxes) {
    // diag(1,0,0): null space is span(e2, e3).
    const DMatrix A = mat(3, 3, {1, 0, 0, 0, 0, 0, 0, 0, 0});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 2, "diag(1,0,0)");
    // Both basis vectors must be orthogonal to e1, the only row-space
    // direction.
    for (size_t j = 0; j < N->cols(); ++j) {
        EXPECT_NEAR((*N)(0, j), 0.0, 1e-12);
    }
}

TEST(NullBasis, SingleExactlyZeroSingularValue) {
    // [[0,0],[0,1]]: rank 1, null space span(e1). The returned column must BE
    // e1 (up to sign), not the zero vector.
    const DMatrix A = mat(2, 2, {0, 0, 0, 1});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 1, "[[0,0],[0,1]]");
    EXPECT_NEAR(std::abs((*N)(0, 0)), 1.0, 1e-12);
    EXPECT_NEAR((*N)(1, 0), 0.0, 1e-12);
}

TEST(NullBasis, RankOneSquareHasTwoIndependentNullVectors) {
    // [1,2,3; 2,4,6; 3,6,9] is rank 1, so the nullity is 2 and BOTH columns
    // have to carry a direction. The old code returned one good vector and one
    // zero vector.
    const DMatrix A = mat(3, 3, {1, 2, 3, 2, 4, 6, 3, 6, 9});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 2, "rank-1 3x3");
}

TEST(NullBasis, RankTwoFourByFour) {
    const DMatrix A = mat(4, 4, {1, 0, 1, 0,
                                 0, 1, 0, 1,
                                 1, 0, 1, 0,
                                 0, 1, 0, 1});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 2, "rank-2 4x4");
}

TEST(NullBasis, RankOneFiveByFive) {
    // The rank-1 5x5 outer product used by the existing
    // svd_rank_deficient_null_and_orth_regression test, which only checked
    // ||A*N|| and therefore passed with four zero columns.
    std::vector<double> u{0.2, 0.4, 0.4, 0.6, 0.5196152423};
    std::vector<double> v{0.5, -0.3, 0.4, -0.5, 0.5};
    auto normalize = [](std::vector<double>& x) {
        double s = 0.0;
        for (double t : x) {
            s += t * t;
        }
        s = std::sqrt(s);
        for (double& t : x) {
            t /= s;
        }
    };
    normalize(u);
    normalize(v);
    DMatrix A(5, 5, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            A(i, j) = 9.0 * u[i] * v[j];
        }
    }
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 4, "rank-1 5x5");
}

TEST(NullBasis, TallRankDeficientMatrix) {
    // 7 x 5 of rank 2 built as B*C, so the nullity is exactly 3.
    const DMatrix B = mat(7, 2, {1, 0,
                                 0, 1,
                                 1, 1,
                                 2, -1,
                                 -1, 3,
                                 0.5, 0.25,
                                 3, 2});
    const DMatrix C = mat(2, 5, {1, 2, 0, -1, 3,
                                 0, 1, 1, 2, -2});
    const DMatrix A = mul(B, C);
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 3, "rank-2 7x5");
}

TEST(NullBasis, FullRankStillReturnsNoColumns) {
    const DMatrix A = mat(3, 3, {2, 0, 1, 0, 3, 0, 1, 0, 2});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    EXPECT_EQ(N->cols(), 0u);
}

TEST(NullBasis, WideZeroMatrixIsAlsoOrthonormal) {
    // The m < n Gram path, for symmetry with the tall cases above.
    const DMatrix A(2, 4, 0.0);
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    expect_null_basis(A, *N, 4, "zeros(2,4)");
}
