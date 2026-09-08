// Regression tests for orth(), null() and norm().
//
//   * orth() took the leading columns of an UNPIVOTED QR's Q, which is not
//     rank revealing, so it returned vectors outside range(A) whenever the
//     deficiency was not in the trailing columns: orth([0,1;0,1]) was [1;0],
//     which is orthogonal to the column space, and orth([0,1,1;0,1,1]) was
//     the all-zero "basis vector" [0;0].
//   * null() applied the sigma_max scaling twice, so its cutoff was
//     sigma <= 1e-10*max(m,n)*sigma_max^2 -- not scale invariant.
//     null([1e6,0; 0,10]) returned [0;1] for a nonsingular diagonal matrix.
//   * norm() returned a successful S(0) for every p outside {2, 1, -1}.

#include <algorithm>
#include <cmath>
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

DMatrix tp(const DMatrix& A) {
    DMatrix T(A.cols(), A.rows(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            T(j, i) = A(i, j);
        }
    }
    return T;
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

// The property that defines orth: the columns are orthonormal AND they span
// range(A), i.e. the orthogonal projector Q*Q^T leaves A alone.
void expect_orthonormal_basis_of_range(const DMatrix& A, const DMatrix& Q) {
    const DMatrix QtQ = mul(tp(Q), Q);
    for (size_t i = 0; i < Q.cols(); ++i) {
        for (size_t j = 0; j < Q.cols(); ++j) {
            EXPECT_NEAR(QtQ(i, j), (i == j) ? 1.0 : 0.0, 1e-10)
                << "Q^T Q not I at (" << i << "," << j << ")";
        }
    }
    DMatrix diff = mul(mul(Q, tp(Q)), A);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            diff(i, j) -= A(i, j);
        }
    }
    EXPECT_LT(max_abs(diff), 1e-10) << "columns of A are not in span(Q)";
}

} // namespace

// ---------------------------------------------------------------------------
// orth
// ---------------------------------------------------------------------------

TEST(OrthRange, LeadingDeficientColumn) {
    // range([0,1; 0,1]) = span([1;1]/sqrt(2)). The old answer, [1;0], is
    // orthogonal to that span.
    const DMatrix A = mat(2, 2, {0, 1, 0, 1});
    const auto Q = orth(A);
    ASSERT_TRUE(Q.has_value());
    ASSERT_EQ(Q->cols(), 1u);
    EXPECT_NEAR(std::abs((*Q)(0, 0)), 1.0 / std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(std::abs((*Q)(1, 0)), 1.0 / std::sqrt(2.0), 1e-10);
    expect_orthonormal_basis_of_range(A, *Q);
}

TEST(OrthRange, WideDeficientMatrixHasNoZeroBasisVector) {
    // The Gram-Schmidt fallback used to emit a literal zero column here.
    const DMatrix A = mat(2, 3, {0, 1, 1, 0, 1, 1});
    const auto Q = orth(A);
    ASSERT_TRUE(Q.has_value());
    ASSERT_EQ(Q->cols(), 1u);
    double nrm = 0.0;
    for (size_t i = 0; i < Q->rows(); ++i) {
        nrm += (*Q)(i, 0) * (*Q)(i, 0);
    }
    EXPECT_NEAR(std::sqrt(nrm), 1.0, 1e-10) << "basis vector must not be zero";
    expect_orthonormal_basis_of_range(A, *Q);
}

TEST(OrthRange, RankDeficientSquare) {
    const DMatrix A = mat(3, 3, {1, 2, 0, 1, 2, 1, 1, 2, 0});
    const auto Q = orth(A);
    ASSERT_TRUE(Q.has_value());
    EXPECT_EQ(Q->cols(), 2u);
    expect_orthonormal_basis_of_range(A, *Q);
}

TEST(OrthRange, ProportionalColumns) {
    const DMatrix A = mat(2, 2, {1, 2, 2, 4});
    const auto Q = orth(A);
    ASSERT_TRUE(Q.has_value());
    ASSERT_EQ(Q->cols(), 1u);
    expect_orthonormal_basis_of_range(A, *Q);
}

TEST(OrthRange, FullRankTallKeepsAllColumns) {
    const DMatrix A = mat(5, 3, {1, 0, 0,
                                 0, 1, 0,
                                 0, 0, 1,
                                 1, 1, 0,
                                 0, 1, 1});
    const auto Q = orth(A);
    ASSERT_TRUE(Q.has_value());
    EXPECT_EQ(Q->rows(), 5u);
    EXPECT_EQ(Q->cols(), 3u);
    expect_orthonormal_basis_of_range(A, *Q);
}

TEST(OrthRange, ZeroMatrixHasEmptyBasis) {
    const auto Q = orth(zeros<double>(3, 3));
    ASSERT_TRUE(Q.has_value());
    EXPECT_EQ(Q->cols(), 0u);
}

// ---------------------------------------------------------------------------
// null
// ---------------------------------------------------------------------------

TEST(NullTolerance, NonsingularScaledMatrixHasEmptyNullSpace) {
    // det = 1e7 and cond = 1e5, yet the doubly-scaled cutoff declared [0;1] a
    // null vector even though A*[0;1] = [0;10].
    const DMatrix A = mat(2, 2, {1000000, 0, 0, 10});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    EXPECT_EQ(N->cols(), 0u);
}

TEST(NullTolerance, IsScaleInvariant) {
    // Same matrix scaled over 12 orders of magnitude must give the same
    // nullity; rank([1,2,3;4,5,6;7,8,9]) is 2, so the nullity is 1.
    for (double c : {1e-6, 1.0, 1e6}) {
        DMatrix A = mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                A(i, j) *= c;
            }
        }
        const auto N = null(A);
        ASSERT_TRUE(N.has_value()) << "c = " << c;
        EXPECT_EQ(N->cols(), 1u) << "c = " << c;
        EXPECT_LT(max_abs(mul(A, *N)), 1e-8 * max_abs(A)) << "c = " << c;
    }
}

TEST(NullTolerance, RankOneSquare) {
    const DMatrix A = mat(2, 2, {1, 2, 2, 4});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    ASSERT_EQ(N->cols(), 1u);
    EXPECT_LT(max_abs(mul(A, *N)), 1e-10);
}

TEST(NullTolerance, WideRankOneKeepsFullNullity) {
    // rank 1 in R^4 -> nullity 3; this is the m < n path.
    const DMatrix A = mat(2, 4, {1, 2, 3, 4, 2, 4, 6, 8});
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    EXPECT_EQ(N->rows(), 4u);
    EXPECT_EQ(N->cols(), 3u);
    EXPECT_LT(max_abs(mul(A, *N)), 1e-8);
}

TEST(NullTolerance, IdentityHasEmptyNullSpace) {
    const auto N = null(eye<double>(4));
    ASSERT_TRUE(N.has_value());
    EXPECT_EQ(N->cols(), 0u);
}

// ---------------------------------------------------------------------------
// norm
// ---------------------------------------------------------------------------

TEST(NormContract, UnsupportedPIsAnErrorNotZero) {
    const DMatrix A = mat(2, 2, {1, 2, 3, 4});
    const auto r = norm(A, -3);
    ASSERT_FALSE(r.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(r.error()));
    const auto r2 = norm(A, -7);
    EXPECT_FALSE(r2.has_value());
}

TEST(NormContract, ZeroPIsTheL0Count) {
    DMatrix v(4, 1, 0.0);
    v(0, 0) = 1.0;
    v(1, 0) = -5.0;
    v(2, 0) = 3.0;
    v(3, 0) = -2.0;
    const auto r = norm(v, 0);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 4.0, 0.0) << "old behaviour returned a fabricated 0.0";

    DMatrix w(3, 1, 0.0);
    w(1, 0) = 7.0;
    const auto r2 = norm(w, 0);
    ASSERT_TRUE(r2.has_value());
    EXPECT_NEAR(*r2, 1.0, 0.0);
}

TEST(NormContract, DocumentedConventionsAreStable) {
    const DMatrix A = mat(2, 2, {1, 2, 3, 4});
    EXPECT_NEAR(*norm(A, 2), std::sqrt(30.0), 1e-12);  // Frobenius
    EXPECT_NEAR(*norm(A, 1), 10.0, 1e-12);             // entrywise sum
    EXPECT_NEAR(*norm(A, -1), 7.0, 1e-12);             // induced infinity
}

TEST(NormContract, GeneralEntrywisePNorm) {
    const DMatrix A = mat(2, 2, {1, 2, 3, 4});
    const auto r3 = norm(A, 3);
    ASSERT_TRUE(r3.has_value());
    EXPECT_NEAR(*r3, std::cbrt(1.0 + 8.0 + 27.0 + 64.0), 1e-12);
    const auto r4 = norm(A, 4);
    ASSERT_TRUE(r4.has_value());
    EXPECT_NEAR(*r4, std::pow(1.0 + 16.0 + 81.0 + 256.0, 0.25), 1e-12);
}

TEST(NormContract, FrobeniusDoesNotOverflow) {
    DMatrix v(2, 1, 0.0);
    v(0, 0) = 1e200;
    v(1, 0) = 1e200;
    const auto r = norm(v, 2);
    ASSERT_TRUE(r.has_value());
    ASSERT_TRUE(std::isfinite(*r));
    EXPECT_NEAR(*r / 1e200, std::sqrt(2.0), 1e-12);
}
