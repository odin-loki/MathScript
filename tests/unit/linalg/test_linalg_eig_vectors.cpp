// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for ms::eig.
//
// The audit found that for non-symmetric input eig() ran an unshifted,
// un-deflated QR iteration and returned the accumulated Q product -- the
// approximate SCHUR basis -- as `vectors`, then permuted its columns while
// sorting, so A*v_j != lambda_j*v_j for essentially every column. Complex
// spectra were flattened onto the real axis with no indication
// (eig([0,1;-1,0]) reported {0,0}).
//
// Every assertion below fails against that behaviour.

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

// Checks the defining property for every column, honouring the LAPACK-style
// packing of a complex-conjugate pair:
//   real  lambda_j : A v_j == lambda_j v_j
//   pair (j, j+1) : with v = vectors(:,j) + i*vectors(:,j+1) and
//                   lambda = re + i*im,
//                   A*Re v == re*Re v - im*Im v  and
//                   A*Im v == re*Im v + im*Re v.
void expect_eigenpairs(const DMatrix& A, const EigResult& r, double tol) {
    const size_t n = A.rows();
    ASSERT_EQ(r.values.rows(), n);
    ASSERT_EQ(r.values_imag.rows(), n);
    ASSERT_EQ(r.vectors.rows(), n);
    ASSERT_EQ(r.vectors.cols(), n);

    for (size_t j = 0; j < n;) {
        const double im = r.values_imag(j, 0);
        if (im == 0.0) {
            const double lam = r.values(j, 0);
            double vn = 0.0;
            for (size_t i = 0; i < n; ++i) {
                vn += r.vectors(i, j) * r.vectors(i, j);
            }
            EXPECT_NEAR(std::sqrt(vn), 1.0, 1e-10) << "column " << j << " not normalised";
            for (size_t i = 0; i < n; ++i) {
                double av = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    av += A(i, k) * r.vectors(k, j);
                }
                EXPECT_NEAR(av, lam * r.vectors(i, j), tol)
                    << "A*v != lambda*v at row " << i << ", column " << j;
            }
            ++j;
        } else {
            ASSERT_LT(j + 1, n) << "a conjugate pair needs two columns";
            const double re = r.values(j, 0);
            EXPECT_NEAR(r.values(j + 1, 0), re, 1e-12);
            EXPECT_NEAR(r.values_imag(j + 1, 0), -im, 1e-12);
            for (size_t i = 0; i < n; ++i) {
                double avr = 0.0;
                double avi = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    avr += A(i, k) * r.vectors(k, j);
                    avi += A(i, k) * r.vectors(k, j + 1);
                }
                EXPECT_NEAR(avr, re * r.vectors(i, j) - im * r.vectors(i, j + 1), tol);
                EXPECT_NEAR(avi, re * r.vectors(i, j + 1) + im * r.vectors(i, j), tol);
            }
            j += 2;
        }
    }
}

} // namespace

TEST(EigVectors, UpperTriangular2x2) {
    // A = [2,1; 0,3]: eigenvalues 3 and 2, eigenvectors [1,1]/sqrt(2) and
    // [1,0]. The old code returned vectors [[0,1],[1,0]], for which
    // A*[0;1] = [1;3] is not 3*[0;1].
    const DMatrix A = mat(2, 2, {2, 1, 0, 3});
    const auto r = eig(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->values(0, 0), 3.0, 1e-12);
    EXPECT_NEAR(r->values(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(r->values_imag(0, 0), 0.0, 0.0);
    EXPECT_NEAR(r->values_imag(1, 0), 0.0, 0.0);
    expect_eigenpairs(A, *r, 1e-10);
    const double s = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::abs(r->vectors(0, 0)), s, 1e-10);
    EXPECT_NEAR(std::abs(r->vectors(1, 0)), s, 1e-10);
    EXPECT_NEAR(std::abs(r->vectors(0, 1)), 1.0, 1e-10);
    EXPECT_NEAR(std::abs(r->vectors(1, 1)), 0.0, 1e-10);
}

TEST(EigVectors, GeneralTwoByTwoMatchesClosedForm) {
    // A = [1,2; 3,4]: lambda = (5 +- sqrt(33))/2.
    const DMatrix A = mat(2, 2, {1, 2, 3, 4});
    const auto r = eig(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->values(0, 0), (5.0 + std::sqrt(33.0)) / 2.0, 1e-12);
    EXPECT_NEAR(r->values(1, 0), (5.0 - std::sqrt(33.0)) / 2.0, 1e-12);
    expect_eigenpairs(A, *r, 1e-10);
}

TEST(EigVectors, RotationHasConjugatePair) {
    // [0,1; -1,0] has eigenvalues +-i. The old code reported {0, 0} with no
    // sign that a complex pair had been flattened.
    const DMatrix A = mat(2, 2, {0, 1, -1, 0});
    const auto r = eig(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->values(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(r->values(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(r->values_imag(0, 0)), 1.0, 1e-12);
    EXPECT_NEAR(std::abs(r->values_imag(1, 0)), 1.0, 1e-12);
    EXPECT_NEAR(r->values_imag(0, 0) + r->values_imag(1, 0), 0.0, 1e-12);
    expect_eigenpairs(A, *r, 1e-10);
}

TEST(EigVectors, ComplexPairAndRealEigenvalue) {
    // [2,-1,0; 1,2,0; 0,0,3] has spectrum {3, 2+i, 2-i}. The old code
    // reported {3, 2, 2}.
    const DMatrix A = mat(3, 3, {2, -1, 0, 1, 2, 0, 0, 0, 3});
    const auto r = eig(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->values(0, 0), 3.0, 1e-12);
    EXPECT_NEAR(r->values_imag(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(r->values(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(r->values(2, 0), 2.0, 1e-12);
    EXPECT_NEAR(std::abs(r->values_imag(1, 0)), 1.0, 1e-12);
    EXPECT_NEAR(r->values_imag(1, 0) + r->values_imag(2, 0), 0.0, 1e-12);
    expect_eigenpairs(A, *r, 1e-10);
}

TEST(EigVectors, CompanionMatrixOfChosenRoots) {
    // Companion of (x-1)(x-2)(x-3)(x-4)(x-5).
    DMatrix C(5, 5, 0.0);
    for (size_t i = 1; i < 5; ++i) {
        C(i, i - 1) = 1.0;
    }
    const double coeff[5] = {120.0, -274.0, 225.0, -85.0, 15.0};
    for (size_t i = 0; i < 5; ++i) {
        C(i, 4) = coeff[i];
    }
    const auto r = eig(C);
    ASSERT_TRUE(r.has_value());
    std::vector<double> d;
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(r->values_imag(i, 0), 0.0, 1e-9);
        d.push_back(r->values(i, 0));
    }
    std::sort(d.begin(), d.end());
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(d[i], static_cast<double>(i + 1), 1e-7);
    }
    expect_eigenpairs(C, *r, 1e-7);
}

TEST(EigVectors, TriangularSpectrumIsTheDiagonal) {
    const DMatrix A = mat(3, 3, {1, 2, 0, 0, 3, 1, 0, 0, 4});
    const auto r = eig(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->values(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(r->values(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(r->values(2, 0), 1.0, 1e-12);
    expect_eigenpairs(A, *r, 1e-10);
}

TEST(EigVectors, DescendingRealPartOrder) {
    const DMatrix A = mat(3, 3, {1, 1, 0, 0, 1, 1, 1, 0, 1});
    const auto r = eig(A);
    ASSERT_TRUE(r.has_value());
    for (size_t i = 1; i < 3; ++i) {
        EXPECT_GE(r->values(i - 1, 0), r->values(i, 0) - 1e-12);
    }
    // I + cyclic permutation: spectrum {2, 1 + w, 1 + w^2} with w a primitive
    // cube root of unity, i.e. {2, 0.5 +- i*sqrt(3)/2}.
    EXPECT_NEAR(r->values(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(r->values(1, 0), 0.5, 1e-10);
    EXPECT_NEAR(std::abs(r->values_imag(1, 0)), std::sqrt(3.0) / 2.0, 1e-10);
    expect_eigenpairs(A, *r, 1e-10);
}

TEST(EigVectors, SymmetricStillHasZeroImaginaryParts) {
    const DMatrix A = mat(3, 3, {4, 1, 0, 1, 3, 1, 0, 1, 2});
    const auto r = eig(A);
    ASSERT_TRUE(r.has_value());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(r->values_imag(i, 0), 0.0);
    }
    EXPECT_NEAR(r->values(0, 0), 3.0 + std::sqrt(3.0), 1e-10);
    EXPECT_NEAR(r->values(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(r->values(2, 0), 3.0 - std::sqrt(3.0), 1e-10);
    expect_eigenpairs(A, *r, 1e-10);
}
