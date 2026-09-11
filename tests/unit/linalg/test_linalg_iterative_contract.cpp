// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for the iterative solvers.
//
//   * minres()'s direction-vector recurrence was wrong (the v term was never
//     divided by gamma and the delta/eps coefficients sat on the wrong history
//     vectors), and it exited on the recursive estimate alone, so it returned
//     grossly wrong iterates as successful Results:
//       minres([2,0;0,3],[2;3])            -> [2.478136; 2.618939]  (exact [1;1])
//       minres([4,1,0;1,3,1;0,1,2],[1;2;3]) -> [0.7058; 0.7725; 3.5017]
//         (exact [2/9; 1/9; 13/9]); the residual was larger than ||b||.
//   * cg/jacobi/bicgstab/gmres/minres all accepted a multi-column b and none
//     solved more than the first right-hand side: jacobi returned zeros in the
//     extra columns, cg/bicgstab returned b unchanged there, and gmres/minres
//     silently returned an n x 1 result (minres additionally read past the end
//     of its heap buffer).
//   * bicgstab() ended every path with an unconditional `return x;`, so
//     bicgstab([1,1;1,1],[1;2]) reported [nan; nan] as a success.
//
// The ground truth is a dense Gaussian-elimination solve written out here, so
// it does not depend on ms::solve.

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

// Reference solve: Gaussian elimination with partial pivoting, multi-RHS.
DMatrix gauss_solve(DMatrix A, DMatrix b) {
    const size_t n = A.rows();
    const size_t nr = b.cols();
    for (size_t k = 0; k < n; ++k) {
        size_t piv = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(A(i, k)) > std::abs(A(piv, k))) {
                piv = i;
            }
        }
        for (size_t j = 0; j < n; ++j) {
            std::swap(A(k, j), A(piv, j));
        }
        for (size_t j = 0; j < nr; ++j) {
            std::swap(b(k, j), b(piv, j));
        }
        for (size_t i = k + 1; i < n; ++i) {
            const double f = A(i, k) / A(k, k);
            for (size_t j = k; j < n; ++j) {
                A(i, j) -= f * A(k, j);
            }
            for (size_t j = 0; j < nr; ++j) {
                b(i, j) -= f * b(k, j);
            }
        }
    }
    DMatrix x(n, nr, 0.0);
    for (size_t j = 0; j < nr; ++j) {
        for (size_t ii = n; ii-- > 0;) {
            double s = b(ii, j);
            for (size_t k = ii + 1; k < n; ++k) {
                s -= A(ii, k) * x(k, j);
            }
            x(ii, j) = s / A(ii, ii);
        }
    }
    return x;
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

double residual_norm(const DMatrix& A, const DMatrix& x, const DMatrix& b) {
    const DMatrix Ax = mul(A, x);
    double s = 0.0;
    for (size_t i = 0; i < b.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            const double d = b(i, j) - Ax(i, j);
            s += d * d;
        }
    }
    return std::sqrt(s);
}

DMatrix col(std::initializer_list<double> v) {
    DMatrix c(v.size(), 1, 0.0);
    size_t i = 0;
    for (double x : v) {
        c(i++, 0) = x;
    }
    return c;
}

} // namespace

// ---------------------------------------------------------------------------
// MINRES
// ---------------------------------------------------------------------------

TEST(MinresCorrectness, DiagonalSystem) {
    const DMatrix A = mat(2, 2, {2, 0, 0, 3});
    const DMatrix b = col({2.0, 3.0});
    const auto x = minres(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 1.0, 1e-11);
    EXPECT_NEAR((*x)(1, 0), 1.0, 1e-11);
}

TEST(MinresCorrectness, SpdTridiagonalSystemMatchesExactSolution) {
    // Exact solution of [4,1,0; 1,3,1; 0,1,2] x = [1;2;3] is
    // [2/9; 1/9; 13/9] = [0.2222...; 0.1111...; 1.4444...].
    const DMatrix A = mat(3, 3, {4, 1, 0, 1, 3, 1, 0, 1, 2});
    const DMatrix b = col({1.0, 2.0, 3.0});
    const auto x = minres(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 2.0 / 9.0, 1e-10);
    EXPECT_NEAR((*x)(1, 0), 1.0 / 9.0, 1e-10);
    EXPECT_NEAR((*x)(2, 0), 13.0 / 9.0, 1e-10);
    EXPECT_LT(residual_norm(A, *x, b), 1e-10);
}

TEST(MinresCorrectness, IndefiniteSymmetricSystem) {
    // MINRES, unlike CG, is meant to handle an indefinite symmetric A.
    const DMatrix A = mat(3, 3, {2, 1, 0, 1, -3, 1, 0, 1, 1});
    const DMatrix b = col({1.0, -2.0, 3.0});
    const DMatrix exact = gauss_solve(A, b);
    const auto x = minres(A, b, 500, 1e-13);
    ASSERT_TRUE(x.has_value());
    EXPECT_LT(max_abs_diff(*x, exact), 1e-9);
    EXPECT_LT(residual_norm(A, *x, b), 1e-10);
}

TEST(MinresCorrectness, LargerSpdSystem) {
    const size_t n = 8;
    DMatrix A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = 4.0 + static_cast<double>(i);
        if (i + 1 < n) {
            A(i, i + 1) = -1.0;
            A(i + 1, i) = -1.0;
        }
    }
    DMatrix b(n, 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        b(i, 0) = std::sin(static_cast<double>(i) + 1.0);
    }
    const DMatrix exact = gauss_solve(A, b);
    const auto x = minres(A, b, 500, 1e-13);
    ASSERT_TRUE(x.has_value());
    EXPECT_LT(max_abs_diff(*x, exact), 1e-9);
}

TEST(MinresCorrectness, ZeroRightHandSideGivesZero) {
    const DMatrix A = mat(2, 2, {2, 0, 0, 3});
    const auto x = minres(A, zeros<double>(2, 1));
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ((*x)(0, 0), 0.0);
    EXPECT_EQ((*x)(1, 0), 0.0);
}

// ---------------------------------------------------------------------------
// Multiple right-hand sides
// ---------------------------------------------------------------------------

TEST(IterativeMultiRhs, DiagonalTwoColumns) {
    const DMatrix A = mat(3, 3, {1, 0, 0, 0, 2, 0, 0, 0, 3});
    DMatrix B(3, 2, 0.0);
    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 1) = 1.0;
    const DMatrix exact = gauss_solve(A, B);  // [[1,0],[0,0.5],[0,1/3]]

    const auto xc = cg(A, B);
    ASSERT_TRUE(xc.has_value());
    EXPECT_EQ(xc->cols(), 2u);
    EXPECT_LT(max_abs_diff(*xc, exact), 1e-9);

    const auto xg = gmres(A, B);
    ASSERT_TRUE(xg.has_value());
    EXPECT_EQ(xg->cols(), 2u) << "gmres used to narrow the result to n x 1";
    EXPECT_LT(max_abs_diff(*xg, exact), 1e-9);

    const auto xm = minres(A, B);
    ASSERT_TRUE(xm.has_value());
    EXPECT_EQ(xm->cols(), 2u) << "minres used to narrow the result to n x 1";
    EXPECT_LT(max_abs_diff(*xm, exact), 1e-9);

    const auto xb = bicgstab(A, B);
    ASSERT_TRUE(xb.has_value());
    EXPECT_EQ(xb->cols(), 2u);
    EXPECT_LT(max_abs_diff(*xb, exact), 1e-9);
}

TEST(IterativeMultiRhs, JacobiSolvesEveryColumn) {
    // Old behaviour: [[0.0909, 0], [0.6364, 0]] -- second column all zeros.
    const DMatrix A = mat(2, 2, {4, 1, 1, 3});
    DMatrix B(2, 2, 0.0);
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(1, 0) = 2.0;
    B(1, 1) = 1.0;
    const DMatrix exact = gauss_solve(A, B);
    const auto x = jacobi(A, B, 20000, 1e-13);
    ASSERT_TRUE(x.has_value());
    ASSERT_EQ(x->cols(), 2u);
    EXPECT_NEAR((*x)(0, 1), exact(0, 1), 1e-9);
    EXPECT_NEAR((*x)(1, 1), exact(1, 1), 1e-9);
    EXPECT_LT(max_abs_diff(*x, exact), 1e-9);
    EXPECT_GT(std::abs((*x)(0, 1)), 0.1) << "second RHS must not come back zero";
}

TEST(IterativeMultiRhs, NonSymmetricThreeColumns) {
    const DMatrix A = mat(3, 3, {4, 1, 2, 0, 3, 1, 1, 0, 5});
    DMatrix B(3, 3, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            B(i, j) = 0.37 * static_cast<double>((i + 1) * (j + 2));
        }
    }
    const DMatrix exact = gauss_solve(A, B);

    const auto xb = bicgstab(A, B, 5000, 1e-13);
    ASSERT_TRUE(xb.has_value());
    EXPECT_EQ(xb->cols(), 3u);
    EXPECT_LT(max_abs_diff(*xb, exact), 1e-8);

    const auto xg = gmres(A, B, 10, 5000, 1e-13);
    ASSERT_TRUE(xg.has_value());
    EXPECT_EQ(xg->cols(), 3u);
    EXPECT_LT(max_abs_diff(*xg, exact), 1e-8);
}

// ---------------------------------------------------------------------------
// BiCGSTAB failure signalling
// ---------------------------------------------------------------------------

TEST(BicgstabFailure, SingularInconsistentSystemIsReported) {
    // [1,1;1,1] x = [1;2] has no solution; the old code returned [nan; nan]
    // as a successful Result while cg/gmres/jacobi reported the failure.
    const DMatrix A = mat(2, 2, {1, 1, 1, 1});
    const DMatrix b = col({1.0, 2.0});
    const auto x = bicgstab(A, b);
    ASSERT_FALSE(x.has_value()) << "an unsolvable system must not report success";
    EXPECT_TRUE(std::holds_alternative<ConvergenceFail>(x.error()));
}

TEST(BicgstabFailure, ExhaustedIterationsAreReported) {
    const DMatrix A = mat(3, 3, {4, 1, 2, 0, 3, 1, 1, 0, 5});
    const DMatrix b = col({1.0, -2.0, 3.0});
    const auto x = bicgstab(A, b, 1, 1e-14);
    if (!x.has_value()) {
        EXPECT_TRUE(std::holds_alternative<ConvergenceFail>(x.error()));
    } else {
        // If one step happens to be enough, the answer must actually be right.
        EXPECT_LT(residual_norm(A, *x, b), 1e-12);
    }
}

TEST(BicgstabFailure, StillSolvesWellPosedSystems) {
    const DMatrix A = mat(3, 3, {4, 1, 2, 0, 3, 1, 1, 0, 5});
    const DMatrix b = col({1.0, -2.0, 3.0});
    const DMatrix exact = gauss_solve(A, b);
    const auto x = bicgstab(A, b, 1000, 1e-13);
    ASSERT_TRUE(x.has_value());
    EXPECT_LT(max_abs_diff(*x, exact), 1e-9);
}

TEST(BicgstabFailure, ZeroRightHandSideGivesZero) {
    const DMatrix A = mat(2, 2, {2, 0, 0, 3});
    const auto x = bicgstab(A, zeros<double>(2, 1));
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ((*x)(0, 0), 0.0);
    EXPECT_EQ((*x)(1, 0), 0.0);
}
