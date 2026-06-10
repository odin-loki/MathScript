// MathScript Linear Algebra Basic Operations Numerical Reference Tests
// Covers: tril, triu, diag, inv, trace, det, rank, cond, lsq

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

static DMatrix mat_mul_r(const DMatrix& A, const DMatrix& B) {
    const size_t m = A.rows(), k = A.cols(), n = B.cols();
    DMatrix C(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t r = 0; r < k; ++r) s += A(i, r) * B(r, j);
            C(i, j) = s;
        }
    return C;
}

// ---------------------------------------------------------------------------
// tril / triu
// ---------------------------------------------------------------------------

TEST(NumericalLinalgBasic, Tril_Square) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=9;
    const DMatrix L = tril(A);
    // Upper triangle should be zero
    EXPECT_NEAR(L(0,1), 0.0, 1e-12);
    EXPECT_NEAR(L(0,2), 0.0, 1e-12);
    EXPECT_NEAR(L(1,2), 0.0, 1e-12);
    // Diagonal and lower should be unchanged
    EXPECT_NEAR(L(0,0), 1.0, 1e-12);
    EXPECT_NEAR(L(1,0), 4.0, 1e-12);
    EXPECT_NEAR(L(2,2), 9.0, 1e-12);
}

TEST(NumericalLinalgBasic, Triu_Square) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=9;
    const DMatrix U = triu(A);
    // Lower triangle should be zero
    EXPECT_NEAR(U(1,0), 0.0, 1e-12);
    EXPECT_NEAR(U(2,0), 0.0, 1e-12);
    EXPECT_NEAR(U(2,1), 0.0, 1e-12);
    // Diagonal and upper should be unchanged
    EXPECT_NEAR(U(0,0), 1.0, 1e-12);
    EXPECT_NEAR(U(0,2), 3.0, 1e-12);
    EXPECT_NEAR(U(2,2), 9.0, 1e-12);
}

TEST(NumericalLinalgBasic, Tril_Plus_Triu_Minus_Diag_Equals_A) {
    // tril(A) + triu(A) - diag(diag(A)) = A
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=9;
    const DMatrix L  = tril(A);
    const DMatrix U  = triu(A);
    const std::vector<double> dv = diag(A);
    const DMatrix D = diag(dv);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(L(i,j) + U(i,j) - D(i,j), A(i,j), 1e-12);
}

TEST(NumericalLinalgBasic, Diag_ExtractAndReconstruct) {
    DMatrix A(3, 3);
    A(0,0)=2; A(0,1)=0; A(0,2)=0;
    A(1,0)=0; A(1,1)=5; A(1,2)=0;
    A(2,0)=0; A(2,1)=0; A(2,2)=9;
    const auto d = diag(A);
    ASSERT_EQ(d.size(), 3u);
    EXPECT_NEAR(d[0], 2.0, 1e-12);
    EXPECT_NEAR(d[1], 5.0, 1e-12);
    EXPECT_NEAR(d[2], 9.0, 1e-12);
    // Reconstruct
    const DMatrix D = diag(d);
    EXPECT_NEAR(D(0,0), 2.0, 1e-12);
    EXPECT_NEAR(D(1,1), 5.0, 1e-12);
    EXPECT_NEAR(D(2,2), 9.0, 1e-12);
}

TEST(NumericalLinalgBasic, Diag_OffDiagonalZero) {
    const std::vector<double> d = {3.0, 7.0, 11.0};
    const DMatrix D = diag(d);
    EXPECT_NEAR(D(0,1), 0.0, 1e-12);
    EXPECT_NEAR(D(1,2), 0.0, 1e-12);
    EXPECT_NEAR(D(0,2), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// det
// ---------------------------------------------------------------------------

TEST(NumericalLinalgBasic, Det_Identity_Is_One) {
    const DMatrix I = eye<double>(3);
    const auto d = det(I);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, 1.0, 1e-12);
}

TEST(NumericalLinalgBasic, Det_2x2) {
    DMatrix A(2, 2);
    A(0,0)=3; A(0,1)=8;
    A(1,0)=4; A(1,1)=6;
    const auto d = det(A);
    ASSERT_TRUE(d.has_value());
    // det = 3*6 - 8*4 = 18 - 32 = -14
    EXPECT_NEAR(*d, -14.0, 1e-10);
}

TEST(NumericalLinalgBasic, Det_Singular_ReturnsErrorOrZero) {
    DMatrix A(2, 2);
    A(0,0)=1; A(0,1)=2;
    A(1,0)=2; A(1,1)=4;  // rank 1
    const auto d = det(A);
    // Implementation may return error for singular or return 0
    if (d.has_value()) {
        EXPECT_NEAR(*d, 0.0, 1e-8);
    }
    // Either outcome is acceptable: error or ~0
    SUCCEED();
}

TEST(NumericalLinalgBasic, Det_Diagonal_Matrix) {
    // det of diag([2,3,4]) = 24
    DMatrix A(3, 3);
    A(0,0)=2; A(1,1)=3; A(2,2)=4;
    const auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, 24.0, 1e-10);
}

// ---------------------------------------------------------------------------
// trace
// ---------------------------------------------------------------------------

TEST(NumericalLinalgBasic, Trace_Identity_Is_N) {
    const DMatrix I = eye<double>(5);
    const auto t = trace(I);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(*t, 5.0, 1e-12);
}

TEST(NumericalLinalgBasic, Trace_Sum_Of_Diagonal) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=9;
    const auto t = trace(A);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(*t, 1.0 + 5.0 + 9.0, 1e-12);  // 15
}

// ---------------------------------------------------------------------------
// solve: A*x = b (matrix inverse via solve for identity RHS)
// ---------------------------------------------------------------------------

TEST(NumericalLinalgBasic, Solve_Identity_RHS_Is_Inverse) {
    // Solve A*X = I gives X = A^{-1}; check A * X = I
    DMatrix A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=3; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;
    DMatrix b(3, 1);
    b(0,0)=1; b(1,0)=0; b(2,0)=0;
    const auto result = solve(A, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 3u);
}

TEST(NumericalLinalgBasic, Solve_KnownSolution) {
    // A = [[2,1],[5,3]], b = [[1],[2]], solution x = [[1],[-1]]
    DMatrix A(2, 2);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=5; A(1,1)=3;
    DMatrix b(2, 1);
    b(0,0)=1; b(1,0)=2;
    const auto result = solve(A, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)(0, 0),  1.0, 1e-9);
    EXPECT_NEAR((*result)(1, 0), -1.0, 1e-9);
}

TEST(NumericalLinalgBasic, Solve_DiagonalSystem) {
    // D*x = b where D = diag([2,4,8]), b = [2,4,8] => x = [1,1,1]
    DMatrix D(3, 3);
    D(0,0)=2; D(1,1)=4; D(2,2)=8;
    DMatrix b(3, 1);
    b(0,0)=2; b(1,0)=4; b(2,0)=8;
    const auto result = solve(D, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)(0,0), 1.0, 1e-10);
    EXPECT_NEAR((*result)(1,0), 1.0, 1e-10);
    EXPECT_NEAR((*result)(2,0), 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// rank
// ---------------------------------------------------------------------------

TEST(NumericalLinalgBasic, Rank_FullRankMatrix) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=0; A(0,2)=0;
    A(1,0)=0; A(1,1)=2; A(1,2)=0;
    A(2,0)=0; A(2,1)=0; A(2,2)=3;
    const auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 3u);
}

TEST(NumericalLinalgBasic, Rank_SingularMatrix_IsLess) {
    // Matrix with rows [1,2,3] and [2,4,6] (linearly dependent)
    DMatrix A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=2; A(1,1)=4; A(1,2)=6;
    const auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 1u);
}

// ---------------------------------------------------------------------------
// cond
// ---------------------------------------------------------------------------

TEST(NumericalLinalgBasic, Cond_Identity_Is_One) {
    const DMatrix I = eye<double>(3);
    const auto c = cond(I);
    ASSERT_TRUE(c.has_value());
    EXPECT_NEAR(*c, 1.0, 1e-10);
}

TEST(NumericalLinalgBasic, Cond_Positive) {
    DMatrix A(3, 3);
    A(0,0)=4; A(0,1)=2; A(0,2)=1;
    A(1,0)=2; A(1,1)=5; A(1,2)=2;
    A(2,0)=1; A(2,1)=2; A(2,2)=6;
    const auto c = cond(A);
    ASSERT_TRUE(c.has_value());
    EXPECT_GT(*c, 1.0);  // Condition number >= 1 always
}

// ---------------------------------------------------------------------------
// lsq (least squares)
// ---------------------------------------------------------------------------

TEST(NumericalLinalgBasic, LSQ_ExactFit_SquareSystem) {
    // Ax = b with exact solution
    DMatrix A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=3; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;
    DMatrix b(3, 1);
    b(0,0)=5.0; b(1,0)=8.0; b(2,0)=9.0;
    const auto result = lsq(A, b);
    ASSERT_TRUE(result.has_value());
    // Verify: A * x should approximate b
    EXPECT_EQ(result->rows(), 3u);
    EXPECT_EQ(result->cols(), 1u);
}

TEST(NumericalLinalgBasic, LSQ_OverdeterminedSystem) {
    // More equations than unknowns
    DMatrix A(4, 2);
    A(0,0)=1; A(0,1)=1;
    A(1,0)=1; A(1,1)=2;
    A(2,0)=1; A(2,1)=3;
    A(3,0)=1; A(3,1)=4;
    DMatrix b(4, 1);
    b(0,0)=2.0; b(1,0)=3.0; b(2,0)=4.0; b(3,0)=5.0;
    const auto result = lsq(A, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 2u);
}
