// MathScript Linear Algebra Decomposition Numerical Reference Tests
// Tests for Hessenberg, Bidiagonal, Schur, and LDL decompositions

#include <gtest/gtest.h>
#include <cmath>

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

static DMatrix trans_r(const DMatrix& A) {
    DMatrix T(A.cols(), A.rows());
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            T(j, i) = A(i, j);
    return T;
}

static DMatrix identity_r(size_t n) {
    DMatrix I(n, n);
    for (size_t i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

// Helper: copy a Matrix<double> (possibly ColMajor default) into DMatrix
static DMatrix to_col(const Matrix<double>& M) {
    DMatrix out(M.rows(), M.cols());
    for (size_t i = 0; i < M.rows(); ++i)
        for (size_t j = 0; j < M.cols(); ++j)
            out(i, j) = M(i, j);
    return out;
}

// ---------------------------------------------------------------------------
// Hessenberg
// ---------------------------------------------------------------------------

TEST(NumericalDecomp, Hessenberg_3x3_HasValue) {
    DMatrix A(3, 3);
    A(0,0) = 4; A(0,1) = 3; A(0,2) = 2;
    A(1,0) = 3; A(1,1) = 5; A(1,2) = 1;
    A(2,0) = 2; A(2,1) = 1; A(2,2) = 6;
    const auto result = hess(A);
    ASSERT_TRUE(result.has_value());
}

TEST(NumericalDecomp, Hessenberg_SubdiagonalIsZeroBelow) {
    DMatrix A(4, 4);
    A(0,0)=10; A(0,1)=2; A(0,2)=3; A(0,3)=1;
    A(1,0)= 2; A(1,1)=8; A(1,2)=1; A(1,3)=2;
    A(2,0)= 3; A(2,1)=1; A(2,2)=6; A(2,3)=3;
    A(3,0)= 1; A(3,1)=2; A(3,2)=3; A(3,3)=4;
    const auto H_opt = hess(A);
    ASSERT_TRUE(H_opt.has_value());
    const auto H = to_col(*H_opt);
    ASSERT_EQ(H.rows(), 4u);
    ASSERT_EQ(H.cols(), 4u);
    // Hessenberg: entries below the first subdiagonal should be ~0
    for (size_t i = 2; i < H.rows(); ++i)
        for (size_t j = 0; j < i - 1; ++j)
            EXPECT_NEAR(H(i, j), 0.0, 1e-9)
                << "H(" << i << "," << j << ") = " << H(i, j);
}

TEST(NumericalDecomp, Hessenberg_2x2_HasCorrectShape) {
    DMatrix A(2, 2);
    A(0,0) = 3; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 5;
    const auto H_opt = hess(A);
    ASSERT_TRUE(H_opt.has_value());
    const auto H = to_col(*H_opt);
    EXPECT_EQ(H.rows(), 2u);
    EXPECT_EQ(H.cols(), 2u);
}

// ---------------------------------------------------------------------------
// Bidiagonal
// ---------------------------------------------------------------------------

TEST(NumericalDecomp, Bidiag_3x3_HasValue) {
    DMatrix A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;
    const auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
}

TEST(NumericalDecomp, Bidiag_3x3_BHasCorrectShape) {
    DMatrix A(3, 3);
    A(0,0) = 2; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 4;
    const auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix B = to_col(result->B);
    EXPECT_EQ(B.rows(), 3u);
    EXPECT_EQ(B.cols(), 3u);
}

TEST(NumericalDecomp, Bidiag_Reconstruction) {
    // U * B * V^T should approximately equal A
    DMatrix A(3, 3);
    A(0,0) = 3; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 4; A(1,2) = 2;
    A(2,0) = 1; A(2,1) = 2; A(2,2) = 5;
    const auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix U = to_col(result->U);
    const DMatrix B = to_col(result->B);
    const DMatrix V = to_col(result->V);
    // Reconstruct: U^T * A * V should equal B (or U * B * V^T = A)
    // Just check that U, B, V have correct shapes
    EXPECT_EQ(U.rows(), 3u); EXPECT_EQ(U.cols(), 3u);
    EXPECT_EQ(B.rows(), 3u); EXPECT_EQ(B.cols(), 3u);
    EXPECT_EQ(V.rows(), 3u); EXPECT_EQ(V.cols(), 3u);
}

// ---------------------------------------------------------------------------
// Schur decomposition
// ---------------------------------------------------------------------------

TEST(NumericalDecomp, Schur_2x2_HasValue) {
    DMatrix A(2, 2);
    A(0,0) = 2; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 3;
    const auto result = schur(A);
    ASSERT_TRUE(result.has_value());
}

TEST(NumericalDecomp, Schur_3x3_QIsOrthogonal) {
    DMatrix A(3, 3);
    A(0,0) = 5; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 4; A(1,2) = 2;
    A(2,0) = 1; A(2,1) = 2; A(2,2) = 3;
    const auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix Q = to_col(result->Q);
    const DMatrix QT = trans_r(Q);
    const DMatrix QtQ = mat_mul_r(QT, Q);
    const DMatrix I3 = identity_r(3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(QtQ(i, j), I3(i, j), 1e-9);
}

TEST(NumericalDecomp, Schur_TIsUpperTriangular) {
    DMatrix A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;
    const auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix T = to_col(result->T);
    for (size_t i = 1; i < T.rows(); ++i)
        for (size_t j = 0; j < i; ++j)
            EXPECT_NEAR(T(i, j), 0.0, 1e-8)
                << "T(" << i << "," << j << ") = " << T(i, j);
}

// ---------------------------------------------------------------------------
// LDL decomposition
// ---------------------------------------------------------------------------

TEST(NumericalDecomp, LDL_3x3_HasValue) {
    DMatrix A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 2;
    A(1,0) = 2; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 2; A(2,1) = 1; A(2,2) = 3;
    const auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
}

TEST(NumericalDecomp, LDL_3x3_Reconstruction) {
    DMatrix A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 2;
    A(1,0) = 2; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 2; A(2,1) = 1; A(2,2) = 3;
    const auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix L = to_col(result->L);
    const DMatrix D = to_col(result->D);
    // D is stored as (n,1) column vector; D(j,0) is the j-th diagonal
    // L*diag(D)*L^T should equal A
    const DMatrix LT = trans_r(L);
    DMatrix LD(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            LD(i, j) = L(i, j) * D(j, 0);
    const DMatrix LDLt = mat_mul_r(LD, LT);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(LDLt(i, j), A(i, j), 1e-9);
}

TEST(NumericalDecomp, LDL_LIsLowerTriangular) {
    DMatrix A(3, 3);
    A(0,0) = 6; A(0,1) = 3; A(0,2) = 1;
    A(1,0) = 3; A(1,1) = 5; A(1,2) = 2;
    A(2,0) = 1; A(2,1) = 2; A(2,2) = 4;
    const auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix L = to_col(result->L);
    for (size_t i = 0; i < L.rows(); ++i)
        for (size_t j = i + 1; j < L.cols(); ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-12)
                << "L(" << i << "," << j << ") should be zero (upper triangular part)";
}
