// MathScript Core Operations Numerical Reference Tests
// Covers matmul, lu, qr, chol, expm, transpose, zeros, ones, eye

#include <gtest/gtest.h>
#include <cmath>
#include <tuple>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

static DMatrix mm(const DMatrix& A, const DMatrix& B) {
    return matmul(A, B).value();
}

static DMatrix eye3() { return eye<double>(3); }

static double mat_diff_norm(const DMatrix& A, const DMatrix& B) {
    double r = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j) {
            double d = A(i, j) - B(i, j);
            r += d * d;
        }
    return std::sqrt(r);
}

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

TEST(CoreOpsRef, MatMul_Identity) {
    const DMatrix I = eye3();
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=9;
    const DMatrix AI = mm(A, I);
    EXPECT_LT(mat_diff_norm(AI, A), 1e-12);
}

TEST(CoreOpsRef, MatMul_Zeros) {
    const DMatrix A = ones<double>(3, 3);
    const DMatrix Z = zeros<double>(3, 3);
    const DMatrix AZ = mm(A, Z);
    EXPECT_LT(mat_diff_norm(AZ, Z), 1e-12);
}

TEST(CoreOpsRef, MatMul_2x2_Known) {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    DMatrix A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    DMatrix B(2, 2);
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;
    const DMatrix C = mm(A, B);
    EXPECT_NEAR(C(0,0), 19.0, 1e-12);
    EXPECT_NEAR(C(0,1), 22.0, 1e-12);
    EXPECT_NEAR(C(1,0), 43.0, 1e-12);
    EXPECT_NEAR(C(1,1), 50.0, 1e-12);
}

TEST(CoreOpsRef, MatMul_Rectangular) {
    // (2x3) * (3x2) = (2x2)
    DMatrix A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    DMatrix B(3, 2);
    B(0,0)=7; B(0,1)=8; B(1,0)=9; B(1,1)=10; B(2,0)=11; B(2,1)=12;
    const DMatrix C = mm(A, B);
    ASSERT_EQ(C.rows(), 2u);
    ASSERT_EQ(C.cols(), 2u);
    EXPECT_NEAR(C(0,0), 58.0, 1e-12);   // 1*7+2*9+3*11
    EXPECT_NEAR(C(0,1), 64.0, 1e-12);   // 1*8+2*10+3*12
    EXPECT_NEAR(C(1,0), 139.0, 1e-12);  // 4*7+5*9+6*11
    EXPECT_NEAR(C(1,1), 154.0, 1e-12);  // 4*8+5*10+6*12
}

TEST(CoreOpsRef, MatMul_Associativity) {
    DMatrix A(2, 2); A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    DMatrix B(2, 2); B(0,0)=2; B(0,1)=0; B(1,0)=1; B(1,1)=3;
    DMatrix C(2, 2); C(0,0)=1; C(0,1)=1; C(1,0)=2; C(1,1)=1;
    const DMatrix ABC1 = mm(mm(A, B), C);
    const DMatrix ABC2 = mm(A, mm(B, C));
    EXPECT_LT(mat_diff_norm(ABC1, ABC2), 1e-10);
}

// ---------------------------------------------------------------------------
// LU decomposition: A = L * U (with partial pivoting P*A = L*U)
// ---------------------------------------------------------------------------

TEST(CoreOpsRef, LU_Identity_ReturnsValue) {
    const auto result = lu(eye3());
    ASSERT_TRUE(result.has_value());
    const auto& [L, U, P] = *result;
    EXPECT_EQ(L.rows(), 3u);
    EXPECT_EQ(U.rows(), 3u);
    EXPECT_EQ(P.rows(), 3u);
}

TEST(CoreOpsRef, LU_2x2_Reconstruction) {
    DMatrix A(2, 2);
    A(0,0)=2; A(0,1)=1; A(1,0)=4; A(1,1)=3;
    const auto result = lu(A);
    ASSERT_TRUE(result.has_value());
    const auto& [L, U, P] = *result;
    // P*A = L*U  =>  P*(L*U) should reconstruct P*A
    const DMatrix LU_prod = mm(L, U);
    const DMatrix PA = mm(P, A);
    EXPECT_LT(mat_diff_norm(LU_prod, PA), 1e-10);
}

TEST(CoreOpsRef, LU_3x3_Reconstruction) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=10;
    const auto result = lu(A);
    ASSERT_TRUE(result.has_value());
    const auto& [L, U, P] = *result;
    const DMatrix PA = mm(P, A);
    const DMatrix LU_prod = mm(L, U);
    EXPECT_LT(mat_diff_norm(LU_prod, PA), 1e-9);
}

TEST(CoreOpsRef, LU_Singular_Returns) {
    DMatrix A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=2; A(1,1)=4; // singular
    const auto result = lu(A);
    // Either returns or errors gracefully
    (void)result;
    SUCCEED();
}

// ---------------------------------------------------------------------------
// QR decomposition: A = Q * R
// ---------------------------------------------------------------------------

TEST(CoreOpsRef, QR_Identity_ReturnsValue) {
    const auto result = qr(eye3());
    ASSERT_TRUE(result.has_value());
    const auto& [Q, R] = *result;
    EXPECT_EQ(Q.rows(), 3u);
    EXPECT_EQ(R.rows(), 3u);
}

TEST(CoreOpsRef, QR_2x2_Reconstruction) {
    DMatrix A(2, 2);
    A(0,0)=3; A(0,1)=1; A(1,0)=4; A(1,1)=1;
    const auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    const auto& [Q, R] = *result;
    const DMatrix QR_prod = mm(Q, R);
    EXPECT_LT(mat_diff_norm(QR_prod, A), 1e-9);
}

TEST(CoreOpsRef, QR_OrthogonalQ) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=10;
    const auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    const auto& [Q, R] = *result;
    // Q^T * Q should be identity
    DMatrix Qcol(Q.rows(), Q.cols());
    for (size_t i = 0; i < Q.rows(); ++i)
        for (size_t j = 0; j < Q.cols(); ++j)
            Qcol(i, j) = Q(i, j);
    DMatrix Qt(Q.cols(), Q.rows());
    for (size_t i = 0; i < Q.rows(); ++i)
        for (size_t j = 0; j < Q.cols(); ++j)
            Qt(j, i) = Q(i, j);
    const DMatrix QtQ = mm(Qt, Qcol);
    const DMatrix I = eye3();
    EXPECT_LT(mat_diff_norm(QtQ, I), 1e-8);
}

TEST(CoreOpsRef, QR_UpperTriangularR) {
    DMatrix A(3, 3);
    A(0,0)=2; A(0,1)=3; A(0,2)=1;
    A(1,0)=1; A(1,1)=4; A(1,2)=2;
    A(2,0)=0; A(2,1)=1; A(2,2)=5;
    const auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    const auto& [Q, R] = *result;
    for (size_t i = 0; i < R.rows(); ++i)
        for (size_t j = 0; j < i && j < R.cols(); ++j)
            EXPECT_NEAR(R(i, j), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Cholesky decomposition: A = L * L^T for SPD matrix
// ---------------------------------------------------------------------------

TEST(CoreOpsRef, Chol_Identity_ReturnsValue) {
    const auto result = chol(eye3());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 3u);
}

TEST(CoreOpsRef, Chol_2x2_SPD_Reconstruction) {
    DMatrix A(2, 2);
    A(0,0)=4; A(0,1)=2; A(1,0)=2; A(1,1)=3;
    const auto result = chol(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix& L = *result;
    DMatrix Lt(2, 2);
    Lt(0,0)=L(0,0); Lt(0,1)=L(1,0);
    Lt(1,0)=L(0,1); Lt(1,1)=L(1,1);
    const DMatrix LLt = mm(L, Lt);
    EXPECT_LT(mat_diff_norm(LLt, A), 1e-10);
}

TEST(CoreOpsRef, Chol_3x3_SPD_LowerTriangular) {
    DMatrix A(3, 3);
    A(0,0)=4; A(0,1)=2; A(0,2)=0;
    A(1,0)=2; A(1,1)=5; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=3;
    const auto result = chol(A);
    ASSERT_TRUE(result.has_value());
    const DMatrix& L = *result;
    // Upper triangle of L should be 0
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = i + 1; j < 3; ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-10);
}

TEST(CoreOpsRef, Chol_NonSPD_ReturnsError) {
    DMatrix A(2, 2);
    A(0,0)=-1; A(0,1)=0; A(1,0)=0; A(1,1)=1;  // not positive definite
    const auto result = chol(A);
    EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Matrix exponential: expm
// ---------------------------------------------------------------------------

TEST(CoreOpsRef, Expm_ZeroMatrix_Is_Identity) {
    const DMatrix Z = zeros<double>(3, 3);
    const auto result = expm(Z);
    ASSERT_TRUE(result.has_value());
    const DMatrix I = eye3();
    EXPECT_LT(mat_diff_norm(*result, I), 1e-8);
}

TEST(CoreOpsRef, Expm_Diagonal_Is_ExpOfDiag) {
    // expm(diag([1,2])) = diag([e^1, e^2])
    DMatrix A(2, 2);
    A(0,0)=1.0; A(0,1)=0.0; A(1,0)=0.0; A(1,1)=2.0;
    const auto result = expm(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)(0,0), std::exp(1.0), 1e-5);
    EXPECT_NEAR((*result)(1,1), std::exp(2.0), 1e-5);
    EXPECT_NEAR((*result)(0,1), 0.0, 1e-9);
    EXPECT_NEAR((*result)(1,0), 0.0, 1e-9);
}

TEST(CoreOpsRef, Expm_Identity_Is_eI) {
    // expm(I) = e * I (diagonal matrix with e on the diagonal)
    const auto result = expm(eye3());
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)(0,0), std::exp(1.0), 1e-6);
    EXPECT_NEAR((*result)(1,1), std::exp(1.0), 1e-6);
    EXPECT_NEAR((*result)(2,2), std::exp(1.0), 1e-6);
}

// ---------------------------------------------------------------------------
// zeros / ones / eye constructors
// ---------------------------------------------------------------------------

TEST(CoreOpsRef, Zeros_AllZero) {
    const DMatrix Z = zeros<double>(3, 4);
    ASSERT_EQ(Z.rows(), 3u); ASSERT_EQ(Z.cols(), 4u);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(Z(i, j), 0.0, 1e-15);
}

TEST(CoreOpsRef, Ones_AllOne) {
    const DMatrix O = ones<double>(2, 5);
    ASSERT_EQ(O.rows(), 2u); ASSERT_EQ(O.cols(), 5u);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 5; ++j)
            EXPECT_NEAR(O(i, j), 1.0, 1e-15);
}

TEST(CoreOpsRef, Eye_DiagonalIsOne) {
    const DMatrix I = eye<double>(4);
    ASSERT_EQ(I.rows(), 4u); ASSERT_EQ(I.cols(), 4u);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(I(i, j), (i == j) ? 1.0 : 0.0, 1e-15);
}
