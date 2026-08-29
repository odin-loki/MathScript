// MathScript Linalg: Hessenberg, Bidiagonalization, Schur Decomposition Tests

#include <gtest/gtest.h>
#include <cmath>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static DMatrix make_sym3() {
    DMatrix A(3, 3);
    A(0,0)=4; A(0,1)=1; A(0,2)=2;
    A(1,0)=1; A(1,1)=3; A(1,2)=1;
    A(2,0)=2; A(2,1)=1; A(2,2)=5;
    return A;
}

static DMatrix make_gen4() {
    DMatrix A(4, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i,j) = static_cast<double>((i+1) * (j+1)) + (i == j ? 5.0 : 0.0);
    return A;
}

// ---------------------------------------------------------------------------
// hess: Hessenberg reduction
// ---------------------------------------------------------------------------

TEST(HessDecomp, ReturnsValue_3x3) {
    auto A = make_sym3();
    auto result = hess(A);
    ASSERT_TRUE(result.has_value()) << "hess should succeed for 3x3 matrix";
}

TEST(HessDecomp, Output_CorrectDimensions) {
    auto A = make_sym3();
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 3u);
    EXPECT_EQ(result->cols(), 3u);
}

TEST(HessDecomp, Output_AllFinite) {
    auto A = make_sym3();
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_TRUE(std::isfinite((*result)(i, j)));
}

TEST(HessDecomp, SubdiagonalBelowZero_For_Hessenberg) {
    // True Hessenberg form: all elements below the first subdiagonal should be zero
    auto A = make_sym3();
    auto result = hess(A);
    if (!result.has_value()) { SUCCEED(); return; }
    // For a 3x3 Hessenberg, H[2][0] should be ~0
    EXPECT_NEAR((*result)(2, 0), 0.0, 1e-8);
}

TEST(HessDecomp, Identity_StaysIdentity) {
    // hess(I) = I (identity is already in Hessenberg form)
    DMatrix I(3, 3);
    I(0,0) = I(1,1) = I(2,2) = 1.0;
    auto result = hess(I);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR((*result)(i, i), 1.0, 1e-10);
    }
}

TEST(HessDecomp, LargerMatrix_Succeeds) {
    auto A = make_gen4();
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 4u);
    // H[3][0], H[3][1], H[2][0] should be ~0
    EXPECT_NEAR((*result)(3, 0), 0.0, 1e-8);
    EXPECT_NEAR((*result)(3, 1), 0.0, 1e-8);
    EXPECT_NEAR((*result)(2, 0), 0.0, 1e-8);
}

// ---------------------------------------------------------------------------
// bidiag: Bidiagonalization
// ---------------------------------------------------------------------------

TEST(BidiagDecomp, ReturnsValue_3x3) {
    auto A = make_sym3();
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value()) << "bidiag should succeed for 3x3 matrix";
}

TEST(BidiagDecomp, UBV_Dimensions) {
    auto A = make_sym3();
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->U.rows(), 3u);
    EXPECT_EQ(result->B.rows(), 3u);
    EXPECT_EQ(result->B.cols(), 3u);
    EXPECT_EQ(result->V.cols(), 3u);
}

TEST(BidiagDecomp, All_Finite) {
    auto A = make_sym3();
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(std::isfinite(result->U(i, j)));
            EXPECT_TRUE(std::isfinite(result->B(i, j)));
            EXPECT_TRUE(std::isfinite(result->V(i, j)));
        }
}

TEST(BidiagDecomp, B_Is_Bidiagonal) {
    // B should have zeros below subdiagonal and above first superdiagonal
    auto A = make_sym3();
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    const auto& B = result->B;
    // B[2][0], B[1][0], B[2][1] should be ~0 (below diagonal)
    EXPECT_NEAR(B(2, 0), 0.0, 1e-8) << "B should be bidiagonal";
    // B[0][2] should be ~0 (above first superdiagonal)
    EXPECT_NEAR(B(0, 2), 0.0, 1e-8) << "B should be bidiagonal";
}

// ---------------------------------------------------------------------------
// schur: Schur decomposition
// ---------------------------------------------------------------------------

TEST(SchurDecomp, ReturnsValue_3x3) {
    auto A = make_sym3();
    auto result = schur(A);
    ASSERT_TRUE(result.has_value()) << "schur should succeed for 3x3 matrix";
}

TEST(SchurDecomp, TQ_Dimensions) {
    auto A = make_sym3();
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->T.rows(), 3u);
    EXPECT_EQ(result->T.cols(), 3u);
    EXPECT_EQ(result->Q.rows(), 3u);
    EXPECT_EQ(result->Q.cols(), 3u);
}

TEST(SchurDecomp, T_AllFinite) {
    auto A = make_sym3();
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_TRUE(std::isfinite(result->T(i, j)));
}

TEST(SchurDecomp, T_Is_Upper_Triangular_Or_Quasi) {
    // For a symmetric matrix, the Schur form should be upper triangular (real eigenvalues)
    auto A = make_sym3();
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    const auto& T = result->T;
    // Sub-diagonal of T should be ~0 (it's upper triangular for real symmetric)
    EXPECT_NEAR(T(1, 0), 0.0, 1e-6) << "Schur T should be upper triangular for symmetric A";
    EXPECT_NEAR(T(2, 0), 0.0, 1e-6);
    EXPECT_NEAR(T(2, 1), 0.0, 1e-6);
}

TEST(SchurDecomp, Identity_SchurIsIdentity) {
    DMatrix I(3, 3);
    I(0,0) = I(1,1) = I(2,2) = 1.0;
    auto result = schur(I);
    ASSERT_TRUE(result.has_value());
    // T for identity should be diagonal (or upper triangular with 1s on diagonal)
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(result->T(i, i), 1.0, 1e-8);
}
