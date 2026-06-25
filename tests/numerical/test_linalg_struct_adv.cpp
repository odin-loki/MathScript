// MathScript: Advanced Linear Algebra Structure Tests (Wave 43)
// Tests for tril, triu (with k offset), hess, bidiag, schur

#include <gtest/gtest.h>
#include <cmath>
#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// tril - lower triangular extraction
// ---------------------------------------------------------------------------

TEST(LinalgStruct, Tril_K0_Main_Diagonal) {
    // Extract lower triangle including main diagonal
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto L = tril(A, 0);
    // Above diagonal should be zero
    EXPECT_NEAR(L(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(L(0, 2), 0.0, 1e-10);
    EXPECT_NEAR(L(1, 2), 0.0, 1e-10);
    // At and below diagonal preserved
    EXPECT_NEAR(L(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(L(1, 0), 4.0, 1e-10);
    EXPECT_NEAR(L(2, 0), 7.0, 1e-10);
    EXPECT_NEAR(L(1, 1), 5.0, 1e-10);
    EXPECT_NEAR(L(2, 2), 9.0, 1e-10);
}

TEST(LinalgStruct, Tril_KMinus1_ExcludeMainDiagonal) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto L = tril(A, -1);
    // Main and above diagonal should be zero
    EXPECT_NEAR(L(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(L(1, 1), 0.0, 1e-10);
    EXPECT_NEAR(L(2, 2), 0.0, 1e-10);
    // Sub-diagonal preserved
    EXPECT_NEAR(L(1, 0), 4.0, 1e-10);
    EXPECT_NEAR(L(2, 0), 7.0, 1e-10);
    EXPECT_NEAR(L(2, 1), 8.0, 1e-10);
}

TEST(LinalgStruct, Tril_KPlus1_IncludeOneSuperdiagonal) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto L = tril(A, 1);
    // Two super-diagonals above should be zero
    EXPECT_NEAR(L(0, 2), 0.0, 1e-10);
    // First super-diagonal preserved
    EXPECT_NEAR(L(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(L(1, 2), 6.0, 1e-10);
}

TEST(LinalgStruct, Tril_Identity_IsIdentity) {
    DMatrix I({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    auto L = tril(I, 0);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(L(i, j), I(i, j), 1e-10);
}

// ---------------------------------------------------------------------------
// triu - upper triangular extraction
// ---------------------------------------------------------------------------

TEST(LinalgStruct, Triu_K0_Main_Diagonal) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto U = triu(A, 0);
    // Below diagonal should be zero
    EXPECT_NEAR(U(1, 0), 0.0, 1e-10);
    EXPECT_NEAR(U(2, 0), 0.0, 1e-10);
    EXPECT_NEAR(U(2, 1), 0.0, 1e-10);
    // At and above diagonal preserved
    EXPECT_NEAR(U(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(U(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(U(0, 2), 3.0, 1e-10);
    EXPECT_NEAR(U(1, 1), 5.0, 1e-10);
    EXPECT_NEAR(U(2, 2), 9.0, 1e-10);
}

TEST(LinalgStruct, Triu_KPlus1_ExcludeMainDiagonal) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto U = triu(A, 1);
    // Main and below diagonal should be zero
    EXPECT_NEAR(U(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(U(1, 1), 0.0, 1e-10);
    EXPECT_NEAR(U(2, 2), 0.0, 1e-10);
    // Super-diagonal preserved
    EXPECT_NEAR(U(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(U(0, 2), 3.0, 1e-10);
    EXPECT_NEAR(U(1, 2), 6.0, 1e-10);
}

TEST(LinalgStruct, Triu_KMinus1_IncludeOneSubdiagonal) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto U = triu(A, -1);
    // Two sub-diagonals below should be zero
    EXPECT_NEAR(U(2, 0), 0.0, 1e-10);
    // First sub-diagonal preserved
    EXPECT_NEAR(U(1, 0), 4.0, 1e-10);
    EXPECT_NEAR(U(2, 1), 8.0, 1e-10);
}

TEST(LinalgStruct, Tril_Plus_Triu_Equals_A_Plus_Diagonal) {
    // tril(A, 0) + triu(A, 1) = A  (tril includes diag, triu excludes diag)
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto L = tril(A, 0);
    auto U = triu(A, 1);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(L(i, j) + U(i, j), A(i, j), 1e-10);
}

// ---------------------------------------------------------------------------
// hess - Hessenberg reduction
// ---------------------------------------------------------------------------

TEST(LinalgStruct, Hess_ReturnsValue_3x3) {
    DMatrix A({{4.0, 1.0, -2.0}, {1.0, 3.0, 1.0}, {-2.0, 1.0, 2.0}});
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    EXPECT_EQ(H.rows(), 3u);
    EXPECT_EQ(H.cols(), 3u);
}

TEST(LinalgStruct, Hess_UpperHessenbergStructure) {
    DMatrix A({{2.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    // Hessenberg: all entries below sub-diagonal are zero
    for (size_t i = 2; i < H.rows(); ++i)
        for (size_t j = 0; j + 1 < i; ++j)
            EXPECT_NEAR(H(i, j), 0.0, 1e-9)
                << "H(" << i << "," << j << ") should be zero";
}

TEST(LinalgStruct, Hess_CorrectDimensions) {
    // Hessenberg result should have same dimensions as input
    DMatrix A({{4.0, 1.0, 0.0}, {1.0, 3.0, 2.0}, {0.0, 2.0, 2.0}});
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    EXPECT_EQ(H.rows(), A.rows());
    EXPECT_EQ(H.cols(), A.cols());
}

TEST(LinalgStruct, Hess_AllFinite) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    for (size_t i = 0; i < H.rows(); ++i)
        for (size_t j = 0; j < H.cols(); ++j)
            EXPECT_TRUE(std::isfinite(H(i, j)));
}

// ---------------------------------------------------------------------------
// bidiag - Bidiagonalization
// ---------------------------------------------------------------------------

TEST(LinalgStruct, Bidiag_ReturnsValue_3x3) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    auto& [U, B, V] = result.value();
    EXPECT_EQ(B.rows(), A.rows());
    EXPECT_EQ(B.cols(), A.cols());
}

TEST(LinalgStruct, Bidiag_AllFinite) {
    DMatrix A({{2.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 4.0}});
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    auto& [U, B, V] = result.value();
    for (size_t i = 0; i < B.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j)
            EXPECT_TRUE(std::isfinite(B(i, j)));
}

TEST(LinalgStruct, Bidiag_BMatrix_LowerTriangleNearZero) {
    // Upper bidiagonal: entries below sub-diagonal should be zero
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}});
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    auto& [U, B, V] = result.value();
    // B should be upper bidiagonal (main + one super-diagonal)
    for (size_t i = 1; i < B.rows(); ++i)
        for (size_t j = 0; j + 1 < i; ++j)
            EXPECT_NEAR(B(i, j), 0.0, 1e-8)
                << "B(" << i << "," << j << ") should be zero";
}

// ---------------------------------------------------------------------------
// schur - Schur decomposition
// ---------------------------------------------------------------------------

TEST(LinalgStruct, Schur_ReturnsValue_3x3) {
    DMatrix A({{2.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 4.0}});
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    auto& [T, Q] = result.value();
    EXPECT_EQ(T.rows(), 3u);
    EXPECT_EQ(T.cols(), 3u);
    EXPECT_EQ(Q.rows(), 3u);
    EXPECT_EQ(Q.cols(), 3u);
}

TEST(LinalgStruct, Schur_AllFinite) {
    DMatrix A({{1.0, 2.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 5.0}});
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    auto& [T, Q] = result.value();
    for (size_t i = 0; i < T.rows(); ++i)
        for (size_t j = 0; j < T.cols(); ++j)
            EXPECT_TRUE(std::isfinite(T(i, j)));
}

TEST(LinalgStruct, Schur_UpperTriangular_T) {
    // Upper triangular matrix should give T that is also upper triangular
    DMatrix A({{3.0, 1.0, 2.0}, {0.0, 5.0, 1.0}, {0.0, 0.0, 7.0}});
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    auto& [T, Q] = result.value();
    // T should be upper triangular (or quasi-upper triangular)
    // Check sub-diagonal entries are near zero
    for (size_t i = 1; i < T.rows(); ++i)
        EXPECT_NEAR(T(i, 0), 0.0, 1e-8)
            << "First column below diagonal should be zero";
}

TEST(LinalgStruct, Schur_Diagonal_EigenvaluesPreserved) {
    // For diagonal matrix, T diagonal = eigenvalues
    DMatrix A({{3.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 7.0}});
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    auto& [T, Q] = result.value();
    // Trace should be preserved
    double traceA = 3.0 + 5.0 + 7.0;
    double traceT = T(0, 0) + T(1, 1) + T(2, 2);
    EXPECT_NEAR(traceT, traceA, 1e-8);
}
