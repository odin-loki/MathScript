// MathScript Tests: tril, triu, diag construction/extraction (Wave 55)

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Helper: check if matrix is lower triangular (all above diagonal = 0)
static bool is_lower_tri(const DMatrix& A, double tol = 1e-10) {
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = i + 1; j < A.cols(); ++j)
            if (std::abs(A(i, j)) > tol) return false;
    return true;
}

// Helper: check if matrix is upper triangular (all below diagonal = 0)
static bool is_upper_tri(const DMatrix& A, double tol = 1e-10) {
    for (size_t i = 1; i < A.rows(); ++i)
        for (size_t j = 0; j < i && j < A.cols(); ++j)
            if (std::abs(A(i, j)) > tol) return false;
    return true;
}

// Helper: build full 4x4 test matrix
static DMatrix full4x4() {
    DMatrix A(4, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);
    return A;
}

// ---------------------------------------------------------------------------
// tril: lower triangular extraction
// ---------------------------------------------------------------------------

TEST(LinalgTrilTriu, Tril_IsLowerTriangular) {
    auto A = full4x4();
    auto L = tril(A);
    EXPECT_TRUE(is_lower_tri(L)) << "tril should produce lower triangular matrix";
}

TEST(LinalgTrilTriu, Tril_PreservesLowerElements) {
    auto A = full4x4();
    auto L = tril(A);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j <= i; ++j)
            EXPECT_NEAR(L(i, j), A(i, j), 1e-12)
                << "tril should preserve lower elements at (" << i << "," << j << ")";
}

TEST(LinalgTrilTriu, Tril_ZerosAboveDiag) {
    auto A = full4x4();
    auto L = tril(A);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = i + 1; j < 4; ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-12)
                << "tril should zero out (" << i << "," << j << ")";
}

TEST(LinalgTrilTriu, Tril_OfLowerTri_Unchanged) {
    // tril of a lower triangular matrix should be unchanged
    DMatrix A(3, 3);
    A(0, 0) = 1.0;
    A(1, 0) = 2.0; A(1, 1) = 3.0;
    A(2, 0) = 4.0; A(2, 1) = 5.0; A(2, 2) = 6.0;
    auto L = tril(A);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(L(i, j), A(i, j), 1e-12);
}

TEST(LinalgTrilTriu, Tril_Shape_Preserved) {
    auto A = full4x4();
    auto L = tril(A);
    EXPECT_EQ(L.rows(), 4u);
    EXPECT_EQ(L.cols(), 4u);
}

TEST(LinalgTrilTriu, Tril_IdentityMatrix) {
    DMatrix I(3, 3);
    for (size_t i = 0; i < 3; ++i) I(i, i) = 1.0;
    auto L = tril(I);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(L(i, j), expected, 1e-12);
        }
}

// ---------------------------------------------------------------------------
// triu: upper triangular extraction
// ---------------------------------------------------------------------------

TEST(LinalgTrilTriu, Triu_IsUpperTriangular) {
    auto A = full4x4();
    auto U = triu(A);
    EXPECT_TRUE(is_upper_tri(U)) << "triu should produce upper triangular matrix";
}

TEST(LinalgTrilTriu, Triu_PreservesUpperElements) {
    auto A = full4x4();
    auto U = triu(A);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = i; j < 4; ++j)
            EXPECT_NEAR(U(i, j), A(i, j), 1e-12)
                << "triu should preserve upper elements at (" << i << "," << j << ")";
}

TEST(LinalgTrilTriu, Triu_ZerosBelowDiag) {
    auto A = full4x4();
    auto U = triu(A);
    for (size_t i = 1; i < 4; ++i)
        for (size_t j = 0; j < i; ++j)
            EXPECT_NEAR(U(i, j), 0.0, 1e-12);
}

TEST(LinalgTrilTriu, Triu_OfUpperTri_Unchanged) {
    DMatrix A(3, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 1) = 4.0; A(1, 2) = 5.0;
    A(2, 2) = 6.0;
    auto U = triu(A);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(U(i, j), A(i, j), 1e-12);
}

TEST(LinalgTrilTriu, Triu_Shape_Preserved) {
    auto A = full4x4();
    auto U = triu(A);
    EXPECT_EQ(U.rows(), 4u);
    EXPECT_EQ(U.cols(), 4u);
}

TEST(LinalgTrilTriu, TrilPlusTriu_EqualsPlusOrigDiag) {
    // tril(A) + triu(A) = A + diag(A) (diagonal counted twice)
    DMatrix A(3, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
    auto L = tril(A);
    auto U = triu(A);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            double expected = A(i, j) + ((i == j) ? A(i, j) : 0.0);
            EXPECT_NEAR(L(i, j) + U(i, j), expected, 1e-12)
                << "tril + triu mismatch at (" << i << "," << j << ")";
        }
}

// ---------------------------------------------------------------------------
// diag: vector→matrix and matrix→vector
// ---------------------------------------------------------------------------

TEST(LinalgTrilTriu, Diag_VecToMatrix_Size) {
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto D = diag<double>(v);
    EXPECT_EQ(D.rows(), 4u);
    EXPECT_EQ(D.cols(), 4u);
}

TEST(LinalgTrilTriu, Diag_VecToMatrix_Values) {
    std::vector<double> v = {10.0, 20.0, 30.0};
    auto D = diag<double>(v);
    for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(D(i, i), v[i], 1e-14);
}

TEST(LinalgTrilTriu, Diag_MatToVec_ExtractsDiagonal) {
    DMatrix A(4, 4);
    for (size_t i = 0; i < 4; ++i) A(i, i) = static_cast<double>(i + 1) * 2.0;
    auto d = diag(A);
    ASSERT_EQ(d.size(), 4u);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(d[i], static_cast<double>(i + 1) * 2.0, 1e-14);
}

TEST(LinalgTrilTriu, Diag_Roundtrip) {
    // diag(diag(D)) = D for diagonal matrix
    std::vector<double> v = {3.0, 7.0, 11.0};
    auto D = diag<double>(v);
    auto v2 = diag(D);
    ASSERT_EQ(v2.size(), v.size());
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_NEAR(v2[i], v[i], 1e-14);
}

TEST(LinalgTrilTriu, Tril_Diag_EqualsDiagExtract) {
    // tril(diag(v)) diagonal should equal v
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto D = diag<double>(v);
    auto L = tril(D);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_NEAR(L(i, i), v[i], 1e-14);
}
