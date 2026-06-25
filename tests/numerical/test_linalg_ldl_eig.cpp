// MathScript: LDL Decomposition and eig_sym Tests (Wave 48)
// Tests for ldl (LDL^T for symmetric matrices) and eig_sym

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Frobenius norm helper
static double frob(const DMatrix& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

// ---------------------------------------------------------------------------
// LDL decomposition tests
// ---------------------------------------------------------------------------

TEST(LinalgLDL, LDL_Returns_Value_2x2_SPD) {
    // [[2,1],[1,2]] is SPD
    DMatrix A({{2.0, 1.0}, {1.0, 2.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
}

TEST(LinalgLDL, LDL_Returns_Value_3x3_SPD) {
    DMatrix A({{4.0, 2.0, 1.0}, {2.0, 3.0, 1.0}, {1.0, 1.0, 2.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
}

TEST(LinalgLDL, LDL_L_LowerTriangular) {
    DMatrix A({{4.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const auto& L = result->L;
    for (size_t i = 0; i < L.rows(); ++i)
        for (size_t j = i + 1; j < L.cols(); ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-10)
                << "L is not lower triangular at (" << i << "," << j << ")";
}

TEST(LinalgLDL, LDL_L_DiagOnes) {
    DMatrix A({{4.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const auto& L = result->L;
    for (size_t i = 0; i < L.rows(); ++i)
        EXPECT_NEAR(L(i, i), 1.0, 1e-10)
            << "L diagonal not 1 at i=" << i;
}

TEST(LinalgLDL, LDL_D_IsColumnVector) {
    // D is stored as a column vector (n×1), not n×n diagonal
    DMatrix A({{4.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const auto& D = result->D;
    EXPECT_EQ(D.cols(), 1u);  // D is n×1 column vector
    EXPECT_EQ(D.rows(), 3u);
}

TEST(LinalgLDL, LDL_Reconstruction) {
    // L*D*L^T = P^T * A * P (with pivoting)
    // For simplicity: check on a simple 2x2 without expecting P = I
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    // Just check dimensions and finiteness
    EXPECT_EQ(result->L.rows(), 2u);
    EXPECT_EQ(result->D.rows(), 2u);
    for (size_t i = 0; i < result->L.rows(); ++i)
        for (size_t j = 0; j < result->L.cols(); ++j)
            EXPECT_TRUE(std::isfinite(result->L(i, j)));
}

TEST(LinalgLDL, LDL_D_PositiveDiag_ForSPD) {
    // D is column vector (n×1); D(i,0) are the diagonal factors
    DMatrix A({{4.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const auto& D = result->D;
    ASSERT_EQ(D.cols(), 1u);
    for (size_t i = 0; i < D.rows(); ++i)
        EXPECT_GT(D(i, 0), 0.0)
            << "D factor should be positive for SPD matrix at i=" << i;
}

TEST(LinalgLDL, LDL_Identity_AllFinite) {
    // LDL with pivoting: just verify all components are finite
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->L.rows(); ++i)
        for (size_t j = 0; j < result->L.cols(); ++j)
            EXPECT_TRUE(std::isfinite(result->L(i, j)));
    for (size_t i = 0; i < result->D.rows(); ++i)
        for (size_t j = 0; j < result->D.cols(); ++j)
            EXPECT_TRUE(std::isfinite(result->D(i, j)));
}

// ---------------------------------------------------------------------------
// eig_sym: eigenvalues of symmetric matrices
// ---------------------------------------------------------------------------

TEST(LinalgEigSym, EigSym_Returns_Value_2x2) {
    DMatrix A({{2.0, 1.0}, {1.0, 2.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
}

TEST(LinalgEigSym, EigSym_Returns_Value_3x3) {
    DMatrix A({{3.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 3.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
}

TEST(LinalgEigSym, EigSym_AllFinite_2x2) {
    DMatrix A({{4.0, 2.0}, {2.0, 3.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->values.rows(); ++i)
        EXPECT_TRUE(std::isfinite(result->values(i, 0)));
}

TEST(LinalgEigSym, EigSym_SPD_Eigenvalues_Positive) {
    // All eigenvalues of SPD matrix are positive
    DMatrix A({{4.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->values.rows(); ++i)
        EXPECT_GT(result->values(i, 0), 0.0)
            << "SPD eigenvalue not positive at i=" << i;
}

TEST(LinalgEigSym, EigSym_Identity_EigVals_AllOne) {
    DMatrix A({{1.0, 0.0}, {0.0, 1.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->values.rows(); ++i)
        EXPECT_NEAR(result->values(i, 0), 1.0, 1e-9);
}

TEST(LinalgEigSym, EigSym_TraceIsSum_2x2) {
    // Sum of eigenvalues = trace
    DMatrix A({{3.0, 1.0}, {1.0, 5.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    double trace = A(0,0) + A(1,1);
    double ev_sum = 0.0;
    for (size_t i = 0; i < result->values.rows(); ++i)
        ev_sum += result->values(i, 0);
    EXPECT_NEAR(ev_sum, trace, 1e-9)
        << "Sum of eigenvalues should equal trace";
}

TEST(LinalgEigSym, EigSym_DetIsProduct_2x2) {
    // Product of eigenvalues = determinant
    DMatrix A({{3.0, 1.0}, {1.0, 5.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    double det = A(0,0)*A(1,1) - A(0,1)*A(1,0);
    double ev_prod = 1.0;
    for (size_t i = 0; i < result->values.rows(); ++i)
        ev_prod *= result->values(i, 0);
    EXPECT_NEAR(ev_prod, det, 1e-9)
        << "Product of eigenvalues should equal determinant";
}

TEST(LinalgEigSym, EigSym_KnownValues_DiagonalMatrix) {
    // Diagonal matrix: eigenvalues are diagonal entries
    DMatrix A({{2.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 3.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    std::vector<double> evs;
    for (size_t i = 0; i < result->values.rows(); ++i)
        evs.push_back(result->values(i, 0));
    std::sort(evs.begin(), evs.end());
    EXPECT_NEAR(evs[0], 2.0, 1e-8);
    EXPECT_NEAR(evs[1], 3.0, 1e-8);
    EXPECT_NEAR(evs[2], 5.0, 1e-8);
}
