// MathScript: Advanced SVD Tests (Wave 47)
// Tests for SVD properties: U/V orthogonality, sigma ordering, Frobenius norm
// Note: SvdResult.S is a column vector of singular values (S(i,0)).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Frobenius norm
static double frob(const DMatrix& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

// Check ||M^T M - I||_F (column orthogonality)
static double ortho_col_error(const DMatrix& M) {
    size_t m = M.rows(), n = M.cols();
    double err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < m; ++k)
                dot += M(k, i) * M(k, j);
            double exp = (i == j) ? 1.0 : 0.0;
            err += (dot - exp) * (dot - exp);
        }
    }
    return std::sqrt(err);
}

// ---------------------------------------------------------------------------
// Basic: returns value, sizes, finiteness
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_Returns_Value_3x3) {
    DMatrix A({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {1.0, 1.0, 1.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->S.rows(), 0u);
}

TEST(LinalgSVDAdv, SVD_AllFinite_3x3) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->U.rows(); ++i)
        for (size_t j = 0; j < result->U.cols(); ++j)
            EXPECT_TRUE(std::isfinite(result->U(i, j)));
    for (size_t i = 0; i < result->S.rows(); ++i)
        EXPECT_TRUE(std::isfinite(result->S(i, 0)));
}

// ---------------------------------------------------------------------------
// Singular values non-negative and ordered
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_SingularValues_NonNegative_3x3) {
    DMatrix A({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {1.0, 1.0, 1.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->S.rows(); ++i)
        EXPECT_GE(result->S(i, 0), 0.0) << "sigma_" << i << " is negative";
}

TEST(LinalgSVDAdv, SVD_SingularValues_Ordered_3x3) {
    DMatrix A({{3.0, 1.0, 0.5}, {1.5, 2.0, 1.0}, {0.5, 1.0, 1.5}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    size_t k = result->S.rows();
    for (size_t i = 0; i + 1 < k; ++i)
        EXPECT_GE(result->S(i, 0), result->S(i+1, 0) - 1e-9)
            << "Singular values not ordered at i=" << i;
}

// ---------------------------------------------------------------------------
// Identity: all singular values should be 1
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_Identity_SingVals_AllOne) {
    DMatrix A({{1.0, 0.0}, {0.0, 1.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->S.rows(); ++i)
        EXPECT_NEAR(result->S(i, 0), 1.0, 1e-9);
}

TEST(LinalgSVDAdv, SVD_ScaledIdentity_SingVals_EqualScale) {
    DMatrix A({{5.0, 0.0}, {0.0, 5.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->S.rows(); ++i)
        EXPECT_NEAR(result->S(i, 0), 5.0, 1e-8);
}

// ---------------------------------------------------------------------------
// Rank-1 matrix: only one non-zero singular value
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_Rank1_OneNonZeroSingVal) {
    DMatrix A({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->S(0, 0), 1e-10);
    if (result->S.rows() > 1) EXPECT_NEAR(result->S(1, 0), 0.0, 1e-8);
    if (result->S.rows() > 2) EXPECT_NEAR(result->S(2, 0), 0.0, 1e-8);
}

// ---------------------------------------------------------------------------
// Frobenius norm: ||A||_F = ||sigma||_2
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_FrobNorm_Conservation_2x3) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    double frob_A = frob(A);
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    double sigma_norm = 0.0;
    for (size_t i = 0; i < result->S.rows(); ++i)
        sigma_norm += result->S(i, 0) * result->S(i, 0);
    EXPECT_NEAR(frob_A, std::sqrt(sigma_norm), 1e-7);
}

// ---------------------------------------------------------------------------
// Orthogonality of U and V for 2x2
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_U_Orthogonal_2x2) {
    DMatrix A({{2.0, 1.0}, {1.0, 3.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    double err = ortho_col_error(result->U);
    EXPECT_LT(err, 1e-8) << "U not orthogonal: err=" << err;
}

TEST(LinalgSVDAdv, SVD_V_Orthogonal_2x2) {
    DMatrix A({{2.0, 1.0}, {1.0, 3.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    double err = ortho_col_error(result->V);
    EXPECT_LT(err, 1e-8) << "V not orthogonal: err=" << err;
}

// ---------------------------------------------------------------------------
// 4x4 SVD
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_4x4_ReturnsValue) {
    DMatrix A({{4.0, 1.0, 0.0, 0.5},
               {1.0, 3.0, 1.0, 0.0},
               {0.0, 1.0, 2.0, 1.0},
               {0.5, 0.0, 1.0, 3.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->U.rows(), 4u);
    EXPECT_EQ(result->V.rows(), 4u);
}

TEST(LinalgSVDAdv, SVD_4x4_SingVals_Finite) {
    DMatrix A({{4.0, 1.0, 0.0, 0.5},
               {1.0, 3.0, 1.0, 0.0},
               {0.0, 1.0, 2.0, 1.0},
               {0.5, 0.0, 1.0, 3.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->S.rows(); ++i)
        EXPECT_TRUE(std::isfinite(result->S(i, 0)));
}

// ---------------------------------------------------------------------------
// Existing working cases to confirm compatibility
// ---------------------------------------------------------------------------

TEST(LinalgSVDAdv, SVD_2x2_Diagonal_SingVals) {
    // [[3,1],[1,2]]: known SVD
    DMatrix A({{3.0, 1.0}, {1.0, 2.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(0, 0), 3.618033988749895, 1e-6);
    EXPECT_NEAR(result->S(1, 0), 1.381966011250105, 1e-6);
}
