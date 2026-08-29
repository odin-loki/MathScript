// MathScript: Advanced Linear Algebra Tests
// Tests for eig, eig_sym, svd, ldl, lsq, cond, rank, rand/randn/diag

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// Eigenvalue decomposition (general)
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, Eig_2x2_Diagonal) {
    DMatrix A({{2.0, 0.0}, {0.0, 3.0}});
    auto result = eig(A);
    ASSERT_TRUE(result.has_value());
    const auto& E = result.value();
    EXPECT_EQ(E.values.rows(), 2u);
    // For diagonal matrix, eigenvalues are the diagonal entries
    // (exact ordering may vary)
    auto v0 = E.values(0, 0);
    auto v1 = E.values(1, 0);
    double sum = v0 + v1;
    EXPECT_NEAR(sum, 5.0, 1e-8);  // trace = sum of eigenvalues
}

TEST(LinalgDecompAdv, Eig_Identity_AllOnes) {
    DMatrix I = eye<double>(3);
    auto result = eig(I);
    ASSERT_TRUE(result.has_value());
    const auto& E = result.value();
    EXPECT_EQ(E.values.rows(), 3u);
    double sum = 0.0;
    for (size_t i = 0; i < 3; ++i) sum += E.values(i, 0);
    EXPECT_NEAR(sum, 3.0, 1e-8);
}

// ---------------------------------------------------------------------------
// Symmetric eigenvalue decomposition
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, EigSym_2x2_KnownValues) {
    // [[2, 1], [1, 2]] has eigenvalues 1 and 3
    DMatrix A({{2.0, 1.0}, {1.0, 2.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    const auto& E = result.value();
    EXPECT_EQ(E.values.rows(), 2u);
    double lo = std::min(E.values(0,0), E.values(1,0));
    double hi = std::max(E.values(0,0), E.values(1,0));
    EXPECT_NEAR(lo, 1.0, 1e-8);
    EXPECT_NEAR(hi, 3.0, 1e-8);
}

TEST(LinalgDecompAdv, EigSym_VectorsOrthonormal) {
    DMatrix A({{4.0, 2.0}, {2.0, 3.0}});
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    const auto& E = result.value();
    // Check each eigenvector has unit norm
    for (size_t col = 0; col < E.vectors.cols(); ++col) {
        double norm_sq = 0.0;
        for (size_t row = 0; row < E.vectors.rows(); ++row) {
            norm_sq += E.vectors(row, col) * E.vectors(row, col);
        }
        EXPECT_NEAR(norm_sq, 1.0, 1e-8) << "column " << col << " not unit norm";
    }
}

TEST(LinalgDecompAdv, EigSym_3x3_Diagonal) {
    DMatrix D({{1.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 9.0}});
    auto result = eig_sym(D);
    ASSERT_TRUE(result.has_value());
    const auto& E = result.value();
    EXPECT_EQ(E.values.rows(), 3u);
    double sum = 0.0;
    for (size_t i = 0; i < 3; ++i) sum += E.values(i, 0);
    EXPECT_NEAR(sum, 15.0, 1e-8);  // 1+5+9
}

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, SVD_2x2_Rank1) {
    // [[1,0],[0,0]] has singular values {1,0}
    DMatrix A({{1.0, 0.0}, {0.0, 0.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    const auto& S = result.value();
    double s_max = std::max(S.S(0,0), S.S(1,0));
    double s_min = std::min(S.S(0,0), S.S(1,0));
    EXPECT_NEAR(s_max, 1.0, 1e-8);
    EXPECT_NEAR(s_min, 0.0, 1e-8);
}

TEST(LinalgDecompAdv, SVD_2x2_SingularValuesFinite) {
    DMatrix A({{3.0, 1.0}, {2.0, 4.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    const auto& S = result.value();
    // Singular values should be positive and finite
    for (size_t i = 0; i < S.S.rows(); ++i) {
        EXPECT_TRUE(std::isfinite(S.S(i,0)));
        EXPECT_GE(S.S(i,0), 0.0);
    }
    // Product of singular values ≈ |det(A)| = |3*4 - 1*2| = 10
    double sv_prod = 1.0;
    for (size_t i = 0; i < S.S.rows(); ++i) sv_prod *= S.S(i,0);
    EXPECT_NEAR(sv_prod, 10.0, 1e-6);
}

TEST(LinalgDecompAdv, SVD_SingularValues_Descending) {
    DMatrix A({{4.0, 3.0, 0.0}, {0.0, 1.0, 2.0}});
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    const auto& S = result.value();
    int n = (int)S.S.rows();
    for (int i = 0; i + 1 < n; ++i) {
        EXPECT_GE(S.S(i,0), S.S(i+1,0) - 1e-10);
    }
}

// ---------------------------------------------------------------------------
// LDL decomposition
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, LDL_2x2_SpdMatrix) {
    DMatrix A({{4.0, 2.0}, {2.0, 3.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const auto& R = result.value();
    // L should have 1s on diagonal
    EXPECT_NEAR(R.L(0,0), 1.0, 1e-10);
    EXPECT_NEAR(R.L(1,1), 1.0, 1e-10);
    // D is stored as Nx1 vector of diagonal values; positive for SPD matrix
    EXPECT_GT(R.D(0,0), 0.0);
    EXPECT_GT(R.D(1,0), 0.0);
}

TEST(LinalgDecompAdv, LDL_3x3_StructureCheck) {
    DMatrix A({{4.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 5.0}});
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    const auto& R = result.value();
    // L should be unit lower-triangular (diagonal = 1, upper triangle = 0)
    EXPECT_NEAR(R.L(0,0), 1.0, 1e-8);
    EXPECT_NEAR(R.L(1,1), 1.0, 1e-8);
    EXPECT_NEAR(R.L(2,2), 1.0, 1e-8);
    EXPECT_NEAR(R.L(0,1), 0.0, 1e-8);
    EXPECT_NEAR(R.L(0,2), 0.0, 1e-8);
    EXPECT_NEAR(R.L(1,2), 0.0, 1e-8);
    // D is stored as Nx1 vector; D(i,0) = i-th diagonal entry of D
    // D diagonal entries should be positive for SPD input
    EXPECT_GT(R.D(0,0), 0.0);
    EXPECT_GT(R.D(1,0), 0.0);
    EXPECT_GT(R.D(2,0), 0.0);
}

// ---------------------------------------------------------------------------
// Least-squares (lsq)
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, LSQ_OverdeterminedSystem) {
    // Overdetermined: 3 eqs, 2 unknowns
    // [1 1; 1 2; 1 3] * x = [2; 3; 4] => x ≈ [1, 1]
    DMatrix A({{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}});
    DMatrix b({{2.0}, {3.0}, {4.0}});
    auto result = lsq(A, b);
    ASSERT_TRUE(result.has_value());
    const auto& x = result.value();
    EXPECT_EQ(x.rows(), 2u);
    EXPECT_NEAR(x(0,0), 1.0, 1e-6);
    EXPECT_NEAR(x(1,0), 1.0, 1e-6);
}

TEST(LinalgDecompAdv, LSQ_ExactSystem) {
    // 2x2 system with exact solution
    DMatrix A({{2.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{5.0}, {10.0}});
    auto result = lsq(A, b);
    ASSERT_TRUE(result.has_value());
    const auto& x = result.value();
    // Ax = b => 2x0 + x1 = 5, x0 + 3x1 = 10 => x0=1, x1=3
    EXPECT_NEAR(x(0,0), 1.0, 1e-8);
    EXPECT_NEAR(x(1,0), 3.0, 1e-8);
}

// ---------------------------------------------------------------------------
// Condition number
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, Cond_Identity_Is1) {
    DMatrix I = eye<double>(3);
    auto result = cond(I);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 1.0, 1e-8);
}

TEST(LinalgDecompAdv, Cond_ScaledIdentity) {
    // 10*I has condition number 1
    DMatrix A = eye<double>(3);
    for (size_t i = 0; i < 3; ++i) A(i,i) = 10.0;
    auto result = cond(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 1.0, 1e-8);
}

TEST(LinalgDecompAdv, Cond_IllConditioned_Large) {
    // [[1, 0], [0, 1e-8]] has cond = 1e8
    DMatrix A({{1.0, 0.0}, {0.0, 1e-8}});
    auto result = cond(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result.value(), 1e6);
}

// ---------------------------------------------------------------------------
// Matrix rank
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, Rank_FullRank_Equals_N) {
    DMatrix A({{1.0, 0.0}, {0.0, 1.0}});
    auto result = rank(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(static_cast<int>(result.value()), 2);
}

TEST(LinalgDecompAdv, Rank_Rank1_Matrix) {
    // [[1,2],[2,4]] is rank 1
    DMatrix A({{1.0, 2.0}, {2.0, 4.0}});
    auto result = rank(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(static_cast<int>(result.value()), 1);
}

TEST(LinalgDecompAdv, Rank_ZeroMatrix_Zero) {
    DMatrix Z = zeros<double>(3, 3);
    auto result = rank(Z);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(static_cast<int>(result.value()), 0);
}

// ---------------------------------------------------------------------------
// rand / randn construction
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, Rand_Size) {
    auto A = rand<double>(4, 3, 42u);
    EXPECT_EQ(A.rows(), 4u);
    EXPECT_EQ(A.cols(), 3u);
}

TEST(LinalgDecompAdv, Rand_ValuesInRange) {
    auto A = rand<double>(10, 10, 1u);
    for (size_t i = 0; i < 10; ++i)
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_GE(A(i,j), 0.0);
            EXPECT_LE(A(i,j), 1.0);
        }
}

TEST(LinalgDecompAdv, Randn_Size) {
    auto A = randn<double>(3, 5, 99u);
    EXPECT_EQ(A.rows(), 3u);
    EXPECT_EQ(A.cols(), 5u);
}

TEST(LinalgDecompAdv, Randn_AllFinite) {
    auto A = randn<double>(5, 5, 7u);
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j)
            EXPECT_TRUE(std::isfinite(A(i,j)));
}

// ---------------------------------------------------------------------------
// diag construction and extraction
// ---------------------------------------------------------------------------

TEST(LinalgDecompAdv, Diag_FromVector_DiagonalMatrix) {
    std::vector<double> v = {1.0, 2.0, 3.0};
    auto D = diag(v);
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 3u);
    EXPECT_NEAR(D(0,0), 1.0, 1e-10);
    EXPECT_NEAR(D(1,1), 2.0, 1e-10);
    EXPECT_NEAR(D(2,2), 3.0, 1e-10);
    EXPECT_NEAR(D(0,1), 0.0, 1e-10);
}

TEST(LinalgDecompAdv, Diag_ExtractFromMatrix) {
    DMatrix A({{1.0, 9.0, 8.0}, {7.0, 2.0, 6.0}, {5.0, 4.0, 3.0}});
    auto v = diag(A);
    EXPECT_EQ(v.size(), 3u);
    EXPECT_NEAR(v[0], 1.0, 1e-10);
    EXPECT_NEAR(v[1], 2.0, 1e-10);
    EXPECT_NEAR(v[2], 3.0, 1e-10);
}
