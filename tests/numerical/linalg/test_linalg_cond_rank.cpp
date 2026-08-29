// MathScript: Condition Number and Matrix Rank Tests (Wave 49, Wave 217)
// Tests for cond(), rank(), and matrix_rank() functions in ms::linalg

#include <gtest/gtest.h>
#include <cmath>
#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// rank() tests
// ---------------------------------------------------------------------------

TEST(LinalgCondRank, Rank_Identity_3x3_IsN) {
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 3.0, 0.5);
}

TEST(LinalgCondRank, Rank_Rank1_Matrix) {
    // Rank-1: all rows are multiples of first row
    DMatrix A({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}});
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 1.0, 0.5);
}

TEST(LinalgCondRank, Rank_Rank2_Matrix) {
    // Two independent rows, third is linear combination
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}});
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 2.0, 0.5);
}

TEST(LinalgCondRank, Rank_FullRank_2x2) {
    DMatrix A({{3.0, 1.0}, {1.0, 2.0}});
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 2.0, 0.5);
}

TEST(LinalgCondRank, Rank_ZeroMatrix_IsZero) {
    DMatrix A(3, 3, 0.0);
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 0.5);
}

TEST(LinalgCondRank, Rank_IsNonNegative) {
    for (auto& A_init : std::vector<DMatrix>{
        DMatrix({{1.0, 2.0}, {3.0, 4.0}}),
        DMatrix({{1.0, 0.0}, {0.0, 0.0}}),
        DMatrix({{5.0, 3.0, 1.0}, {1.0, 2.0, 3.0}, {4.0, 6.0, 7.0}})
    }) {
        auto r = rank(A_init);
        if (r.has_value()) EXPECT_GE(*r, 0.0);
    }
}

// ---------------------------------------------------------------------------
// matrix_rank() tests (Wave 217)
// ---------------------------------------------------------------------------

TEST(LinalgCondRank, MatrixRank_Identity_3x3_IsN) {
    DMatrix A = eye<double>(3);
    EXPECT_EQ(matrix_rank(A), 3);
}

TEST(LinalgCondRank, MatrixRank_Identity_5x5_IsN) {
    DMatrix A = eye<double>(5);
    EXPECT_EQ(matrix_rank(A), 5);
}

TEST(LinalgCondRank, MatrixRank_ZeroMatrix_IsZero) {
    DMatrix A(4, 4, 0.0);
    EXPECT_EQ(matrix_rank(A), 0);
}

TEST(LinalgCondRank, MatrixRank_Rank1_Matrix) {
    DMatrix A({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}});
    EXPECT_EQ(matrix_rank(A), 1);
}

TEST(LinalgCondRank, MatrixRank_Rank2_Matrix) {
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}});
    EXPECT_EQ(matrix_rank(A), 2);
}

TEST(LinalgCondRank, MatrixRank_FullRank_2x2) {
    DMatrix A({{3.0, 1.0}, {1.0, 2.0}});
    EXPECT_EQ(matrix_rank(A), 2);
}

TEST(LinalgCondRank, MatrixRank_Tall_RankDeficient) {
    DMatrix A({{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}});
    EXPECT_EQ(matrix_rank(A), 1);
}

TEST(LinalgCondRank, MatrixRank_Wide_RankDeficient) {
    DMatrix A({{1.0, 2.0, 3.0, 4.0}, {2.0, 4.0, 6.0, 8.0}});
    EXPECT_EQ(matrix_rank(A), 1);
}

TEST(LinalgCondRank, MatrixRank_CustomTolerance) {
    DMatrix A({{1000.0, 0.0}, {0.0, 1e-12}});
    EXPECT_EQ(matrix_rank(A, 1e-10), 1);
    EXPECT_EQ(matrix_rank(A, 1e-14), 2);
}

TEST(LinalgCondRank, MatrixRank_MatchesRankForWellConditioned) {
    DMatrix A({{2.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(matrix_rank(A), static_cast<int>(*r));
}

// ---------------------------------------------------------------------------
// cond() tests
// ---------------------------------------------------------------------------

TEST(LinalgCondRank, Cond_Identity_IsOne) {
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    auto c = cond(A);
    ASSERT_TRUE(c.has_value());
    EXPECT_NEAR(*c, 1.0, 1e-6);
}

TEST(LinalgCondRank, Cond_ScaledIdentity_IsOne) {
    // cond(alpha * I) = 1 (scale cancels)
    DMatrix A({{5.0, 0.0}, {0.0, 5.0}});
    auto c = cond(A);
    ASSERT_TRUE(c.has_value());
    EXPECT_NEAR(*c, 1.0, 1e-6);
}

TEST(LinalgCondRank, Cond_IsAtLeastOne) {
    for (auto& A_init : std::vector<DMatrix>{
        DMatrix({{2.0, 1.0}, {1.0, 3.0}}),
        DMatrix({{4.0, 0.0}, {0.0, 2.0}}),
        DMatrix({{3.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}})
    }) {
        auto c = cond(A_init);
        if (c.has_value()) EXPECT_GE(*c, 1.0 - 1e-9);
    }
}

TEST(LinalgCondRank, Cond_WellConditioned_IsSmall) {
    // Well-conditioned matrix: small condition number
    DMatrix A({{3.0, 0.0}, {0.0, 2.0}});  // cond = 3/2
    auto c = cond(A);
    ASSERT_TRUE(c.has_value());
    EXPECT_LT(*c, 10.0) << "Well-conditioned matrix should have small cond";
    EXPECT_GT(*c, 0.0);
}

TEST(LinalgCondRank, Cond_IllConditioned_IsLarge) {
    // Ill-conditioned: large ratio of singular values
    DMatrix A({{1000.0, 0.0}, {0.0, 0.001}});  // cond = 1e6
    auto c = cond(A);
    ASSERT_TRUE(c.has_value());
    EXPECT_GT(*c, 1e4) << "Ill-conditioned matrix should have large cond";
}

TEST(LinalgCondRank, Cond_Finite) {
    DMatrix A({{2.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}});
    auto c = cond(A);
    ASSERT_TRUE(c.has_value());
    EXPECT_TRUE(std::isfinite(*c));
}

TEST(LinalgCondRank, Cond_Diagonal_IsRatioOfExtreme) {
    // For diagonal matrix, cond = max(|d_i|) / min(|d_i|)
    DMatrix A({{4.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 1.0}});
    auto c = cond(A);
    ASSERT_TRUE(c.has_value());
    // cond = 4/1 = 4
    EXPECT_NEAR(*c, 4.0, 0.1);
}
