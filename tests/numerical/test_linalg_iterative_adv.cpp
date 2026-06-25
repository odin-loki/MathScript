// MathScript: Advanced Iterative Linear Solver Tests (Wave 45)
// Tests for CG, BiCGSTAB, GMRES on various SPD and non-symmetric systems

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Helper: compute residual norm ||Ax - b||
static double residual_norm(const DMatrix& A, const DMatrix& x, const DMatrix& b) {
    double norm2 = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < A.cols(); ++j)
            row_sum += A(i, j) * x(j, 0);
        double r = row_sum - b(i, 0);
        norm2 += r * r;
    }
    return std::sqrt(norm2);
}

// ---------------------------------------------------------------------------
// Conjugate Gradient (CG) — for SPD systems
// ---------------------------------------------------------------------------

TEST(LinalgIterative, CG_2x2_SPD_System) {
    // A = [[4, 1], [1, 3]], b = [1, 2] — SPD system
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});
    auto result = cg(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    EXPECT_EQ(x.rows(), 2u);
    double res = residual_norm(A, x, b);
    EXPECT_LT(res, 1e-8) << "CG residual too large: " << res;
}

TEST(LinalgIterative, CG_Identity_System) {
    // A = I, b = [3, 5, 7] => x = b exactly
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    DMatrix b({{3.0}, {5.0}, {7.0}});
    auto result = cg(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    EXPECT_NEAR(x(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(x(1, 0), 5.0, 1e-8);
    EXPECT_NEAR(x(2, 0), 7.0, 1e-8);
}

TEST(LinalgIterative, CG_Diagonal_SPD) {
    // A = diag(2, 4, 6), b = [2, 4, 6] => x = [1, 1, 1]
    DMatrix A({{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 6.0}});
    DMatrix b({{2.0}, {4.0}, {6.0}});
    auto result = cg(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    EXPECT_NEAR(x(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(x(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(x(2, 0), 1.0, 1e-8);
}

TEST(LinalgIterative, CG_4x4_SPD_System) {
    // Construct a small 4x4 SPD matrix: A = B^T * B + 4*I
    DMatrix A({{5.0, 1.0, 0.0, 0.5},
               {1.0, 6.0, 1.0, 0.0},
               {0.0, 1.0, 5.0, 1.0},
               {0.5, 0.0, 1.0, 4.0}});
    DMatrix b({{1.0}, {2.0}, {3.0}, {4.0}});
    auto result = cg(A, b, 1000, 1e-10);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    double res = residual_norm(A, x, b);
    EXPECT_LT(res, 1e-7) << "CG 4x4 residual too large: " << res;
}

TEST(LinalgIterative, CG_Solution_AllFinite) {
    DMatrix A({{3.0, 1.0}, {1.0, 2.0}});
    DMatrix b({{5.0}, {5.0}});
    auto result = cg(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < x.rows(); ++i)
        EXPECT_TRUE(std::isfinite(x(i, 0)));
}

// ---------------------------------------------------------------------------
// BiCGSTAB — for general (possibly non-symmetric) systems
// ---------------------------------------------------------------------------

TEST(LinalgIterative, BiCGSTAB_Identity_System) {
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    DMatrix b({{2.0}, {3.0}, {4.0}});
    auto result = bicgstab(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    EXPECT_NEAR(x(0, 0), 2.0, 1e-8);
    EXPECT_NEAR(x(1, 0), 3.0, 1e-8);
    EXPECT_NEAR(x(2, 0), 4.0, 1e-8);
}

TEST(LinalgIterative, BiCGSTAB_2x2_General) {
    // A = [[2, 1], [1, 3]], b = [5, 10] => x = [1, 3]
    DMatrix A({{2.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{5.0}, {10.0}});
    auto result = bicgstab(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    double res = residual_norm(A, x, b);
    EXPECT_LT(res, 1e-7) << "BiCGSTAB residual too large: " << res;
}

TEST(LinalgIterative, BiCGSTAB_Diagonal) {
    DMatrix A({{3.0, 0.0, 0.0}, {0.0, 6.0, 0.0}, {0.0, 0.0, 9.0}});
    DMatrix b({{6.0}, {6.0}, {9.0}});
    auto result = bicgstab(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    EXPECT_NEAR(x(0, 0), 2.0, 1e-7);
    EXPECT_NEAR(x(1, 0), 1.0, 1e-7);
    EXPECT_NEAR(x(2, 0), 1.0, 1e-7);
}

TEST(LinalgIterative, BiCGSTAB_AllFinite) {
    DMatrix A({{4.0, 1.0}, {2.0, 3.0}});
    DMatrix b({{1.0}, {1.0}});
    auto result = bicgstab(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < x.rows(); ++i)
        EXPECT_TRUE(std::isfinite(x(i, 0)));
}

// ---------------------------------------------------------------------------
// GMRES — for general non-symmetric systems
// ---------------------------------------------------------------------------

TEST(LinalgIterative, GMRES_Identity_System) {
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    DMatrix b({{7.0}, {8.0}, {9.0}});
    auto result = gmres(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    EXPECT_NEAR(x(0, 0), 7.0, 1e-7);
    EXPECT_NEAR(x(1, 0), 8.0, 1e-7);
    EXPECT_NEAR(x(2, 0), 9.0, 1e-7);
}

TEST(LinalgIterative, GMRES_2x2_System) {
    DMatrix A({{3.0, 1.0}, {1.0, 2.0}});
    DMatrix b({{5.0}, {5.0}});
    auto result = gmres(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    double res = residual_norm(A, x, b);
    EXPECT_LT(res, 1e-7) << "GMRES residual too large: " << res;
}

TEST(LinalgIterative, GMRES_3x3_System) {
    DMatrix A({{4.0, 1.0, 0.0},
               {1.0, 4.0, 1.0},
               {0.0, 1.0, 4.0}});
    DMatrix b({{1.0}, {2.0}, {3.0}});
    auto result = gmres(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    double res = residual_norm(A, x, b);
    EXPECT_LT(res, 1e-7) << "GMRES 3x3 residual too large: " << res;
}

TEST(LinalgIterative, GMRES_AllFinite) {
    DMatrix A({{5.0, 2.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 2.0, 5.0}});
    DMatrix b({{1.0}, {0.0}, {1.0}});
    auto result = gmres(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < x.rows(); ++i)
        EXPECT_TRUE(std::isfinite(x(i, 0)));
}

// ---------------------------------------------------------------------------
// Comparison: direct solve vs iterative
// ---------------------------------------------------------------------------

TEST(LinalgIterative, CG_MatchesDirect_3x3) {
    DMatrix A({{4.0, 1.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 1.0, 4.0}});
    DMatrix b({{1.0}, {2.0}, {3.0}});
    auto direct = solve(A, b);
    auto iter = cg(A, b, 1000, 1e-12);
    ASSERT_TRUE(direct.has_value());
    ASSERT_TRUE(iter.has_value());
    auto& xd = direct.value();
    auto& xi = iter.value();
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(xd(i, 0), xi(i, 0), 1e-7)
            << "CG/direct mismatch at row " << i;
}

TEST(LinalgIterative, GMRES_MatchesDirect_3x3) {
    DMatrix A({{3.0, 1.0, 0.0}, {1.0, 5.0, 1.0}, {0.0, 1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}, {1.0}});
    auto direct = solve(A, b);
    auto iter = gmres(A, b);
    ASSERT_TRUE(direct.has_value());
    ASSERT_TRUE(iter.has_value());
    auto& xd = direct.value();
    auto& xi = iter.value();
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(xd(i, 0), xi(i, 0), 1e-6)
            << "GMRES/direct mismatch at row " << i;
}
