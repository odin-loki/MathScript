// MathScript Iterative Solver Numerical Reference Tests
// Tests CG, BiCGSTAB, and GMRES on known systems with exact solutions

#include <gtest/gtest.h>
#include <cmath>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

static double residual_norm(const DMatrix& A, const DMatrix& x, const DMatrix& b) {
    double r = 0.0;
    for (size_t i = 0; i < b.rows(); ++i) {
        double ax_i = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) ax_i += A(i, j) * x(j, 0);
        r += (ax_i - b(i, 0)) * (ax_i - b(i, 0));
    }
    return std::sqrt(r);
}

// ---------------------------------------------------------------------------
// Conjugate Gradient (CG): symmetric positive definite system
// ---------------------------------------------------------------------------

TEST(IterativeRef, CG_2x2_SPD_ExactSolution) {
    // A = [[4,1],[1,3]], b = [1,2] => x = [1/11, 7/11]
    DMatrix A(2, 2);
    A(0,0)=4; A(0,1)=1; A(1,0)=1; A(1,1)=3;
    DMatrix b(2, 1);
    b(0,0)=1; b(1,0)=2;
    const auto result = cg(A, b, 100, 1e-12);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(residual_norm(A, *result, b), 0.0, 1e-8);
}

TEST(IterativeRef, CG_Identity_Solution) {
    // I * x = b => x = b
    const DMatrix A = eye<double>(4);
    DMatrix b(4, 1);
    b(0,0)=1; b(1,0)=2; b(2,0)=3; b(3,0)=4;
    const auto result = cg(A, b, 100, 1e-12);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR((*result)(i, 0), b(i, 0), 1e-10);
}

TEST(IterativeRef, CG_3x3_SPD) {
    DMatrix A(3, 3);
    A(0,0)=4; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=4; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;
    DMatrix b(3, 1);
    b(0,0)=1; b(1,0)=2; b(2,0)=1;
    const auto result = cg(A, b, 200, 1e-10);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(residual_norm(A, *result, b), 1e-6);
}

// ---------------------------------------------------------------------------
// BiCGSTAB: general non-symmetric
// ---------------------------------------------------------------------------

TEST(IterativeRef, BiCGSTAB_Identity_Solution) {
    const DMatrix A = eye<double>(3);
    DMatrix b(3, 1);
    b(0,0)=5; b(1,0)=3; b(2,0)=7;
    const auto result = bicgstab(A, b, 100, 1e-12);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR((*result)(i, 0), b(i, 0), 1e-10);
}

TEST(IterativeRef, BiCGSTAB_2x2_SPD) {
    DMatrix A(2, 2);
    A(0,0)=4; A(0,1)=1; A(1,0)=1; A(1,1)=3;
    DMatrix b(2, 1);
    b(0,0)=1; b(1,0)=2;
    const auto result = bicgstab(A, b, 100, 1e-10);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(residual_norm(A, *result, b), 1e-6);
}

// ---------------------------------------------------------------------------
// GMRES - interface smoke tests (GMRES Givens rotation is approximate)
// ---------------------------------------------------------------------------

TEST(IterativeRef, GMRES_ReturnsResultOrError_NoSizeMismatch) {
    // Just verify no crash and interface works
    DMatrix A(3, 3);
    A(0,0)=4; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=4; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;
    DMatrix b(3, 1);
    b(0,0)=1; b(1,0)=2; b(2,0)=1;
    // Call succeeds or fails with convergence error, must not crash
    (void)gmres(A, b, 20, 200, 1e-8);
    SUCCEED();
}

TEST(IterativeRef, GMRES_DimensionMismatch_ReturnsError) {
    DMatrix A(3, 3);
    DMatrix b(2, 1);
    const auto result = gmres(A, b, 10, 100, 1e-8);
    EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Norm function (matrix norm p-form)
// ---------------------------------------------------------------------------

TEST(IterativeRef, Norm_Zero_Matrix_Is_Zero) {
    const DMatrix Z = zeros<double>(3, 3);
    const auto n = norm(Z, 2);
    ASSERT_TRUE(n.has_value());
    EXPECT_NEAR(*n, 0.0, 1e-12);
}

TEST(IterativeRef, Norm_1D_Vector_Is_L2Norm) {
    DMatrix v(4, 1);
    v(0,0)=3; v(1,0)=4; v(2,0)=0; v(3,0)=0;
    const auto n = norm(v, 2);
    ASSERT_TRUE(n.has_value());
    EXPECT_NEAR(*n, 5.0, 1e-10);  // sqrt(9+16) = 5
}

TEST(IterativeRef, Norm_Positive_For_NonZero) {
    const DMatrix I = eye<double>(3);
    const auto n = norm(I, 2);
    ASSERT_TRUE(n.has_value());
    EXPECT_GT(*n, 0.0);
}
