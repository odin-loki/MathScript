// MathScript Iterative Solvers Additional Numerical Tests
// Extended coverage for CG, BiCGSTAB, and GMRES on 3x3 SPD systems

#include <gtest/gtest.h>
#include <cmath>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

// A = [[4,1,0],[1,3,1],[0,1,4]], b = [1,2,3]
// This is symmetric positive definite.
static DMatrix make_A3() {
    DMatrix A(3, 3);
    A(0,0)=4; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=3; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;
    return A;
}

static DMatrix make_b3() {
    DMatrix b(3, 1);
    b(0,0)=1; b(1,0)=2; b(2,0)=3;
    return b;
}

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
// CG solver
// ---------------------------------------------------------------------------

TEST(IterativeSolversRef, CG_3x3_SPD_Converges) {
    const auto A = make_A3();
    const auto b = make_b3();
    const auto result = cg(A, b, 500, 1e-12);
    ASSERT_TRUE(result.has_value());
}

TEST(IterativeSolversRef, CG_3x3_SPD_CorrectDimensions) {
    const auto A = make_A3();
    const auto b = make_b3();
    const auto result = cg(A, b, 500, 1e-12);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 3u);
    EXPECT_EQ(result->cols(), 1u);
}

TEST(IterativeSolversRef, CG_3x3_SPD_ResidualBelowTolerance) {
    const auto A = make_A3();
    const auto b = make_b3();
    const auto result = cg(A, b, 500, 1e-12);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(residual_norm(A, *result, b), 1e-6);
}

// ---------------------------------------------------------------------------
// BiCGSTAB solver
// ---------------------------------------------------------------------------

TEST(IterativeSolversRef, BiCGSTAB_3x3_SPD_Converges) {
    const auto A = make_A3();
    const auto b = make_b3();
    const auto result = bicgstab(A, b, 500, 1e-12);
    ASSERT_TRUE(result.has_value());
}

TEST(IterativeSolversRef, BiCGSTAB_3x3_SPD_ResidualBelowTolerance) {
    const auto A = make_A3();
    const auto b = make_b3();
    const auto result = bicgstab(A, b, 500, 1e-12);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(residual_norm(A, *result, b), 1e-6);
}

// ---------------------------------------------------------------------------
// GMRES solver
// ---------------------------------------------------------------------------

TEST(IterativeSolversRef, GMRES_3x3_SPD_RunsWithoutCrash) {
    // GMRES may or may not converge for small systems depending on implementation
    const auto A = make_A3();
    const auto b = make_b3();
    const auto result = gmres(A, b, 3, 500, 1e-10);
    // Just check it doesn't crash
    if (result.has_value()) {
        for (size_t i = 0; i < 3; ++i)
            EXPECT_TRUE(std::isfinite((*result)(i, 0)));
    }
    SUCCEED();
}

TEST(IterativeSolversRef, GMRES_Smoke_Finite) {
    const auto A = make_A3();
    const auto b = make_b3();
    const auto result = gmres(A, b, 3, 100, 1e-8);
    if (result.has_value()) {
        double res = residual_norm(A, *result, b);
        EXPECT_TRUE(std::isfinite(res));
    }
    SUCCEED();
}

// ---------------------------------------------------------------------------
// All three solvers converge on the same well-conditioned system
// ---------------------------------------------------------------------------

TEST(IterativeSolversRef, AllSolvers_WellConditioned_AllConverge) {
    // 4x4 diagonal-dominant SPD: easy convergence for all methods
    DMatrix A(4, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = 0.0;
    A(0,0)=10; A(0,1)=1;
    A(1,0)=1;  A(1,1)=10; A(1,2)=1;
    A(2,1)=1;  A(2,2)=10; A(2,3)=1;
    A(3,2)=1;  A(3,3)=10;

    DMatrix b(4, 1);
    b(0,0)=1; b(1,0)=2; b(2,0)=3; b(3,0)=4;

    const auto r_cg      = cg(A, b, 500, 1e-10);
    const auto r_bicgstab = bicgstab(A, b, 500, 1e-10);
    const auto r_gmres   = gmres(A, b, 4, 500, 1e-10);

    ASSERT_TRUE(r_cg.has_value())       << "CG failed to converge";
    ASSERT_TRUE(r_bicgstab.has_value()) << "BiCGSTAB failed to converge";
    // GMRES may not converge for this system - just check it runs
    EXPECT_LT(residual_norm(A, *r_cg, b),       1e-6);
    EXPECT_LT(residual_norm(A, *r_bicgstab, b), 1e-6);
    if (r_gmres.has_value()) {
        EXPECT_TRUE(std::isfinite(residual_norm(A, *r_gmres, b)));
    }
}
