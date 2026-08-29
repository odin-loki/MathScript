// MathScript: LinAlg + ODE Integration Pipeline (Wave 50)
// Tests combining matrix operations with ODE solvers and stats

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/ode/ode.hpp"
#include "ms/stats/stats.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// Pipeline 1: expm verifies matrix exponential for diagonal systems
// ---------------------------------------------------------------------------

TEST(LinalgOdePipeline, Expm_DiagonalLambda_MatchesScalar) {
    // For diagonal A = diag(a, b), expm(A) = diag(e^a, e^b)
    // Use small entries within 12-term Taylor series convergence range
    DMatrix A({{0.5, 0.0}, {0.0, 0.3}});
    auto R = expm(A);
    ASSERT_TRUE(R.has_value());
    EXPECT_NEAR((*R)(0, 0), std::exp(0.5), 1e-8);
    EXPECT_NEAR((*R)(1, 1), std::exp(0.3), 1e-8);
    EXPECT_NEAR((*R)(0, 1), 0.0, 1e-8);
    EXPECT_NEAR((*R)(1, 0), 0.0, 1e-8);
}

TEST(LinalgOdePipeline, Expm_tA_MatchesODE_Euler) {
    // y' = lambda * y, y(0) = 1 solved by ODE Euler should match exp(lambda*t)
    double lambda = -0.5;
    auto f = [lambda](double /*t*/, double y) { return lambda * y; };
    auto result = ode_euler(f, 0.0, 1.0, 2.0, 2000);
    double y_final = result.y.back();
    double expected = std::exp(lambda * 2.0);
    EXPECT_NEAR(y_final, expected, 0.01)
        << "Euler ODE should match exp(lambda*t) for y'=lambda*y";
}

TEST(LinalgOdePipeline, Expm_tA_MatchesODE_RK4) {
    // y' = lambda * y, y(0) = 1 solved by RK4 should be more accurate
    double lambda = -1.0;
    auto f = [lambda](double /*t*/, double y) { return lambda * y; };
    auto result = ode_rk4(f, 0.0, 1.0, 3.0, 300);
    double y_final = result.y.back();
    double expected = std::exp(lambda * 3.0);
    EXPECT_NEAR(y_final, expected, 1e-5)
        << "RK4 ODE should closely match exp(lambda*t)";
}

// ---------------------------------------------------------------------------
// Pipeline 2: ODE solution statistics
// ---------------------------------------------------------------------------

TEST(LinalgOdePipeline, ODE_Solution_Mean_DecayingExp) {
    // y' = -y, y(0)=1; solution values should have positive mean
    auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 5.0, 500);
    double m = mean(result.y);
    EXPECT_GT(m, 0.0) << "Decaying exponential mean should be positive";
    EXPECT_LT(m, 1.0) << "Decaying exponential mean should be < 1";
}

TEST(LinalgOdePipeline, ODE_Solution_Monotone_Decrease) {
    // y' = -y monotonically decreasing
    auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 5.0, 500);
    auto& y = result.y;
    for (size_t i = 1; i < y.size(); ++i)
        EXPECT_LT(y[i], y[i-1] + 1e-9)
            << "Solution not monotone at step " << i;
}

// ---------------------------------------------------------------------------
// Pipeline 3: solve Ax=b and verify residual
// ---------------------------------------------------------------------------

TEST(LinalgOdePipeline, Solve_ResidualNear0_3x3) {
    DMatrix A({{3.0, 1.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}, {1.0}});
    auto x_res = solve(A, b);
    ASSERT_TRUE(x_res.has_value());
    auto& x = *x_res;
    // Residual r = A*x - b
    auto Ax_res = matmul<double>(A, x);
    ASSERT_TRUE(Ax_res.has_value());
    auto& Ax = *Ax_res;
    for (size_t i = 0; i < b.rows(); ++i)
        EXPECT_NEAR(Ax(i, 0), b(i, 0), 1e-9)
            << "Residual not near 0 at row " << i;
}

TEST(LinalgOdePipeline, Solve_Identity_GivesB) {
    DMatrix I({{1.0, 0.0}, {0.0, 1.0}});
    DMatrix b({{5.0}, {7.0}});
    auto x_res = solve(I, b);
    ASSERT_TRUE(x_res.has_value());
    auto& x = *x_res;
    EXPECT_NEAR(x(0, 0), 5.0, 1e-10);
    EXPECT_NEAR(x(1, 0), 7.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Pipeline 4: det/trace with matmul
// ---------------------------------------------------------------------------

TEST(LinalgOdePipeline, Det_Product_AB_IsProduct_DetA_DetB) {
    // det(A*B) = det(A) * det(B)
    DMatrix A({{2.0, 1.0}, {1.0, 3.0}});
    DMatrix B({{4.0, 0.0}, {1.0, 2.0}});
    auto dA = det(A);
    auto dB = det(B);
    auto AB_res = matmul<double>(A, B);
    ASSERT_TRUE(dA.has_value());
    ASSERT_TRUE(dB.has_value());
    ASSERT_TRUE(AB_res.has_value());
    auto dAB = det(*AB_res);
    ASSERT_TRUE(dAB.has_value());
    EXPECT_NEAR(*dAB, (*dA) * (*dB), 1e-9)
        << "det(A*B) != det(A)*det(B)";
}

TEST(LinalgOdePipeline, Trace_SumOfEigenvalues_2x2_Symmetric) {
    // For symmetric 2x2, trace = sum of eigenvalues
    DMatrix A({{3.0, 1.0}, {1.0, 5.0}});
    auto t = trace(A);
    auto ev = eig_sym(A);
    ASSERT_TRUE(t.has_value());
    ASSERT_TRUE(ev.has_value());
    double ev_sum = 0.0;
    for (size_t i = 0; i < ev->values.rows(); ++i)
        ev_sum += ev->values(i, 0);
    EXPECT_NEAR(*t, ev_sum, 1e-9);
}
