// MathScript: Advanced Newton-Raphson and Broyden Optimizer Tests (Wave 50)
// Tests for newton_raphson (2D root finder) and broyden (2D root finder)

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/optim/optim.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// newton_raphson: 2D root finding (finds f(x,y) = g(x,y) = 0)
// ---------------------------------------------------------------------------

TEST(OptimNewtonBroyden, NewtonRaphson_SimpleLinear_FiniteResult) {
    // f(x, y) = x + y - 2: zero at any (a, 2-a)
    auto f = [](double x, double y) { return x + y - 2.0; };
    auto [xr, yr] = newton_raphson(f, 1.0, 1.0);
    EXPECT_TRUE(std::isfinite(xr));
    EXPECT_TRUE(std::isfinite(yr));
}

TEST(OptimNewtonBroyden, NewtonRaphson_QuadraticRoot_Converges) {
    // f(x, y) = x^2 + y^2 - 1: roots on unit circle
    auto f = [](double x, double y) { return x * x + y * y - 1.0; };
    auto [xr, yr] = newton_raphson(f, 0.5, 0.5);
    // Should be near unit circle
    double dist = std::sqrt(xr * xr + yr * yr);
    EXPECT_NEAR(dist, 1.0, 0.1) << "Newton-Raphson should converge near unit circle";
}

TEST(OptimNewtonBroyden, NewtonRaphson_ResultIsFinite) {
    auto f = [](double x, double y) { return std::sin(x) + y - 1.0; };
    auto [xr, yr] = newton_raphson(f, 0.0, 0.0);
    EXPECT_TRUE(std::isfinite(xr));
    EXPECT_TRUE(std::isfinite(yr));
}

TEST(OptimNewtonBroyden, NewtonRaphson_Zero_ConstantFunc) {
    // f(x,y) = 0: already at root; should stay at starting point
    auto f = [](double /*x*/, double /*y*/) { return 0.0; };
    auto [xr, yr] = newton_raphson(f, 1.0, 2.0);
    EXPECT_TRUE(std::isfinite(xr));
    EXPECT_TRUE(std::isfinite(yr));
}

// ---------------------------------------------------------------------------
// broyden: 2D quasi-Newton root finding
// ---------------------------------------------------------------------------

TEST(OptimNewtonBroyden, Broyden_LinearSystem_Converges) {
    // f(x, y) = 2x + y - 3: zero at (1.5, 0), (1, 1), etc.
    auto f = [](double x, double y) { return 2.0 * x + y - 3.0; };
    auto [xr, yr] = broyden(f, 1.0, 1.0);
    // Check f(xr, yr) ≈ 0
    double fval = 2.0 * xr + yr - 3.0;
    EXPECT_NEAR(fval, 0.0, 0.1) << "Broyden should find root of linear function";
}

TEST(OptimNewtonBroyden, Broyden_ResultIsFinite) {
    auto f = [](double x, double y) { return x * x - y; };
    auto [xr, yr] = broyden(f, 1.0, 1.0);
    EXPECT_TRUE(std::isfinite(xr));
    EXPECT_TRUE(std::isfinite(yr));
}

TEST(OptimNewtonBroyden, Broyden_Zero_AlreadyRoot) {
    auto f = [](double /*x*/, double /*y*/) { return 0.0; };
    auto [xr, yr] = broyden(f, 2.0, 3.0);
    EXPECT_TRUE(std::isfinite(xr));
    EXPECT_TRUE(std::isfinite(yr));
}

// ---------------------------------------------------------------------------
// gradient_descent: 2D minimizer
// ---------------------------------------------------------------------------

TEST(OptimNewtonBroyden, GradDesc_Converges_ToMinimum_Paraboloid) {
    // f(x,y) = (x-2)^2 + (y-3)^2, minimum at (2, 3)
    auto f = [](double x, double y) { return (x-2.0)*(x-2.0) + (y-3.0)*(y-3.0); };
    auto result = gradient_descent(f, 0.0, 0.0, 0.1, 500);
    EXPECT_NEAR(result.x, 2.0, 0.1)
        << "Gradient descent should converge to x=2";
    EXPECT_NEAR(result.y, 3.0, 0.1)
        << "Gradient descent should converge to y=3";
}

TEST(OptimNewtonBroyden, GradDesc_FVal_Decreases) {
    auto f = [](double x, double y) { return x*x + y*y; };
    auto result1 = gradient_descent(f, 5.0, 5.0, 0.1, 10);
    auto result2 = gradient_descent(f, 5.0, 5.0, 0.1, 100);
    EXPECT_LE(result2.f_val, result1.f_val + 1e-6)
        << "More iterations should give smaller f_val";
}

TEST(OptimNewtonBroyden, GradDesc_ResultIsFinite) {
    auto f = [](double x, double y) { return x*x + 2.0*y*y + x*y; };
    auto result = gradient_descent(f, 1.0, 1.0, 0.05, 200);
    EXPECT_TRUE(std::isfinite(result.x));
    EXPECT_TRUE(std::isfinite(result.y));
    EXPECT_TRUE(std::isfinite(result.f_val));
}

TEST(OptimNewtonBroyden, GradDesc_AlreadyAtMinimum) {
    // At minimum (0,0), gradient is 0, should stay
    auto f = [](double x, double y) { return x*x + y*y; };
    auto result = gradient_descent(f, 0.0, 0.0, 0.1, 50);
    EXPECT_NEAR(result.x, 0.0, 1e-6);
    EXPECT_NEAR(result.y, 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// golden_section: 1D minimizer
// ---------------------------------------------------------------------------

TEST(OptimNewtonBroyden, GoldenSection_QuadraticMinimum) {
    // f(x) = (x-2)^2, minimum at x=2
    auto f = [](double x) { return (x-2.0)*(x-2.0); };
    double xmin = golden_section(f, 0.0, 5.0, 1e-8);
    EXPECT_NEAR(xmin, 2.0, 1e-5);
}

TEST(OptimNewtonBroyden, GoldenSection_SineMinimum) {
    // f(x) = -cos(x) on [-pi/2, pi/2], minimum at x=0
    auto f = [](double x) { return -std::cos(x); };
    double xmin = golden_section(f, -M_PI / 2.0, M_PI / 2.0, 1e-8);
    EXPECT_NEAR(xmin, 0.0, 1e-5);
}

TEST(OptimNewtonBroyden, GoldenSection_ResultInRange) {
    auto f = [](double x) { return x * x - 4.0 * x + 5.0; };
    double xmin = golden_section(f, 0.0, 4.0, 1e-8);
    EXPECT_GE(xmin, 0.0);
    EXPECT_LE(xmin, 4.0);
}

TEST(OptimNewtonBroyden, Newton1D_QuadraticMinimizer) {
    // f(x) = (x-1)^2, f'(x) = 2(x-1) = 0 at x=1
    auto f = [](double x) { return (x-1.0)*(x-1.0); };
    double xmin = newton_1d(f, 2.0);
    EXPECT_NEAR(xmin, 1.0, 1e-5);
}

TEST(OptimNewtonBroyden, Newton1D_ResultIsFinite) {
    auto f = [](double x) { return x*x + std::sin(x); };
    double xmin = newton_1d(f, 0.0);
    EXPECT_TRUE(std::isfinite(xmin));
}
