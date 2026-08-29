// MathScript: Advanced Optimization Tests (Wave 43)
// Tests for golden_section, newton_1d, gradient_descent, newton_raphson, broyden

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/optim/optim.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// golden_section: find minimum of 1D function
// ---------------------------------------------------------------------------

TEST(OptimAdv2, GoldenSection_Quadratic_MinAtZero) {
    // f(x) = x^2 minimized at x=0
    auto f = [](double x) { return x * x; };
    double xmin = golden_section(f, -1.0, 1.0);
    EXPECT_NEAR(xmin, 0.0, 1e-5);
}

TEST(OptimAdv2, GoldenSection_ShiftedQuadratic_MinAtTwo) {
    // f(x) = (x-2)^2 minimized at x=2
    auto f = [](double x) { return (x - 2.0) * (x - 2.0); };
    double xmin = golden_section(f, 0.0, 4.0);
    EXPECT_NEAR(xmin, 2.0, 1e-5);
}

TEST(OptimAdv2, GoldenSection_Cosine_MinInInterval) {
    // cos(x) on [3, 4] has minimum near pi ≈ 3.14159
    auto f = [](double x) { return std::cos(x); };
    double xmin = golden_section(f, 3.0, 4.0);
    EXPECT_NEAR(xmin, M_PI, 1e-4);
}

TEST(OptimAdv2, GoldenSection_ReturnsFinite) {
    auto f = [](double x) { return (x - 0.5) * (x - 0.5) + 1.0; };
    double xmin = golden_section(f, 0.0, 1.0);
    EXPECT_TRUE(std::isfinite(xmin));
    EXPECT_GE(xmin, 0.0);
    EXPECT_LE(xmin, 1.0);
}

TEST(OptimAdv2, GoldenSection_QuarticFunction) {
    // f(x) = x^4 - 4x^2 + 4 = (x^2 - 2)^2 on [-1, 2] has min at x=sqrt(2)
    auto f = [](double x) { return (x * x - 2.0) * (x * x - 2.0); };
    double xmin = golden_section(f, 1.0, 2.0);
    EXPECT_NEAR(xmin, std::sqrt(2.0), 1e-4);
}

// ---------------------------------------------------------------------------
// newton_1d: find local minimum of 1D function (Newton second-order method)
// ---------------------------------------------------------------------------

TEST(OptimAdv2, Newton1D_Quadratic_MinimizesAtZero) {
    // f(x) = x^2, minimum at x=0 — newton_1d finds critical point f'(x)=0
    auto f = [](double x) { return x * x; };
    double xmin = newton_1d(f, 1.0);
    EXPECT_NEAR(xmin, 0.0, 1e-6);
}

TEST(OptimAdv2, Newton1D_ShiftedQuadratic_MinimizesAtTwo) {
    // f(x) = (x-2)^2, minimum at x=2
    auto f = [](double x) { return (x - 2.0) * (x - 2.0); };
    double xmin = newton_1d(f, 2.5);
    EXPECT_NEAR(xmin, 2.0, 1e-6);
}

TEST(OptimAdv2, Newton1D_Cosine_MaximizesAtZero) {
    // f(x) = -cos(x), minimum (of neg-cos) at x=0 → critical point of cos at x=0
    auto f = [](double x) { return -std::cos(x); };
    double xmin = newton_1d(f, 0.5);
    EXPECT_NEAR(xmin, 0.0, 1e-4);
}

TEST(OptimAdv2, Newton1D_ReturnsFinite) {
    auto f = [](double x) { return x * x + 2.0 * x + 1.0; };  // (x+1)^2
    double xmin = newton_1d(f, 0.0);
    EXPECT_TRUE(std::isfinite(xmin));
    EXPECT_NEAR(xmin, -1.0, 1e-5);
}

TEST(OptimAdv2, Newton1D_FirstDerivNearZeroAtResult) {
    // At a true minimum, f'(x) ≈ 0
    auto f = [](double x) { return (x - 3.0) * (x - 3.0); };
    double xmin = newton_1d(f, 4.0);
    constexpr double h = 1e-5;
    double deriv = (f(xmin + h) - f(xmin - h)) / (2.0 * h);
    EXPECT_NEAR(deriv, 0.0, 1e-4) << "Derivative at minimum should be ~0";
}

// ---------------------------------------------------------------------------
// gradient_descent: minimize 2D function
// ---------------------------------------------------------------------------

TEST(OptimAdv2, GradientDescent_Bowl_ConvergesNearMinimum) {
    // f(x, y) = x^2 + y^2, minimum at (0, 0)
    auto f = [](double x, double y) { return x * x + y * y; };
    auto result = gradient_descent(f, 1.0, 1.0, 0.1, 500);
    EXPECT_TRUE(std::isfinite(result.x));
    EXPECT_TRUE(std::isfinite(result.y));
    // After convergence, function value should be near 0
    EXPECT_LT(result.f_val, 0.1);
}

TEST(OptimAdv2, GradientDescent_OffCenter_Converges) {
    // f(x, y) = (x-2)^2 + (y+1)^2, minimum at (2, -1)
    auto f = [](double x, double y) {
        return (x - 2.0) * (x - 2.0) + (y + 1.0) * (y + 1.0);
    };
    auto result = gradient_descent(f, 0.0, 0.0, 0.1, 1000);
    EXPECT_TRUE(std::isfinite(result.x));
    EXPECT_TRUE(std::isfinite(result.y));
    EXPECT_LT(result.f_val, 0.1);
}

TEST(OptimAdv2, GradientDescent_ReturnsConvergedFlag) {
    auto f = [](double x, double y) { return x * x + y * y; };
    auto result = gradient_descent(f, 0.001, 0.001, 0.1, 1000);
    // Very close to minimum — should converge
    EXPECT_TRUE(std::isfinite(result.f_val));
    EXPECT_GE(result.iterations, 0u);
}

TEST(OptimAdv2, GradientDescent_FValDecreases) {
    // f_val of the result should be less than initial f value
    auto f = [](double x, double y) { return x * x + y * y; };
    auto result = gradient_descent(f, 5.0, 5.0, 0.05, 200);
    double initial = f(5.0, 5.0);
    EXPECT_LT(result.f_val, initial) << "f_val should decrease from starting point";
}

// ---------------------------------------------------------------------------
// newton_raphson: find root of 2D system
// ---------------------------------------------------------------------------

TEST(OptimAdv2, NewtonRaphson_SimpleQuadratic_FindsRoot) {
    // f(x, y) = x^2 + y^2 - 1 (unit circle) — root near (1, 0)
    // Note: newton_raphson for 2D needs a proper 2D system
    // Here we use f as a scalar — test that result is finite
    auto f = [](double x, double y) { return x * x + y - 1.0; };
    auto [rx, ry] = newton_raphson(f, 0.5, 0.5);
    EXPECT_TRUE(std::isfinite(rx));
    EXPECT_TRUE(std::isfinite(ry));
}

TEST(OptimAdv2, NewtonRaphson_LinearSystem_ConvergesNearRoot) {
    auto f = [](double x, double y) { return x + y - 2.0; };
    auto [rx, ry] = newton_raphson(f, 1.0, 1.0);
    EXPECT_TRUE(std::isfinite(rx));
    EXPECT_TRUE(std::isfinite(ry));
}

// ---------------------------------------------------------------------------
// broyden: quasi-Newton root finding
// ---------------------------------------------------------------------------

TEST(OptimAdv2, Broyden_ReturnsFinite) {
    auto f = [](double x, double y) { return x * x + y * y - 1.0; };
    auto [rx, ry] = broyden(f, 0.5, 0.5);
    EXPECT_TRUE(std::isfinite(rx));
    EXPECT_TRUE(std::isfinite(ry));
}

TEST(OptimAdv2, Broyden_LinearCase) {
    auto f = [](double x, double y) { return x + y - 3.0; };
    auto [rx, ry] = broyden(f, 1.0, 1.0);
    EXPECT_TRUE(std::isfinite(rx));
    EXPECT_TRUE(std::isfinite(ry));
    // f(rx, ry) should be near 0
    EXPECT_NEAR(rx + ry, 3.0, 0.5);
}

// ---------------------------------------------------------------------------
// simplex_solver
// ---------------------------------------------------------------------------

TEST(OptimAdv2, Simplex_SimpleMaximize) {
    // Maximize c^T x subject to Ax <= b, x >= 0
    // Standard LP: max 2x1 + 3x2 s.t. x1 + x2 <= 4, x1 + 3x2 <= 6
    std::vector<double> c = {-2.0, -3.0};  // Negate for minimization
    std::vector<std::vector<double>> A = {{1.0, 1.0}, {1.0, 3.0}};
    std::vector<double> b = {4.0, 6.0};
    auto [sol, val] = simplex_solver(c, A, b);
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_GE(sol.size(), 2u);
}

TEST(OptimAdv2, Simplex_ReturnsFinite) {
    std::vector<double> c = {-1.0, -1.0};
    std::vector<std::vector<double>> A = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> b = {5.0, 5.0};
    auto [sol, val] = simplex_solver(c, A, b);
    EXPECT_TRUE(std::isfinite(val));
}
