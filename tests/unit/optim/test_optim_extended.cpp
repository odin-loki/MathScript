#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <functional>

#include "ms/optim/optim.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// golden_section: minimize 1D unimodal functions
// ---------------------------------------------------------------------------

TEST(OptimExtTest, GoldenSection_QuadraticMinimum) {
    // f(x) = (x-3)^2 has minimum at x=3
    const auto f = [](double x) { return (x - 3.0) * (x - 3.0); };
    const double xmin = golden_section(f, 0.0, 10.0, 1e-8);
    EXPECT_NEAR(xmin, 3.0, 1e-6);
}

TEST(OptimExtTest, GoldenSection_CubicMinimum) {
    // f(x) = (x-1)^2 + 1 on [0,4], minimum at x=1
    const auto f = [](double x) { return (x - 1.0) * (x - 1.0) + 1.0; };
    const double xmin = golden_section(f, 0.0, 4.0, 1e-8);
    EXPECT_NEAR(xmin, 1.0, 1e-6);
}

TEST(OptimExtTest, GoldenSection_BroadInterval) {
    // f(x) = x^2, minimum at x=0
    const auto f = [](double x) { return x * x; };
    const double xmin = golden_section(f, -5.0, 5.0, 1e-8);
    EXPECT_NEAR(xmin, 0.0, 1e-6);
}

TEST(OptimExtTest, GoldenSection_NearBoundary) {
    // f(x) = (x+2)^2 + 1 on [-3, 0], minimum at x=-2
    const auto f = [](double x) { return (x + 2.0) * (x + 2.0) + 1.0; };
    const double xmin = golden_section(f, -3.0, 0.0, 1e-8);
    EXPECT_NEAR(xmin, -2.0, 1e-6);
}

// ---------------------------------------------------------------------------
// newton_1d: stationary point (minimum) finding via Newton's method
// ---------------------------------------------------------------------------

TEST(OptimExtTest, Newton1D_QuadraticMinimum) {
    // f(x) = (x-3)^2 has minimum at x=3; Newton finds stationary point f'(x)=0
    const auto f = [](double x) { return (x - 3.0) * (x - 3.0); };
    const double xmin = newton_1d(f, 4.0, 1e-10, 100);
    EXPECT_NEAR(xmin, 3.0, 1e-6);
}

TEST(OptimExtTest, Newton1D_ShiftedParabola) {
    // f(x) = (x+5)^2 has minimum at x=-5
    const auto f = [](double x) { return (x + 5.0) * (x + 5.0); };
    const double xmin = newton_1d(f, -4.0, 1e-10, 100);
    EXPECT_NEAR(xmin, -5.0, 1e-6);
}

TEST(OptimExtTest, Newton1D_FlatParabola) {
    // f(x) = 2x^2 has minimum at x=0
    const auto f = [](double x) { return 2.0 * x * x; };
    const double xmin = newton_1d(f, 1.0, 1e-10, 100);
    EXPECT_NEAR(xmin, 0.0, 1e-6);
}

TEST(OptimExtTest, Newton1D_FunctionValueDecreases) {
    // After newton_1d, f at the result should be near the minimum
    const auto f = [](double x) { return (x - 2.0) * (x - 2.0) + 1.0; };
    const double xmin = newton_1d(f, 0.0, 1e-8, 100);
    EXPECT_LT(f(xmin), f(0.0));  // should be closer to minimum than start
}

// ---------------------------------------------------------------------------
// gradient_descent: 2D minimization
// ---------------------------------------------------------------------------

TEST(OptimExtTest, GradientDescent_RosenbrockBasin) {
    // f(x,y) = x^2 + y^2 has minimum at (0,0)
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = gradient_descent(f, 1.0, 1.0, 0.1, 500);
    EXPECT_NEAR(result.x, 0.0, 0.1);
    EXPECT_NEAR(result.y, 0.0, 0.1);
    EXPECT_LT(result.f_val, 0.01);
}

TEST(OptimExtTest, GradientDescent_NegativeStart) {
    // f(x,y) = x^2 + y^2, starting from negative quadrant
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = gradient_descent(f, -2.0, -2.0, 0.1, 500);
    EXPECT_NEAR(result.x, 0.0, 0.1);
    EXPECT_NEAR(result.y, 0.0, 0.1);
}

TEST(OptimExtTest, GradientDescent_ConvergedFlag) {
    // A simple quadratic should converge
    const auto f = [](double x, double y) { return (x - 2.0) * (x - 2.0) + (y - 1.0) * (y - 1.0); };
    const auto result = gradient_descent(f, 0.0, 0.0, 0.05, 1000);
    EXPECT_TRUE(result.converged);
}

TEST(OptimExtTest, GradientDescent_IterationsPositive) {
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = gradient_descent(f, 1.0, 1.0, 0.1, 50);
    EXPECT_GT(result.iterations, 0u);
    EXPECT_LE(result.iterations, 50u);
}

// ---------------------------------------------------------------------------
// newton_raphson: finds x such that f(x, y0) = 0 (root in x, y is fixed)
// ---------------------------------------------------------------------------

TEST(OptimExtTest, NewtonRaphson_LinearRoot) {
    // f(x, y) = x - 3; root in x at x=3 (y is fixed)
    const auto f = [](double x, double /*y*/) { return x - 3.0; };
    const auto [xr, yr] = newton_raphson(f, 0.0, 0.0, 1e-8, 100);
    EXPECT_NEAR(xr, 3.0, 1e-6);
}

TEST(OptimExtTest, NewtonRaphson_QuadraticRoot) {
    // f(x, y) = x*x - 9; root at x=3 (starting from x=2)
    const auto f = [](double x, double /*y*/) { return x * x - 9.0; };
    const auto [xr, yr] = newton_raphson(f, 2.0, 0.0, 1e-8, 100);
    EXPECT_NEAR(std::abs(xr), 3.0, 1e-5);  // may converge to +3 or -3
}

TEST(OptimExtTest, NewtonRaphson_YIsFixed) {
    // newton_raphson should not modify y0
    const auto f = [](double x, double /*y*/) { return x - 5.0; };
    const auto [xr, yr] = newton_raphson(f, 0.0, 7.0, 1e-8, 100);
    EXPECT_NEAR(yr, 7.0, 1e-12);  // y should remain at y0=7.0
}

// ---------------------------------------------------------------------------
// simplex_solver: linear programming
// ---------------------------------------------------------------------------

TEST(OptimExtTest, SimplexSolver_NoConstraints_ReturnsResult) {
    // Minimize -x1 - x2 subject to x1+x2 <= 4
    const std::vector<double> c = {-1.0, -1.0};
    const std::vector<std::vector<double>> A = {{1.0, 1.0}};
    const std::vector<double> b = {4.0};
    const auto [solution, obj] = simplex_solver(c, A, b);
    EXPECT_FALSE(solution.empty());
    EXPECT_LT(obj, 0.0);  // minimizing negative values gives negative objective
}

TEST(OptimExtTest, SimplexSolver_ReturnsSizedSolution) {
    const std::vector<double> c = {-2.0, -3.0};
    const std::vector<std::vector<double>> A = {
        {1.0, 2.0},
        {2.0, 1.0}
    };
    const std::vector<double> b = {6.0, 8.0};
    const auto [solution, obj] = simplex_solver(c, A, b);
    EXPECT_EQ(solution.size(), 2u);
}
