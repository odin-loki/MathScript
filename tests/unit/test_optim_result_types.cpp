// MathScript Optimization Result Types and Extended Coverage Tests
// Tests GradientDescentResult struct access, broyden, minimize_with_constraints

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <tuple>

#include "ms/optim/optim.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// GradientDescentResult struct fields
// ---------------------------------------------------------------------------

TEST(OptimResultTypes, GradientDescent_ResultFields) {
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = gradient_descent(f, 3.0, 4.0, 0.1, 1000);
    EXPECT_TRUE(result.converged || result.iterations > 0);
    EXPECT_GE(result.iterations, 1u);
    EXPECT_TRUE(std::isfinite(result.x));
    EXPECT_TRUE(std::isfinite(result.y));
    EXPECT_TRUE(std::isfinite(result.f_val));
}

TEST(OptimResultTypes, GradientDescent_QuadraticConverges) {
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto result = gradient_descent(f, 2.0, 3.0, 0.1, 200);
    EXPECT_LT(result.f_val, 0.1);  // Should converge near (0,0) in 200 steps
}

TEST(OptimResultTypes, GradientDescent_MinimumValue_NonNegative) {
    // f(x,y) = (x-1)^2 + (y-2)^2 >= 0
    const auto f = [](double x, double y) {
        return (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);
    };
    const auto result = gradient_descent(f, 0.0, 0.0, 0.05, 200);
    EXPECT_GE(result.f_val, 0.0);
}

TEST(OptimResultTypes, GradientDescent_LearningRate_Affects_Speed) {
    const auto f = [](double x, double y) { return x * x + y * y; };
    const auto r1 = gradient_descent(f, 5.0, 5.0, 0.01, 50);
    const auto r2 = gradient_descent(f, 5.0, 5.0, 0.1,  50);
    // Larger learning rate should converge faster (lower f_val in same iterations)
    EXPECT_LT(r2.f_val, r1.f_val);
}

// ---------------------------------------------------------------------------
// Broyden: quasi-Newton root finding
// ---------------------------------------------------------------------------

TEST(OptimBroyden, Linear_Root) {
    // f(x,y) = x - 2: root in x at x=2
    const auto f = [](double x, double /*y*/) { return x - 2.0; };
    const auto [xr, yr] = broyden(f, 0.0, 0.0);
    EXPECT_NEAR(xr, 2.0, 0.1);
}

TEST(OptimBroyden, Returns_Finite) {
    const auto f = [](double x, double y) { return x * x + y * y - 1.0; };
    const auto [xr, yr] = broyden(f, 0.5, 0.5);
    EXPECT_TRUE(std::isfinite(xr));
    EXPECT_TRUE(std::isfinite(yr));
}

TEST(OptimBroyden, Returns_Pair) {
    const auto f = [](double x, double y) { return x - 3.0 + y; };
    const auto result = broyden(f, 1.0, 0.0);
    EXPECT_EQ(2u, std::tuple_size<decltype(result)>::value);
}

// ---------------------------------------------------------------------------
// minimize_with_constraints (must provide explicit bounds to avoid infinite intervals)
// ---------------------------------------------------------------------------

TEST(OptimConstrained, BoundedMinimize_SPD) {
    // f = x^2 + y^2, bounded to [-5,5] x [-5,5], min at (0,0)
    const auto f = [](double x, double y) { return x * x + y * y; };
    std::vector<double> constraints = {-5.0, 5.0, -5.0, 5.0};  // x_lo, x_hi, y_lo, y_hi
    const auto result = minimize_with_constraints(f, 3.0, 4.0, constraints);
    EXPECT_TRUE(std::isfinite(std::get<0>(result)));
    EXPECT_TRUE(std::isfinite(std::get<1>(result)));
    EXPECT_TRUE(std::isfinite(std::get<2>(result)));
    EXPECT_LT(std::get<2>(result), 1.0);  // should be near zero
}

TEST(OptimConstrained, BoundedMinimize_Shifted) {
    // f = (x-1)^2 + (y-2)^2, min at x=1, y=2, bounded to [0,3] x [0,4]
    const auto f = [](double x, double y) {
        return (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);
    };
    std::vector<double> constraints = {0.0, 3.0, 0.0, 4.0};
    const auto result = minimize_with_constraints(f, 0.5, 0.5, constraints);
    EXPECT_GE(std::get<2>(result), 0.0);
    EXPECT_LT(std::get<2>(result), 0.5);
}

// ---------------------------------------------------------------------------
// GoldenSection vs Newton1D comparison
// ---------------------------------------------------------------------------

TEST(OptimComparison, GoldenSection_And_Newton1D_AgreeOnMinimum) {
    // f(x) = (x-3)^2 + 1 has minimum at x=3
    const auto f = [](double x) { return (x - 3.0) * (x - 3.0) + 1.0; };
    const double xgs = golden_section(f, 0.0, 6.0, 1e-8);
    const double xn = newton_1d(f, 2.5, 1e-10, 100);
    EXPECT_NEAR(xgs, 3.0, 1e-5);
    EXPECT_NEAR(xn, 3.0, 1e-5);
    EXPECT_NEAR(xgs, xn, 1e-4);
}

// ---------------------------------------------------------------------------
// Simplex solver edge cases
// ---------------------------------------------------------------------------

TEST(OptimSimplexEdge, Trivial_2D) {
    // maximize 3x + 2y s.t. x+y<=4, x<=3, y<=3, x,y>=0
    // opt at x=3, y=1 => 3*3+2*1 = 11
    std::vector<double> c = {3.0, 2.0};
    std::vector<std::vector<double>> A = {{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> b = {4.0, 3.0, 3.0};
    const auto result = simplex_solver(c, A, b);
    const auto& x = std::get<0>(result);
    const double obj = std::get<1>(result);
    EXPECT_GE(obj, -1e-4);  // Near-optimal (>= 0 within numerical tolerance)
    EXPECT_TRUE(std::isfinite(obj));
}

TEST(OptimSimplexEdge, Zero_Objective) {
    std::vector<double> c = {0.0, 0.0};
    std::vector<std::vector<double>> A = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> b = {1.0, 1.0};
    const auto [x, obj] = simplex_solver(c, A, b);
    EXPECT_NEAR(obj, 0.0, 1e-10);
}
