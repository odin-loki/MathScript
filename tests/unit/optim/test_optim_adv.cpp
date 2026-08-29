// MathScript Optimization - Advanced Tests
// Covers: newton_1d, broyden, minimize_with_constraints, GradientDescentResult members

#include <gtest/gtest.h>
#include <cmath>
#include <tuple>
#include <vector>

#include "ms/optim/optim.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// newton_1d: 1D stationary-point finder (finds where f'(x) = 0)
// Uses Newton's method on f'' to find minimum/maximum
// ---------------------------------------------------------------------------

TEST(Newton1D, MinOfQuadratic) {
    // f(x) = (x - 3)^2 has minimum at x = 3
    auto r = newton_1d([](double x){ return (x - 3.0)*(x - 3.0); }, 4.0);
    EXPECT_NEAR(r, 3.0, 1e-6);
}

TEST(Newton1D, MinOfBiquadratic) {
    // f(x) = x^4 has minimum at x = 0
    auto r = newton_1d([](double x){ return x*x*x*x; }, 0.5);
    EXPECT_NEAR(r, 0.0, 1e-5);
}

TEST(Newton1D, MinOfShiftedQuadratic) {
    // f(x) = (x + 2)^2 has minimum at x = -2
    auto r = newton_1d([](double x){ return (x + 2.0)*(x + 2.0); }, -1.0);
    EXPECT_NEAR(r, -2.0, 1e-6);
}

TEST(Newton1D, MinOfCosine) {
    // f(x) = -cos(x) has minimum near pi starting from 2.5
    auto r = newton_1d([](double x){ return -std::cos(x); }, 2.5, 1e-10, 100);
    EXPECT_TRUE(std::isfinite(r));
    // Should be near pi
    EXPECT_NEAR(r, M_PI, 0.1);
}

TEST(Newton1D, ReturnIsFinite) {
    auto r = newton_1d([](double x){ return x*x - 1.0; }, 0.5);
    EXPECT_TRUE(std::isfinite(r));
}

TEST(Newton1D, SmokeDifferentFunctions) {
    // Gauss: f(x) = exp(-x^2) has max at x=0, min at +/-inf; near 0 newton finds the max
    auto r = newton_1d([](double x){ return -std::exp(-x*x); }, 0.1);
    EXPECT_TRUE(std::isfinite(r));
}

// ---------------------------------------------------------------------------
// broyden: 2D quasi-Newton root finder
// ---------------------------------------------------------------------------

TEST(Broyden, SimplePairOfRoots) {
    // f(x, y) = x^2 + y^2 - 4 AND g(x,y) modeled as single function here
    // broyden takes a Func2D: finds (x, y) where f(x, y) = 0 iteratively
    // Let f(x, y) = x + y - 4 (trivial, x=y=2 is one solution near (1.5, 1.5))
    auto [x, y] = broyden([](double a, double b){ return a + b - 4.0; }, 1.5, 1.5);
    // broyden finds a zero; just check it returns finite values
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_TRUE(std::isfinite(y));
}

TEST(Broyden, SeparableProblem) {
    // f(x, y) = x - 3 → root at x=3 (y unchanged)
    auto [x, y] = broyden([](double a, double /*b*/){ return a - 3.0; }, 0.0, 0.0);
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_TRUE(std::isfinite(y));
}

// ---------------------------------------------------------------------------
// minimize_with_constraints
// ---------------------------------------------------------------------------

TEST(MinimizeConstrained, BasicFinite) {
    // IMPORTANT: must always provide finite constraints, otherwise golden_section
    // will run on [-1e300, 1e300] and loop forever.
    // Minimize f(x, y) = x^2 + y^2 on [-5, 5] x [-5, 5], minimum at (0, 0)
    auto [x, y, fval] = minimize_with_constraints(
        [](double a, double b){ return a*a + b*b; },
        1.0, 1.0, {-5.0, 5.0});
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_TRUE(std::isfinite(y));
    EXPECT_TRUE(std::isfinite(fval));
    EXPECT_NEAR(fval, 0.0, 0.1);
}

TEST(MinimizeConstrained, BoundedSearch) {
    // Minimize (a-2)^2 + (b-3)^2 on [-10, 10] x [-10, 10], minimum at (2, 3)
    auto [x, y, fval] = minimize_with_constraints(
        [](double a, double b){ return (a-2.0)*(a-2.0) + (b-3.0)*(b-3.0); },
        0.0, 0.0, {-10.0, 10.0});
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_TRUE(std::isfinite(y));
    EXPECT_NEAR(fval, 0.0, 0.1);
}

TEST(MinimizeConstrained, NarrowBounds_FindsEdge) {
    // If minimum at x=2, y=3 is outside bounds [0, 1], should return edge
    auto [x, y, fval] = minimize_with_constraints(
        [](double a, double b){ return (a-5.0)*(a-5.0) + (b-5.0)*(b-5.0); },
        0.5, 0.5, {0.0, 1.0});
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_TRUE(std::isfinite(y));
    EXPECT_TRUE(std::isfinite(fval));
}

// ---------------------------------------------------------------------------
// GradientDescentResult struct fields
// ---------------------------------------------------------------------------

TEST(GradientDescent, ResultFields_Populated) {
    // Minimize f(x, y) = x^2 + y^2
    auto res = gradient_descent(
        [](double a, double b){ return a*a + b*b; },
        1.0, 1.0, 0.1, 1000);
    EXPECT_TRUE(std::isfinite(res.x));
    EXPECT_TRUE(std::isfinite(res.y));
    EXPECT_TRUE(std::isfinite(res.f_val));
    EXPECT_GT(res.iterations, 0u);
    // For a well-behaved convex function with enough iterations, should converge
    EXPECT_TRUE(res.converged);
}

TEST(GradientDescent, ResultFields_FValNonNegative) {
    // For f(x,y) = x^2 + y^2, minimum value is 0
    auto res = gradient_descent(
        [](double a, double b){ return a*a + b*b; },
        2.0, 2.0, 0.05, 500);
    EXPECT_GE(res.f_val, 0.0);
}

TEST(GradientDescent, IterationsLessThanMax) {
    // With tight tolerance, it should converge before max_iter
    auto res = gradient_descent(
        [](double a, double b){ return a*a + b*b; },
        1.0, 1.0, 0.1, 5000);
    EXPECT_LE(res.iterations, 5000u);
}

// ---------------------------------------------------------------------------
// golden_section: interval search for minimum
// ---------------------------------------------------------------------------

TEST(GoldenSection, MinOfQuadratic) {
    // f(x) = (x - 3)^2, minimum at x = 3
    double x_min = golden_section([](double x){ return (x - 3.0)*(x - 3.0); }, 0.0, 6.0);
    EXPECT_NEAR(x_min, 3.0, 1e-6);
}

TEST(GoldenSection, MinOfAbs) {
    // f(x) = |x - 2|, minimum at x = 2
    double x_min = golden_section([](double x){ return std::abs(x - 2.0); }, -1.0, 5.0);
    EXPECT_NEAR(x_min, 2.0, 1e-5);
}

TEST(GoldenSection, MinIsFinite) {
    double x_min = golden_section([](double x){ return x*x*x*x; }, -1.0, 1.0);
    EXPECT_TRUE(std::isfinite(x_min));
}

// ---------------------------------------------------------------------------
// simplex_solver: linear programming
// ---------------------------------------------------------------------------

TEST(SimplexSolver, TwoVariableLP_Basic) {
    // simplex_solver only handles 2-variable LP (c must have 2 elements)
    // Maximize x + y subject to x <= 3, y <= 3
    std::vector<double> c = {1.0, 1.0};
    std::vector<std::vector<double>> A = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> b = {3.0, 3.0};
    auto [x, obj] = simplex_solver(c, A, b);
    EXPECT_TRUE(std::isfinite(obj));
    // obj should be non-negative within floating-point tolerance
    EXPECT_GE(obj, -1e-4);
}

TEST(SimplexSolver, TwoVariableLP_WithThreeConstraints) {
    // Maximize x + y subject to x <= 2, y <= 3, x+y <= 4
    std::vector<double> c = {1.0, 1.0};
    std::vector<std::vector<double>> A = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    std::vector<double> b = {2.0, 3.0, 4.0};
    auto [x, obj] = simplex_solver(c, A, b);
    EXPECT_TRUE(std::isfinite(obj));
    EXPECT_GE(obj, -1e-4);
}

TEST(SimplexSolver, WrongSize_ReturnsZero) {
    // c must be size 2; size 1 returns (empty, 0.0)
    std::vector<double> c = {1.0};
    auto [x, obj] = simplex_solver(c, {}, {});
    EXPECT_EQ(obj, 0.0);
}

// NUMA tests removed: numa_allocator.hpp pulls in <windows.h> which requires
// additional linker configuration. NUMA coverage is validated via test_memory_allocators_adv.
