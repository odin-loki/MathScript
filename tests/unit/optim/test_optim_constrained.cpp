// MathScript Advanced Constrained Optimization Tests (Wave 53)
// Tests for minimize_with_constraints and simplex_solver

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>

#include "ms/optim/optim.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// minimize_with_constraints
// ---------------------------------------------------------------------------

TEST(OptimConstrained, MinimizeConstrained_Paraboloid_FindsMin) {
    // f(x,y) = x^2 + y^2 min near origin; constraint: x + y = 0 → min at (0,0)
    auto f = [](double x, double y) { return x * x + y * y; };
    std::vector<double> constraints = {0.0, 0.0};
    auto [xmin, ymin, fmin] = minimize_with_constraints(f, 2.0, 2.0, constraints);
    EXPECT_TRUE(std::isfinite(xmin));
    EXPECT_TRUE(std::isfinite(ymin));
    EXPECT_TRUE(std::isfinite(fmin));
    EXPECT_LT(fmin, 8.0) << "Constrained minimization should reduce f from (2,2)";
}

TEST(OptimConstrained, MinimizeConstrained_FvalLessThanStart) {
    auto f = [](double x, double y) { return (x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0); };
    std::vector<double> constraints = {0.0, 0.0};
    auto [xmin, ymin, fmin] = minimize_with_constraints(f, 5.0, 5.0, constraints);
    double f_start = f(5.0, 5.0);
    EXPECT_LT(fmin, f_start) << "Constrained minimize should reduce f value";
}

TEST(OptimConstrained, MinimizeConstrained_AllFinite) {
    auto f = [](double x, double y) { return x * x + y * y + x * y; };
    std::vector<double> constraints = {-1.0, 1.0};
    auto [xmin, ymin, fmin] = minimize_with_constraints(f, 0.0, 0.0, constraints);
    EXPECT_TRUE(std::isfinite(xmin));
    EXPECT_TRUE(std::isfinite(ymin));
    EXPECT_TRUE(std::isfinite(fmin));
}

TEST(OptimConstrained, MinimizeConstrained_AtMinimum_Stays) {
    // At true minimum (0,0), should stay there
    auto f = [](double x, double y) { return x * x + y * y; };
    std::vector<double> constraints = {0.0, 0.0};
    auto [xmin, ymin, fmin] = minimize_with_constraints(f, 0.0, 0.0, constraints);
    EXPECT_NEAR(fmin, 0.0, 0.1) << "Starting at minimum should give ~0";
}

// ---------------------------------------------------------------------------
// simplex_solver (Linear Programming)
// ---------------------------------------------------------------------------

TEST(OptimConstrained, Simplex_SimpleLP_TwoVars) {
    // maximize c^T x s.t. Ax <= b, x >= 0
    // Example: max x1 + x2 s.t. x1 + x2 <= 4, 2x1 + x2 <= 6
    // Optimal: x1=2, x2=2, obj=4
    // For simplex_solver (minimization), negate: min -x1 - x2
    std::vector<double> c = {-1.0, -1.0};
    std::vector<std::vector<double>> A = {{1.0, 1.0}, {2.0, 1.0}};
    std::vector<double> b_vec = {4.0, 6.0};
    auto [x, obj] = simplex_solver(c, A, b_vec);
    EXPECT_FALSE(x.empty()) << "simplex_solver should return a solution";
    EXPECT_TRUE(std::isfinite(obj)) << "objective should be finite";
}

TEST(OptimConstrained, Simplex_Feasible_TwoVars_ObjectiveFinite) {
    // min -2x1 - x2 s.t. x1 + x2 <= 4, x1 <= 3
    std::vector<double> c = {-2.0, -1.0};
    std::vector<std::vector<double>> A = {{1.0, 1.0}, {1.0, 0.0}};
    std::vector<double> b_vec = {4.0, 3.0};
    auto [x, obj] = simplex_solver(c, A, b_vec);
    EXPECT_TRUE(std::isfinite(obj));
    EXPECT_FALSE(x.empty());
}

TEST(OptimConstrained, Simplex_Solution_SatisfiesConstraints) {
    // min -2x1 - x2 s.t. x1 <= 4, x2 <= 3, x1 + x2 <= 5
    std::vector<double> c = {-2.0, -1.0};
    std::vector<std::vector<double>> A = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    std::vector<double> b_vec = {4.0, 3.0, 5.0};
    auto [x, obj] = simplex_solver(c, A, b_vec);
    ASSERT_GE(x.size(), 2u);
    // x1 <= 4, x2 <= 3, x1+x2 <= 5
    EXPECT_LE(x[0], 4.0 + 1e-6) << "x1 should satisfy x1 <= 4";
    EXPECT_LE(x[1], 3.0 + 1e-6) << "x2 should satisfy x2 <= 3";
    EXPECT_LE(x[0] + x[1], 5.0 + 1e-6) << "x1+x2 should satisfy x1+x2 <= 5";
}

TEST(OptimConstrained, Simplex_AllVariablesNonneg) {
    std::vector<double> c = {-3.0, -2.0};
    std::vector<std::vector<double>> A = {{1.0, 1.0}, {2.0, 1.0}};
    std::vector<double> b_vec = {4.0, 6.0};
    auto [x, obj] = simplex_solver(c, A, b_vec);
    ASSERT_GE(x.size(), 2u);
    EXPECT_GE(x[0], -1e-8) << "x1 should be nonneg";
    EXPECT_GE(x[1], -1e-8) << "x2 should be nonneg";
}

// ---------------------------------------------------------------------------
// golden_section: additional cases
// ---------------------------------------------------------------------------

TEST(OptimConstrained, GoldenSection_AbsoluteValue_FindsZero) {
    // min |x| on [-1, 2] → x=0
    auto f = [](double x) { return std::abs(x); };
    double xmin = golden_section(f, -1.0, 2.0);
    EXPECT_NEAR(xmin, 0.0, 0.01);
}

TEST(OptimConstrained, GoldenSection_Cubic_FindsLocalMin) {
    // f(x) = x^3 - 3x on [0, 3] → local min at x=1 (f'(x) = 3x^2-3=0 → x=1)
    auto f = [](double x) { return x * x * x - 3.0 * x; };
    double xmin = golden_section(f, 0.0, 3.0);
    // Local min at x=1 in this interval
    EXPECT_NEAR(xmin, 1.0, 0.05);
}

TEST(OptimConstrained, GoldenSection_ReturnsInBounds) {
    auto f = [](double x) { return (x - 2.5) * (x - 2.5); };
    double xmin = golden_section(f, 0.0, 5.0);
    EXPECT_GE(xmin, 0.0);
    EXPECT_LE(xmin, 5.0);
    EXPECT_NEAR(xmin, 2.5, 0.01);
}

// ---------------------------------------------------------------------------
// newton_1d: additional edge cases
// ---------------------------------------------------------------------------

TEST(OptimConstrained, Newton1D_CubicMin) {
    // f(x) = x^4 - 4x^2, f'(x) = 4x^3 - 8x = 0 → x=0 or x=±sqrt(2)
    // Local min at x=sqrt(2) ≈ 1.414
    auto f = [](double x) { return x * x * x * x - 4.0 * x * x; };
    double xmin = newton_1d(f, 1.0); // start near sqrt(2)
    EXPECT_TRUE(std::isfinite(xmin));
    EXPECT_NEAR(std::abs(xmin), std::sqrt(2.0), 0.1)
        << "newton_1d should converge near sqrt(2)";
}

TEST(OptimConstrained, Newton1D_Finiteoutput) {
    auto f = [](double x) { return std::exp(x) + std::exp(-x); }; // min at x=0
    double xmin = newton_1d(f, 0.5);
    EXPECT_TRUE(std::isfinite(xmin));
    EXPECT_NEAR(xmin, 0.0, 0.01);
}
