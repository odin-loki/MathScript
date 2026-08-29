// MathScript: Symbolic → Numeric Pipeline Integration Tests (Wave 46)
// Tests combining symbolic differentiation with numerical verification
// Note: SymExpr is move-only — re-build expressions for each use.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/symbolic/symbolic.hpp"
#include "ms/stats/stats.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Pipeline 1: Symbolic differentiation matches finite differences
// ---------------------------------------------------------------------------

TEST(SymNumPipeline, Polynomial_DerivativeMatchesFD) {
    // f(x) = x^3, f'(x) = 3x^2
    // Build the expr and derivative separately (move-only)
    for (double x : {-2.0, -1.0, 0.5, 1.0, 2.0}) {
        double h = 1e-5;
        double fp = sym_eval(sym_pow(sym_var("x"), sym_const(3.0)), {{"x", x + h}});
        double fm = sym_eval(sym_pow(sym_var("x"), sym_const(3.0)), {{"x", x - h}});
        double fd = (fp - fm) / (2.0 * h);

        auto df = sym_diff(sym_pow(sym_var("x"), sym_const(3.0)), "x");
        EXPECT_NEAR(sym_eval(df, {{"x", x}}), fd, 1e-5)
            << "Gradient mismatch at x=" << x;
    }
}

TEST(SymNumPipeline, SinExp_DerivativeMatchesFD) {
    // f(x) = sin(x)*exp(x)
    for (double x : {0.0, 0.5, 1.0, 1.5}) {
        double h = 1e-5;
        double fp = sym_eval(sym_mul(sym_sin(sym_var("x")), sym_exp(sym_var("x"))),
                             {{"x", x + h}});
        double fm = sym_eval(sym_mul(sym_sin(sym_var("x")), sym_exp(sym_var("x"))),
                             {{"x", x - h}});
        double fd = (fp - fm) / (2.0 * h);

        auto df = sym_diff(sym_mul(sym_sin(sym_var("x")), sym_exp(sym_var("x"))), "x");
        EXPECT_NEAR(sym_eval(df, {{"x", x}}), fd, 1e-5)
            << "sin*exp gradient mismatch at x=" << x;
    }
}

TEST(SymNumPipeline, Cos_Derivative_IsNegSin) {
    // f(x) = cos(x), f'(x) = -sin(x)
    for (double x : {0.0, 1.0, 2.0, M_PI / 4.0}) {
        auto df = sym_diff(sym_cos(sym_var("x")), "x");
        EXPECT_NEAR(sym_eval(df, {{"x", x}}), -std::sin(x), 1e-9)
            << "cos derivative error at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// Pipeline 2: Symbolic evaluation consistency
// ---------------------------------------------------------------------------

TEST(SymNumPipeline, Quadratic_EvalAtMultiplePoints) {
    // f(x) = (x+1)^2 = x^2 + 2x + 1
    for (double x : {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}) {
        auto f = sym_add(
            sym_add(sym_pow(sym_var("x"), sym_const(2.0)),
                    sym_mul(sym_const(2.0), sym_var("x"))),
            sym_const(1.0));
        double expected = (x + 1.0) * (x + 1.0);
        EXPECT_NEAR(sym_eval(f, {{"x", x}}), expected, 1e-10)
            << "Eval error at x=" << x;
    }
}

TEST(SymNumPipeline, TrigIdentity_Sin2PlusCos2) {
    // sin^2(x) + cos^2(x) = 1
    for (double x : {0.0, 0.5, 1.0, 1.5, 2.0, M_PI}) {
        auto f = sym_add(
            sym_pow(sym_sin(sym_var("x")), sym_const(2.0)),
            sym_pow(sym_cos(sym_var("x")), sym_const(2.0)));
        EXPECT_NEAR(sym_eval(f, {{"x", x}}), 1.0, 1e-9)
            << "sin^2 + cos^2 != 1 at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// Pipeline 3: Gradient descent guided by symbolic gradient
// ---------------------------------------------------------------------------

TEST(SymNumPipeline, GradientDescent_QuadraticMinimum) {
    // f(x) = (x-3)^2, gradient = 2*(x-3), minimum at x=3
    double x = 0.0;
    double lr = 0.1;
    for (int i = 0; i < 60; ++i) {
        auto df = sym_diff(
            sym_pow(sym_add(sym_var("x"), sym_const(-3.0)), sym_const(2.0)), "x");
        double grad = sym_eval(df, {{"x", x}});
        x -= lr * grad;
    }
    EXPECT_NEAR(x, 3.0, 0.01) << "Gradient descent should converge to x=3";
}

TEST(SymNumPipeline, GradientDescent_CosineMinimum) {
    // f(x) = -cos(x), minimum at x=0, gradient = sin(x)
    double x = 0.3;
    double lr = 0.1;
    for (int i = 0; i < 100; ++i) {
        auto df = sym_diff(sym_mul(sym_const(-1.0), sym_cos(sym_var("x"))), "x");
        double grad = sym_eval(df, {{"x", x}});
        x -= lr * grad;
    }
    EXPECT_NEAR(x, 0.0, 0.05) << "Gradient descent on -cos should converge to 0";
}

// ---------------------------------------------------------------------------
// Pipeline 4: Symbolic expressions + statistics
// ---------------------------------------------------------------------------

TEST(SymNumPipeline, SquareFunction_StatsMeanAndVar) {
    // f(x) = x^2 at x = 1..5 → values = {1, 4, 9, 16, 25}, mean = 11
    std::vector<double> values;
    for (double x = 1.0; x <= 5.0; x += 1.0) {
        values.push_back(sym_eval(sym_pow(sym_var("x"), sym_const(2.0)), {{"x", x}}));
    }
    EXPECT_NEAR(mean(values), 11.0, 1e-10);
    EXPECT_GT(var(values), 0.0);
}

TEST(SymNumPipeline, Derivative_IsZeroAtMinimum) {
    // f(x) = x^2, df/dx = 2x. At x=0, derivative = 0
    auto df = sym_diff(sym_pow(sym_var("x"), sym_const(2.0)), "x");
    EXPECT_NEAR(sym_eval(df, {{"x", 0.0}}), 0.0, 1e-10);
}

TEST(SymNumPipeline, SecondDerivative_IsPositiveAtMinimum) {
    // d2/dx2(x^2) = 2 > 0 (confirms minimum)
    auto df = sym_diff(sym_pow(sym_var("x"), sym_const(2.0)), "x");
    auto d2f = sym_diff(std::move(df), "x");
    EXPECT_GT(sym_eval(d2f, {{"x", 0.0}}), 0.0);
}

TEST(SymNumPipeline, ExponentialGrowth_Values_Increasing) {
    // f(x) = exp(x) at x=0,1,2,3 should be increasing
    std::vector<double> vals;
    for (double x = 0.0; x <= 3.0; x += 1.0)
        vals.push_back(sym_eval(sym_exp(sym_var("x")), {{"x", x}}));
    for (size_t i = 1; i < vals.size(); ++i)
        EXPECT_GT(vals[i], vals[i-1]) << "exp should be monotone increasing";
}
