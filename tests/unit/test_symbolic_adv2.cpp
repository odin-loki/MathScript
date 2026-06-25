// MathScript: Advanced Symbolic Math Tests (Wave 46)
// Tests for sym_eval, sym_diff, sym_simplify, sym_to_string correctness
// Note: SymExpr is move-only (unique_ptr members), so expressions must be moved.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <string>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// sym_eval: evaluate expressions
// ---------------------------------------------------------------------------

TEST(SymbolicAdv2, Eval_Constant) {
    EXPECT_NEAR(sym_eval(sym_const(3.14), {}), 3.14, 1e-10);
}

TEST(SymbolicAdv2, Eval_Variable) {
    EXPECT_NEAR(sym_eval(sym_var("x"), {{"x", 2.5}}), 2.5, 1e-10);
}

TEST(SymbolicAdv2, Eval_Add) {
    auto expr = sym_add(sym_var("x"), sym_const(3.0));
    EXPECT_NEAR(sym_eval(expr, {{"x", 2.0}}), 5.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Mul) {
    auto expr = sym_mul(sym_var("x"), sym_var("y"));
    EXPECT_NEAR(sym_eval(expr, {{"x", 3.0}, {"y", 4.0}}), 12.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Sin_AtZero) {
    EXPECT_NEAR(sym_eval(sym_sin(sym_const(0.0)), {}), 0.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Cos_AtZero) {
    EXPECT_NEAR(sym_eval(sym_cos(sym_const(0.0)), {}), 1.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Exp_AtZero) {
    EXPECT_NEAR(sym_eval(sym_exp(sym_const(0.0)), {}), 1.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Log_AtOne) {
    EXPECT_NEAR(sym_eval(sym_log(sym_const(1.0)), {}), 0.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Pow) {
    // 2^3 = 8
    EXPECT_NEAR(sym_eval(sym_pow(sym_const(2.0), sym_const(3.0)), {}), 8.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Nested_Expression) {
    // x^2 + 3*x + 2 at x=1 => 6
    auto expr = sym_add(
        sym_add(
            sym_pow(sym_var("x"), sym_const(2.0)),
            sym_mul(sym_const(3.0), sym_var("x"))),
        sym_const(2.0));
    EXPECT_NEAR(sym_eval(expr, {{"x", 1.0}}), 6.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_TwoVariables) {
    // a*x + b*y at (a=1, x=2, b=2, y=3) => 8
    auto expr = sym_add(
        sym_mul(sym_var("a"), sym_var("x")),
        sym_mul(sym_var("b"), sym_var("y")));
    std::map<std::string, double> env = {{"a", 1.0}, {"x", 2.0}, {"b", 2.0}, {"y", 3.0}};
    EXPECT_NEAR(sym_eval(expr, env), 8.0, 1e-10);
}

TEST(SymbolicAdv2, Eval_Sin_AtPiOver2) {
    EXPECT_NEAR(sym_eval(sym_sin(sym_const(M_PI / 2.0)), {}), 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// sym_diff: symbolic differentiation
// ---------------------------------------------------------------------------

TEST(SymbolicAdv2, Diff_Const_IsZero) {
    auto dexpr = sym_diff(sym_const(5.0), "x");
    EXPECT_NEAR(sym_eval(dexpr, {{"x", 1.0}}), 0.0, 1e-10);
}

TEST(SymbolicAdv2, Diff_Var_IsOne) {
    auto dexpr = sym_diff(sym_var("x"), "x");
    EXPECT_NEAR(sym_eval(dexpr, {{"x", 3.0}}), 1.0, 1e-10);
}

TEST(SymbolicAdv2, Diff_DiffVar_IsZero) {
    auto dexpr = sym_diff(sym_var("x"), "y");
    EXPECT_NEAR(sym_eval(dexpr, {{"x", 3.0}, {"y", 2.0}}), 0.0, 1e-10);
}

TEST(SymbolicAdv2, Diff_Add_SumRule) {
    // d/dx(x + 3) = 1
    auto dexpr = sym_diff(sym_add(sym_var("x"), sym_const(3.0)), "x");
    EXPECT_NEAR(sym_eval(dexpr, {{"x", 5.0}}), 1.0, 1e-10);
}

TEST(SymbolicAdv2, Diff_Product_ProductRule) {
    // d/dx(x * x) = 2x
    auto dexpr = sym_diff(sym_mul(sym_var("x"), sym_var("x")), "x");
    double x_val = 4.0;
    EXPECT_NEAR(sym_eval(dexpr, {{"x", x_val}}), 2.0 * x_val, 1e-9);
}

TEST(SymbolicAdv2, Diff_Sin_IsCos) {
    // d/dx(sin(x)) = cos(x)
    auto dexpr = sym_diff(sym_sin(sym_var("x")), "x");
    double x_val = 1.0;
    EXPECT_NEAR(sym_eval(dexpr, {{"x", x_val}}), std::cos(x_val), 1e-9);
}

TEST(SymbolicAdv2, Diff_Cos_IsNegSin) {
    // d/dx(cos(x)) = -sin(x)
    auto dexpr = sym_diff(sym_cos(sym_var("x")), "x");
    double x_val = 1.0;
    EXPECT_NEAR(sym_eval(dexpr, {{"x", x_val}}), -std::sin(x_val), 1e-9);
}

TEST(SymbolicAdv2, Diff_Exp_IsExp) {
    // d/dx(exp(x)) = exp(x)
    auto dexpr = sym_diff(sym_exp(sym_var("x")), "x");
    double x_val = 2.0;
    EXPECT_NEAR(sym_eval(dexpr, {{"x", x_val}}), std::exp(x_val), 1e-9);
}

TEST(SymbolicAdv2, Diff_Log_IsReciprocal) {
    // d/dx(log(x)) = 1/x
    auto dexpr = sym_diff(sym_log(sym_var("x")), "x");
    double x_val = 3.0;
    EXPECT_NEAR(sym_eval(dexpr, {{"x", x_val}}), 1.0 / x_val, 1e-9);
}

TEST(SymbolicAdv2, Diff_ChainRule_SinOfLinear) {
    // d/dx(sin(2*x)) = 2*cos(2*x)
    auto dexpr = sym_diff(
        sym_sin(sym_mul(sym_const(2.0), sym_var("x"))), "x");
    double x_val = 1.0;
    EXPECT_NEAR(sym_eval(dexpr, {{"x", x_val}}), 2.0 * std::cos(2.0 * x_val), 1e-9);
}

// Numeric gradient via finite differences using sym_eval
static double finite_diff_eval(double xm, double xp, double h) {
    return (xp - xm) / (2.0 * h);
}

TEST(SymbolicAdv2, Diff_Matches_FiniteDiff_Quadratic) {
    // f(x) = x^3, f'(x) = 3x^2
    double x_val = 2.5, h = 1e-5;
    double fp = sym_eval(sym_pow(sym_var("x"), sym_const(3.0)), {{"x", x_val + h}});
    double fm = sym_eval(sym_pow(sym_var("x"), sym_const(3.0)), {{"x", x_val - h}});
    double fd = finite_diff_eval(fm, fp, h);

    auto df = sym_diff(sym_pow(sym_var("x"), sym_const(3.0)), "x");
    EXPECT_NEAR(sym_eval(df, {{"x", x_val}}), fd, 1e-5);
}

TEST(SymbolicAdv2, Diff_Matches_FiniteDiff_SinCos) {
    // f(x) = sin(x)*cos(x)
    double x_val = 1.3, h = 1e-5;
    double fp = sym_eval(sym_mul(sym_sin(sym_var("x")), sym_cos(sym_var("x"))),
                         {{"x", x_val + h}});
    double fm = sym_eval(sym_mul(sym_sin(sym_var("x")), sym_cos(sym_var("x"))),
                         {{"x", x_val - h}});
    double fd = finite_diff_eval(fm, fp, h);

    auto df = sym_diff(sym_mul(sym_sin(sym_var("x")), sym_cos(sym_var("x"))), "x");
    EXPECT_NEAR(sym_eval(df, {{"x", x_val}}), fd, 1e-5);
}

// ---------------------------------------------------------------------------
// sym_to_string: readable output
// ---------------------------------------------------------------------------

TEST(SymbolicAdv2, ToString_Const_IsNonEmpty) {
    EXPECT_FALSE(sym_to_string(sym_const(3.0)).empty());
}

TEST(SymbolicAdv2, ToString_Var_ContainsName) {
    auto s = sym_to_string(sym_var("theta"));
    EXPECT_NE(s.find("theta"), std::string::npos);
}

TEST(SymbolicAdv2, ToString_NonEmpty_ForAllOps) {
    std::vector<std::string> results = {
        sym_to_string(sym_const(1.0)),
        sym_to_string(sym_var("x")),
        sym_to_string(sym_add(sym_var("x"), sym_const(1.0))),
        sym_to_string(sym_mul(sym_var("x"), sym_const(2.0))),
        sym_to_string(sym_sin(sym_var("x"))),
        sym_to_string(sym_cos(sym_var("x"))),
        sym_to_string(sym_exp(sym_var("x"))),
        sym_to_string(sym_log(sym_var("x"))),
    };
    for (auto& r : results)
        EXPECT_FALSE(r.empty()) << "sym_to_string returned empty string";
}
