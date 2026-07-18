#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <numbers>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

namespace {

void expect_eval_equivalent(
    const SymExpr& original, const SymExpr& transformed, const std::map<std::string, double>& env,
    double tol = 1e-9) {
    EXPECT_NEAR(sym_eval(original, env), sym_eval(transformed, env), tol);
}

void expect_laplace_pair(
    const SymExpr& time_expr, const SymExpr& s_expr, const std::map<std::string, double>& s_env) {
    const auto forward = sym_simplify(sym_laplace(time_expr, "t", "s"));
    expect_eval_equivalent(forward, s_expr, s_env);
}

void expect_ilaplace_pair(
    const SymExpr& s_expr, const SymExpr& time_expr, const std::map<std::string, double>& t_env) {
    const auto inverse = sym_simplify(sym_ilaplace(s_expr, "s", "t"));
    expect_eval_equivalent(inverse, time_expr, t_env);
}

} // namespace

TEST(SymbolicTransformsTest, laplace_constant) {
    const auto time_expr = sym_const(3.0);
    const auto expected = sym_div(sym_const(3.0), sym_var("s"));
    expect_laplace_pair(time_expr, expected, {{"s", 2.0}});
}

TEST(SymbolicTransformsTest, laplace_exponential) {
    const auto time_expr = sym_exp(sym_mul(sym_const(2.0), sym_var("t")));
    const auto expected = sym_div(sym_const(1.0), sym_sub(sym_var("s"), sym_const(2.0)));
    expect_laplace_pair(time_expr, expected, {{"s", 5.0}});
}

TEST(SymbolicTransformsTest, laplace_sine) {
    const auto time_expr = sym_sin(sym_mul(sym_const(3.0), sym_var("t")));
    const auto expected = sym_div(
        sym_const(3.0),
        sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_pow(sym_const(3.0), sym_const(2.0))));
    expect_laplace_pair(time_expr, expected, {{"s", 4.0}});
}

TEST(SymbolicTransformsTest, laplace_cosine) {
    const auto time_expr = sym_cos(sym_mul(sym_const(2.0), sym_var("t")));
    const auto expected = sym_div(
        sym_var("s"),
        sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))));
    expect_laplace_pair(time_expr, expected, {{"s", 1.5}});
}

TEST(SymbolicTransformsTest, laplace_power_of_t) {
    const auto time_expr = sym_pow(sym_var("t"), sym_const(2.0));
    const auto expected = sym_div(sym_const(2.0), sym_pow(sym_var("s"), sym_const(3.0)));
    expect_laplace_pair(time_expr, expected, {{"s", 3.0}});
}

TEST(SymbolicTransformsTest, laplace_unit_ramp) {
    const auto time_expr = sym_var("t");
    const auto expected = sym_div(sym_const(1.0), sym_pow(sym_var("s"), sym_const(2.0)));
    expect_laplace_pair(time_expr, expected, {{"s", 2.0}});
}

TEST(SymbolicTransformsTest, ilaplace_one_over_s_minus_a) {
    const auto s_expr = sym_div(sym_const(1.0), sym_sub(sym_var("s"), sym_const(1.5)));
    const auto expected = sym_exp(sym_mul(sym_const(1.5), sym_var("t")));
    expect_ilaplace_pair(s_expr, expected, {{"t", 0.5}});
}

TEST(SymbolicTransformsTest, ilaplace_one_over_s_squared) {
    const auto s_expr = sym_div(sym_const(1.0), sym_pow(sym_var("s"), sym_const(2.0)));
    const auto expected = sym_var("t");
    expect_ilaplace_pair(s_expr, expected, {{"t", 2.5}});
}

TEST(SymbolicTransformsTest, ilaplace_cosine_form) {
    const auto s_expr = sym_div(
        sym_var("s"),
        sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_pow(sym_const(4.0), sym_const(2.0))));
    const auto expected = sym_cos(sym_mul(sym_const(4.0), sym_var("t")));
    expect_ilaplace_pair(s_expr, expected, {{"t", 0.25}});
}

TEST(SymbolicTransformsTest, ilaplace_sine_form) {
    const auto s_expr = sym_div(
        sym_const(5.0),
        sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_pow(sym_const(5.0), sym_const(2.0))));
    const auto expected = sym_sin(sym_mul(sym_const(5.0), sym_var("t")));
    expect_ilaplace_pair(s_expr, expected, {{"t", 0.1}});
}

TEST(SymbolicTransformsTest, laplace_ilaplace_roundtrip) {
    const auto original = sym_exp(sym_mul(sym_const(-1.0), sym_var("t")));
    const auto in_s = sym_simplify(sym_laplace(original, "t", "s"));
    const auto recovered = sym_simplify(sym_ilaplace(in_s, "s", "t"));
    expect_eval_equivalent(original, recovered, {{"t", 1.25}});

    const auto sin_t = sym_sin(sym_var("t"));
    const auto sin_s = sym_simplify(sym_laplace(sin_t, "t", "s"));
    const auto sin_back = sym_simplify(sym_ilaplace(sin_s, "s", "t"));
    expect_eval_equivalent(sin_t, sin_back, {{"t", std::numbers::pi / 6.0}});
}

TEST(SymbolicTransformsTest, unsupported_returns_deriv_sentinel) {
    const auto unsupported_laplace = sym_laplace(sym_log(sym_var("t")), "t", "s");
    EXPECT_EQ(unsupported_laplace.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_laplace.name, "t");

    const auto unsupported_ilaplace = sym_ilaplace(sym_log(sym_var("s")), "s", "t");
    EXPECT_EQ(unsupported_ilaplace.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_ilaplace.name, "s");
}
