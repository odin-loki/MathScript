#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <numbers>
#include <string>

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

void expect_mellin_pair(
    const SymExpr& time_expr, const SymExpr& s_expr, const std::map<std::string, double>& s_env) {
    const auto forward = sym_simplify(sym_mellin(time_expr, "t", "s"));
    expect_eval_equivalent(forward, s_expr, s_env);
}

void expect_imellin_pair(
    const SymExpr& s_expr, const SymExpr& time_expr, const std::map<std::string, double>& t_env) {
    const auto inverse = sym_simplify(sym_imellin(s_expr, "s", "t"));
    expect_eval_equivalent(inverse, time_expr, t_env);
}

SymExpr clone_expr(const SymExpr& expr) {
    SymExpr copy;
    copy.op = expr.op;
    copy.value = expr.value;
    copy.name = expr.name;
    if (expr.left) {
        copy.left = std::make_unique<SymExpr>(clone_expr(*expr.left));
    }
    if (expr.right) {
        copy.right = std::make_unique<SymExpr>(clone_expr(*expr.right));
    }
    return copy;
}

bool is_deriv_sentinel(const SymExpr& original, const SymExpr& result, const std::string& var) {
    const SymExpr expected = sym_deriv(clone_expr(original), var);
    return sym_to_string(result) == sym_to_string(expected);
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

TEST(SymbolicTransformsTest, unsupported_laplace_returns_deriv_sentinel) {
    const auto unsupported_laplace = sym_laplace(sym_log(sym_var("t")), "t", "s");
    EXPECT_EQ(unsupported_laplace.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_laplace.name, "t");

    const auto unsupported_ilaplace = sym_ilaplace(sym_log(sym_var("s")), "s", "t");
    EXPECT_EQ(unsupported_ilaplace.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_ilaplace.name, "s");
}

TEST(SymbolicTransformsTest, fourier_exponential_decay) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t"))));
    const SymExpr spectrum = sym_simplify(sym_fourier(time, "t", "omega"));

    EXPECT_NE(sym_to_string(spectrum).find("omega"), std::string::npos);
    const double value = sym_eval(spectrum, {{"omega", 1.0}});
    EXPECT_NEAR(value, 4.0 / (4.0 + 1.0), 1e-9);
}

TEST(SymbolicTransformsTest, ifourier_exponential_decay) {
    const SymExpr spectrum = sym_div(
        sym_const(4.0),
        sym_add(sym_pow(sym_const(2.0), sym_const(2.0)), sym_pow(sym_var("omega"), sym_const(2.0))));
    const SymExpr time = sym_simplify(sym_ifourier(spectrum, "omega", "t"));

    EXPECT_NE(sym_to_string(time).find("exp"), std::string::npos);
    EXPECT_NEAR(sym_eval(time, {{"t", 0.5}}), std::exp(-1.0), 1e-9);
}

TEST(SymbolicTransformsTest, fourier_ifourier_roundtrip_decay) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("t"))));
    const SymExpr roundtrip = sym_simplify(sym_ifourier(sym_fourier(time, "t", "omega"), "omega", "t"));
    EXPECT_EQ(sym_to_string(roundtrip), sym_to_string(time));
}

TEST(SymbolicTransformsTest, fourier_gaussian_pair) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(1.5), sym_pow(sym_var("t"), sym_const(2.0)))));
    const SymExpr spectrum = sym_simplify(sym_fourier(time, "t", "omega"));
    const double expected_scale = std::sqrt(std::numbers::pi / 1.5);
    const double expected = expected_scale * std::exp(-1.0 / (4.0 * 1.5));

    EXPECT_NEAR(sym_eval(spectrum, {{"omega", 1.0}}), expected, 1e-9);
}

TEST(SymbolicTransformsTest, ifourier_gaussian_roundtrip) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_pow(sym_var("t"), sym_const(2.0)))));
    const SymExpr roundtrip = sym_simplify(sym_ifourier(sym_fourier(time, "t", "omega"), "omega", "t"));
    EXPECT_EQ(sym_to_string(roundtrip), sym_to_string(time));
}

TEST(SymbolicTransformsTest, fourier_unsupported_returns_deriv_sentinel) {
    const SymExpr expr = sym_sin(sym_var("t"));
    const SymExpr result = sym_fourier(expr, "t", "omega");
    EXPECT_TRUE(is_deriv_sentinel(expr, result, "t"));
}

TEST(SymbolicTransformsTest, ztransform_geometric_sequence) {
    const SymExpr seq = sym_pow(sym_const(0.5), sym_var("n"));
    const SymExpr zdomain = sym_simplify(sym_ztransform(seq, "n", "z"));

    EXPECT_NE(sym_to_string(zdomain).find("z"), std::string::npos);
    const double value = sym_eval(zdomain, {{"z", 2.0}});
    EXPECT_NEAR(value, 2.0 / (2.0 - 0.5), 1e-9);
}

TEST(SymbolicTransformsTest, ztransform_scaled_geometric_sequence) {
    const SymExpr seq = sym_mul(sym_const(3.0), sym_pow(sym_const(0.25), sym_var("n")));
    const SymExpr zdomain = sym_simplify(sym_ztransform(seq, "n", "z"));
    const double value = sym_eval(zdomain, {{"z", 1.0}});
    EXPECT_NEAR(value, 3.0 * 1.0 / (1.0 - 0.25), 1e-9);
}

TEST(SymbolicTransformsTest, iztransform_geometric_sequence) {
    const SymExpr zdomain = sym_div(sym_var("z"), sym_sub(sym_var("z"), sym_const(0.5)));
    const SymExpr seq = sym_simplify(sym_iztransform(zdomain, "z", "n"));

    EXPECT_NE(sym_to_string(seq).find("n"), std::string::npos);
    EXPECT_NEAR(sym_eval(seq, {{"n", 2.0}}), std::pow(0.5, 2.0), 1e-9);
}

TEST(SymbolicTransformsTest, ztransform_iztransform_roundtrip) {
    const SymExpr seq = sym_pow(sym_const(0.75), sym_var("n"));
    const SymExpr roundtrip = sym_simplify(sym_iztransform(sym_ztransform(seq, "n", "z"), "z", "n"));
    EXPECT_EQ(sym_to_string(roundtrip), sym_to_string(seq));
}

TEST(SymbolicTransformsTest, ztransform_constant_sequence) {
    const SymExpr seq = sym_const(4.0);
    const SymExpr zdomain = sym_simplify(sym_ztransform(seq, "n", "z"));
    const double value = sym_eval(zdomain, {{"z", 3.0}});
    EXPECT_NEAR(value, 4.0 * 3.0 / (3.0 - 1.0), 1e-9);
}

TEST(SymbolicTransformsTest, iztransform_constant_sequence) {
    const SymExpr zdomain = sym_div(sym_var("z"), sym_sub(sym_var("z"), sym_const(1.0)));
    const SymExpr seq = sym_simplify(sym_iztransform(zdomain, "z", "n"));
    EXPECT_NEAR(sym_eval(seq, {{"n", 5.0}}), 1.0, 1e-9);
}

TEST(SymbolicTransformsTest, mellin_constant) {
    const auto time_expr = sym_const(5.0);
    const auto expected = sym_div(sym_const(5.0), sym_var("s"));
    expect_mellin_pair(time_expr, expected, {{"s", 2.0}});
}

TEST(SymbolicTransformsTest, mellin_power_of_t) {
    const auto time_expr = sym_pow(sym_var("t"), sym_const(2.0));
    const auto expected = sym_div(sym_const(1.0), sym_add(sym_var("s"), sym_const(2.0)));
    expect_mellin_pair(time_expr, expected, {{"s", 1.0}});
}

TEST(SymbolicTransformsTest, mellin_exponential_decay) {
    const auto time_expr = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t"))));
    const auto expected = sym_div(sym_const(1.0), sym_pow(sym_const(2.0), sym_var("s")));
    expect_mellin_pair(time_expr, expected, {{"s", 1.0}});
}

TEST(SymbolicTransformsTest, mellin_power_exponential) {
    const auto time_expr = sym_mul(
        sym_pow(sym_var("t"), sym_const(2.0)),
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("t")))));
    const auto expected = sym_div(
        sym_const(2.0),
        sym_pow(sym_const(3.0), sym_add(sym_var("s"), sym_const(2.0))));
    expect_mellin_pair(time_expr, expected, {{"s", 1.0}});
}

TEST(SymbolicTransformsTest, mellin_one_over_one_plus_t) {
    const auto time_expr = sym_div(sym_const(1.0), sym_add(sym_const(1.0), sym_var("t")));
    const auto expected = sym_div(
        sym_const(std::numbers::pi),
        sym_sin(sym_mul(sym_const(std::numbers::pi), sym_var("s"))));
    expect_mellin_pair(time_expr, expected, {{"s", 0.5}});
}

TEST(SymbolicTransformsTest, imellin_constant_over_s) {
    const auto s_expr = sym_div(sym_const(4.0), sym_var("s"));
    const auto expected = sym_const(4.0);
    expect_imellin_pair(s_expr, expected, {{"t", 1.0}});
}

TEST(SymbolicTransformsTest, imellin_power_form) {
    const auto s_expr = sym_div(sym_const(1.0), sym_add(sym_var("s"), sym_const(1.5)));
    const auto expected = sym_pow(sym_var("t"), sym_const(1.5));
    expect_imellin_pair(s_expr, expected, {{"t", 2.0}});
}

TEST(SymbolicTransformsTest, mellin_imellin_roundtrip) {
    const auto original = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t"))));
    const auto in_s = sym_simplify(sym_mellin(original, "t", "s"));
    const auto recovered = sym_simplify(sym_imellin(in_s, "s", "t"));
    expect_eval_equivalent(original, recovered, {{"t", 0.75}});

    const auto power = sym_pow(sym_var("t"), sym_const(1.0));
    const auto power_s = sym_simplify(sym_mellin(power, "t", "s"));
    const auto power_back = sym_simplify(sym_imellin(power_s, "s", "t"));
    expect_eval_equivalent(power, power_back, {{"t", 1.25}});
}

TEST(SymbolicTransformsTest, unsupported_mellin_returns_deriv_sentinel) {
    const auto unsupported_mellin = sym_mellin(sym_log(sym_var("t")), "t", "s");
    EXPECT_EQ(unsupported_mellin.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_mellin.name, "t");

    const auto unsupported_imellin = sym_imellin(sym_log(sym_var("s")), "s", "t");
    EXPECT_EQ(unsupported_imellin.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_imellin.name, "s");
}
