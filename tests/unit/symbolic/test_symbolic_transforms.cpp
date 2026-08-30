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

void expect_hankel_pair(
    const SymExpr& r_expr, const SymExpr& k_expr, const std::map<std::string, double>& k_env) {
    const auto forward = sym_simplify(sym_hankel(r_expr, "r", "k"));
    expect_eval_equivalent(forward, k_expr, k_env);
}

void expect_ihankel_pair(
    const SymExpr& k_expr, const SymExpr& r_expr, const std::map<std::string, double>& r_env) {
    const auto inverse = sym_simplify(sym_ihankel(k_expr, "k", "r"));
    expect_eval_equivalent(inverse, r_expr, r_env);
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

TEST(SymbolicTransformsTest, hankel_exponential_decay) {
    const auto r_expr = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r"))));
    const auto expected = sym_div(
        sym_const(2.0),
        sym_pow(
            sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
            sym_const(1.5)));
    expect_hankel_pair(r_expr, expected, {{"k", 1.5}});
}

TEST(SymbolicTransformsTest, hankel_power_exponential) {
    // n=1: r*exp(-a*r) -> scale*a / (a^2+k^2)^((n+3)/2) with (n+3)/2 = 2.
    const auto r_expr = sym_mul(
        sym_var("r"),
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("r")))));
    const double numer = std::pow(2.0, 2) * std::tgamma(2.0) / std::sqrt(std::numbers::pi) * 3.0;
    const auto expected = sym_div(
        sym_const(numer),
        sym_pow(
            sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(3.0), sym_const(2.0))),
            sym_const(2.0)));
    expect_hankel_pair(r_expr, expected, {{"k", 2.0}});

    const auto r_pow_expr = sym_mul(
        sym_pow(sym_var("r"), sym_const(1.0)),
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("r")))));
    expect_hankel_pair(r_pow_expr, expected, {{"k", 2.0}});
}

TEST(SymbolicTransformsTest, hankel_one_over_sqrt_r2_plus_a2) {
    const auto r_expr = sym_div(
        sym_const(1.0),
        sym_sqrt(sym_add(sym_pow(sym_var("r"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0)))));
    const auto expected = sym_div(
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("k")))),
        sym_var("k"));
    expect_hankel_pair(r_expr, expected, {{"k", 2.5}});
}

TEST(SymbolicTransformsTest, ihankel_exponential_form) {
    const auto k_expr = sym_div(
        sym_const(2.0),
        sym_pow(
            sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
            sym_const(1.5)));
    const auto expected = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r"))));
    expect_ihankel_pair(k_expr, expected, {{"r", 1.0}});
}

TEST(SymbolicTransformsTest, ihankel_sqrt_kernel_form) {
    const auto k_expr = sym_div(
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("k")))),
        sym_var("k"));
    const auto expected = sym_div(
        sym_const(1.0),
        sym_sqrt(sym_add(sym_pow(sym_var("r"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0)))));
    expect_ihankel_pair(k_expr, expected, {{"r", 3.0}});
}

TEST(SymbolicTransformsTest, hankel_linearity_with_constant_factor) {
    const auto r_expr = sym_mul(
        sym_const(3.0),
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r")))));
    const auto expected = sym_mul(
        sym_const(3.0),
        sym_div(
            sym_const(2.0),
            sym_pow(
                sym_add(
                    sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
                sym_const(1.5))));
    expect_hankel_pair(r_expr, expected, {{"k", 1.0}});
}

TEST(SymbolicTransformsTest, hankel_ihankel_roundtrip) {
    const auto original = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r"))));
    const auto in_k = sym_simplify(sym_hankel(original, "r", "k"));
    const auto recovered = sym_simplify(sym_ihankel(in_k, "k", "r"));
    expect_eval_equivalent(original, recovered, {{"r", 0.75}});

    const auto sqrt_form = sym_div(
        sym_const(1.0),
        sym_sqrt(sym_add(sym_pow(sym_var("r"), sym_const(2.0)), sym_pow(sym_const(3.0), sym_const(2.0)))));
    const auto sqrt_k = sym_simplify(sym_hankel(sqrt_form, "r", "k"));
    const auto sqrt_back = sym_simplify(sym_ihankel(sqrt_k, "k", "r"));
    expect_eval_equivalent(sqrt_form, sqrt_back, {{"r", 2.0}});
}

TEST(SymbolicTransformsTest, unsupported_hankel_returns_deriv_sentinel) {
    const auto unsupported_hankel = sym_hankel(sym_log(sym_var("r")), "r", "k");
    EXPECT_EQ(unsupported_hankel.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_hankel.name, "r");

    const auto unsupported_ihankel = sym_ihankel(sym_log(sym_var("k")), "k", "r");
    EXPECT_EQ(unsupported_ihankel.op, SymOp::Deriv);
    EXPECT_EQ(unsupported_ihankel.name, "k");
}

TEST(SymbolicTransformsTest, laplace_add_sub_neg_and_right_const) {
    const auto time_expr = sym_add(
        sym_sub(sym_neg(sym_var("t")), sym_const(1.0)),
        sym_mul(sym_var("t"), sym_const(2.0)));
    const auto forward = sym_simplify(sym_laplace(time_expr, "t", "s"));
    EXPECT_NE(forward.op, SymOp::Deriv);
    EXPECT_NEAR(sym_eval(forward, {{"s", 2.0}}), -0.25 - 0.5 + 0.5, 1e-9);
}

TEST(SymbolicTransformsTest, laplace_scaled_var_const_on_right) {
    const auto time_expr = sym_sin(sym_mul(sym_var("t"), sym_const(3.0)));
    const auto expected = sym_div(
        sym_const(3.0),
        sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_pow(sym_const(3.0), sym_const(2.0))));
    expect_laplace_pair(time_expr, expected, {{"s", 4.0}});
}

TEST(SymbolicTransformsTest, laplace_other_var_and_large_power_sentinel) {
    const auto other = sym_laplace(sym_var("y"), "t", "s");
    EXPECT_TRUE(is_deriv_sentinel(sym_var("y"), other, "t"));

    const auto large = sym_pow(sym_var("t"), sym_const(9.0));
    EXPECT_TRUE(is_deriv_sentinel(large, sym_laplace(large, "t", "s"), "t"));
}

TEST(SymbolicTransformsTest, ilaplace_zero_and_nonzero_const) {
    EXPECT_NEAR(sym_eval(sym_ilaplace(sym_const(0.0), "s", "t"), {}), 0.0, 1e-12);

    const auto nonzero = sym_ilaplace(sym_const(3.0), "s", "t");
    EXPECT_TRUE(is_deriv_sentinel(sym_const(3.0), nonzero, "s"));
}

TEST(SymbolicTransformsTest, ilaplace_factorial_over_s_power) {
    const auto s_expr = sym_div(sym_const(6.0), sym_pow(sym_var("s"), sym_const(4.0)));
    const auto expected = sym_pow(sym_var("t"), sym_const(3.0));
    expect_ilaplace_pair(s_expr, expected, {{"t", 1.5}});
}

TEST(SymbolicTransformsTest, ilaplace_s2_plus_const_sine) {
    const auto s_expr = sym_div(
        sym_const(3.0),
        sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_const(9.0)));
    const auto expected = sym_sin(sym_mul(sym_const(3.0), sym_var("t")));
    expect_ilaplace_pair(s_expr, expected, {{"t", 0.2}});
}

TEST(SymbolicTransformsTest, ilaplace_linearity) {
    const auto s_expr = sym_add(
        sym_neg(sym_div(sym_const(1.0), sym_pow(sym_var("s"), sym_const(2.0)))),
        sym_mul(sym_const(2.0), sym_div(sym_const(1.0), sym_sub(sym_var("s"), sym_const(1.0)))));
    const auto inverse = sym_simplify(sym_ilaplace(s_expr, "s", "t"));
    EXPECT_NEAR(sym_eval(inverse, {{"t", 0.5}}), -0.5 + 2.0 * std::exp(0.5), 1e-9);
}

TEST(SymbolicTransformsTest, fourier_bare_neg_t) {
    const SymExpr time = sym_exp(sym_neg(sym_var("t")));
    const SymExpr spectrum = sym_simplify(sym_fourier(time, "t", "omega"));
    EXPECT_NEAR(sym_eval(spectrum, {{"omega", 1.0}}), 2.0 / 2.0, 1e-9);
}

TEST(SymbolicTransformsTest, fourier_sub_zero_minus_scaled) {
    const SymExpr time = sym_exp(sym_sub(sym_const(0.0), sym_mul(sym_const(2.0), sym_var("t"))));
    const SymExpr spectrum = sym_simplify(sym_fourier(time, "t", "omega"));
    EXPECT_NEAR(sym_eval(spectrum, {{"omega", 1.0}}), 4.0 / 5.0, 1e-9);
}

TEST(SymbolicTransformsTest, ifourier_unsupported_and_const_a2) {
    const SymExpr bad = sym_sin(sym_var("omega"));
    EXPECT_TRUE(is_deriv_sentinel(bad, sym_ifourier(bad, "omega", "t"), "omega"));

    const SymExpr spectrum = sym_div(
        sym_const(4.0),
        sym_add(sym_const(4.0), sym_pow(sym_var("omega"), sym_const(2.0))));
    const SymExpr time = sym_simplify(sym_ifourier(spectrum, "omega", "t"));
    EXPECT_NEAR(sym_eval(time, {{"t", 0.5}}), std::exp(-1.0), 1e-9);
}

TEST(SymbolicTransformsTest, ztransform_const_on_right_of_geometric) {
    const SymExpr seq = sym_mul(sym_pow(sym_const(0.5), sym_var("n")), sym_const(3.0));
    const SymExpr zdomain = sym_simplify(sym_ztransform(seq, "n", "z"));
    EXPECT_NEAR(sym_eval(zdomain, {{"z", 2.0}}), 3.0 * 2.0 / (2.0 - 0.5), 1e-9);
}

TEST(SymbolicTransformsTest, ztransform_and_iztransform_unsupported) {
    const SymExpr seq = sym_sin(sym_var("n"));
    EXPECT_TRUE(is_deriv_sentinel(seq, sym_ztransform(seq, "n", "z"), "n"));

    const SymExpr zdom = sym_log(sym_var("z"));
    EXPECT_TRUE(is_deriv_sentinel(zdom, sym_iztransform(zdom, "z", "n"), "z"));
}

TEST(SymbolicTransformsTest, iztransform_scaled_on_right) {
    const SymExpr zdomain = sym_mul(
        sym_div(sym_var("z"), sym_sub(sym_var("z"), sym_const(0.25))),
        sym_const(2.0));
    const SymExpr seq = sym_simplify(sym_iztransform(zdomain, "z", "n"));
    EXPECT_NEAR(sym_eval(seq, {{"n", 2.0}}), 2.0 * std::pow(0.25, 2.0), 1e-9);
}

TEST(SymbolicTransformsTest, mellin_bare_t_and_swapped_one_plus_t) {
    const auto bare = sym_var("t");
    const auto expected_bare = sym_div(sym_const(1.0), sym_add(sym_var("s"), sym_const(1.0)));
    expect_mellin_pair(bare, expected_bare, {{"s", 2.0}});

    const auto swapped = sym_div(sym_const(1.0), sym_add(sym_var("t"), sym_const(1.0)));
    const auto expected_pi = sym_div(
        sym_const(std::numbers::pi),
        sym_sin(sym_mul(sym_const(std::numbers::pi), sym_var("s"))));
    expect_mellin_pair(swapped, expected_pi, {{"s", 0.5}});
}

TEST(SymbolicTransformsTest, mellin_t_times_exp_and_linearity) {
    const auto time_expr = sym_mul(
        sym_var("t"),
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t")))));
    const auto expected = sym_div(
        sym_const(1.0),
        sym_pow(sym_const(2.0), sym_add(sym_var("s"), sym_const(1.0))));
    expect_mellin_pair(time_expr, expected, {{"s", 1.0}});

    const auto linear = sym_add(sym_neg(sym_const(2.0)), sym_var("t"));
    const auto forward = sym_simplify(sym_mellin(linear, "t", "s"));
    EXPECT_NE(forward.op, SymOp::Deriv);
}

TEST(SymbolicTransformsTest, imellin_zero_and_pi_over_sin) {
    EXPECT_NEAR(sym_eval(sym_imellin(sym_const(0.0), "s", "t"), {}), 0.0, 1e-12);

    const auto s_expr = sym_div(
        sym_const(std::numbers::pi),
        sym_sin(sym_mul(sym_const(std::numbers::pi), sym_var("s"))));
    const auto expected = sym_div(sym_const(1.0), sym_add(sym_const(1.0), sym_var("t")));
    expect_imellin_pair(s_expr, expected, {{"t", 3.0}});
}

TEST(SymbolicTransformsTest, hankel_add_neg_and_const_sentinel) {
    const auto r_expr = sym_add(
        sym_neg(sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r"))))),
        sym_mul(sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r")))), sym_const(3.0)));
    const auto forward = sym_simplify(sym_hankel(r_expr, "r", "k"));
    EXPECT_NE(forward.op, SymOp::Deriv);

    EXPECT_TRUE(is_deriv_sentinel(sym_const(1.0), sym_hankel(sym_const(1.0), "r", "k"), "r"));
    EXPECT_TRUE(is_deriv_sentinel(sym_var("r"), sym_hankel(sym_var("r"), "r", "k"), "r"));
}

TEST(SymbolicTransformsTest, ihankel_zero_and_linearity) {
    EXPECT_NEAR(sym_eval(sym_ihankel(sym_const(0.0), "k", "r"), {}), 0.0, 1e-12);

    const auto k_expr = sym_neg(sym_div(
        sym_const(2.0),
        sym_pow(
            sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
            sym_const(1.5))));
    const auto inverse = sym_simplify(sym_ihankel(k_expr, "k", "r"));
    EXPECT_NEAR(sym_eval(inverse, {{"r", 1.0}}), -std::exp(-2.0), 1e-9);
}

TEST(SymbolicTransformsTest, laplace_sin_minus_neg_t_times_const) {
    const auto time_expr = sym_sub(
        sym_sin(sym_var("t")),
        sym_neg(sym_mul(sym_var("t"), sym_const(2.0))));
    const auto expected = sym_add(
        sym_div(
            sym_const(1.0),
            sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_const(1.0))),
        sym_div(sym_const(2.0), sym_pow(sym_var("s"), sym_const(2.0))));
    expect_laplace_pair(time_expr, expected, {{"s", 2.0}});
}

TEST(SymbolicTransformsTest, laplace_t_times_right_const) {
    const auto time_expr = sym_mul(sym_var("t"), sym_const(3.0));
    const auto expected = sym_div(sym_const(3.0), sym_pow(sym_var("s"), sym_const(2.0)));
    expect_laplace_pair(time_expr, expected, {{"s", 2.0}});
}

TEST(SymbolicTransformsTest, ilaplace_sub_neg_and_right_const) {
    const auto s_expr = sym_sub(
        sym_div(sym_const(1.0), sym_sub(sym_var("s"), sym_const(1.0))),
        sym_neg(sym_div(sym_const(1.0), sym_pow(sym_var("s"), sym_const(2.0)))));
    const auto inverse = sym_simplify(sym_ilaplace(s_expr, "s", "t"));
    EXPECT_NEAR(sym_eval(inverse, {{"t", 0.5}}), std::exp(0.5) + 0.5, 1e-9);

    const auto scaled = sym_mul(
        sym_div(sym_const(1.0), sym_sub(sym_var("s"), sym_const(1.0))),
        sym_const(2.0));
    expect_ilaplace_pair(
        scaled, sym_mul(sym_const(2.0), sym_exp(sym_var("t"))), {{"t", 0.5}});
}

TEST(SymbolicTransformsTest, ilaplace_two_over_s_cubed) {
    const auto s_expr = sym_div(sym_const(2.0), sym_pow(sym_var("s"), sym_const(3.0)));
    const auto expected = sym_pow(sym_var("t"), sym_const(2.0));
    expect_ilaplace_pair(s_expr, expected, {{"t", 1.5}});
}

TEST(SymbolicTransformsTest, ilaplace_swapped_a2_plus_s2) {
    const auto sine = sym_div(
        sym_const(3.0),
        sym_add(sym_pow(sym_const(3.0), sym_const(2.0)), sym_pow(sym_var("s"), sym_const(2.0))));
    expect_ilaplace_pair(sine, sym_sin(sym_mul(sym_const(3.0), sym_var("t"))), {{"t", 0.2}});

    const auto cosine = sym_div(
        sym_var("s"),
        sym_add(sym_pow(sym_const(4.0), sym_const(2.0)), sym_pow(sym_var("s"), sym_const(2.0))));
    expect_ilaplace_pair(cosine, sym_cos(sym_mul(sym_const(4.0), sym_var("t"))), {{"t", 0.25}});
}

TEST(SymbolicTransformsTest, mellin_sub_t_minus_neg_const) {
    const auto time_expr = sym_sub(sym_var("t"), sym_neg(sym_const(1.0)));
    const auto expected = sym_sub(
        sym_div(sym_const(1.0), sym_add(sym_var("s"), sym_const(1.0))),
        sym_neg(sym_div(sym_const(1.0), sym_var("s"))));
    expect_mellin_pair(time_expr, expected, {{"s", 2.0}});
}

TEST(SymbolicTransformsTest, mellin_right_const_and_neg_scale_exp) {
    const auto scaled = sym_mul(
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t")))),
        sym_const(3.0));
    expect_mellin_pair(
        scaled, sym_div(sym_const(3.0), sym_pow(sym_const(2.0), sym_var("s"))), {{"s", 1.0}});

    const auto neg_scale = sym_exp(sym_mul(sym_const(-2.0), sym_var("t")));
    expect_mellin_pair(
        neg_scale, sym_div(sym_const(1.0), sym_pow(sym_const(2.0), sym_var("s"))), {{"s", 1.0}});

    const auto left_const = sym_mul(sym_const(4.0), sym_pow(sym_var("t"), sym_const(2.0)));
    expect_mellin_pair(
        left_const, sym_div(sym_const(4.0), sym_add(sym_var("s"), sym_const(2.0))), {{"s", 1.0}});
}

TEST(SymbolicTransformsTest, mellin_exp_times_t_and_tpow) {
    const auto t_on_right = sym_mul(
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t")))),
        sym_var("t"));
    expect_mellin_pair(
        t_on_right,
        sym_div(sym_const(1.0), sym_pow(sym_const(2.0), sym_add(sym_var("s"), sym_const(1.0)))),
        {{"s", 1.0}});

    const auto tpow_on_right = sym_mul(
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("t")))),
        sym_pow(sym_var("t"), sym_const(2.0)));
    expect_mellin_pair(
        tpow_on_right,
        sym_div(sym_const(2.0), sym_pow(sym_const(3.0), sym_add(sym_var("s"), sym_const(2.0)))),
        {{"s", 1.0}});
}

TEST(SymbolicTransformsTest, imellin_sub_neg_and_right_const) {
    const auto s_expr = sym_sub(
        sym_div(sym_const(1.0), sym_var("s")),
        sym_neg(sym_div(sym_const(1.0), sym_add(sym_var("s"), sym_const(1.0)))));
    const auto inverse = sym_simplify(sym_imellin(s_expr, "s", "t"));
    EXPECT_NEAR(sym_eval(inverse, {{"t", 2.0}}), 1.0 + 2.0, 1e-9);

    const auto scaled = sym_mul(sym_div(sym_const(1.0), sym_var("s")), sym_const(4.0));
    expect_imellin_pair(scaled, sym_const(4.0), {{"t", 1.0}});

    const auto added = sym_add(
        sym_div(sym_const(1.0), sym_var("s")),
        sym_div(sym_const(1.0), sym_add(sym_var("s"), sym_const(1.0))));
    expect_imellin_pair(added, sym_add(sym_const(1.0), sym_var("t")), {{"t", 2.0}});
}

TEST(SymbolicTransformsTest, imellin_factorial_over_a_pow_s_plus_n) {
    const auto s_expr = sym_div(
        sym_const(2.0),
        sym_pow(sym_const(3.0), sym_add(sym_var("s"), sym_const(2.0))));
    const auto expected = sym_mul(
        sym_pow(sym_var("t"), sym_const(2.0)),
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("t")))));
    expect_imellin_pair(s_expr, expected, {{"t", 0.5}});
}

TEST(SymbolicTransformsTest, imellin_a_pow_s_plus_zero_and_one) {
    const auto n0 = sym_div(
        sym_const(1.0),
        sym_pow(sym_const(2.0), sym_add(sym_var("s"), sym_const(0.0))));
    expect_imellin_pair(
        n0, sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t")))), {{"t", 0.75}});

    const auto n1 = sym_div(
        sym_const(1.0),
        sym_pow(sym_const(2.0), sym_add(sym_var("s"), sym_const(1.0))));
    const auto expected_n1 = sym_mul(
        sym_pow(sym_var("t"), sym_const(1.0)),
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t")))));
    expect_imellin_pair(n1, expected_n1, {{"t", 0.5}});
}

TEST(SymbolicTransformsTest, hankel_sub_scaled_exponentials) {
    const auto r_expr = sym_sub(
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r")))),
        sym_mul(sym_const(3.0), sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r"))))));
    const auto expected = sym_sub(
        sym_div(
            sym_const(2.0),
            sym_pow(
                sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
                sym_const(1.5))),
        sym_mul(
            sym_const(3.0),
            sym_div(
                sym_const(2.0),
                sym_pow(
                    sym_add(
                        sym_pow(sym_var("k"), sym_const(2.0)),
                        sym_pow(sym_const(2.0), sym_const(2.0))),
                    sym_const(1.5)))));
    expect_hankel_pair(r_expr, expected, {{"k", 1.0}});
}

TEST(SymbolicTransformsTest, hankel_exp_times_r_and_swapped_sqrt) {
    const auto r_expr = sym_mul(
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("r")))),
        sym_var("r"));
    const double numer = std::pow(2.0, 2) * std::tgamma(2.0) / std::sqrt(std::numbers::pi) * 3.0;
    const auto expected = sym_div(
        sym_const(numer),
        sym_pow(
            sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(3.0), sym_const(2.0))),
            sym_const(2.0)));
    expect_hankel_pair(r_expr, expected, {{"k", 2.0}});

    const auto tpow_on_right = sym_mul(
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("r")))),
        sym_pow(sym_var("r"), sym_const(1.0)));
    expect_hankel_pair(tpow_on_right, expected, {{"k", 2.0}});

    const auto swapped_sqrt = sym_div(
        sym_const(1.0),
        sym_sqrt(sym_add(
            sym_pow(sym_const(2.0), sym_const(2.0)),
            sym_pow(sym_var("r"), sym_const(2.0)))));
    const auto expected_sqrt = sym_div(
        sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("k")))),
        sym_var("k"));
    expect_hankel_pair(swapped_sqrt, expected_sqrt, {{"k", 2.5}});
}

TEST(SymbolicTransformsTest, ihankel_sub_exponential_forms) {
    const auto k_expr = sym_sub(
        sym_div(
            sym_const(2.0),
            sym_pow(
                sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
                sym_const(1.5))),
        sym_mul(
            sym_const(3.0),
            sym_div(
                sym_const(2.0),
                sym_pow(
                    sym_add(
                        sym_pow(sym_var("k"), sym_const(2.0)),
                        sym_pow(sym_const(2.0), sym_const(2.0))),
                    sym_const(1.5)))));
    const auto inverse = sym_simplify(sym_ihankel(k_expr, "k", "r"));
    EXPECT_NEAR(sym_eval(inverse, {{"r", 1.0}}), -2.0 * std::exp(-2.0), 1e-9);
}

TEST(SymbolicTransformsTest, ihankel_n1_form_and_right_const) {
    const double numer = std::pow(2.0, 2) * std::tgamma(2.0) / std::sqrt(std::numbers::pi) * 3.0;
    const auto k_n1 = sym_div(
        sym_const(numer),
        sym_pow(
            sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(3.0), sym_const(2.0))),
            sym_const(2.0)));
    const auto expected_n1 = sym_mul(
        sym_var("r"),
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("r")))));
    expect_ihankel_pair(k_n1, expected_n1, {{"r", 1.0}});

    const auto scaled = sym_mul(
        sym_div(
            sym_const(2.0),
            sym_pow(
                sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
                sym_const(1.5))),
        sym_const(2.0));
    expect_ihankel_pair(
        scaled,
        sym_mul(sym_const(2.0), sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r"))))),
        {{"r", 1.0}});

    const auto mul_numer = sym_div(
        sym_mul(sym_const(1.0), sym_const(2.0)),
        sym_pow(
            sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
            sym_const(1.5)));
    expect_ihankel_pair(
        mul_numer, sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r")))), {{"r", 1.0}});
}

TEST(SymbolicTransformsTest, ihankel_add_and_nonzero_sentinel) {
    const auto k_expr = sym_add(
        sym_div(
            sym_const(2.0),
            sym_pow(
                sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
                sym_const(1.5))),
        sym_div(
            sym_const(2.0),
            sym_pow(
                sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
                sym_const(1.5))));
    const auto inverse = sym_simplify(sym_ihankel(k_expr, "k", "r"));
    EXPECT_NEAR(sym_eval(inverse, {{"r", 1.0}}), 2.0 * std::exp(-2.0), 1e-9);

    EXPECT_TRUE(is_deriv_sentinel(sym_const(1.0), sym_ihankel(sym_const(1.0), "k", "r"), "k"));
}

TEST(SymbolicTransformsTest, transform_unmatched_mul_and_nonzero_sentinels) {
    EXPECT_TRUE(is_deriv_sentinel(
        sym_mul(sym_var("t"), sym_sin(sym_var("t"))),
        sym_laplace(sym_mul(sym_var("t"), sym_sin(sym_var("t"))), "t", "s"),
        "t"));
    EXPECT_TRUE(is_deriv_sentinel(sym_var("y"), sym_mellin(sym_var("y"), "t", "s"), "t"));
    EXPECT_TRUE(is_deriv_sentinel(
        sym_mul(sym_var("t"), sym_sin(sym_var("t"))),
        sym_mellin(sym_mul(sym_var("t"), sym_sin(sym_var("t"))), "t", "s"),
        "t"));
    EXPECT_TRUE(is_deriv_sentinel(sym_const(3.0), sym_imellin(sym_const(3.0), "s", "t"), "s"));
    EXPECT_TRUE(is_deriv_sentinel(
        sym_exp(sym_var("r")),
        sym_hankel(sym_exp(sym_var("r")), "r", "k"),
        "r"));
}
