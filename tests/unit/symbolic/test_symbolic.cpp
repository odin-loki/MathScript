#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <numbers>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

TEST(SymbolicCasTest, diff_polynomial) {
    auto expr = sym_add(sym_mul(sym_var("x"), sym_var("x")), sym_mul(sym_const(3.0), sym_var("x")));
    auto dx = sym_simplify(sym_diff(std::move(expr), "x"));
    auto expected = sym_simplify(sym_add(sym_mul(sym_const(2.0), sym_var("x")), sym_const(3.0)));

    std::map<std::string, double> probe{{"x", 4.0}};
    EXPECT_NEAR(sym_eval(dx, probe), sym_eval(expected, probe), 1e-12);
    EXPECT_NEAR(sym_eval(dx, probe), 11.0, 1e-12);
}

TEST(SymbolicCasTest, eval_and_simplify_constants) {
    auto expr = sym_simplify(sym_add(sym_const(2.0), sym_const(3.0)));
    EXPECT_DOUBLE_EQ(sym_eval(expr, {}), 5.0);

    auto product = sym_simplify(sym_mul(sym_const(0.0), sym_var("x")));
    EXPECT_DOUBLE_EQ(sym_eval(product, {{"x", 123.0}}), 0.0);
}

TEST(SymbolicCasTest, diff_product_rule) {
    auto expr = sym_mul(sym_var("x"), sym_var("x"));
    auto dx = sym_simplify(sym_diff(std::move(expr), "x"));
    EXPECT_NEAR(sym_eval(dx, {{"x", 3.0}}), 6.0, 1e-12);
}

TEST(SymbolicCasTest, sin_and_exp) {
    EXPECT_NEAR(sym_eval(sym_add(sym_sin(sym_var("x")), sym_exp(sym_const(0.0))), {{"x", 0.0}}), 1.0, 1e-12);

    auto dx = sym_simplify(sym_diff(sym_add(sym_sin(sym_var("x")), sym_exp(sym_const(0.0))), "x"));
    EXPECT_NEAR(sym_eval(dx, {{"x", 0.0}}), 1.0, 1e-12);
}

TEST(SymbolicCasTest, log_and_pow) {
    auto dlog = sym_simplify(sym_diff(sym_log(sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(dlog, {{"x", 2.0}}), 0.5, 1e-12);

    auto dx2 = sym_simplify(sym_diff(sym_pow(sym_var("x"), sym_const(2.0)), "x"));
    EXPECT_NEAR(sym_eval(dx2, {{"x", 3.0}}), 6.0, 1e-12);

    auto simplified = sym_simplify(sym_log(sym_exp(sym_var("x"))));
    EXPECT_NEAR(sym_eval(simplified, {{"x", 1.5}}), 1.5, 1e-12);
}

TEST(SymbolicCasTest, simplify_add_zero_on_left) {
    const auto simplified = sym_simplify(sym_add(sym_const(0.0), sym_var("x")));
    EXPECT_EQ(simplified.op, SymOp::Var);
    EXPECT_EQ(simplified.name, "x");
    EXPECT_NEAR(sym_eval(simplified, {{"x", 4.5}}), 4.5, 1e-12);
}

TEST(SymbolicCasTest, simplify_mul_zero_and_one_on_right) {
    const auto times_zero = sym_simplify(sym_mul(sym_var("x"), sym_const(0.0)));
    EXPECT_EQ(times_zero.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(times_zero, {{"x", 11.0}}), 0.0, 1e-12);

    const auto times_one = sym_simplify(sym_mul(sym_var("x"), sym_const(1.0)));
    EXPECT_EQ(times_one.op, SymOp::Var);
    EXPECT_NEAR(sym_eval(times_one, {{"x", -2.5}}), -2.5, 1e-12);
}

TEST(SymbolicCasTest, eval_deriv_node_matches_diff) {
    const auto wrapped = sym_deriv(sym_pow(sym_var("x"), sym_const(3.0)), "x");
    EXPECT_NEAR(sym_eval(wrapped, {{"x", 2.0}}), 12.0, 1e-12);
}

TEST(SymbolicCasTest, diff_tan_sqrt_and_power_identities) {
    const auto d_tan = sym_simplify(sym_diff(sym_tan(sym_mul(sym_const(2.0), sym_var("x"))), "x"));
    const double x_tan = 0.3;
    EXPECT_NEAR(sym_eval(d_tan, {{"x", x_tan}}), 2.0 / (std::cos(2.0 * x_tan) * std::cos(2.0 * x_tan)), 1e-9);

    const auto d_sqrt = sym_simplify(sym_diff(sym_sqrt(sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(d_sqrt, {{"x", 4.0}}), 0.25, 1e-12);

    const auto d_xx = sym_simplify(sym_diff(sym_pow(sym_var("x"), sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(d_xx, {{"x", 2.0}}), 4.0 * (1.0 + std::log(2.0)), 1e-8);

    const auto d_quot = sym_simplify(sym_diff(
        sym_div(sym_pow(sym_var("x"), sym_const(2.0)), sym_add(sym_var("x"), sym_const(1.0))), "x"));
    EXPECT_NEAR(sym_eval(d_quot, {{"x", 1.0}}), 0.75, 1e-9);
}

TEST(SymbolicCasTest, construct_sub_div_neg_and_cos) {
    const auto sub = sym_sub(sym_const(9.0), sym_const(4.0));
    EXPECT_EQ(sub.op, SymOp::Sub);
    EXPECT_NEAR(sym_eval(sub, {}), 5.0, 1e-12);

    const auto quot = sym_div(sym_const(8.0), sym_const(2.0));
    EXPECT_EQ(quot.op, SymOp::Div);
    EXPECT_NEAR(sym_eval(quot, {}), 4.0, 1e-12);

    const auto neg = sym_neg(sym_const(3.5));
    EXPECT_EQ(neg.op, SymOp::Neg);
    EXPECT_NEAR(sym_eval(neg, {}), -3.5, 1e-12);

    const auto c = sym_cos(sym_const(0.0));
    EXPECT_EQ(c.op, SymOp::Cos);
    EXPECT_NEAR(sym_eval(c, {}), 1.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_add_zero_on_right) {
    const auto simplified = sym_simplify(sym_add(sym_var("x"), sym_const(0.0)));
    EXPECT_EQ(simplified.op, SymOp::Var);
    EXPECT_EQ(simplified.name, "x");
    EXPECT_NEAR(sym_eval(simplified, {{"x", -1.25}}), -1.25, 1e-12);
}

TEST(SymbolicCasTest, simplify_sub_identities) {
    const auto folded = sym_simplify(sym_sub(sym_const(11.0), sym_const(4.0)));
    EXPECT_EQ(folded.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(folded, {}), 7.0, 1e-12);

    const auto minus_zero = sym_simplify(sym_sub(sym_var("x"), sym_const(0.0)));
    EXPECT_EQ(minus_zero.op, SymOp::Var);
    EXPECT_NEAR(sym_eval(minus_zero, {{"x", 8.0}}), 8.0, 1e-12);

    const auto zero_minus = sym_simplify(sym_sub(sym_const(0.0), sym_var("x")));
    EXPECT_EQ(zero_minus.op, SymOp::Neg);
    EXPECT_NEAR(sym_eval(zero_minus, {{"x", 3.0}}), -3.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_mul_one_on_left_and_const_fold) {
    const auto times_one = sym_simplify(sym_mul(sym_const(1.0), sym_var("x")));
    EXPECT_EQ(times_one.op, SymOp::Var);
    EXPECT_NEAR(sym_eval(times_one, {{"x", 6.5}}), 6.5, 1e-12);

    const auto folded = sym_simplify(sym_mul(sym_const(3.0), sym_const(7.0)));
    EXPECT_EQ(folded.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(folded, {}), 21.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_div_identities) {
    const auto folded = sym_simplify(sym_div(sym_const(15.0), sym_const(3.0)));
    EXPECT_EQ(folded.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(folded, {}), 5.0, 1e-12);

    const auto over_one = sym_simplify(sym_div(sym_var("x"), sym_const(1.0)));
    EXPECT_EQ(over_one.op, SymOp::Var);
    EXPECT_NEAR(sym_eval(over_one, {{"x", -4.0}}), -4.0, 1e-12);

    const auto zero_over = sym_simplify(sym_div(sym_const(0.0), sym_var("x")));
    EXPECT_EQ(zero_over.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(zero_over, {{"x", 9.0}}), 0.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_neg_const_and_double_neg) {
    const auto folded = sym_simplify(sym_neg(sym_const(2.5)));
    EXPECT_EQ(folded.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(folded, {}), -2.5, 1e-12);

    const auto double_neg = sym_simplify(sym_neg(sym_neg(sym_var("x"))));
    EXPECT_EQ(double_neg.op, SymOp::Var);
    EXPECT_NEAR(sym_eval(double_neg, {{"x", 1.75}}), 1.75, 1e-12);
}

TEST(SymbolicCasTest, simplify_trig_sqrt_const_fold) {
    EXPECT_NEAR(sym_eval(sym_simplify(sym_sin(sym_const(0.0))), {}), 0.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_cos(sym_const(0.0))), {}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_tan(sym_const(0.0))), {}), 0.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_sqrt(sym_const(9.0))), {}), 3.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_exp_log_and_pow_identities) {
    EXPECT_NEAR(sym_eval(sym_simplify(sym_exp(sym_const(0.0))), {}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_log(sym_const(1.0))), {}), 0.0, 1e-12);

    const auto exp_log = sym_simplify(sym_exp(sym_log(sym_var("x"))));
    EXPECT_EQ(exp_log.op, SymOp::Var);
    EXPECT_NEAR(sym_eval(exp_log, {{"x", 2.25}}), 2.25, 1e-12);

    const auto pow_zero = sym_simplify(sym_pow(sym_var("x"), sym_const(0.0)));
    EXPECT_EQ(pow_zero.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(pow_zero, {{"x", 11.0}}), 1.0, 1e-12);

    const auto pow_one = sym_simplify(sym_pow(sym_var("x"), sym_const(1.0)));
    EXPECT_EQ(pow_one.op, SymOp::Var);
    EXPECT_NEAR(sym_eval(pow_one, {{"x", -3.0}}), -3.0, 1e-12);

    const auto one_pow = sym_simplify(sym_pow(sym_const(1.0), sym_var("x")));
    EXPECT_EQ(one_pow.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(one_pow, {{"x", 40.0}}), 1.0, 1e-12);

    const auto both_const = sym_simplify(sym_pow(sym_const(2.0), sym_const(5.0)));
    EXPECT_EQ(both_const.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(both_const, {}), 32.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_rejects_invalid_const_folds) {
    const auto neg_sqrt = sym_simplify(sym_sqrt(sym_const(-4.0)));
    EXPECT_EQ(neg_sqrt.op, SymOp::Sqrt);

    const auto log_zero = sym_simplify(sym_log(sym_const(0.0)));
    EXPECT_EQ(log_zero.op, SymOp::Log);

    const auto zero_over_zero = sym_simplify(sym_div(sym_const(0.0), sym_const(0.0)));
    EXPECT_EQ(zero_over_zero.op, SymOp::Div);
}

TEST(SymbolicCasTest, diff_const_and_unrelated_var) {
    const auto d_const = sym_simplify(sym_diff(sym_const(9.0), "x"));
    EXPECT_NEAR(sym_eval(d_const, {}), 0.0, 1e-12);

    const auto d_other = sym_simplify(sym_diff(sym_var("y"), "x"));
    EXPECT_NEAR(sym_eval(d_other, {{"y", 4.0}}), 0.0, 1e-12);

    const auto d_self = sym_simplify(sym_diff(sym_var("x"), "x"));
    EXPECT_NEAR(sym_eval(d_self, {{"x", 99.0}}), 1.0, 1e-12);
}

TEST(SymbolicCasTest, diff_sub_and_neg) {
    const auto d_sub = sym_simplify(sym_diff(
        sym_sub(sym_pow(sym_var("x"), sym_const(2.0)), sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(d_sub, {{"x", 3.0}}), 5.0, 1e-12);

    const auto d_neg = sym_simplify(sym_diff(sym_neg(sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(d_neg, {{"x", 2.0}}), -1.0, 1e-12);
}

TEST(SymbolicCasTest, diff_cos_and_exp_of_var) {
    const auto d_cos = sym_simplify(sym_diff(sym_cos(sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(d_cos, {{"x", 0.0}}), 0.0, 1e-12);
    EXPECT_NEAR(sym_eval(d_cos, {{"x", 1.0}}), -std::sin(1.0), 1e-12);

    const auto d_exp = sym_simplify(sym_diff(sym_exp(sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(d_exp, {{"x", 0.0}}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(d_exp, {{"x", 1.0}}), std::exp(1.0), 1e-12);
}

TEST(SymbolicCasTest, diff_const_base_power_and_nested_deriv) {
    const auto d_base = sym_simplify(sym_diff(sym_pow(sym_const(2.0), sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(d_base, {{"x", 3.0}}), 8.0 * std::log(2.0), 1e-9);

    const auto d_of_deriv = sym_diff(sym_deriv(sym_mul(sym_var("x"), sym_var("x")), "x"), "x");
    EXPECT_EQ(d_of_deriv.op, SymOp::Deriv);
    EXPECT_NEAR(sym_eval(d_of_deriv, {{"x", 4.0}}), 8.0, 1e-12);
}

TEST(SymbolicCasTest, eval_missing_var_and_arithmetic) {
    EXPECT_DOUBLE_EQ(sym_eval(sym_var("missing"), {}), 0.0);
    EXPECT_NEAR(sym_eval(sym_sub(sym_const(10.0), sym_var("x")), {{"x", 3.0}}), 7.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_div(sym_var("x"), sym_const(4.0)), {{"x", 10.0}}), 2.5, 1e-12);
    EXPECT_NEAR(sym_eval(sym_neg(sym_var("x")), {{"x", 6.0}}), -6.0, 1e-12);
}

TEST(SymbolicCasTest, eval_trig_log_sqrt_and_pow) {
    const double x = 0.6;
    EXPECT_NEAR(sym_eval(sym_cos(sym_var("x")), {{"x", x}}), std::cos(x), 1e-12);
    EXPECT_NEAR(sym_eval(sym_tan(sym_var("x")), {{"x", x}}), std::tan(x), 1e-12);
    EXPECT_NEAR(sym_eval(sym_log(sym_var("x")), {{"x", x}}), std::log(x), 1e-12);
    EXPECT_NEAR(sym_eval(sym_sqrt(sym_var("x")), {{"x", 6.25}}), 2.5, 1e-12);
    EXPECT_NEAR(sym_eval(sym_pow(sym_var("x"), sym_const(3.0)), {{"x", 2.0}}), 8.0, 1e-12);
}

TEST(SymbolicCasTest, eval_and_diff_unknown_op_is_zero) {
    SymExpr bogus;
    bogus.op = static_cast<SymOp>(127);
    EXPECT_DOUBLE_EQ(sym_eval(bogus, {}), 0.0);
    const auto d = sym_diff(std::move(bogus), "x");
    EXPECT_EQ(d.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(d, {}), 0.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_log_of_positive_const) {
    const auto folded = sym_simplify(sym_log(sym_const(std::exp(1.0))));
    EXPECT_EQ(folded.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(folded, {}), 1.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_div_by_zero_const_stays_div) {
    const auto expr = sym_simplify(sym_div(sym_const(1.0), sym_const(0.0)));
    EXPECT_EQ(expr.op, SymOp::Div);
}

TEST(SymbolicCasTest, simplify_tan_and_sqrt_of_var_stay) {
    const auto t = sym_simplify(sym_tan(sym_var("x")));
    EXPECT_EQ(t.op, SymOp::Tan);
    EXPECT_NEAR(sym_eval(t, {{"x", 0.3}}), std::tan(0.3), 1e-12);

    const auto s = sym_simplify(sym_sqrt(sym_var("x")));
    EXPECT_EQ(s.op, SymOp::Sqrt);
    EXPECT_NEAR(sym_eval(s, {{"x", 16.0}}), 4.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_pow_zero_base_positive_exp) {
    const auto folded = sym_simplify(sym_pow(sym_const(0.0), sym_const(3.0)));
    EXPECT_EQ(folded.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(folded, {}), 0.0, 1e-12);
}

TEST(SymbolicCasTest, simplify_neg_of_var_and_deriv_node) {
    const auto neg = sym_simplify(sym_neg(sym_var("x")));
    EXPECT_EQ(neg.op, SymOp::Neg);
    EXPECT_NEAR(sym_eval(neg, {{"x", 4.0}}), -4.0, 1e-12);

    const auto d = sym_simplify(sym_deriv(sym_var("x"), "x"));
    EXPECT_EQ(d.op, SymOp::Deriv);
}

TEST(SymbolicCasTest, simplify_exp_of_nonzero_const) {
    const auto folded = sym_simplify(sym_exp(sym_const(2.0)));
    EXPECT_EQ(folded.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(folded, {}), std::exp(2.0), 1e-12);
}

TEST(SymbolicCasTest, diff_zero_base_variable_exponent) {
    const auto d = sym_diff(sym_pow(sym_const(0.0), sym_var("x")), "x");
    EXPECT_FALSE(sym_to_string(d).empty());
}

TEST(SymbolicCasTest, eval_nested_div_neg_tan) {
    const auto expr = sym_div(sym_neg(sym_tan(sym_var("x"))), sym_const(2.0));
    EXPECT_NEAR(sym_eval(expr, {{"x", 0.2}}), -std::tan(0.2) / 2.0, 1e-12);
}

TEST(SymbolicCasTest, substitute_into_pow_and_div) {
    const auto expr = sym_div(sym_pow(sym_var("x"), sym_const(3.0)), sym_var("x"));
    const auto replaced = sym_substitute(expr, "x", sym_const(2.0));
    EXPECT_NEAR(sym_eval(replaced, {}), 4.0, 1e-12);
}

TEST(SymbolicCasTest, expand_power_zero_and_one) {
    const auto z = sym_expand(sym_pow(sym_add(sym_var("x"), sym_const(1.0)), sym_const(0.0)));
    EXPECT_EQ(z.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(z, {{"x", 9.0}}), 1.0, 1e-12);

    const auto one = sym_expand(sym_pow(sym_add(sym_var("x"), sym_const(2.0)), sym_const(1.0)));
    EXPECT_NEAR(sym_eval(one, {{"x", 3.0}}), 5.0, 1e-12);
}

TEST(SymbolicCasTest, collect_empty_var_returns_simplified) {
    const auto expr = sym_add(sym_const(2.0), sym_const(5.0));
    const auto collected = sym_collect(expr, "");
    EXPECT_EQ(collected.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(collected, {}), 7.0, 1e-12);
}

TEST(SymbolicCasTest, integrate_div_form_is_sentinel) {
    const auto unsupported = sym_integrate(sym_div(sym_var("x"), sym_const(2.0)), "x");
    EXPECT_EQ(unsupported.op, SymOp::Deriv);
    EXPECT_EQ(unsupported.name, "x");
}

TEST(SymbolicCasTest, laplace_exp_t_times_const_and_cos_bare) {
    const auto exp_right = sym_laplace(sym_exp(sym_mul(sym_var("t"), sym_const(3.0))), "t", "s");
    EXPECT_NEAR(sym_eval(exp_right, {{"s", 5.0}}), 1.0 / (5.0 - 3.0), 1e-12);

    const auto cos_bare = sym_laplace(sym_cos(sym_var("t")), "t", "s");
    EXPECT_NEAR(sym_eval(cos_bare, {{"s", 2.0}}), 2.0 / (4.0 + 1.0), 1e-12);
}

TEST(SymbolicCasTest, ilaplace_const_a2_cosine_and_numer_mismatch) {
    const auto cosine = sym_ilaplace(
        sym_div(sym_var("s"), sym_add(sym_const(9.0), sym_pow(sym_var("s"), sym_const(2.0)))),
        "s", "t");
    EXPECT_NEAR(sym_eval(cosine, {{"t", 0.25}}), std::cos(3.0 * 0.25), 1e-12);

    const auto mismatch = sym_ilaplace(
        sym_div(sym_const(2.0), sym_add(sym_pow(sym_var("s"), sym_const(2.0)), sym_const(9.0))),
        "s", "t");
    EXPECT_EQ(mismatch.op, SymOp::Deriv);
    EXPECT_EQ(mismatch.name, "s");
}

TEST(SymbolicCasTest, mellin_t_squared_exp_neg_at_and_bad_reciprocal) {
    const auto matched = sym_mellin(
        sym_mul(
            sym_pow(sym_var("t"), sym_const(2.0)),
            sym_exp(sym_mul(sym_const(-2.0), sym_var("t")))),
        "t", "s");
    EXPECT_NEAR(sym_eval(matched, {{"s", 1.0}}), 2.0 / std::pow(2.0, 3.0), 1e-12);

    const auto miss = sym_mellin(
        sym_div(sym_const(2.0), sym_add(sym_const(1.0), sym_var("t"))), "t", "s");
    EXPECT_EQ(miss.op, SymOp::Deriv);
    EXPECT_EQ(miss.name, "t");
}

TEST(SymbolicCasTest, hankel_r_squared_exp_and_sqrt_const_sum) {
    const auto rpow = sym_hankel(
        sym_mul(
            sym_pow(sym_var("r"), sym_const(2.0)),
            sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("r"))))),
        "r", "k");
    const double expected_n2 = 8.0 * std::tgamma(2.5) / std::sqrt(std::numbers::pi) * 2.0 /
                               std::pow(4.0 + 1.0, 2.5);
    EXPECT_NEAR(sym_eval(rpow, {{"k", 1.0}}), expected_n2, 1e-9);

    const auto sqrt_form = sym_hankel(
        sym_div(
            sym_const(1.0),
            sym_sqrt(sym_add(sym_pow(sym_var("r"), sym_const(2.0)), sym_const(4.0)))),
        "r", "k");
    EXPECT_NEAR(sym_eval(sqrt_form, {{"k", 2.0}}), std::exp(-4.0) / 2.0, 1e-12);
}

TEST(SymbolicCasTest, ihankel_k_times_const_decay_and_scale_miss) {
    const auto decay = sym_ihankel(
        sym_div(sym_exp(sym_neg(sym_mul(sym_var("k"), sym_const(2.0)))), sym_var("k")),
        "k", "r");
    EXPECT_NEAR(sym_eval(decay, {{"r", 3.0}}), 1.0 / std::sqrt(9.0 + 4.0), 1e-12);

    const auto miss = sym_ihankel(
        sym_div(
            sym_const(99.0),
            sym_pow(
                sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_pow(sym_const(2.0), sym_const(2.0))),
                sym_const(1.5))),
        "k", "r");
    EXPECT_EQ(miss.op, SymOp::Deriv);
    EXPECT_EQ(miss.name, "k");
}

TEST(SymbolicCasTest, fourier_neg_wrapped_const_and_t2_on_left) {
    const auto linear = sym_fourier(sym_exp(sym_mul(sym_neg(sym_const(2.0)), sym_var("t"))), "t", "w");
    EXPECT_NEAR(sym_eval(linear, {{"w", 1.0}}), 4.0 / (4.0 + 1.0), 1e-9);

    const auto linear_right =
        sym_fourier(sym_exp(sym_mul(sym_var("t"), sym_neg(sym_const(2.0)))), "t", "w");
    EXPECT_NEAR(sym_eval(linear_right, {{"w", 1.0}}), 4.0 / 5.0, 1e-9);

    const auto gaussian = sym_fourier(
        sym_exp(sym_neg(sym_mul(sym_pow(sym_var("t"), sym_const(2.0)), sym_const(1.5)))),
        "t", "w");
    const double scale = std::sqrt(std::numbers::pi / 1.5);
    EXPECT_NEAR(
        sym_eval(gaussian, {{"w", 1.0}}), scale * std::exp(-1.0 / (4.0 * 1.5)), 1e-9);
}

TEST(SymbolicCasTest, ifourier_two_const_numer_and_gaussian_zero_minus) {
    const auto rational = sym_ifourier(
        sym_div(
            sym_mul(sym_const(2.0), sym_const(3.0)),
            sym_add(sym_const(9.0), sym_pow(sym_var("w"), sym_const(2.0)))),
        "w", "t");
    EXPECT_NEAR(sym_eval(rational, {{"t", 0.5}}), std::exp(-1.5), 1e-9);

    const double a = std::numbers::pi / 4.0;
    const auto spectrum = sym_mul(
        sym_exp(sym_sub(
            sym_const(0.0),
            sym_div(sym_pow(sym_var("w"), sym_const(2.0)), sym_const(4.0 * a)))),
        sym_const(2.0));
    const auto time = sym_ifourier(spectrum, "w", "t");
    EXPECT_NEAR(sym_eval(time, {{"t", 0.0}}), 1.0, 1e-9);
}

TEST(SymbolicCasTest, dsolve_linear_affine_and_xy_product) {
    const auto affine = sym_dsolve(
        sym_add(sym_mul(sym_const(2.0), sym_var("y")), sym_const(6.0)), "x", "y");
    EXPECT_NEAR(sym_eval(affine, {{"x", 0.0}, {"C", 1.0}}), -3.0 + 1.0, 1e-12);

    const auto product = sym_dsolve(sym_mul(sym_var("x"), sym_var("y")), "x", "y");
    EXPECT_NEAR(sym_eval(product, {{"x", 0.0}, {"C", 2.0}}), 2.0, 1e-12);
}

TEST(SymbolicCasTest, laplace_sin_of_t_squared_is_sentinel) {
    const auto miss = sym_laplace(sym_sin(sym_pow(sym_var("t"), sym_const(2.0))), "t", "s");
    EXPECT_EQ(miss.op, SymOp::Deriv);
    EXPECT_EQ(miss.name, "t");
}

TEST(SymbolicCasTest, integrate_bare_trig_and_hankel_div_miss) {
    const auto sine = sym_integrate(sym_sin(sym_var("x")), "x");
    EXPECT_NEAR(sym_eval(sine, {{"x", 0.0}}), -1.0, 1e-12);

    const auto cosine = sym_integrate(sym_cos(sym_var("x")), "x");
    EXPECT_NEAR(sym_eval(cosine, {{"x", 0.0}}), 0.0, 1e-12);

    const auto hankel_miss = sym_hankel(
        sym_div(
            sym_const(2.0),
            sym_sqrt(sym_add(sym_pow(sym_var("r"), sym_const(2.0)), sym_const(4.0)))),
        "r", "k");
    EXPECT_EQ(hankel_miss.op, SymOp::Deriv);
    EXPECT_EQ(hankel_miss.name, "r");
}

TEST(SymbolicCasTest, integrate_const_other_var_pow_and_reciprocal_miss) {
    const auto c = sym_integrate(sym_const(3.0), "x");
    EXPECT_NEAR(sym_eval(c, {{"x", 2.0}}), 6.0, 1e-12);

    const auto other = sym_integrate(sym_var("y"), "x");
    EXPECT_NEAR(sym_eval(other, {{"x", 2.0}, {"y", 4.0}}), 8.0, 1e-12);

    const auto pow3 = sym_integrate(sym_pow(sym_var("x"), sym_const(3.0)), "x");
    EXPECT_NEAR(sym_eval(pow3, {{"x", 2.0}}), 4.0, 1e-12);

    const auto recip = sym_integrate(sym_pow(sym_var("x"), sym_const(-1.0)), "x");
    EXPECT_EQ(recip.op, SymOp::Deriv);
    EXPECT_EQ(recip.name, "x");
}

TEST(SymbolicCasTest, dsolve_independent_rhs_and_imellin_c_over_s) {
    const auto sol = sym_dsolve(sym_const(2.0), "x", "y");
    EXPECT_NEAR(sym_eval(sol, {{"x", 3.0}, {"C", 1.0}}), 7.0, 1e-12);

    auto five = sym_const(5.0);
    const auto im = sym_imellin(sym_div(std::move(five), sym_var("s")), "s", "t");
    EXPECT_NEAR(sym_eval(im, {}), 5.0, 1e-12);
}

TEST(SymbolicCasTest, ilaplace_swapped_const_a2_and_laplace_exp_neg) {
    // 2 / (4 + s^2): swapped addends, const a^2 (not pow), sine pair with a = 2.
    const auto sine = sym_ilaplace(
        sym_div(sym_const(2.0), sym_add(sym_const(4.0), sym_pow(sym_var("s"), sym_const(2.0)))),
        "s", "t");
    EXPECT_NEAR(sym_eval(sine, {{"t", 0.3}}), std::sin(2.0 * 0.3), 1e-12);

    // exp(-3*t) = exp((-3)*t): match_scaled_var const-on-left, L{e^{at}} with a = -3.
    const auto decay = sym_laplace(
        sym_exp(sym_mul(sym_const(-3.0), sym_var("t"))), "t", "s");
    EXPECT_NEAR(sym_eval(decay, {{"s", 5.0}}), 1.0 / (5.0 + 3.0), 1e-12);

    // exp(t*(-3)): same scale with const on the right of Mul.
    const auto decay_right = sym_laplace(
        sym_exp(sym_mul(sym_var("t"), sym_const(-3.0))), "t", "s");
    EXPECT_NEAR(sym_eval(decay_right, {{"s", 5.0}}), 1.0 / 8.0, 1e-12);

    // exp(-(3*t)): Laplace matches scaled var only, so Neg-wrapped product is a sentinel.
    const auto wrapped = sym_laplace(
        sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("t")))), "t", "s");
    EXPECT_EQ(wrapped.op, SymOp::Deriv);
    EXPECT_EQ(wrapped.name, "t");
}

TEST(SymbolicCasTest, mellin_one_plus_t_and_exp_neg_both_shapes) {
    const double pi_over_sin = std::numbers::pi / std::sin(std::numbers::pi * 0.5);

    const auto left_one = sym_mellin(
        sym_div(sym_const(1.0), sym_add(sym_const(1.0), sym_var("t"))), "t", "s");
    EXPECT_NEAR(sym_eval(left_one, {{"s", 0.5}}), pi_over_sin, 1e-12);

    const auto right_one = sym_mellin(
        sym_div(sym_const(1.0), sym_add(sym_var("t"), sym_const(1.0))), "t", "s");
    EXPECT_NEAR(sym_eval(right_one, {{"s", 0.5}}), pi_over_sin, 1e-12);

    // exp((-4)*t) vs exp(-(4*t)): both match_exp_neg_at, M{e^{-a t}} = a^{-s}.
    const auto scale_neg = sym_mellin(
        sym_exp(sym_mul(sym_const(-4.0), sym_var("t"))), "t", "s");
    EXPECT_NEAR(sym_eval(scale_neg, {{"s", 2.0}}), 1.0 / 16.0, 1e-12);

    const auto neg_product = sym_mellin(
        sym_exp(sym_neg(sym_mul(sym_const(4.0), sym_var("t")))), "t", "s");
    EXPECT_NEAR(sym_eval(neg_product, {{"s", 2.0}}), 1.0 / 16.0, 1e-12);
}

TEST(SymbolicCasTest, hankel_sqrt_const_first_and_ihankel_mul_numer) {
    // 1/sqrt(9 + r^2): match_one_over_sqrt with swapped addends and const a^2.
    const auto sqrt_swapped = sym_hankel(
        sym_div(
            sym_const(1.0),
            sym_sqrt(sym_add(sym_const(9.0), sym_pow(sym_var("r"), sym_const(2.0))))),
        "r", "k");
    EXPECT_NEAR(sym_eval(sqrt_swapped, {{"k", 1.0}}), std::exp(-3.0), 1e-12);

    // n=0 k-domain: (1*2) / (k^2 + 4)^{3/2} — const*const numerator, a = 2.
    const auto mul_numer = sym_ihankel(
        sym_div(
            sym_mul(sym_const(1.0), sym_const(2.0)),
            sym_pow(
                sym_add(sym_pow(sym_var("k"), sym_const(2.0)), sym_const(4.0)),
                sym_const(1.5))),
        "k", "r");
    EXPECT_NEAR(sym_eval(mul_numer, {{"r", 1.0}}), std::exp(-2.0), 1e-9);

    // exp(-(3*k))/k: match_exp_neg_ak_over_k with const on the left of Mul.
    const auto decay = sym_ihankel(
        sym_div(sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("k")))), sym_var("k")),
        "k", "r");
    EXPECT_NEAR(sym_eval(decay, {{"r", 4.0}}), 1.0 / 5.0, 1e-12);
}
