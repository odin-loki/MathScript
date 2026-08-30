#include <cmath>
#include <gtest/gtest.h>
#include <map>

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
