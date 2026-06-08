#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

TEST(SymbolicExtTest, to_string_nested) {
    const auto expr = sym_mul(sym_add(sym_var("x"), sym_const(1.0)), sym_var("y"));
    const auto text = sym_to_string(expr);
    EXPECT_NE(text.find("x"), std::string::npos);
    EXPECT_NE(text.find("y"), std::string::npos);
}

TEST(SymbolicExtTest, deriv_alias) {
    auto expr = sym_mul(sym_var("x"), sym_var("x"));
    const auto dx = sym_simplify(sym_deriv(std::move(expr), "x"));
    EXPECT_NEAR(sym_eval(dx, {{"x", 4.0}}), 8.0, 1e-12);
}

TEST(SymbolicExtTest, simplify_nested_trig) {
    auto expr = sym_add(sym_sin(sym_var("x")), sym_sin(sym_const(0.0)));
    const auto simplified = sym_simplify(std::move(expr));
    EXPECT_NEAR(sym_eval(simplified, {{"x", 0.5}}), std::sin(0.5), 1e-12);
}

TEST(SymbolicExtTest, cos_eval_and_pow_simplify) {
    EXPECT_NEAR(sym_eval(sym_cos(sym_const(0.0)), {}), 1.0, 1e-12);
    auto pow_expr = sym_pow(sym_var("x"), sym_const(2.0));
    const auto simplified = sym_simplify(std::move(pow_expr));
    EXPECT_NEAR(sym_eval(simplified, {{"x", 3.0}}), 9.0, 1e-12);
    auto deriv = sym_deriv(sym_pow(sym_var("x"), sym_const(3.0)), "x");
    EXPECT_NEAR(sym_eval(sym_simplify(std::move(deriv)), {{"x", 2.0}}), 12.0, 1e-12);
}

TEST(SymbolicExtTest, deriv_to_string) {
    auto expr = sym_mul(sym_var("x"), sym_const(2.0));
    const auto d = sym_deriv(std::move(expr), "x");
    EXPECT_NE(sym_to_string(d).find("d/d"), std::string::npos);
}

TEST(SymbolicExtTest, exp_log_and_diff_alias) {
    const auto e = sym_simplify(sym_exp(sym_const(0.0)));
    EXPECT_NEAR(sym_eval(e, {}), 1.0, 1e-12);
    const auto l = sym_simplify(sym_log(sym_const(1.0)));
    EXPECT_NEAR(sym_eval(l, {}), 0.0, 1e-12);
    auto expr1 = sym_mul(sym_var("x"), sym_var("x"));
    const auto d1 = sym_simplify(sym_deriv(std::move(expr1), "x"));
    auto expr2 = sym_mul(sym_var("x"), sym_var("x"));
    const auto d2 = sym_simplify(sym_diff(std::move(expr2), "x"));
    EXPECT_NEAR(sym_eval(d1, {{"x", 2.0}}), sym_eval(d2, {{"x", 2.0}}), 1e-12);
}

TEST(SymbolicExtTest, diff_rules_for_elementary_ops) {
    auto sin_x = sym_sin(sym_var("x"));
    const auto d_sin = sym_simplify(sym_diff(std::move(sin_x), "x"));
    EXPECT_NEAR(sym_eval(d_sin, {{"x", 0.0}}), 1.0, 1e-12);

    auto cos_x = sym_cos(sym_var("x"));
    const auto d_cos = sym_simplify(sym_diff(std::move(cos_x), "x"));
    EXPECT_NEAR(sym_eval(d_cos, {{"x", 0.0}}), 0.0, 1e-12);

    auto exp_x = sym_exp(sym_var("x"));
    const auto d_exp = sym_simplify(sym_diff(std::move(exp_x), "x"));
    EXPECT_NEAR(sym_eval(d_exp, {{"x", 1.0}}), std::exp(1.0), 1e-12);

    auto log_x = sym_log(sym_var("x"));
    const auto d_log = sym_simplify(sym_diff(std::move(log_x), "x"));
    EXPECT_NEAR(sym_eval(d_log, {{"x", 2.0}}), 0.5, 1e-12);

    auto pow_x = sym_pow(sym_var("x"), sym_const(4.0));
    const auto d_pow = sym_simplify(sym_diff(std::move(pow_x), "x"));
    EXPECT_NEAR(sym_eval(d_pow, {{"x", 2.0}}), 32.0, 1e-12);
}

TEST(SymbolicExtTest, simplify_mul_by_zero_and_one) {
    auto zero_mul = sym_simplify(sym_mul(sym_const(0.0), sym_var("x")));
    EXPECT_NEAR(sym_eval(zero_mul, {{"x", 99.0}}), 0.0, 1e-12);
    auto one_mul = sym_simplify(sym_mul(sym_const(1.0), sym_var("x")));
    EXPECT_NEAR(sym_eval(one_mul, {{"x", 2.5}}), 2.5, 1e-12);
    auto add_zero = sym_simplify(sym_add(sym_var("x"), sym_const(0.0)));
    EXPECT_NEAR(sym_eval(add_zero, {{"x", 1.5}}), 1.5, 1e-12);
}

TEST(SymbolicExtTest, to_string_all_ops) {
    EXPECT_FALSE(sym_to_string(sym_const(3.0)).empty());
    EXPECT_FALSE(sym_to_string(sym_var("t")).empty());
    EXPECT_NE(sym_to_string(sym_sin(sym_var("x"))).find("sin"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_cos(sym_var("x"))).find("cos"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_exp(sym_var("x"))).find("exp"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_log(sym_var("x"))).find("log"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_pow(sym_var("x"), sym_const(2.0))).find("^"), std::string::npos);
}

TEST(SymbolicExtTest, simplify_elementary_identities) {
    EXPECT_NEAR(sym_eval(sym_simplify(sym_sin(sym_const(0.0))), {}), 0.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_cos(sym_const(0.0))), {}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_pow(sym_var("x"), sym_const(1.0))), {{"x", 2.0}}), 2.0, 1e-12);
    auto nested = sym_add(sym_mul(sym_var("x"), sym_const(2.0)), sym_mul(sym_var("y"), sym_const(3.0)));
    EXPECT_NEAR(sym_eval(nested, {{"x", 1.0}, {"y", 2.0}}), 8.0, 1e-12);
}

TEST(SymbolicExtTest, deriv_of_deriv_smoke) {
    auto inner = sym_mul(sym_var("x"), sym_var("x"));
    const auto d2 = sym_simplify(sym_deriv(sym_deriv(std::move(inner), "x"), "x"));
    EXPECT_NEAR(sym_eval(d2, {{"x", 1.0}}), 2.0, 1e-12);
}

TEST(SymbolicExtTest, simplify_const_folding_and_identities) {
    auto add_const = sym_simplify(sym_add(sym_const(2.0), sym_const(3.0)));
    EXPECT_NEAR(sym_eval(add_const, {}), 5.0, 1e-12);
    auto mul_const = sym_simplify(sym_mul(sym_const(2.0), sym_const(4.0)));
    EXPECT_NEAR(sym_eval(mul_const, {}), 8.0, 1e-12);
    auto log_exp = sym_simplify(sym_log(sym_exp(sym_var("x"))));
    EXPECT_NEAR(sym_eval(log_exp, {{"x", 2.0}}), 2.0, 1e-12);
    auto pow_one_base = sym_simplify(sym_pow(sym_const(1.0), sym_var("x")));
    EXPECT_NEAR(sym_eval(pow_one_base, {{"x", 99.0}}), 1.0, 1e-12);
    auto pow_both_const = sym_simplify(sym_pow(sym_const(2.0), sym_const(3.0)));
    EXPECT_NEAR(sym_eval(pow_both_const, {}), 8.0, 1e-12);
}

TEST(SymbolicExtTest, diff_pow_and_general_branches) {
    auto const_base = sym_pow(sym_const(4.0), sym_var("x"));
    const auto d_const_base = sym_simplify(sym_diff(std::move(const_base), "x"));
    EXPECT_NEAR(sym_eval(d_const_base, {{"x", 1.0}}), 4.0 * std::log(4.0), 1e-9);

    auto general_pow = sym_pow(sym_var("x"), sym_var("y"));
    const auto d_general = sym_simplify(sym_diff(std::move(general_pow), "x"));
    EXPECT_NEAR(sym_eval(d_general, {{"x", 2.0}, {"y", 3.0}}), 12.0, 1e-6);

    auto deriv_wrap = sym_deriv(sym_var("x"), "x");
    const auto d_deriv = sym_simplify(sym_diff(std::move(deriv_wrap), "x"));
    EXPECT_NEAR(sym_eval(d_deriv, {{"x", 5.0}}), 1.0, 1e-12);
}

TEST(SymbolicExtTest, eval_missing_var_and_diff_other_var) {
    EXPECT_DOUBLE_EQ(sym_eval(sym_var("z"), {}), 0.0);
    auto expr = sym_mul(sym_var("x"), sym_var("y"));
    const auto d_y = sym_simplify(sym_diff(std::move(expr), "y"));
    EXPECT_NEAR(sym_eval(d_y, {{"x", 3.0}, {"y", 2.0}}), 3.0, 1e-12);
    const auto d_const = sym_simplify(sym_diff(sym_const(7.0), "x"));
    EXPECT_NEAR(sym_eval(d_const, {}), 0.0, 1e-12);
}

TEST(SymbolicExtTest, simplify_exp_log_inverse) {
    auto exp_log = sym_simplify(sym_exp(sym_log(sym_var("x"))));
    EXPECT_NEAR(sym_eval(exp_log, {{"x", 2.5}}), 2.5, 1e-12);
}

TEST(SymbolicExtTest, simplify_trig_const_fold) {
    EXPECT_NEAR(sym_eval(sym_simplify(sym_sin(sym_const(std::numbers::pi / 2.0))), {}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_cos(sym_const(std::numbers::pi))), {}), -1.0, 1e-12);
}

TEST(SymbolicExtTest, simplify_exp_log_const_fold) {
    EXPECT_NEAR(sym_eval(sym_simplify(sym_exp(sym_const(1.0))), {}), std::exp(1.0), 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_log(sym_const(std::exp(2.0)))), {}), 2.0, 1e-12);
}

TEST(SymbolicExtTest, simplify_pow_zero_exponent) {
    auto zero_exp = sym_simplify(sym_pow(sym_var("x"), sym_const(0.0)));
    EXPECT_NEAR(sym_eval(zero_exp, {{"x", 7.0}}), 1.0, 1e-12);
}

TEST(SymbolicExtTest, diff_const_trig_is_zero) {
    const auto d_sin = sym_simplify(sym_diff(sym_sin(sym_const(1.0)), "x"));
    EXPECT_NEAR(sym_eval(d_sin, {{"x", 3.0}}), 0.0, 1e-12);
    const auto d_cos = sym_simplify(sym_diff(sym_cos(sym_const(0.5)), "x"));
    EXPECT_NEAR(sym_eval(d_cos, {{"x", 3.0}}), 0.0, 1e-12);
}
