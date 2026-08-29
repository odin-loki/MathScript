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
