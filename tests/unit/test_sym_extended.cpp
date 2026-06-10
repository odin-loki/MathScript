#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include "ms/symbolic/symbolic.hpp"
#include <cmath>
#include <map>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// sym_eval: evaluation of trig, exp, log, pow
// ---------------------------------------------------------------------------

TEST(SymExtTest, eval_sin_pi_half) {
    const std::map<std::string, double> env;
    auto expr = sym_sin(sym_const(M_PI / 2.0));
    EXPECT_NEAR(sym_eval(expr, env), 1.0, 1e-12);
}

TEST(SymExtTest, eval_cos_zero) {
    const std::map<std::string, double> env;
    auto expr = sym_cos(sym_const(0.0));
    EXPECT_NEAR(sym_eval(expr, env), 1.0, 1e-12);
}

TEST(SymExtTest, eval_exp_zero) {
    const std::map<std::string, double> env;
    auto expr = sym_exp(sym_const(0.0));
    EXPECT_NEAR(sym_eval(expr, env), 1.0, 1e-12);
}

TEST(SymExtTest, eval_log_e) {
    const std::map<std::string, double> env;
    auto expr = sym_log(sym_const(std::exp(1.0)));
    EXPECT_NEAR(sym_eval(expr, env), 1.0, 1e-12);
}

TEST(SymExtTest, eval_pow_two_three) {
    const std::map<std::string, double> env;
    auto expr = sym_pow(sym_const(2.0), sym_const(3.0));
    EXPECT_NEAR(sym_eval(expr, env), 8.0, 1e-12);
}

TEST(SymExtTest, eval_variable_lookup) {
    const std::map<std::string, double> env = {{"x", 5.0}, {"y", 3.0}};
    auto expr = sym_add(sym_var("x"), sym_var("y"));
    EXPECT_NEAR(sym_eval(expr, env), 8.0, 1e-12);
}

// ---------------------------------------------------------------------------
// sym_diff: derivatives of various functions
// ---------------------------------------------------------------------------

TEST(SymExtTest, diff_constant_is_zero) {
    auto expr = sym_const(42.0);
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(d, env), 0.0, 1e-12);
}

TEST(SymExtTest, diff_variable_wrt_self_is_one) {
    auto expr = sym_var("x");
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(d, env), 1.0, 1e-12);
}

TEST(SymExtTest, diff_variable_wrt_other_is_zero) {
    auto expr = sym_var("y");
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(d, env), 0.0, 1e-12);
}

TEST(SymExtTest, diff_sum_linearity) {
    // d/dx (x + x) = 2
    auto expr = sym_add(sym_var("x"), sym_var("x"));
    auto d = sym_diff(std::move(expr), "x");
    auto ds = sym_simplify(std::move(d));
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(ds, env), 2.0, 1e-12);
}

TEST(SymExtTest, diff_product_rule) {
    // d/dx (x * x) = x*1 + 1*x = 2x; at x=3 => 6
    auto expr = sym_mul(sym_var("x"), sym_var("x"));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", 3.0}};
    EXPECT_NEAR(sym_eval(d, env), 6.0, 1e-12);
}

TEST(SymExtTest, diff_sin_x_is_cos_x) {
    // d/dx sin(x) = cos(x); at x=0 => 1
    auto expr = sym_sin(sym_var("x"));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", 0.0}};
    EXPECT_NEAR(sym_eval(d, env), 1.0, 1e-12);
}

TEST(SymExtTest, diff_cos_x_at_pi) {
    // d/dx cos(x) = -sin(x); at x=pi => 0
    auto expr = sym_cos(sym_var("x"));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", M_PI}};
    EXPECT_NEAR(sym_eval(d, env), 0.0, 1e-10);
}

TEST(SymExtTest, diff_exp_x) {
    // d/dx exp(x) = exp(x); at x=0 => 1
    auto expr = sym_exp(sym_var("x"));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", 0.0}};
    EXPECT_NEAR(sym_eval(d, env), 1.0, 1e-12);
}

TEST(SymExtTest, diff_log_x) {
    // d/dx log(x) = 1/x = x^(-1); at x=1 => 1
    auto expr = sym_log(sym_var("x"));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", 1.0}};
    EXPECT_NEAR(sym_eval(d, env), 1.0, 1e-12);
}

TEST(SymExtTest, diff_pow_x_squared) {
    // d/dx x^2 = 2x; at x=4 => 8
    auto expr = sym_pow(sym_var("x"), sym_const(2.0));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", 4.0}};
    EXPECT_NEAR(sym_eval(d, env), 8.0, 1e-12);
}

TEST(SymExtTest, diff_pow_x_cubed) {
    // d/dx x^3 = 3x^2; at x=2 => 12
    auto expr = sym_pow(sym_var("x"), sym_const(3.0));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", 2.0}};
    EXPECT_NEAR(sym_eval(d, env), 12.0, 1e-12);
}

// ---------------------------------------------------------------------------
// sym_simplify: algebraic identity simplifications
// ---------------------------------------------------------------------------

TEST(SymExtTest, simplify_add_zero_left) {
    // 0 + x => x; eval at x=7 => 7
    auto expr = sym_add(sym_const(0.0), sym_var("x"));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Var);
}

TEST(SymExtTest, simplify_add_zero_right) {
    auto expr = sym_add(sym_var("x"), sym_const(0.0));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Var);
}

TEST(SymExtTest, simplify_mul_one_left) {
    auto expr = sym_mul(sym_const(1.0), sym_var("x"));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Var);
}

TEST(SymExtTest, simplify_mul_one_right) {
    auto expr = sym_mul(sym_var("x"), sym_const(1.0));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Var);
}

TEST(SymExtTest, simplify_mul_zero_gives_zero) {
    auto expr = sym_mul(sym_var("x"), sym_const(0.0));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Const);
    const std::map<std::string, double> env = {{"x", 999.0}};
    EXPECT_NEAR(sym_eval(s, env), 0.0, 1e-12);
}

TEST(SymExtTest, simplify_constant_fold_add) {
    auto expr = sym_add(sym_const(3.0), sym_const(4.0));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Const);
    EXPECT_NEAR(s.value, 7.0, 1e-12);
}

TEST(SymExtTest, simplify_constant_fold_mul) {
    auto expr = sym_mul(sym_const(5.0), sym_const(6.0));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Const);
    EXPECT_NEAR(s.value, 30.0, 1e-12);
}

TEST(SymExtTest, simplify_sin_constant) {
    auto expr = sym_sin(sym_const(0.0));
    auto s = sym_simplify(std::move(expr));
    EXPECT_EQ(s.op, SymOp::Const);
    EXPECT_NEAR(s.value, 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// sym_to_string: smoke tests
// ---------------------------------------------------------------------------

TEST(SymExtTest, to_string_const) {
    const auto s = sym_to_string(sym_const(3.14));
    EXPECT_FALSE(s.empty());
}

TEST(SymExtTest, to_string_add_not_empty) {
    const auto expr = sym_add(sym_var("a"), sym_var("b"));
    const auto s = sym_to_string(expr);
    EXPECT_FALSE(s.empty());
}

TEST(SymExtTest, to_string_mul_not_empty) {
    const auto expr = sym_mul(sym_var("x"), sym_const(2.0));
    const auto s = sym_to_string(expr);
    EXPECT_FALSE(s.empty());
}

// ---------------------------------------------------------------------------
// Chained: differentiate, simplify, evaluate pipeline
// ---------------------------------------------------------------------------

TEST(SymExtTest, chain_diff_simplify_eval) {
    // d/dx (3*x + 5) = 3; simplified from product rule result
    auto three = sym_const(3.0);
    auto five = sym_const(5.0);
    auto expr = sym_add(sym_mul(sym_const(3.0), sym_var("x")), std::move(five));
    auto d = sym_diff(std::move(expr), "x");
    auto ds = sym_simplify(std::move(d));
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(ds, env), 3.0, 1e-12);
    (void)three;
}

TEST(SymExtTest, chain_nested_derivatives) {
    // d/dx sin(x*x) = cos(x^2) * 2x; at x=0 => 0
    auto inner = sym_mul(sym_var("x"), sym_var("x"));
    auto expr = sym_sin(std::move(inner));
    auto d = sym_diff(std::move(expr), "x");
    const std::map<std::string, double> env = {{"x", 0.0}};
    EXPECT_NEAR(sym_eval(d, env), 0.0, 1e-12);
}
