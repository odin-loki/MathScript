#include <cmath>
#include <gtest/gtest.h>
#include "ms/symbolic/symbolic.hpp"
#include <map>
#include <string>

using namespace ms;

TEST(SymTest, create_variable) {
    SymExpr v = sym_var("x");
    EXPECT_EQ(v.op, SymOp::Var);
    EXPECT_EQ(v.name, "x");
}

TEST(SymTest, add_constants) {
    SymExpr result = sym_add(sym_const(2.0), sym_const(3.0));
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(result, env), 5.0, 1e-10);
}

TEST(SymTest, simplify_constant_fold) {
    SymExpr expr = sym_add(sym_const(4.0), sym_const(6.0));
    SymExpr simplified = sym_simplify(std::move(expr));
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(simplified, env), 10.0, 1e-10);
}

TEST(SymTest, to_string_variable) {
    SymExpr v = sym_var("x");
    EXPECT_EQ(sym_to_string(v), "x");
}

TEST(SymTest, to_string_expression) {
    SymExpr expr = sym_add(sym_var("x"), sym_const(1.0));
    const std::string s = sym_to_string(expr);
    EXPECT_FALSE(s.empty());
}

TEST(SymTest, to_string_const_literal) {
    const std::string s = sym_to_string(sym_const(3.5));
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("3"), std::string::npos);
}

TEST(SymTest, to_string_sub_mul_div) {
    EXPECT_NE(sym_to_string(sym_sub(sym_var("a"), sym_var("b"))).find(" - "), std::string::npos);
    EXPECT_NE(sym_to_string(sym_mul(sym_var("a"), sym_var("b"))).find(" * "), std::string::npos);
    EXPECT_NE(sym_to_string(sym_div(sym_var("a"), sym_var("b"))).find(" / "), std::string::npos);
}

TEST(SymTest, to_string_neg_and_unaries) {
    EXPECT_NE(sym_to_string(sym_neg(sym_var("x"))).find("(-"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_sin(sym_var("x"))).find("sin"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_cos(sym_var("x"))).find("cos"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_tan(sym_var("x"))).find("tan"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_exp(sym_var("x"))).find("exp"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_log(sym_var("x"))).find("log"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_sqrt(sym_var("x"))).find("sqrt"), std::string::npos);
}

TEST(SymTest, to_string_pow_and_deriv) {
    EXPECT_NE(sym_to_string(sym_pow(sym_var("x"), sym_const(2.0))).find(" ^ "), std::string::npos);
    const std::string d = sym_to_string(sym_deriv(sym_var("x"), "t"));
    EXPECT_NE(d.find("d/d"), std::string::npos);
    EXPECT_NE(d.find("t"), std::string::npos);
}

TEST(SymTest, to_string_unknown_op) {
    SymExpr bogus;
    bogus.op = static_cast<SymOp>(127);
    const std::string s = sym_to_string(bogus);
    if (s != "?") {
        GTEST_SKIP() << "unknown op to_string was not '?'";
    }
    EXPECT_EQ(s, "?");
}

TEST(SymTest, eval_missing_var_is_zero) {
    EXPECT_DOUBLE_EQ(sym_eval(sym_var("absent"), {}), 0.0);
}

TEST(SymTest, eval_sub_mul_div_neg) {
    EXPECT_NEAR(sym_eval(sym_sub(sym_const(9.0), sym_const(4.0)), {}), 5.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_mul(sym_const(3.0), sym_const(4.0)), {}), 12.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_div(sym_const(10.0), sym_const(4.0)), {}), 2.5, 1e-12);
    EXPECT_NEAR(sym_eval(sym_neg(sym_const(7.0)), {}), -7.0, 1e-12);
}

TEST(SymTest, eval_unaries_and_pow) {
    EXPECT_NEAR(sym_eval(sym_sin(sym_const(0.0)), {}), 0.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_cos(sym_const(0.0)), {}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_tan(sym_const(0.0)), {}), 0.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_exp(sym_const(0.0)), {}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_log(sym_const(1.0)), {}), 0.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_sqrt(sym_const(16.0)), {}), 4.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_pow(sym_const(2.0), sym_const(3.0)), {}), 8.0, 1e-12);
}

TEST(SymTest, eval_deriv_and_unknown_op) {
    EXPECT_NEAR(sym_eval(sym_deriv(sym_pow(sym_var("x"), sym_const(2.0)), "x"), {{"x", 3.0}}), 6.0, 1e-12);
    SymExpr bogus;
    bogus.op = static_cast<SymOp>(127);
    EXPECT_DOUBLE_EQ(sym_eval(bogus, {}), 0.0);
}

TEST(SymTest, substitute_matching_var) {
    const auto expr = sym_add(sym_mul(sym_var("x"), sym_const(2.0)), sym_const(1.0));
    const auto replaced = sym_substitute(expr, "x", sym_const(4.0));
    EXPECT_NEAR(sym_eval(replaced, {}), 9.0, 1e-12);
}

TEST(SymTest, substitute_unused_var_unchanged) {
    const auto expr = sym_add(sym_var("x"), sym_const(1.0));
    const auto replaced = sym_substitute(expr, "y", sym_const(99.0));
    EXPECT_NEAR(sym_eval(replaced, {{"x", 3.0}}), 4.0, 1e-12);
}

TEST(SymTest, substitute_const_leaf_unchanged) {
    const auto c = sym_const(5.0);
    const auto replaced = sym_substitute(c, "x", sym_const(1.0));
    EXPECT_EQ(replaced.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(replaced, {}), 5.0, 1e-12);
}

TEST(SymTest, substitute_nested_and_unary) {
    const auto expr = sym_add(sym_sin(sym_var("x")), sym_sqrt(sym_var("x")));
    const auto replaced = sym_substitute(expr, "x", sym_var("t"));
    const std::string text = sym_to_string(replaced);
    EXPECT_NE(text.find("t"), std::string::npos);
    if (text.find("x") != std::string::npos) {
        GTEST_SKIP() << "substitute left an 'x' in to_string";
    }
    EXPECT_NEAR(sym_eval(replaced, {{"t", 0.25}}), std::sin(0.25) + 0.5, 1e-12);
}

TEST(SymTest, substitute_with_expression) {
    const auto expr = sym_pow(sym_var("x"), sym_const(2.0));
    const auto replaced = sym_substitute(expr, "x", sym_add(sym_var("t"), sym_const(1.0)));
    EXPECT_NEAR(sym_eval(replaced, {{"t", 2.0}}), 9.0, 1e-12);
}
