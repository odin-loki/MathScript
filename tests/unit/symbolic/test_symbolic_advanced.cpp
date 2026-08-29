#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <numbers>
#include <string>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

namespace {

double eval_at(const SymExpr& expr, double x_val, const std::string& var = "x") {
    return sym_eval(expr, {{var, x_val}});
}

double eval2(const SymExpr& expr, double x_val, double y_val) {
    return sym_eval(expr, {{"x", x_val}, {"y", y_val}});
}

} // namespace

TEST(SymAdvancedTest, differentiate_product_rule) {
    auto expr = sym_mul(sym_var("x"), sym_sin(sym_var("x")));
    const auto dx = sym_simplify(sym_diff(std::move(expr), "x"));
    const auto expected = sym_add(sym_sin(sym_var("x")), sym_mul(sym_var("x"), sym_cos(sym_var("x"))));
    for (double x = 0.1; x < 2.0; x += 0.37) {
        EXPECT_NEAR(eval_at(dx, x), eval_at(expected, x), 1e-9);
    }
}

TEST(SymAdvancedTest, differentiate_chain_rule) {
    auto inner = sym_pow(sym_var("x"), sym_const(2.0));
    auto expr = sym_sin(std::move(inner));
    const auto dx = sym_simplify(sym_diff(std::move(expr), "x"));
    const auto expected =
        sym_mul(sym_mul(sym_const(2.0), sym_var("x")), sym_cos(sym_pow(sym_var("x"), sym_const(2.0))));
    for (double x = 0.2; x < 2.5; x += 0.41) {
        EXPECT_NEAR(eval_at(dx, x), eval_at(expected, x), 1e-9);
    }
}

TEST(SymAdvancedTest, simplify_trig_identity) {
    auto expr = sym_add(sym_pow(sym_sin(sym_var("x")), sym_const(2.0)),
                        sym_pow(sym_cos(sym_var("x")), sym_const(2.0)));
    const auto simplified = sym_simplify(std::move(expr));
    for (double x = 0.0; x < 3.0; x += 0.5) {
        EXPECT_NEAR(eval_at(simplified, x), 1.0, 1e-9);
    }
}

TEST(SymAdvancedTest, nested_simplify) {
    auto make_expr = [] {
        return sym_add(sym_mul(sym_const(2.0), sym_var("x")), sym_mul(sym_const(3.0), sym_var("x")));
    };
    const auto once = sym_simplify(make_expr());
    const auto twice = sym_simplify(sym_simplify(make_expr()));
    EXPECT_NEAR(sym_eval(once, {{"x", 4.0}}), sym_eval(twice, {{"x", 4.0}}), 1e-12);
    EXPECT_NEAR(sym_eval(once, {{"x", 1.0}}), 5.0, 1e-12);
}

TEST(SymAdvancedTest, eval_with_variable) {
    const auto expr = sym_add(sym_var("x"), sym_const(1.0));
    EXPECT_DOUBLE_EQ(sym_eval(expr, {{"x", 3.0}}), 4.0);
}

TEST(SymAdvancedTest, to_string_roundtrip) {
    auto expr = sym_add(sym_mul(sym_var("x"), sym_const(2.0)), sym_var("y"));
    const auto text = sym_to_string(expr);
    EXPECT_NE(text.find("x"), std::string::npos);
    EXPECT_NE(text.find("y"), std::string::npos);
    EXPECT_NE(text.find("2"), std::string::npos);
}

TEST(SymAdvancedTest, poly_degree_zero) {
    const auto dx = sym_simplify(sym_diff(sym_const(7.0), "x"));
    EXPECT_NEAR(sym_eval(dx, {}), 0.0, 1e-12);
}

TEST(SymAdvancedTest, large_polynomial) {
    auto expr = sym_pow(sym_var("x"), sym_const(10.0));
    const auto dx = sym_simplify(sym_diff(std::move(expr), "x"));
    const auto expected = sym_mul(sym_const(10.0), sym_pow(sym_var("x"), sym_const(9.0)));
    for (double x = 0.5; x < 3.0; x += 0.7) {
        EXPECT_NEAR(eval_at(dx, x), eval_at(expected, x), 1e-6);
    }
}

TEST(SymAdvancedTest, multiple_variables) {
    const auto d_x = sym_simplify(sym_diff(sym_mul(sym_var("x"), sym_var("y")), "x"));
    const auto d_y = sym_simplify(sym_diff(sym_mul(sym_var("x"), sym_var("y")), "y"));
    EXPECT_NEAR(eval2(d_x, 3.0, 5.0), 5.0, 1e-12);
    EXPECT_NEAR(eval2(d_y, 3.0, 5.0), 3.0, 1e-12);
}
