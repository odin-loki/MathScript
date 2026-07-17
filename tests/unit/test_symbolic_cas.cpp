#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <numbers>
#include <vector>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

namespace {

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

void expect_eval_equivalent(
    const SymExpr& original, const SymExpr& transformed, const std::map<std::string, double>& env,
    double tol = 1e-6) {
    EXPECT_NEAR(sym_eval(original, env), sym_eval(transformed, env), tol);
}

template<typename... Args>
std::vector<SymExpr> make_equations(Args&&... args) {
    std::vector<SymExpr> equations;
    equations.reserve(sizeof...(Args));
    (equations.push_back(std::forward<Args>(args)), ...);
    return equations;
}

} // namespace

TEST(SymbolicLimitTest, sin_over_x_at_zero) {
    const auto expr = sym_div(sym_sin(sym_var("x")), sym_var("x"));
    EXPECT_NEAR(sym_limit(expr, "x", 0.0), 1.0, 1e-6);
}

TEST(SymbolicLimitTest, polynomial_at_point) {
    const auto expr = sym_add(sym_pow(sym_var("x"), sym_const(2.0)), sym_const(3.0));
    EXPECT_NEAR(sym_limit(expr, "x", 2.0), 7.0, 1e-9);
}

TEST(SymbolicLimitTest, removable_singularity) {
    const auto expr = sym_div(
        sym_sub(sym_pow(sym_var("x"), sym_const(2.0)), sym_const(1.0)),
        sym_sub(sym_var("x"), sym_const(1.0)));
    EXPECT_NEAR(sym_limit(expr, "x", 1.0), 2.0, 1e-5);
}

TEST(SymbolicLimitTest, constant_expression) {
    const auto expr = sym_const(5.5);
    EXPECT_NEAR(sym_limit(expr, "x", 100.0), 5.5, 1e-12);
}

TEST(SymbolicLimitTest, exp_at_zero) {
    const auto expr = sym_exp(sym_var("x"));
    EXPECT_NEAR(sym_limit(expr, "x", 0.0), 1.0, 1e-9);
}

TEST(SymbolicSeriesTest, sin_at_zero_order_four) {
    const auto expr = sym_sin(sym_var("x"));
    const auto series = sym_series(expr, "x", 0.0, 4);
    const auto expected = sym_sub(sym_var("x"), sym_div(sym_pow(sym_var("x"), sym_const(3.0)), sym_const(6.0)));

    for (const double x : {0.0, 0.1, 0.2, -0.15}) {
        expect_eval_equivalent(expected, series, {{"x", x}}, 1e-4);
    }
}

TEST(SymbolicSeriesTest, cos_at_zero_order_four) {
    const auto expr = sym_cos(sym_var("x"));
    const auto series = sym_series(expr, "x", 0.0, 4);
    const auto expected = sym_sub(
        sym_const(1.0),
        sym_div(sym_pow(sym_var("x"), sym_const(2.0)), sym_const(2.0)));

    for (const double x : {0.0, 0.1, 0.25}) {
        expect_eval_equivalent(expected, series, {{"x", x}}, 1e-4);
    }
}

TEST(SymbolicSeriesTest, exp_at_zero_order_four) {
    const auto expr = sym_exp(sym_var("x"));
    const auto series = sym_series(expr, "x", 0.0, 4);
    const auto expected = sym_add(
        sym_add(sym_const(1.0), sym_var("x")),
        sym_add(
            sym_div(sym_pow(sym_var("x"), sym_const(2.0)), sym_const(2.0)),
            sym_div(sym_pow(sym_var("x"), sym_const(3.0)), sym_const(6.0))));

    for (const double x : {0.0, 0.05, 0.2}) {
        expect_eval_equivalent(expected, series, {{"x", x}}, 1e-4);
    }
}

TEST(SymbolicSeriesTest, log_at_one_order_three) {
    const auto expr = sym_log(sym_var("x"));
    const auto series = sym_series(expr, "x", 1.0, 3);
    const auto expected = sym_sub(
        sym_sub(sym_var("x"), sym_const(1.0)),
        sym_div(sym_pow(sym_sub(sym_var("x"), sym_const(1.0)), sym_const(2.0)), sym_const(2.0)));

    for (const double x : {0.9, 1.0, 1.1, 1.2}) {
        expect_eval_equivalent(expected, series, {{"x", x}}, 1e-3);
    }
}

TEST(SymbolicSeriesTest, polynomial_exact_at_zero) {
    const auto expr = sym_add(
        sym_pow(sym_var("x"), sym_const(3.0)),
        sym_mul(sym_const(2.0), sym_var("x")));
    const auto series = sym_series(expr, "x", 0.0, 4);
    expect_eval_equivalent(expr, series, {{"x", 0.5}}, 1e-9);
    expect_eval_equivalent(expr, series, {{"x", -1.0}}, 1e-9);
}

TEST(SymbolicSeriesTest, zero_order_is_zero) {
    const auto expr = sym_sin(sym_var("x"));
    const auto series = sym_series(expr, "x", 0.0, 0);
    EXPECT_NEAR(sym_eval(series, {{"x", 1.0}}), 0.0, 1e-12);
}

TEST(SymbolicSolveLinearTest, single_variable_numeric) {
    auto eq = sym_add(sym_mul(sym_const(2.0), sym_var("x")), sym_const(4.0));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x"});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(sym_eval(result->at("x"), {}), -2.0, 1e-12);
}

TEST(SymbolicSolveLinearTest, single_variable_symbolic) {
    auto eq = sym_add(sym_mul(sym_var("a"), sym_var("x")), sym_var("b"));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x"});
    ASSERT_TRUE(result.has_value());
    const auto& x_expr = result->at("x");
    EXPECT_NEAR(sym_eval(x_expr, {{"a", 3.0}, {"b", 6.0}}), -2.0, 1e-12);
    EXPECT_NEAR(sym_eval(x_expr, {{"a", -2.0}, {"b", 4.0}}), 2.0, 1e-12);
}

TEST(SymbolicSolveLinearTest, two_by_two_numeric) {
    auto eq1 = sym_sub(
        sym_add(sym_mul(sym_const(2.0), sym_var("x")), sym_mul(sym_const(3.0), sym_var("y"))),
        sym_const(8.0));
    auto eq2 = sym_sub(
        sym_add(sym_var("x"), sym_neg(sym_var("y"))),
        sym_const(1.0));
    const auto result = sym_solve_linear(make_equations(std::move(eq1), std::move(eq2)), {"x", "y"});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(sym_eval(result->at("x"), {}), 2.2, 1e-9);
    EXPECT_NEAR(sym_eval(result->at("y"), {}), 1.2, 1e-9);
}

TEST(SymbolicSolveLinearTest, two_by_two_symbolic) {
    auto eq1 = sym_add(
        sym_add(sym_mul(sym_var("a"), sym_var("x")), sym_mul(sym_var("b"), sym_var("y"))),
        sym_neg(sym_var("e")));
    auto eq2 = sym_add(
        sym_add(sym_mul(sym_var("c"), sym_var("x")), sym_mul(sym_var("d"), sym_var("y"))),
        sym_neg(sym_var("f")));
    const auto result = sym_solve_linear(make_equations(std::move(eq1), std::move(eq2)), {"x", "y"});
    ASSERT_TRUE(result.has_value());

    const std::map<std::string, double> env{
        {"a", 2.0}, {"b", 3.0}, {"c", 1.0}, {"d", -1.0}, {"e", 8.0}, {"f", 1.0}};
    EXPECT_NEAR(sym_eval(result->at("x"), env), 2.2, 1e-9);
    EXPECT_NEAR(sym_eval(result->at("y"), env), 1.2, 1e-9);
}

TEST(SymbolicSolveLinearTest, mismatched_counts_fail) {
    auto eq = sym_add(sym_var("x"), sym_const(1.0));
    auto eq_copy = clone_expr(eq);
    const auto result = sym_solve_linear(make_equations(std::move(eq), std::move(eq_copy)), {"x"});
    ASSERT_FALSE(result.has_value());
}

TEST(SymbolicSolveLinearTest, singular_system_fails) {
    auto eq1 = sym_add(sym_var("x"), sym_var("y"));
    auto eq2 = sym_add(sym_mul(sym_const(2.0), sym_var("x")), sym_mul(sym_const(2.0), sym_var("y")));
    const auto result = sym_solve_linear(make_equations(std::move(eq1), std::move(eq2)), {"x", "y"});
    ASSERT_FALSE(result.has_value());
}
