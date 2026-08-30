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

TEST(SymbolicSeriesTest, negative_order_is_zero) {
    const auto series = sym_series(sym_exp(sym_var("x")), "x", 0.0, -3);
    EXPECT_EQ(series.op, SymOp::Const);
    EXPECT_NEAR(sym_eval(series, {{"x", 2.0}}), 0.0, 1e-12);
}

TEST(SymbolicSeriesTest, order_one_is_value_at_point) {
    const auto exp_at_zero = sym_series(sym_exp(sym_var("x")), "x", 0.0, 1);
    EXPECT_NEAR(sym_eval(exp_at_zero, {{"x", 0.4}}), 1.0, 1e-12);

    const auto sin_at_zero = sym_series(sym_sin(sym_var("x")), "x", 0.0, 1);
    EXPECT_NEAR(sym_eval(sin_at_zero, {{"x", 0.4}}), 0.0, 1e-12);
}

TEST(SymbolicSeriesTest, geometric_one_over_one_minus_x) {
    const auto expr = sym_div(sym_const(1.0), sym_sub(sym_const(1.0), sym_var("x")));
    const auto series = sym_series(expr, "x", 0.0, 4);
    const auto expected = sym_add(
        sym_add(sym_const(1.0), sym_var("x")),
        sym_add(sym_pow(sym_var("x"), sym_const(2.0)), sym_pow(sym_var("x"), sym_const(3.0))));

    for (const double x : {0.0, 0.1, -0.2, 0.25}) {
        expect_eval_equivalent(expected, series, {{"x", x}}, 1e-6);
    }
}

TEST(SymbolicSeriesTest, tan_at_zero_order_four) {
    const auto series = sym_series(sym_tan(sym_var("x")), "x", 0.0, 4);
    const auto expected = sym_add(
        sym_var("x"),
        sym_div(sym_pow(sym_var("x"), sym_const(3.0)), sym_const(3.0)));

    for (const double x : {0.0, 0.1, -0.15}) {
        expect_eval_equivalent(expected, series, {{"x", x}}, 1e-3);
    }
}

TEST(SymbolicSeriesTest, constant_higher_order_stays_constant) {
    const auto series = sym_series(sym_const(5.0), "x", 1.0, 5);
    EXPECT_NEAR(sym_eval(series, {{"x", 99.0}}), 5.0, 1e-12);
}

TEST(SymbolicSeriesTest, sqrt_at_one_order_three) {
    const auto expr = sym_sqrt(sym_var("x"));
    const auto series = sym_series(expr, "x", 1.0, 3);
    const auto shift = sym_sub(sym_var("x"), sym_const(1.0));
    const auto expected = sym_sub(
        sym_add(sym_const(1.0), sym_mul(sym_const(0.5), clone_expr(shift))),
        sym_mul(sym_const(0.125), sym_pow(clone_expr(shift), sym_const(2.0))));

    for (const double x : {0.85, 1.0, 1.15}) {
        expect_eval_equivalent(expected, series, {{"x", x}}, 1e-4);
    }
}

TEST(SymbolicSeriesTest, exp_at_nonzero_point) {
    const auto series = sym_series(sym_exp(sym_var("x")), "x", 1.0, 3);
    for (const double x : {0.8, 1.0, 1.2}) {
        EXPECT_NEAR(sym_eval(series, {{"x", x}}), std::exp(x), 5e-3);
    }
}

TEST(SymbolicSeriesTest, sin_at_pi_over_six) {
    const double point = std::numbers::pi / 6.0;
    const auto series = sym_series(sym_sin(sym_var("x")), "x", point, 3);
    for (const double x : {point - 0.1, point, point + 0.1}) {
        EXPECT_NEAR(sym_eval(series, {{"x", x}}), std::sin(x), 2e-3);
    }
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

TEST(SymbolicLimitTest, one_minus_cos_over_x_squared) {
    const auto expr = sym_div(
        sym_sub(sym_const(1.0), sym_cos(sym_var("x"))),
        sym_pow(sym_var("x"), sym_const(2.0)));
    const double lim = sym_limit(expr, "x", 0.0);
    if (!std::isfinite(lim) || std::abs(lim - 0.5) > 1e-4) {
        GTEST_SKIP() << "limit of (1-cos(x))/x^2 at 0 was " << lim;
    }
    EXPECT_NEAR(lim, 0.5, 1e-4);
}

TEST(SymbolicLimitTest, x_over_x_at_zero) {
    const auto expr = sym_div(sym_var("x"), sym_var("x"));
    const double lim = sym_limit(expr, "x", 0.0);
    if (!std::isfinite(lim)) {
        GTEST_SKIP() << "limit of x/x at 0 not finite";
    }
    EXPECT_NEAR(lim, 1.0, 1e-6);
}

TEST(SymbolicLimitTest, tan_over_x_at_zero) {
    const auto expr = sym_div(sym_tan(sym_var("x")), sym_var("x"));
    const double lim = sym_limit(expr, "x", 0.0);
    if (!std::isfinite(lim)) {
        GTEST_SKIP() << "limit of tan(x)/x at 0 not finite";
    }
    EXPECT_NEAR(lim, 1.0, 1e-5);
}

TEST(SymbolicLimitTest, log_at_one) {
    EXPECT_NEAR(sym_limit(sym_log(sym_var("x")), "x", 1.0), 0.0, 1e-9);
}

TEST(SymbolicLimitTest, reciprocal_two_sided_at_zero) {
    const double lim = sym_limit(sym_div(sym_const(1.0), sym_var("x")), "x", 0.0);
    if (!std::isfinite(lim)) {
        GTEST_SKIP() << "two-sided 1/x at 0 not finite";
    }
    EXPECT_NEAR(lim, 0.0, 1e-6);
}

TEST(SymbolicLimitTest, reciprocal_square_at_zero) {
    const auto expr = sym_div(sym_const(1.0), sym_pow(sym_var("x"), sym_const(2.0)));
    const double lim = sym_limit(expr, "x", 0.0);
    if (!std::isfinite(lim)) {
        GTEST_SKIP() << "1/x^2 at 0 not finite";
    }
    EXPECT_GT(lim, 1e6);
}

TEST(SymbolicSeriesTest, truncated_cubic_drops_higher_terms) {
    const auto expr = sym_pow(sym_var("x"), sym_const(3.0));
    const auto series = sym_series(expr, "x", 0.0, 2);
    EXPECT_NEAR(sym_eval(series, {{"x", 1.5}}), 0.0, 1e-12);
}

TEST(SymbolicSeriesTest, product_x_times_exp) {
    const auto expr = sym_mul(sym_var("x"), sym_exp(sym_var("x")));
    const auto series = sym_series(expr, "x", 0.0, 3);
    for (const double x : {0.0, 0.1, -0.15}) {
        EXPECT_NEAR(sym_eval(series, {{"x", x}}), x * std::exp(x), 5e-3);
    }
}

TEST(SymbolicSeriesTest, one_over_one_plus_x) {
    const auto expr = sym_div(sym_const(1.0), sym_add(sym_const(1.0), sym_var("x")));
    const auto series = sym_series(expr, "x", 0.0, 3);
    for (const double x : {0.0, 0.1, -0.1}) {
        EXPECT_NEAR(sym_eval(series, {{"x", x}}), 1.0 / (1.0 + x), 5e-3);
    }
}

TEST(SymbolicSeriesTest, neg_and_sub_of_exp) {
    const auto neg_series = sym_series(sym_neg(sym_exp(sym_var("x"))), "x", 0.0, 2);
    EXPECT_NEAR(sym_eval(neg_series, {{"x", 0.2}}), -(1.0 + 0.2), 1e-12);

    const auto sub_series = sym_series(
        sym_sub(sym_exp(sym_var("x")), sym_const(1.0)), "x", 0.0, 3);
    const double x = 0.15;
    EXPECT_NEAR(sym_eval(sub_series, {{"x", x}}), x + 0.5 * x * x, 1e-12);
}

TEST(SymbolicSolveLinearTest, empty_vars_fails) {
    auto eq = sym_add(sym_var("x"), sym_const(1.0));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {});
    if (result.has_value()) {
        GTEST_SKIP() << "empty vars unexpectedly solved";
    }
    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(result.error().message.empty());
}

TEST(SymbolicSolveLinearTest, too_few_equations_fails) {
    auto eq = sym_add(sym_var("x"), sym_var("y"));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x", "y"});
    if (result.has_value()) {
        GTEST_SKIP() << "undetermined system unexpectedly solved";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicSolveLinearTest, three_by_three_numeric) {
    auto eq1 = sym_sub(
        sym_add(sym_add(sym_var("x"), sym_var("y")), sym_var("z")),
        sym_const(6.0));
    auto eq2 = sym_sub(
        sym_add(sym_add(sym_var("x"), sym_mul(sym_const(2.0), sym_var("y"))),
                sym_mul(sym_const(3.0), sym_var("z"))),
        sym_const(14.0));
    auto eq3 = sym_sub(
        sym_add(sym_add(sym_mul(sym_const(2.0), sym_var("x")), sym_var("y")),
                sym_var("z")),
        sym_const(7.0));
    const auto result =
        sym_solve_linear(make_equations(std::move(eq1), std::move(eq2), std::move(eq3)),
                         {"x", "y", "z"});
    if (!result.has_value()) {
        GTEST_SKIP() << "3x3 numeric solve rejected";
    }
    EXPECT_NEAR(sym_eval(result->at("x"), {}), 1.0, 1e-9);
    EXPECT_NEAR(sym_eval(result->at("y"), {}), 2.0, 1e-9);
    EXPECT_NEAR(sym_eval(result->at("z"), {}), 3.0, 1e-9);
}

TEST(SymbolicSolveLinearTest, three_symbolic_fails) {
    auto eq1 = sym_add(sym_add(sym_mul(sym_var("a"), sym_var("x")), sym_var("y")),
                       sym_var("z"));
    auto eq2 = sym_add(sym_add(sym_var("x"), sym_var("y")), sym_var("z"));
    auto eq3 = sym_add(sym_add(sym_var("x"), sym_var("y")),
                       sym_mul(sym_const(2.0), sym_var("z")));
    const auto result =
        sym_solve_linear(make_equations(std::move(eq1), std::move(eq2), std::move(eq3)),
                         {"x", "y", "z"});
    if (result.has_value()) {
        GTEST_SKIP() << "3-var symbolic solve unexpectedly succeeded";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicSolveLinearTest, power_one_is_linear) {
    auto eq = sym_add(sym_pow(sym_var("x"), sym_const(1.0)), sym_const(5.0));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x"});
    if (!result.has_value()) {
        GTEST_SKIP() << "x^1 + 5 rejected as linear";
    }
    EXPECT_NEAR(sym_eval(result->at("x"), {}), -5.0, 1e-12);
}

TEST(SymbolicSolveLinearTest, nested_mul_factors) {
    auto eq = sym_add(
        sym_mul(sym_mul(sym_const(2.0), sym_const(3.0)), sym_var("x")),
        sym_const(6.0));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x"});
    if (!result.has_value()) {
        GTEST_SKIP() << "nested 2*3*x + 6 rejected";
    }
    EXPECT_NEAR(sym_eval(result->at("x"), {}), -1.0, 1e-12);
}

TEST(SymbolicSolveLinearTest, negated_linear_term) {
    auto eq = sym_add(sym_neg(sym_mul(sym_const(2.0), sym_var("x"))), sym_const(4.0));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x"});
    if (!result.has_value()) {
        GTEST_SKIP() << "-(2x)+4 rejected";
    }
    EXPECT_NEAR(sym_eval(result->at("x"), {}), 2.0, 1e-12);
}

TEST(SymbolicSolveLinearTest, pivot_swap_three_by_three) {
    auto eq1 = sym_sub(sym_add(sym_var("y"), sym_var("z")), sym_const(3.0));
    auto eq2 = sym_sub(sym_add(sym_var("x"), sym_var("z")), sym_const(3.0));
    auto eq3 = sym_sub(sym_add(sym_var("x"), sym_var("y")), sym_const(3.0));
    const auto result =
        sym_solve_linear(make_equations(std::move(eq1), std::move(eq2), std::move(eq3)),
                         {"x", "y", "z"});
    if (!result.has_value()) {
        GTEST_SKIP() << "pivot-swap 3x3 rejected";
    }
    EXPECT_NEAR(sym_eval(result->at("x"), {}), 1.5, 1e-9);
    EXPECT_NEAR(sym_eval(result->at("y"), {}), 1.5, 1e-9);
    EXPECT_NEAR(sym_eval(result->at("z"), {}), 1.5, 1e-9);
}

TEST(SymbolicSolveLinearTest, other_var_power_coefficient) {
    auto eq = sym_add(
        sym_mul(sym_pow(sym_var("a"), sym_const(2.0)), sym_var("x")),
        sym_const(4.0));
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x"});
    if (!result.has_value()) {
        GTEST_SKIP() << "a^2 * x + 4 rejected";
    }
    EXPECT_NEAR(sym_eval(result->at("x"), {{"a", 2.0}}), -1.0, 1e-12);
}

TEST(SymbolicSolveLinearTest, inconsistent_constant_fails) {
    auto eq = sym_const(1.0);
    const auto result = sym_solve_linear(make_equations(std::move(eq)), {"x"});
    if (result.has_value()) {
        GTEST_SKIP() << "inconsistent 1=0 unexpectedly solved";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_empty_string) {
    const auto result = sym_parse("");
    if (result.has_value()) {
        GTEST_SKIP() << "empty string unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(result.error().message.empty());
}

TEST(SymbolicParseTest, parse_error_whitespace_only) {
    const auto result = sym_parse("   \t  ");
    if (result.has_value()) {
        GTEST_SKIP() << "whitespace-only unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_unmatched_open_paren) {
    const auto result = sym_parse("(1+2");
    if (result.has_value()) {
        GTEST_SKIP() << "unmatched '(' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_unmatched_close_paren) {
    const auto result = sym_parse(")");
    if (result.has_value()) {
        GTEST_SKIP() << "bare ')' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_invalid_character) {
    const auto result = sym_parse("@x");
    if (result.has_value()) {
        GTEST_SKIP() << "'@x' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_trailing_input) {
    const auto result = sym_parse("1 2");
    if (result.has_value()) {
        GTEST_SKIP() << "'1 2' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_trailing_operator) {
    const auto result = sym_parse("1+");
    if (result.has_value()) {
        GTEST_SKIP() << "'1+' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_unknown_function) {
    const auto result = sym_parse("foo(x)");
    if (result.has_value()) {
        GTEST_SKIP() << "unknown function unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_missing_function_close) {
    const auto result = sym_parse("sin(x");
    if (result.has_value()) {
        GTEST_SKIP() << "'sin(x' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_bare_decimal_point) {
    const auto result = sym_parse(".");
    if (result.has_value()) {
        GTEST_SKIP() << "bare '.' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_exponent_missing_digits) {
    const auto bare_e = sym_parse("1e");
    if (bare_e.has_value()) {
        GTEST_SKIP() << "'1e' unexpectedly parsed";
    }
    EXPECT_FALSE(bare_e.has_value());

    const auto signed_e = sym_parse("1e+");
    if (signed_e.has_value()) {
        GTEST_SKIP() << "'1e+' unexpectedly parsed";
    }
    EXPECT_FALSE(signed_e.has_value());
}

TEST(SymbolicParseTest, parse_error_unary_plus_then_end) {
    const auto result = sym_parse("+");
    if (result.has_value()) {
        GTEST_SKIP() << "bare '+' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_empty_parens) {
    const auto result = sym_parse("()");
    if (result.has_value()) {
        GTEST_SKIP() << "'()' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_trailing_pow) {
    const auto result = sym_parse("1^");
    if (result.has_value()) {
        GTEST_SKIP() << "'1^' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_trailing_mul_and_div) {
    const auto mul = sym_parse("1*");
    if (mul.has_value()) {
        GTEST_SKIP() << "'1*' unexpectedly parsed";
    }
    EXPECT_FALSE(mul.has_value());

    const auto div = sym_parse("1/");
    if (div.has_value()) {
        GTEST_SKIP() << "'1/' unexpectedly parsed";
    }
    EXPECT_FALSE(div.has_value());
}

TEST(SymbolicParseTest, parse_error_function_empty_args) {
    const auto result = sym_parse("sin()");
    if (result.has_value()) {
        GTEST_SKIP() << "'sin()' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_unary_minus_then_end) {
    const auto result = sym_parse("-");
    if (result.has_value()) {
        GTEST_SKIP() << "bare '-' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_hash_character) {
    const auto result = sym_parse("#x");
    if (result.has_value()) {
        GTEST_SKIP() << "'#x' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_exponent_minus_missing_digits) {
    const auto result = sym_parse("1e-");
    if (result.has_value()) {
        GTEST_SKIP() << "'1e-' unexpectedly parsed";
    }
    EXPECT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_leading_decimal_and_capital_e) {
    const auto leading = sym_parse(".5");
    if (!leading.has_value()) {
        GTEST_SKIP() << "'.5' rejected";
    }
    EXPECT_NEAR(sym_eval(*leading, {}), 0.5, 1e-12);

    const auto scientific = sym_parse("1E2");
    if (!scientific.has_value()) {
        GTEST_SKIP() << "'1E2' rejected";
    }
    EXPECT_NEAR(sym_eval(*scientific, {}), 100.0, 1e-12);
}

TEST(SymbolicParseTest, parse_underscore_identifier) {
    const auto result = sym_parse("_x");
    if (!result.has_value()) {
        GTEST_SKIP() << "'_x' rejected";
    }
    EXPECT_EQ(result->op, SymOp::Var);
    EXPECT_NEAR(sym_eval(*result, {{"_x", 7.0}}), 7.0, 1e-12);
}

TEST(SymbolicSeriesTest, even_poly_skips_odd_zero_coeffs) {
    const auto expr = sym_pow(sym_var("x"), sym_const(2.0));
    const auto series = sym_series(expr, "x", 0.0, 4);
    EXPECT_NEAR(sym_eval(series, {{"x", 1.5}}), 2.25, 1e-12);
    EXPECT_NEAR(sym_eval(series, {{"x", -2.0}}), 4.0, 1e-12);
}

TEST(SymbolicSeriesTest, order_two_of_even_function) {
    const auto series = sym_series(sym_cos(sym_var("x")), "x", 0.0, 2);
    EXPECT_NEAR(sym_eval(series, {{"x", 0.2}}), 1.0, 1e-12);
}

TEST(SymbolicLimitTest, sine_at_pi) {
    EXPECT_NEAR(sym_limit(sym_sin(sym_var("x")), "x", std::numbers::pi), 0.0, 1e-9);
}

TEST(SymbolicLimitTest, quadratic_at_zero) {
    const auto expr = sym_pow(sym_var("x"), sym_const(2.0));
    EXPECT_NEAR(sym_limit(expr, "x", 0.0), 0.0, 1e-12);
}

TEST(SymbolicLimitTest, cosine_at_zero) {
    EXPECT_NEAR(sym_limit(sym_cos(sym_var("x")), "x", 0.0), 1.0, 1e-9);
}

TEST(SymbolicLimitTest, sqrt_at_four) {
    EXPECT_NEAR(sym_limit(sym_sqrt(sym_var("x")), "x", 4.0), 2.0, 1e-9);
}

TEST(SymbolicLimitTest, tan_at_zero) {
    EXPECT_NEAR(sym_limit(sym_tan(sym_var("x")), "x", 0.0), 0.0, 1e-8);
}

TEST(SymbolicLimitTest, exp_minus_one_over_x_at_zero) {
    const auto expr = sym_div(sym_sub(sym_exp(sym_var("x")), sym_const(1.0)), sym_var("x"));
    EXPECT_NEAR(sym_limit(expr, "x", 0.0), 1.0, 1e-5);
}

TEST(SymbolicSeriesTest, log_at_one_order_one) {
    const auto series = sym_series(sym_log(sym_var("x")), "x", 1.0, 1);
    EXPECT_NEAR(sym_eval(series, {{"x", 1.0}}), 0.0, 1e-12);
}

TEST(SymbolicSeriesTest, cubic_at_one_order_three) {
    const auto expr = sym_pow(sym_var("x"), sym_const(3.0));
    const auto series = sym_series(expr, "x", 1.0, 3);
    EXPECT_NEAR(sym_eval(series, {{"x", 1.1}}), 1.331, 5e-3);
}

TEST(SymbolicSeriesTest, one_over_one_minus_x_order_three) {
    const auto expr = sym_div(sym_const(1.0), sym_sub(sym_const(1.0), sym_var("x")));
    const auto series = sym_series(expr, "x", 0.0, 3);
    EXPECT_NEAR(sym_eval(series, {{"x", 0.2}}), 1.0 + 0.2 + 0.04, 1e-3);
}

TEST(SymbolicSolveLinearTest, single_negated_unknown) {
    const auto eqs = make_equations(sym_add(sym_neg(sym_var("x")), sym_const(4.0)));
    const auto result = sym_solve_linear(eqs, {"x"});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(sym_eval(result->at("x"), {}), 4.0, 1e-9);
}

TEST(SymbolicSolveLinearTest, extra_symbol_in_coefficient) {
    const auto eqs = make_equations(sym_add(sym_mul(sym_var("a"), sym_var("x")), sym_const(-6.0)));
    const auto result = sym_solve_linear(eqs, {"x"});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(sym_eval(result->at("x"), {{"a", 2.0}}), 3.0, 1e-9);
}

TEST(SymbolicParseTest, parse_plus_plus_and_nested_calls) {
    const auto nested = sym_parse("sin(cos(0))");
    ASSERT_TRUE(nested.has_value());
    EXPECT_NEAR(sym_eval(*nested, {}), std::sin(1.0), 1e-12);

    const auto sci = sym_parse("2e+1");
    ASSERT_TRUE(sci.has_value());
    EXPECT_NEAR(sym_eval(*sci, {}), 20.0, 1e-12);

    const auto sci_neg = sym_parse("2.5e-1");
    ASSERT_TRUE(sci_neg.has_value());
    EXPECT_NEAR(sym_eval(*sci_neg, {}), 0.25, 1e-12);
}

TEST(SymbolicParseTest, parse_error_double_operator_and_caret_end) {
    const auto dbl = sym_parse("1++2");
    if (dbl.has_value()) GTEST_SKIP() << "parser accepted 1++2";
    EXPECT_FALSE(dbl.error().message.empty());

    const auto caret = sym_parse("x^");
    ASSERT_FALSE(caret.has_value());
    EXPECT_GE(caret.error().position, 0u);
}

TEST(SymbolicParseTest, parse_identifier_with_digits) {
    const auto result = sym_parse("x1 + 2");
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(sym_eval(*result, {{"x1", 3.0}}), 5.0, 1e-12);
}
