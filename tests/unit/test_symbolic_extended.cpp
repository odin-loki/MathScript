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

double finite_diff(const SymExpr& expr, const std::string& var, double x_val, double h = 1e-6) {
    const std::map<std::string, double> env_plus{{var, x_val + h}};
    const std::map<std::string, double> env_minus{{var, x_val - h}};
    return (sym_eval(expr, env_plus) - sym_eval(expr, env_minus)) / (2.0 * h);
}

void expect_deriv_matches_finite_diff(
    const SymExpr& expr, const std::string& var, double x_val, double tol = 1e-5) {
    const auto deriv = sym_simplify(sym_diff(clone_expr(expr), var));
    const double analytic = sym_eval(deriv, {{var, x_val}});
    const double numeric = finite_diff(expr, var, x_val);
    EXPECT_NEAR(analytic, numeric, tol);
}

double trapezoid_integral(const SymExpr& expr, const std::string& var, double a, double b, int n) {
    const double h = (b - a) / static_cast<double>(n);
    double sum = 0.5 * (sym_eval(expr, {{var, a}}) + sym_eval(expr, {{var, b}}));
    for (int i = 1; i < n; ++i) {
        sum += sym_eval(expr, {{var, a + h * static_cast<double>(i)}});
    }
    return sum * h;
}

void expect_integrate_roundtrip(const SymExpr& expr, const std::string& var, double x_val) {
    const auto integral = sym_simplify(sym_integrate(expr, var));
    const auto d_integral = sym_simplify(sym_diff(clone_expr(integral), var));
    EXPECT_NEAR(sym_eval(d_integral, {{var, x_val}}), sym_eval(expr, {{var, x_val}}), 1e-5);
}

bool has_mul_with_sum_factor(const SymExpr& expr) {
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Add || expr.left->op == SymOp::Sub || expr.right->op == SymOp::Add ||
            expr.right->op == SymOp::Sub) {
            return true;
        }
    }
    if (expr.left && has_mul_with_sum_factor(*expr.left)) {
        return true;
    }
    if (expr.right && has_mul_with_sum_factor(*expr.right)) {
        return true;
    }
    return false;
}

void expect_eval_equivalent(
    const SymExpr& original, const SymExpr& transformed, const std::map<std::string, double>& env,
    double tol = 1e-9) {
    EXPECT_NEAR(sym_eval(original, env), sym_eval(transformed, env), tol);
}

} // namespace

TEST(SymbolicExtendedTest, sub_eval_and_to_string) {
    const auto expr = sym_sub(sym_const(7.0), sym_const(2.5));
    EXPECT_NEAR(sym_eval(expr, {}), 4.5, 1e-12);
    EXPECT_NE(sym_to_string(sym_sub(sym_var("x"), sym_const(1.0))).find(" - "), std::string::npos);
}

TEST(SymbolicExtendedTest, sub_simplify_identities) {
    const auto minus_zero = sym_simplify(sym_sub(sym_var("x"), sym_const(0.0)));
    EXPECT_NEAR(sym_eval(minus_zero, {{"x", 3.0}}), 3.0, 1e-12);

    const auto zero_minus = sym_simplify(sym_sub(sym_const(0.0), sym_var("x")));
    EXPECT_NEAR(sym_eval(zero_minus, {{"x", 4.0}}), -4.0, 1e-12);

    const auto folded = sym_simplify(sym_sub(sym_const(9.0), sym_const(4.0)));
    EXPECT_NEAR(sym_eval(folded, {}), 5.0, 1e-12);
}

TEST(SymbolicExtendedTest, sub_diff_finite_difference) {
    const auto expr = sym_sub(sym_pow(sym_var("x"), sym_const(2.0)), sym_mul(sym_const(3.0), sym_var("x")));
    expect_deriv_matches_finite_diff(expr, "x", 1.5);
    expect_deriv_matches_finite_diff(expr, "x", -0.75);
}

TEST(SymbolicExtendedTest, div_eval_and_to_string) {
    const auto expr = sym_div(sym_const(8.0), sym_const(2.0));
    EXPECT_NEAR(sym_eval(expr, {}), 4.0, 1e-12);
    EXPECT_NE(sym_to_string(sym_div(sym_var("x"), sym_const(2.0))).find(" / "), std::string::npos);
}

TEST(SymbolicExtendedTest, div_simplify_identities) {
    const auto over_one = sym_simplify(sym_div(sym_var("x"), sym_const(1.0)));
    EXPECT_NEAR(sym_eval(over_one, {{"x", 5.0}}), 5.0, 1e-12);

    const auto zero_over = sym_simplify(sym_div(sym_const(0.0), sym_var("x")));
    EXPECT_NEAR(sym_eval(zero_over, {{"x", 2.0}}), 0.0, 1e-12);

    const auto folded = sym_simplify(sym_div(sym_const(15.0), sym_const(3.0)));
    EXPECT_NEAR(sym_eval(folded, {}), 5.0, 1e-12);
}

TEST(SymbolicExtendedTest, div_diff_finite_difference) {
    const auto expr = sym_div(sym_var("x"), sym_add(sym_const(1.0), sym_var("x")));
    expect_deriv_matches_finite_diff(expr, "x", 0.5);
    expect_deriv_matches_finite_diff(expr, "x", 2.0);
}

TEST(SymbolicExtendedTest, neg_eval_and_to_string) {
    const auto expr = sym_neg(sym_const(6.0));
    EXPECT_NEAR(sym_eval(expr, {}), -6.0, 1e-12);
    EXPECT_NE(sym_to_string(sym_neg(sym_var("x"))).find("(-"), std::string::npos);
}

TEST(SymbolicExtendedTest, neg_simplify_identities) {
    const auto double_neg = sym_simplify(sym_neg(sym_neg(sym_var("x"))));
    EXPECT_NEAR(sym_eval(double_neg, {{"x", 2.0}}), 2.0, 1e-12);

    const auto folded = sym_simplify(sym_neg(sym_const(2.5)));
    EXPECT_NEAR(sym_eval(folded, {}), -2.5, 1e-12);
}

TEST(SymbolicExtendedTest, neg_diff_finite_difference) {
    const auto expr = sym_neg(sym_sin(sym_var("x")));
    expect_deriv_matches_finite_diff(expr, "x", 0.3);
    expect_deriv_matches_finite_diff(expr, "x", 1.1);
}

TEST(SymbolicExtendedTest, tan_eval_and_to_string) {
    const auto expr = sym_tan(sym_const(0.0));
    EXPECT_NEAR(sym_eval(expr, {}), 0.0, 1e-12);
    EXPECT_NE(sym_to_string(sym_tan(sym_var("x"))).find("tan"), std::string::npos);
}

TEST(SymbolicExtendedTest, tan_simplify_const) {
    const auto folded = sym_simplify(sym_tan(sym_const(std::numbers::pi / 4.0)));
    EXPECT_NEAR(sym_eval(folded, {}), 1.0, 1e-12);
}

TEST(SymbolicExtendedTest, tan_diff_finite_difference) {
    const auto expr = sym_tan(sym_var("x"));
    expect_deriv_matches_finite_diff(expr, "x", 0.2);
    expect_deriv_matches_finite_diff(expr, "x", -0.4);
}

TEST(SymbolicExtendedTest, sqrt_eval_and_to_string) {
    const auto expr = sym_sqrt(sym_const(9.0));
    EXPECT_NEAR(sym_eval(expr, {}), 3.0, 1e-12);
    EXPECT_NE(sym_to_string(sym_sqrt(sym_var("x"))).find("sqrt"), std::string::npos);
}

TEST(SymbolicExtendedTest, sqrt_simplify_const) {
    const auto folded = sym_simplify(sym_sqrt(sym_const(2.0)));
    EXPECT_NEAR(sym_eval(folded, {}), std::sqrt(2.0), 1e-12);
}

TEST(SymbolicExtendedTest, sqrt_diff_finite_difference) {
    const auto expr = sym_sqrt(sym_add(sym_var("x"), sym_const(1.0)));
    expect_deriv_matches_finite_diff(expr, "x", 1.0);
    expect_deriv_matches_finite_diff(expr, "x", 4.0);
}

TEST(SymbolicExtendedTest, integrate_constant) {
    const auto integral = sym_simplify(sym_integrate(sym_const(5.0), "x"));
    EXPECT_NEAR(sym_eval(integral, {{"x", 2.0}}), 10.0, 1e-12);
    expect_integrate_roundtrip(sym_const(5.0), "x", 1.5);
}

TEST(SymbolicExtendedTest, integrate_variable) {
    const auto integral = sym_simplify(sym_integrate(sym_var("x"), "x"));
    EXPECT_NEAR(sym_eval(integral, {{"x", 4.0}}), 8.0, 1e-12);
    expect_integrate_roundtrip(sym_var("x"), "x", 3.0);
}

TEST(SymbolicExtendedTest, integrate_monomial) {
    const auto expr = sym_pow(sym_var("x"), sym_const(3.0));
    const auto integral = sym_simplify(sym_integrate(expr, "x"));
    EXPECT_NEAR(sym_eval(integral, {{"x", 2.0}}), 4.0, 1e-12);
    expect_integrate_roundtrip(expr, "x", 1.5);
}

TEST(SymbolicExtendedTest, integrate_sum_and_const_multiple) {
    const auto expr = sym_add(
        sym_mul(sym_const(2.0), sym_pow(sym_var("x"), sym_const(2.0))),
        sym_mul(sym_const(3.0), sym_var("x")));
    expect_integrate_roundtrip(expr, "x", 2.0);

    auto only_const = sym_integrate(sym_mul(sym_const(4.0), sym_var("x")), "x");
    EXPECT_NEAR(sym_eval(sym_simplify(std::move(only_const)), {{"x", 2.0}}), 8.0, 1e-12);
}

TEST(SymbolicExtendedTest, integrate_sin_and_cos) {
    expect_integrate_roundtrip(sym_sin(sym_var("x")), "x", 0.5);
    expect_integrate_roundtrip(sym_cos(sym_var("x")), "x", 1.0);

    const auto sin_int = sym_simplify(sym_integrate(sym_sin(sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(sin_int, {{"x", 0.0}}), -1.0, 1e-12);
    const auto cos_int = sym_simplify(sym_integrate(sym_cos(sym_var("x")), "x"));
    EXPECT_NEAR(sym_eval(cos_int, {{"x", std::numbers::pi / 2.0}}), 1.0, 1e-12);
}

TEST(SymbolicExtendedTest, integrate_definite_sanity) {
    const auto expr = sym_add(sym_var("x"), sym_const(1.0));
    const auto antideriv = sym_simplify(sym_integrate(expr, "x"));
    const double a = 0.0;
    const double b = 2.0;
    const double ftb = sym_eval(antideriv, {{"x", b}}) - sym_eval(antideriv, {{"x", a}});
    const double numeric = trapezoid_integral(expr, "x", a, b, 200);
    EXPECT_NEAR(ftb, numeric, 1e-3);
}

TEST(SymbolicExtendedTest, integrate_unsupported_returns_deriv_sentinel) {
    const auto unsupported = sym_integrate(sym_sin(sym_mul(sym_const(2.0), sym_var("x"))), "x");
    EXPECT_EQ(unsupported.op, SymOp::Deriv);
}

TEST(SymbolicExtendedTest, substitute_simple_and_nested) {
    const auto expr = sym_add(sym_mul(sym_var("x"), sym_const(2.0)), sym_var("y"));
    const auto replaced = sym_substitute(expr, "x", sym_const(3.0));
    EXPECT_NEAR(sym_eval(replaced, {{"y", 1.0}}), 7.0, 1e-12);

    const auto nested = sym_substitute(
        sym_pow(sym_var("x"), sym_const(2.0)), "x", sym_add(sym_var("t"), sym_const(1.0)));
    EXPECT_NEAR(sym_eval(nested, {{"t", 2.0}}), 9.0, 1e-12);
}

TEST(SymbolicExtendedTest, substitute_preserves_other_variables) {
    const auto expr = sym_add(sym_var("x"), sym_var("y"));
    const auto replaced = sym_substitute(expr, "x", sym_const(1.0));
    EXPECT_NEAR(sym_eval(replaced, {{"y", 4.0}}), 5.0, 1e-12);
}

TEST(SymbolicExtendedTest, mixed_ops_eval) {
    const auto expr = sym_sqrt(sym_add(sym_tan(sym_var("x")), sym_const(1.0)));
    const double x = 0.25;
    const double expected = std::sqrt(std::tan(x) + 1.0);
    EXPECT_NEAR(sym_eval(expr, {{"x", x}}), expected, 1e-12);
}

TEST(SymbolicExtendedTest, sub_div_neg_combined_diff) {
    const auto expr = sym_neg(sym_div(sym_sub(sym_var("x"), sym_const(1.0)), sym_const(2.0)));
    expect_deriv_matches_finite_diff(expr, "x", 1.0);
    expect_deriv_matches_finite_diff(expr, "x", -2.0);
}

TEST(SymbolicExtendedTest, integrate_sub_and_neg) {
    const auto expr = sym_sub(sym_pow(sym_var("x"), sym_const(2.0)), sym_var("x"));
    expect_integrate_roundtrip(expr, "x", 2.0);

    const auto neg_expr = sym_neg(sym_mul(sym_const(2.0), sym_var("x")));
    expect_integrate_roundtrip(neg_expr, "x", 1.0);
}

TEST(SymbolicExtendedTest, integrate_other_variable_is_constant_factor) {
    const auto integral = sym_simplify(sym_integrate(sym_var("y"), "x"));
    EXPECT_NEAR(sym_eval(integral, {{"x", 3.0}, {"y", 2.0}}), 6.0, 1e-12);
}

TEST(SymbolicExtendedTest, to_string_new_ops) {
    EXPECT_NE(sym_to_string(sym_sub(sym_var("a"), sym_var("b"))).find(" - "), std::string::npos);
    EXPECT_NE(sym_to_string(sym_div(sym_var("a"), sym_var("b"))).find(" / "), std::string::npos);
    EXPECT_NE(sym_to_string(sym_neg(sym_var("a"))).find("(-"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_tan(sym_var("a"))).find("tan"), std::string::npos);
    EXPECT_NE(sym_to_string(sym_sqrt(sym_var("a"))).find("sqrt"), std::string::npos);
}

TEST(SymbolicExtendedTest, integrate_pow_minus_one_unsupported) {
    const auto unsupported = sym_integrate(sym_pow(sym_var("x"), sym_const(-1.0)), "x");
    EXPECT_EQ(unsupported.op, SymOp::Deriv);
}

TEST(SymbolicExtendedTest, substitute_in_unary_ops) {
    const auto expr = sym_add(sym_sin(sym_var("x")), sym_sqrt(sym_var("x")));
    const auto replaced = sym_substitute(expr, "x", sym_var("t"));
    EXPECT_NE(sym_to_string(replaced).find("t"), std::string::npos);
    EXPECT_EQ(sym_to_string(replaced).find("x"), std::string::npos);
}

TEST(SymbolicExpandTest, expand_binomial_product) {
    const auto original =
        sym_mul(sym_add(sym_var("x"), sym_const(1.0)), sym_add(sym_var("x"), sym_const(2.0)));
    const auto expanded = sym_expand(clone_expr(original));

    for (const double x : {0.0, 1.0, 2.0, -3.0, 0.5}) {
        expect_eval_equivalent(original, expanded, {{"x", x}});
    }

    EXPECT_EQ(expanded.op, SymOp::Add);
    EXPECT_FALSE(has_mul_with_sum_factor(expanded));
}

TEST(SymbolicExpandTest, expand_difference_of_squares) {
    const auto original =
        sym_mul(sym_add(sym_var("x"), sym_var("y")), sym_sub(sym_var("x"), sym_var("y")));
    const auto expanded = sym_expand(clone_expr(original));
    const auto expected = sym_sub(sym_pow(sym_var("x"), sym_const(2.0)), sym_pow(sym_var("y"), sym_const(2.0)));

    for (const auto& point : std::vector<std::pair<double, double>>{{0.0, 0.0}, {1.0, 2.0}, {3.0, -1.0}, {-2.0, 0.5}}) {
        expect_eval_equivalent(original, expanded, {{"x", point.first}, {"y", point.second}});
        expect_eval_equivalent(expanded, expected, {{"x", point.first}, {"y", point.second}});
    }

    EXPECT_FALSE(has_mul_with_sum_factor(expanded));
}

TEST(SymbolicExpandTest, expand_nested_sum) {
    const auto original = sym_mul(
        sym_add(sym_add(sym_var("x"), sym_const(1.0)), sym_const(2.0)),
        sym_var("x"));
    const auto expanded = sym_expand(clone_expr(original));

    for (const double x : {0.0, 1.0, 2.5, -4.0}) {
        expect_eval_equivalent(original, expanded, {{"x", x}});
    }

    EXPECT_FALSE(has_mul_with_sum_factor(expanded));
}

TEST(SymbolicExpandTest, expand_integer_power) {
    const auto original = sym_pow(sym_add(sym_var("x"), sym_const(1.0)), sym_const(3.0));
    const auto expanded = sym_expand(clone_expr(original));

    for (const double x : {0.0, 1.0, 2.0, -1.0, 0.25}) {
        expect_eval_equivalent(original, expanded, {{"x", x}});
    }

    EXPECT_FALSE(has_mul_with_sum_factor(expanded));
}

TEST(SymbolicExpandTest, expand_triple_product) {
    const auto original = sym_mul(
        sym_mul(sym_add(sym_var("x"), sym_const(1.0)), sym_add(sym_var("x"), sym_const(2.0))),
        sym_add(sym_var("x"), sym_const(3.0)));
    const auto expanded = sym_expand(clone_expr(original));

    for (const double x : {-2.0, 0.0, 1.0, 4.0, 0.75}) {
        expect_eval_equivalent(original, expanded, {{"x", x}});
    }

    EXPECT_FALSE(has_mul_with_sum_factor(expanded));
}

TEST(SymbolicExpandTest, expand_noop_trig_product) {
    const auto original = sym_mul(sym_sin(sym_var("x")), sym_cos(sym_var("y")));
    const auto expanded = sym_expand(clone_expr(original));

    for (const auto& point : std::vector<std::pair<double, double>>{{0.0, 0.0}, {1.0, 0.5}, {-0.5, 2.0}}) {
        expect_eval_equivalent(original, expanded, {{"x", point.first}, {"y", point.second}});
    }

    EXPECT_EQ(expanded.op, SymOp::Mul);
    EXPECT_EQ(expanded.left->op, SymOp::Sin);
    EXPECT_EQ(expanded.right->op, SymOp::Cos);
}

TEST(SymbolicExpandTest, expand_parse_roundtrip) {
    const auto parsed = sym_parse("(x+1)*(x+2)");
    ASSERT_TRUE(parsed.has_value());
    const auto expanded = sym_expand(clone_expr(*parsed));

    for (const double x : {0.0, 1.0, 2.0, -3.0, 0.5}) {
        expect_eval_equivalent(*parsed, expanded, {{"x", x}});
    }

    EXPECT_EQ(expanded.op, SymOp::Add);
    EXPECT_FALSE(has_mul_with_sum_factor(expanded));
}

TEST(SymbolicParseTest, parse_const_and_var) {
    const auto c = sym_parse("42");
    ASSERT_TRUE(c.has_value());
    EXPECT_NEAR(sym_eval(*c, {}), 42.0, 1e-12);

    const auto v = sym_parse("x");
    ASSERT_TRUE(v.has_value());
    EXPECT_NEAR(sym_eval(*v, {{"x", 5.0}}), 5.0, 1e-12);
}

TEST(SymbolicParseTest, parse_add_sub_mul_div) {
    const auto add = sym_parse("2+3");
    ASSERT_TRUE(add.has_value());
    EXPECT_NEAR(sym_eval(*add, {}), 5.0, 1e-12);

    const auto sub = sym_parse("7-2.5");
    ASSERT_TRUE(sub.has_value());
    EXPECT_NEAR(sym_eval(*sub, {}), 4.5, 1e-12);

    const auto mul = sym_parse("3*4");
    ASSERT_TRUE(mul.has_value());
    EXPECT_NEAR(sym_eval(*mul, {}), 12.0, 1e-12);

    const auto div = sym_parse("8/2");
    ASSERT_TRUE(div.has_value());
    EXPECT_NEAR(sym_eval(*div, {}), 4.0, 1e-12);
}

TEST(SymbolicParseTest, parse_pow_right_associative) {
    const auto pow_expr = sym_parse("2^3^2");
    ASSERT_TRUE(pow_expr.has_value());
    EXPECT_NEAR(sym_eval(*pow_expr, {}), 512.0, 1e-12);
}

TEST(SymbolicParseTest, parse_unary_minus) {
    const auto neg = sym_parse("-5");
    ASSERT_TRUE(neg.has_value());
    EXPECT_NEAR(sym_eval(*neg, {}), -5.0, 1e-12);

    const auto expr = sym_parse("-x");
    ASSERT_TRUE(expr.has_value());
    EXPECT_NEAR(sym_eval(*expr, {{"x", 3.0}}), -3.0, 1e-12);
}

TEST(SymbolicParseTest, parse_trig_and_exp_log_sqrt) {
    const double x = 0.5;
    const std::map<std::string, double> env{{"x", x}};

    const auto sin_expr = sym_parse("sin(x)");
    ASSERT_TRUE(sin_expr.has_value());
    EXPECT_NEAR(sym_eval(*sin_expr, env), std::sin(x), 1e-12);

    const auto cos_expr = sym_parse("cos(x)");
    ASSERT_TRUE(cos_expr.has_value());
    EXPECT_NEAR(sym_eval(*cos_expr, env), std::cos(x), 1e-12);

    const auto tan_expr = sym_parse("tan(x)");
    ASSERT_TRUE(tan_expr.has_value());
    EXPECT_NEAR(sym_eval(*tan_expr, env), std::tan(x), 1e-12);

    const auto exp_expr = sym_parse("exp(x)");
    ASSERT_TRUE(exp_expr.has_value());
    EXPECT_NEAR(sym_eval(*exp_expr, env), std::exp(x), 1e-12);

    const auto log_expr = sym_parse("log(x)");
    ASSERT_TRUE(log_expr.has_value());
    EXPECT_NEAR(sym_eval(*log_expr, env), std::log(x), 1e-12);

    const auto sqrt_expr = sym_parse("sqrt(x)");
    ASSERT_TRUE(sqrt_expr.has_value());
    EXPECT_NEAR(sym_eval(*sqrt_expr, env), std::sqrt(x), 1e-12);
}

TEST(SymbolicParseTest, parse_nested_parens_and_precedence) {
    const auto expr = sym_parse("(2+3)*4");
    ASSERT_TRUE(expr.has_value());
    EXPECT_NEAR(sym_eval(*expr, {}), 20.0, 1e-12);

    const auto complex = sym_parse("x^2 + sin(y) - 3*x");
    ASSERT_TRUE(complex.has_value());
    EXPECT_NEAR(sym_eval(*complex, {{"x", 2.0}, {"y", 1.0}}),
                4.0 + std::sin(1.0) - 6.0, 1e-12);
}

TEST(SymbolicParseTest, parse_decimal_and_scientific) {
    const auto decimal = sym_parse("1.5");
    ASSERT_TRUE(decimal.has_value());
    EXPECT_NEAR(sym_eval(*decimal, {}), 1.5, 1e-12);

    const auto scientific = sym_parse("1.5e-3");
    ASSERT_TRUE(scientific.has_value());
    EXPECT_NEAR(sym_eval(*scientific, {}), 0.0015, 1e-15);
}

TEST(SymbolicParseTest, parse_x_squared_plus_one) {
    const auto expr = sym_parse("x^2+1");
    ASSERT_TRUE(expr.has_value());
    EXPECT_NEAR(sym_eval(*expr, {{"x", 3.0}}), 10.0, 1e-12);
}

TEST(SymbolicParseTest, parse_whitespace_tolerance) {
    const auto expr = sym_parse("  x  +  1  ");
    ASSERT_TRUE(expr.has_value());
    EXPECT_NEAR(sym_eval(*expr, {{"x", 2.0}}), 3.0, 1e-12);
}

TEST(SymbolicParseTest, parse_error_empty_string) {
    const auto result = sym_parse("");
    ASSERT_FALSE(result.has_value());
    EXPECT_FALSE(result.error().message.empty());
}

TEST(SymbolicParseTest, parse_error_mismatched_parens) {
    const auto result = sym_parse("(1+2");
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("')'"), std::string::npos);
}

TEST(SymbolicParseTest, parse_error_invalid_character) {
    const auto result = sym_parse("x@y");
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("trailing"), std::string::npos);
}

TEST(SymbolicParseTest, parse_error_trailing_operator) {
    const auto result = sym_parse("1+");
    ASSERT_FALSE(result.has_value());
}

TEST(SymbolicParseTest, parse_error_unknown_function) {
    const auto result = sym_parse("foo(x)");
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("unknown function"), std::string::npos);
}

TEST(SymbolicParseTest, diff_on_parsed_expression) {
    auto parsed = sym_parse("x^2");
    ASSERT_TRUE(parsed.has_value());
    const auto deriv = sym_simplify(sym_diff(std::move(*parsed), "x"));
    EXPECT_NEAR(sym_eval(deriv, {{"x", 3.0}}), 6.0, 1e-12);
}

TEST(SymbolicParseTest, simplify_on_parsed_expression) {
    auto parsed = sym_parse("2+3");
    ASSERT_TRUE(parsed.has_value());
    const auto simplified = sym_simplify(std::move(*parsed));
    EXPECT_NEAR(sym_eval(simplified, {}), 5.0, 1e-12);
}

TEST(SymbolicParseTest, integrate_on_parsed_expression) {
    auto parsed = sym_parse("x");
    ASSERT_TRUE(parsed.has_value());
    const auto integral = sym_simplify(sym_integrate(*parsed, "x"));
    EXPECT_NEAR(sym_eval(integral, {{"x", 4.0}}), 8.0, 1e-12);
}

TEST(SymbolicParseTest, substitute_on_parsed_expression) {
    auto parsed = sym_parse("x+y");
    ASSERT_TRUE(parsed.has_value());
    const auto replaced = sym_substitute(*parsed, "x", sym_const(2.0));
    EXPECT_NEAR(sym_eval(replaced, {{"y", 1.0}}), 3.0, 1e-12);
}

TEST(SymbolicParseTest, roundtrip_to_string_reparse) {
    auto parsed = sym_parse("sin(x)*x^2");
    ASSERT_TRUE(parsed.has_value());
    const double expected = sym_eval(*parsed, {{"x", 2.0}});
    const auto text = sym_to_string(*parsed);
    const auto reparsed = sym_parse(text);
    ASSERT_TRUE(reparsed.has_value());
    EXPECT_NEAR(sym_eval(*reparsed, {{"x", 2.0}}), expected, 1e-12);
}

TEST(SymbolicParseTest, integrate_unsupported_parsed_form) {
    auto parsed = sym_parse("sin(2*x)");
    ASSERT_TRUE(parsed.has_value());
    const auto integral = sym_integrate(*parsed, "x");
    EXPECT_EQ(integral.op, SymOp::Deriv);
}
