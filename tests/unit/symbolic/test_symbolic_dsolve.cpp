#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>

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

bool is_deriv_sentinel(const SymExpr& original, const SymExpr& result, const std::string& var) {
    const SymExpr expected = sym_deriv(clone_expr(original), var);
    return sym_to_string(result) == sym_to_string(expected);
}

void expect_ode_solution(const SymExpr& rhs, const SymExpr& solution, const std::string& indep_var,
                         double x_value = 1.25, double c_value = 2.0) {
    const double y_value = sym_eval(solution, {{indep_var, x_value}, {"C", c_value}});
    const double derivative =
        sym_eval(sym_simplify(sym_diff(clone_expr(solution), indep_var)),
                 {{indep_var, x_value}, {"C", c_value}});
    const double rhs_value = sym_eval(rhs, {{indep_var, x_value}, {"y", y_value}, {"C", c_value}});
    EXPECT_NEAR(derivative, rhs_value, 1e-9);
}

} // namespace

TEST(SymbolicDsolveTest, separable_function_of_independent_var) {
    const SymExpr rhs = sym_var("x");
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    EXPECT_NE(sym_to_string(solution).find("C"), std::string::npos);
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, exponential_growth) {
    const SymExpr rhs = sym_var("y");
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    EXPECT_NE(sym_to_string(solution).find("exp"), std::string::npos);
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, scaled_exponential_growth) {
    const SymExpr rhs = sym_mul(sym_const(2.0), sym_var("y"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, linear_affine_coefficients) {
    const SymExpr rhs = sym_add(sym_var("y"), sym_const(1.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, separable_power_law) {
    const SymExpr rhs = sym_pow(sym_var("y"), sym_const(2.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, multiplier_depends_on_independent_var) {
    const SymExpr rhs = sym_mul(sym_var("x"), sym_var("y"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, unsupported_returns_deriv_sentinel) {
    const SymExpr rhs = sym_sin(sym_var("y"));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, exponential_decay_neg_dep_var) {
    const SymExpr rhs = sym_neg(sym_var("y"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, multiplier_dep_var_on_left) {
    const SymExpr rhs = sym_mul(sym_var("y"), sym_var("x"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, affine_sub_y_minus_const) {
    const SymExpr rhs = sym_sub(sym_var("y"), sym_const(1.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, affine_sub_const_minus_y) {
    const SymExpr rhs = sym_sub(sym_const(1.0), sym_var("y"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, affine_y_plus_y) {
    const SymExpr rhs = sym_add(sym_var("y"), sym_var("y"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, affine_cancelled_y_is_constant) {
    const SymExpr rhs = sym_sub(sym_var("y"), sym_var("y"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, affine_zero_coef_constant_rhs) {
    const SymExpr rhs = sym_add(sym_mul(sym_const(0.0), sym_var("y")), sym_const(5.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, affine_nested_sum) {
    const SymExpr rhs = sym_add(sym_add(sym_var("y"), sym_const(1.0)), sym_const(2.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, separable_power_zero) {
    const SymExpr rhs = sym_pow(sym_var("y"), sym_const(0.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, separable_power_cubic) {
    const SymExpr rhs = sym_pow(sym_var("y"), sym_const(3.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x", 0.4, -1.0);
}

TEST(SymbolicDsolveTest, independent_unsupported_integrate_is_sentinel) {
    const SymExpr rhs = sym_sin(sym_mul(sym_const(2.0), sym_var("x")));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, multiplier_unsupported_integrate_is_sentinel) {
    const SymExpr rhs = sym_mul(sym_sin(sym_mul(sym_const(2.0), sym_var("x"))), sym_var("y"));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, power_one_falls_through_to_sentinel) {
    const SymExpr rhs = sym_pow(sym_var("y"), sym_const(1.0));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, y_squared_plus_y_is_unsupported) {
    const SymExpr rhs = sym_add(sym_pow(sym_var("y"), sym_const(2.0)), sym_var("y"));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, y_plus_independent_var_is_unsupported) {
    const SymExpr rhs = sym_add(sym_var("y"), sym_var("x"));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, constant_rhs_is_linear) {
    const SymExpr rhs = sym_const(5.0);
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, multiplier_const_on_right) {
    const SymExpr rhs = sym_mul(sym_var("y"), sym_const(3.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, affine_nested_difference) {
    const SymExpr rhs = sym_sub(sym_add(sym_var("y"), sym_const(2.0)), sym_const(1.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, independent_power_is_supported) {
    const SymExpr rhs = sym_pow(sym_var("x"), sym_const(3.0));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, multiplier_sin_of_indep) {
    const SymExpr rhs = sym_mul(sym_sin(sym_var("x")), sym_var("y"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, zero_rhs_is_constant_solution) {
    const SymExpr rhs = sym_const(0.0);
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}

TEST(SymbolicDsolveTest, tan_of_dep_is_unsupported) {
    const SymExpr rhs = sym_tan(sym_var("y"));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, product_y_times_y_is_unsupported) {
    const SymExpr rhs = sym_mul(sym_var("y"), sym_var("y"));
    const SymExpr result = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x"));
}

TEST(SymbolicDsolveTest, independent_const_plus_var) {
    const SymExpr rhs = sym_add(sym_const(1.0), sym_var("x"));
    const SymExpr solution = sym_simplify(sym_dsolve(rhs, "x", "y"));
    expect_ode_solution(rhs, solution, "x");
}
