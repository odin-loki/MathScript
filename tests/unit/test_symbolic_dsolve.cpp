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
