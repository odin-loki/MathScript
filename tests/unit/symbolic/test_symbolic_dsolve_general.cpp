// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <utility>

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
    return sym_to_string(result) == sym_to_string(sym_deriv(clone_expr(original), var));
}

// Primary check: differentiate the returned solution symbolically and substitute it back
// into dy/dx = rhs at several (x, C) points; the residual must vanish. The sample points
// are deliberately different from the ones the solver verifies internally.
void expect_solves_ode(const SymExpr& rhs, const SymExpr& solution, const std::string& indep_var,
                       const std::string& dep_var,
                       std::initializer_list<double> xs = {0.8, 1.6},
                       std::initializer_list<double> cs = {2.0, -0.7}) {
    ASSERT_NE(solution.op, SymOp::Deriv) << "unsolved: " << sym_to_string(solution);
    const SymExpr derivative = sym_simplify(sym_diff(clone_expr(solution), indep_var));
    int checked = 0;
    for (const double xv : xs) {
        for (const double cv : cs) {
            const std::map<std::string, double> env{{indep_var, xv}, {"C", cv}};
            const double yv = sym_eval(solution, env);
            if (!std::isfinite(yv)) {
                continue;
            }
            const double dv = sym_eval(derivative, env);
            std::map<std::string, double> full = env;
            full[dep_var] = yv;
            const double rv = sym_eval(rhs, full);
            if (!std::isfinite(dv) || !std::isfinite(rv)) {
                continue;
            }
            ++checked;
            EXPECT_NEAR(dv, rv, 1e-7 * (1.0 + std::abs(dv) + std::abs(rv)))
                << "x=" << xv << " C=" << cv << " sol=" << sym_to_string(solution);
        }
    }
    EXPECT_GE(checked, 2);
}

void expect_solves_ode2(double a, double b, double c, const SymExpr& forcing,
                        const SymExpr& solution, const std::string& indep_var,
                        std::initializer_list<double> xs = {0.6, 1.7},
                        std::initializer_list<std::pair<double, double>> consts = {{1.0, 2.0},
                                                                                  {-0.5, 0.3}}) {
    ASSERT_NE(solution.op, SymOp::Deriv) << "unsolved: " << sym_to_string(solution);
    const SymExpr d1 = sym_simplify(sym_diff(clone_expr(solution), indep_var));
    const SymExpr d2 = sym_simplify(sym_diff(clone_expr(d1), indep_var));
    for (const double xv : xs) {
        for (const auto& pair : consts) {
            const std::map<std::string, double> env{
                {indep_var, xv}, {"C1", pair.first}, {"C2", pair.second}};
            const double second = sym_eval(d2, env);
            const double residual =
                a * second + b * sym_eval(d1, env) + c * sym_eval(solution, env) -
                sym_eval(forcing, env);
            EXPECT_NEAR(residual, 0.0, 1e-7 * (1.0 + std::abs(a * second)))
                << "x=" << xv << " sol=" << sym_to_string(solution);
        }
    }
}

SymExpr x_var() {
    return sym_var("x");
}

SymExpr y_var() {
    return sym_var("y");
}

} // namespace

// --- Group A: backwards compatibility ---------------------------------------

TEST(SymbolicDsolveGeneralTest, legacy_sentinels_are_frozen) {
    // sym_dsolve is a frozen v1 contract: every rhs shape it declined before must still be
    // declined, even the four that sym_dsolve_ode now solves outright.
    SymExpr declined[] = {
        sym_sin(y_var()),
        sym_tan(y_var()),
        sym_mul(y_var(), sym_sin(y_var())),
        // sin(2*x) and tan(x) integrate now, so the legacy table solves both. What
        // it still cannot do is integration by parts.
        sym_mul(x_var(), sym_sin(x_var())),
        sym_mul(sym_mul(x_var(), sym_sin(x_var())), y_var()),
        sym_mul(x_var(), sym_exp(x_var())),
        sym_pow(y_var(), sym_const(1.0)),
        sym_mul(y_var(), y_var()),
        sym_add(sym_pow(y_var(), sym_const(2.0)), y_var()),
        sym_add(y_var(), x_var()),
    };
    for (const SymExpr& rhs : declined) {
        const SymExpr result = sym_dsolve(rhs, "x", "y");
        EXPECT_TRUE(is_deriv_sentinel(rhs, result, "x")) << sym_to_string(rhs);
    }
}

TEST(SymbolicDsolveGeneralTest, ode_matches_legacy_where_legacy_succeeds) {
    // sym_dsolve_ode runs the frozen separable table first, so it is a strict superset.
    SymExpr solved[] = {
        x_var(),
        y_var(),
        sym_mul(sym_const(2.0), y_var()),
        sym_add(y_var(), sym_const(1.0)),
        sym_pow(y_var(), sym_const(2.0)),
        sym_mul(x_var(), y_var()),
        sym_neg(y_var()),
        sym_pow(y_var(), sym_const(3.0)),
        sym_const(5.0),
        sym_pow(x_var(), sym_const(3.0)),
        sym_mul(sym_sin(x_var()), y_var()),
        sym_const(0.0),
        sym_sub(y_var(), sym_const(1.0)),
        sym_sub(sym_const(1.0), y_var()),
        sym_add(y_var(), y_var()),
        sym_sub(y_var(), y_var()),
        sym_pow(y_var(), sym_const(0.0)),
        sym_mul(y_var(), sym_const(3.0)),
        sym_add(sym_const(1.0), x_var()),
    };
    for (const SymExpr& rhs : solved) {
        EXPECT_EQ(sym_to_string(sym_dsolve_ode(rhs, "x", "y")),
                  sym_to_string(sym_dsolve(rhs, "x", "y")))
            << sym_to_string(rhs);
    }
}

TEST(SymbolicDsolveGeneralTest, constant_is_named_C) {
    // sym_eval silently substitutes 0.0 for unknown names, so the constant's name is part
    // of the contract: a different name would produce plausible-but-wrong numbers.
    const SymExpr growth = sym_dsolve_ode(y_var(), "x", "y");
    EXPECT_NE(sym_to_string(growth).find("C"), std::string::npos);
    const SymExpr linear = sym_dsolve_ode(sym_sub(x_var(), y_var()), "x", "y");
    EXPECT_NE(sym_to_string(linear).find("C"), std::string::npos);
}

// --- Group B: the named ODEs -------------------------------------------------

TEST(SymbolicDsolveGeneralTest, exponential_growth) {
    const SymExpr rhs = y_var();
    const SymExpr solution = sym_simplify(sym_dsolve_ode(rhs, "x", "y"));
    EXPECT_NE(sym_to_string(solution).find("exp"), std::string::npos);
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C", 2.0}}), 2.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, separable_x_times_y) {
    const SymExpr rhs = sym_mul(x_var(), y_var());
    const SymExpr solution = sym_simplify(sym_dsolve_ode(rhs, "x", "y"));
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C", 2.0}}), 2.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, linear_constant_coefficient) {
    // y' + y = x  ->  y = x - 1 + C e^-x
    const SymExpr rhs = sym_sub(x_var(), y_var());
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C", 2.0}}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(solution, {{"x", 1.0}, {"C", 2.0}}), 0.7357588823428847, 1e-9);
}

TEST(SymbolicDsolveGeneralTest, linear_variable_coefficient) {
    // y' + y/x = x^2  ->  y = x^3/4 + C/x
    const SymExpr rhs = sym_sub(sym_pow(x_var(), sym_const(2.0)), sym_div(y_var(), x_var()));
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 1.0}, {"C", 2.0}}), 2.25, 1e-12);
    EXPECT_NEAR(sym_eval(solution, {{"x", 2.0}, {"C", 2.0}}), 3.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, bernoulli_n_two) {
    // y' + y = y^2  ->  y = 1 / (1 + C e^x)
    const SymExpr rhs = sym_sub(sym_pow(y_var(), sym_const(2.0)), y_var());
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C", 2.0}}), 1.0 / 3.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, bernoulli_n_three_variable_coefficient) {
    // y' = -x y^3  ->  y = (x^2 + C)^(-1/2)
    const SymExpr rhs = sym_neg(sym_mul(x_var(), sym_pow(y_var(), sym_const(3.0))));
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y", {0.8, 1.6}, {2.0, 3.0});
    EXPECT_NEAR(sym_eval(solution, {{"x", 1.0}, {"C", 3.0}}), 0.5, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, bernoulli_legacy_sentinel_shape_now_solved) {
    // y' = y^2 + y stays a sentinel through the frozen entry point but is solved by the
    // general one; there is no predicate separating it from y^2 - y, which is why the
    // frozen table was not widened in place.
    const SymExpr rhs = sym_add(sym_pow(y_var(), sym_const(2.0)), y_var());
    EXPECT_TRUE(is_deriv_sentinel(rhs, sym_dsolve(rhs, "x", "y"), "x"));
    expect_solves_ode(rhs, sym_dsolve_ode(rhs, "x", "y"), "x", "y");
}

TEST(SymbolicDsolveGeneralTest, exact_equation) {
    // (2xy + y^2) dx + (x^2 + 2xy) dy = 0  ->  x^2 y + x y^2 = C
    const SymExpr m =
        sym_add(sym_mul(sym_mul(sym_const(2.0), x_var()), y_var()),
                sym_pow(y_var(), sym_const(2.0)));
    const SymExpr n = sym_add(sym_pow(x_var(), sym_const(2.0)),
                              sym_mul(sym_mul(sym_const(2.0), x_var()), y_var()));
    const SymExpr relation = sym_dsolve_exact(m, n, "x", "y");
    ASSERT_NE(relation.op, SymOp::Deriv) << sym_to_string(relation);
    EXPECT_NEAR(sym_eval(relation, {{"x", 1.0}, {"y", 1.0}, {"C", 0.0}}), 2.0, 1e-12);

    const SymExpr fx = sym_simplify(sym_diff(clone_expr(relation), "x"));
    const SymExpr fy = sym_simplify(sym_diff(clone_expr(relation), "y"));
    for (const auto& point : {std::pair<double, double>{1.3, 0.7}, {2.1, -0.4}}) {
        const std::map<std::string, double> env{{"x", point.first}, {"y", point.second}};
        EXPECT_NEAR(sym_eval(fx, env), sym_eval(m, env), 1e-9);
        EXPECT_NEAR(sym_eval(fy, env), sym_eval(n, env), 1e-9);
    }
}

TEST(SymbolicDsolveGeneralTest, homogeneous_degree_zero) {
    // y' = (x + y)/x  ->  y = x (ln x + C)
    const SymExpr rhs = sym_div(sym_add(x_var(), y_var()), x_var());
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 1.0}, {"C", 2.0}}), 2.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, second_order_distinct_real_roots) {
    const SymExpr forcing = sym_const(0.0);
    const SymExpr solution = sym_dsolve_linear2(1.0, -3.0, 2.0, forcing, "x");
    expect_solves_ode2(1.0, -3.0, 2.0, forcing, solution, "x");
    EXPECT_NE(sym_to_string(solution).find("exp"), std::string::npos);
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C1", 1.0}, {"C2", 1.0}}), 2.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, second_order_complex_roots) {
    const SymExpr forcing = sym_const(0.0);
    const SymExpr solution = sym_dsolve_linear2(1.0, 0.0, 4.0, forcing, "x");
    expect_solves_ode2(1.0, 0.0, 4.0, forcing, solution, "x");
    EXPECT_NE(sym_to_string(solution).find("cos"), std::string::npos);
    EXPECT_NE(sym_to_string(solution).find("sin"), std::string::npos);
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C1", 1.0}, {"C2", 2.0}}), 1.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, second_order_repeated_root) {
    const SymExpr forcing = sym_const(0.0);
    const SymExpr solution = sym_dsolve_linear2(1.0, 2.0, 1.0, forcing, "x");
    expect_solves_ode2(1.0, 2.0, 1.0, forcing, solution, "x");
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C1", 3.0}, {"C2", 5.0}}), 3.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, unsupported_rhs_returns_sentinel) {
    const SymExpr rhs = sym_sin(y_var());
    const SymExpr legacy = sym_dsolve(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, legacy, "x"));
    const SymExpr general = sym_dsolve_ode(rhs, "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(rhs, general, "x"));
    EXPECT_EQ(general.op, SymOp::Deriv);
    EXPECT_EQ(general.name, "x");
}

// --- Group C: the rest of the pipeline ---------------------------------------

TEST(SymbolicDsolveGeneralTest, quadrature_beyond_the_legacy_integrator) {
    // x*sin(x) integrates by parts to sin(x) - x*cos(x), which the general
    // quadrature finds and the legacy separable table does not.
    const SymExpr rhs = sym_mul(x_var(), sym_sin(x_var()));
    EXPECT_TRUE(is_deriv_sentinel(rhs, sym_dsolve(rhs, "x", "y"), "x"));
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y");
    // sin(0) - 0*cos(0) + C = C.
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C", 1.0}}), 1.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, linear_exponential_forcing) {
    const SymExpr rhs =
        sym_sub(sym_exp(sym_mul(sym_const(3.0), x_var())), sym_mul(sym_const(2.0), y_var()));
    expect_solves_ode(rhs, sym_dsolve_ode(rhs, "x", "y"), "x", "y");
}

TEST(SymbolicDsolveGeneralTest, linear_sinusoidal_forcing) {
    const SymExpr rhs = sym_sub(sym_sin(x_var()), y_var());
    expect_solves_ode(rhs, sym_dsolve_ode(rhs, "x", "y"), "x", "y");
}

TEST(SymbolicDsolveGeneralTest, linear_scale_invariant) {
    const SymExpr rhs = sym_div(y_var(), x_var());
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 3.0}, {"C", 2.0}}), 6.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, linear_scaled_variable_coefficient) {
    // y' = 2y/x -> y = C x^2; exercises the log-shaped integrating-factor rewrite.
    const SymExpr rhs = sym_div(sym_mul(sym_const(2.0), y_var()), x_var());
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y");
    EXPECT_NEAR(sym_eval(solution, {{"x", 3.0}, {"C", 2.0}}), 18.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, bernoulli_negative_exponent) {
    // y' = x/y -> y = sqrt(x^2 + C), i.e. Bernoulli with n = -1.
    const SymExpr rhs = sym_div(x_var(), y_var());
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y", {0.8, 1.6}, {2.0, 3.0});
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C", 4.0}}), 2.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, homogeneous_power_inversion) {
    const SymExpr rhs = sym_div(sym_add(sym_mul(x_var(), x_var()), sym_mul(y_var(), y_var())),
                                sym_mul(x_var(), y_var()));
    expect_solves_ode(rhs, sym_dsolve_ode(rhs, "x", "y"), "x", "y", {1.5, 2.4}, {1.0, 2.0});
}

TEST(SymbolicDsolveGeneralTest, homogeneous_repeated_root_inversion) {
    const SymExpr rhs = sym_div(sym_add(sym_mul(y_var(), y_var()), sym_mul(x_var(), y_var())),
                                sym_mul(x_var(), x_var()));
    const SymExpr solution = sym_dsolve_ode(rhs, "x", "y");
    expect_solves_ode(rhs, solution, "x", "y", {0.8, 1.6}, {2.0, 3.0});
    EXPECT_NEAR(sym_eval(solution, {{"x", 1.0}, {"C", 2.0}}), -0.5, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, exact_made_explicit) {
    const SymExpr numerator =
        sym_add(sym_mul(sym_mul(sym_const(2.0), x_var()), y_var()),
                sym_pow(y_var(), sym_const(2.0)));
    const SymExpr denominator = sym_add(sym_pow(x_var(), sym_const(2.0)),
                                        sym_mul(sym_mul(sym_const(2.0), x_var()), y_var()));
    const SymExpr rhs = sym_neg(sym_div(clone_expr(numerator), clone_expr(denominator)));
    expect_solves_ode(rhs, sym_dsolve_ode(rhs, "x", "y"), "x", "y", {0.8, 1.6}, {2.0, 5.0});
}

TEST(SymbolicDsolveGeneralTest, exact_non_exact_pair_is_sentinel) {
    const SymExpr m = sym_mul(y_var(), y_var());
    const SymExpr result = sym_dsolve_exact(m, x_var(), "x", "y");
    EXPECT_TRUE(is_deriv_sentinel(m, result, "x"));
}

TEST(SymbolicDsolveGeneralTest, homogeneous_without_atan_is_sentinel) {
    // (x + y)/(x - y) is homogeneous of degree 0, but inverting its potential needs an
    // arctangent and the AST has no Atan node, so no closed form is emitted.
    const SymExpr rhs = sym_div(sym_add(x_var(), y_var()), sym_sub(x_var(), y_var()));
    EXPECT_TRUE(is_deriv_sentinel(rhs, sym_dsolve_ode(rhs, "x", "y"), "x"));
}

TEST(SymbolicDsolveGeneralTest, second_order_polynomial_forcing) {
    const SymExpr forcing = x_var();
    const SymExpr solution = sym_dsolve_linear2(1.0, -3.0, 2.0, forcing, "x");
    expect_solves_ode2(1.0, -3.0, 2.0, forcing, solution, "x");
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C1", 0.0}, {"C2", 0.0}}), 0.75, 1e-12);
    EXPECT_NEAR(sym_eval(solution, {{"x", 2.0}, {"C1", 0.0}, {"C2", 0.0}}), 1.75, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, second_order_resonant_forcing) {
    const SymExpr exponential = sym_exp(x_var());
    expect_solves_ode2(1.0, 0.0, -1.0, exponential,
                       sym_dsolve_linear2(1.0, 0.0, -1.0, exponential, "x"), "x");
    const SymExpr sinusoid = sym_sin(x_var());
    expect_solves_ode2(1.0, 0.0, 1.0, sinusoid, sym_dsolve_linear2(1.0, 0.0, 1.0, sinusoid, "x"),
                       "x");
}

TEST(SymbolicDsolveGeneralTest, second_order_nonresonant_sinusoid) {
    const SymExpr forcing = sym_mul(sym_const(2.0), sym_cos(x_var()));
    const SymExpr solution = sym_dsolve_linear2(1.0, 3.0, 2.0, forcing, "x");
    expect_solves_ode2(1.0, 3.0, 2.0, forcing, solution, "x");
    EXPECT_NEAR(sym_eval(solution, {{"x", 0.0}, {"C1", 0.0}, {"C2", 0.0}}), 0.2, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, second_order_edge_cases) {
    const SymExpr linear_forcing = x_var();
    expect_solves_ode2(1.0, 1.0, 0.0, linear_forcing,
                       sym_dsolve_linear2(1.0, 1.0, 0.0, linear_forcing, "x"), "x");

    const SymExpr quadratic_forcing = sym_pow(x_var(), sym_const(2.0));
    expect_solves_ode2(1.0, 0.0, 0.0, quadratic_forcing,
                       sym_dsolve_linear2(1.0, 0.0, 0.0, quadratic_forcing, "x"), "x");

    const SymExpr mixed_forcing =
        sym_add(x_var(), sym_exp(sym_mul(sym_const(3.0), x_var())));
    expect_solves_ode2(1.0, -3.0, 2.0, mixed_forcing,
                       sym_dsolve_linear2(1.0, -3.0, 2.0, mixed_forcing, "x"), "x");

    const SymExpr tangent = sym_tan(x_var());
    const SymExpr unsupported = sym_dsolve_linear2(1.0, 0.0, 4.0, tangent, "x");
    EXPECT_TRUE(is_deriv_sentinel(tangent, unsupported, "x"));

    const SymExpr first_order = sym_dsolve_linear2(0.0, 1.0, 1.0, x_var(), "x");
    EXPECT_TRUE(is_deriv_sentinel(x_var(), first_order, "x"));
}

TEST(SymbolicDsolveGeneralTest, ivp_pins_the_constant) {
    const SymExpr general = sym_mul(sym_var("C"), sym_exp(x_var()));
    const SymExpr pinned = sym_dsolve_ivp(general, "x", 0.0, 3.0);
    EXPECT_EQ(sym_to_string(pinned).find("C"), std::string::npos);
    EXPECT_NEAR(sym_eval(pinned, {{"x", 0.0}}), 3.0, 1e-12);
    EXPECT_NEAR(sym_eval(pinned, {{"x", 1.0}}), 8.154845485377136, 1e-9);

    // Nonlinear in C: resolved by the bracket-and-bisect path.
    const SymExpr bernoulli =
        sym_dsolve_ode(sym_sub(sym_pow(y_var(), sym_const(2.0)), y_var()), "x", "y");
    const SymExpr particular = sym_dsolve_ivp(bernoulli, "x", 0.0, 1.0 / 3.0);
    ASSERT_NE(particular.op, SymOp::Deriv) << sym_to_string(particular);
    EXPECT_NEAR(sym_eval(particular, {{"x", 0.0}}), 1.0 / 3.0, 1e-9);

    const SymExpr quadrature =
        sym_add(sym_div(sym_pow(x_var(), sym_const(2.0)), sym_const(2.0)), sym_var("C"));
    const SymExpr shifted = sym_dsolve_ivp(quadrature, "x", 1.0, 5.0);
    EXPECT_NEAR(sym_eval(shifted, {{"x", 1.0}}), 5.0, 1e-12);
}

TEST(SymbolicDsolveGeneralTest, ivp_without_a_constant_is_sentinel) {
    const SymExpr general = x_var();
    const SymExpr result = sym_dsolve_ivp(general, "x", 0.0, 5.0);
    EXPECT_TRUE(is_deriv_sentinel(general, result, "x"));

    // A sentinel handed in is propagated unchanged rather than wrapped again.
    const SymExpr sentinel = sym_deriv(sym_sin(y_var()), "x");
    const SymExpr propagated = sym_dsolve_ivp(sentinel, "x", 0.0, 1.0);
    EXPECT_EQ(sym_to_string(propagated), sym_to_string(sentinel));
}

TEST(SymbolicDsolveGeneralTest, ivp2_pins_both_constants) {
    const SymExpr distinct = sym_dsolve_linear2(1.0, -3.0, 2.0, sym_const(0.0), "x");
    const SymExpr pinned = sym_dsolve_ivp2(distinct, "x", 0.0, 2.0, 3.0);
    ASSERT_NE(pinned.op, SymOp::Deriv) << sym_to_string(pinned);
    EXPECT_EQ(sym_to_string(pinned).find("C1"), std::string::npos);
    EXPECT_NEAR(sym_eval(pinned, {{"x", 0.0}}), 2.0, 1e-12);
    EXPECT_NEAR(sym_eval(sym_simplify(sym_diff(clone_expr(pinned), "x")), {{"x", 0.0}}), 3.0,
                1e-12);

    const SymExpr repeated = sym_dsolve_linear2(1.0, 2.0, 1.0, sym_const(0.0), "x");
    const SymExpr critical = sym_dsolve_ivp2(repeated, "x", 0.0, 1.0, 0.0);
    ASSERT_NE(critical.op, SymOp::Deriv) << sym_to_string(critical);
    EXPECT_NEAR(sym_eval(critical, {{"x", 0.0}}), 1.0, 1e-12);
    EXPECT_NEAR(sym_eval(critical, {{"x", 1.0}}), 0.7357588823428847, 1e-9);
}
