// A contract sweep over the first-order ODE classifier.
//
// sym_dsolve_ode documents a hard guarantee: "a wrong closed form is never emitted, and no
// result ever contains an unevaluated integral" -- an unsupported form comes back as the
// explicit sentinel sym_deriv(rhs, indep_var). That is a property every right-hand side
// must satisfy, not just the handful each shape-specific test covers, and it is exactly the
// property that the classifier's many internal branches (rational integration, log- and
// power-shaped matching, the exact-equation potential recovery, the IVP constant search)
// have to preserve on the inputs they decline as much as on the ones they solve.
//
// So this sweeps a wide table of right-hand sides and asserts, for each: either the result
// is the sentinel, or the returned closed form actually solves the ODE when differentiated
// and substituted back. The check never inspects which branch produced the answer, so it
// stays valid as the classifier grows.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

namespace {

SymExpr clone_expr(const SymExpr& expr) {
    SymExpr copy;
    copy.op = expr.op;
    copy.value = expr.value;
    copy.name = expr.name;
    if (expr.left) copy.left = std::make_unique<SymExpr>(clone_expr(*expr.left));
    if (expr.right) copy.right = std::make_unique<SymExpr>(clone_expr(*expr.right));
    return copy;
}

SymExpr parse(const std::string& text) {
    auto parsed = sym_parse(text);
    EXPECT_TRUE(parsed.has_value()) << "could not parse: " << text;
    if (!parsed) return SymExpr{};
    return std::move(*parsed);
}

// Returns the number of sample points at which the residual was finite and checked.
// Fails the test if any finite residual is non-zero.
int residual_is_zero(const SymExpr& rhs, const SymExpr& solution, const std::string& label) {
    const SymExpr derivative = sym_simplify(sym_diff(clone_expr(solution), "x"));
    int checked = 0;
    for (const double xv : {0.7, 1.3, 2.1}) {
        for (const double cv : {1.5, -0.6}) {
            const std::map<std::string, double> env{{"x", xv}, {"C", cv}};
            const double yv = sym_eval(solution, env);
            if (!std::isfinite(yv)) continue;
            const double dv = sym_eval(derivative, env);
            std::map<std::string, double> full = env;
            full["y"] = yv;
            const double rv = sym_eval(rhs, full);
            if (!std::isfinite(dv) || !std::isfinite(rv)) continue;
            ++checked;
            EXPECT_NEAR(dv, rv, 1e-6 * (1.0 + std::abs(dv) + std::abs(rv)))
                << label << " -> " << sym_to_string(solution) << " at x=" << xv << " C=" << cv;
        }
    }
    return checked;
}

bool is_sentinel(const SymExpr& result) { return result.op == SymOp::Deriv; }

}  // namespace

TEST(SymDsolveSweep, EveryAnswerIsEitherTheSentinelOrActuallySolvesTheOde) {
    const std::vector<std::string> rhs_table{
        // quadrature: rhs free of y
        "1", "x", "x^2", "x^3 + 2*x", "1/x", "1/(x^2)", "1/(x+1)", "1/(x^2+1)",
        "1/(x^2-1)", "1/(x^2+2*x+1)", "1/(x^2+2*x+5)", "x/(x^2+1)", "(2*x+1)/(x^2+x)",
        "exp(x)", "exp(2*x)", "sin(x)", "cos(x)", "sin(2*x)", "exp(x)*sin(x)",
        "log(x)", "x*log(x)", "1/(x*log(x))",
        // separable and linear in y
        "y", "2*y", "-y", "x*y", "y/x", "y*sin(x)", "y + x", "y - x", "2*y + 3",
        "x*y + x", "(y+1)/x", "y/(x+1)",
        // Bernoulli
        "y + y^2", "y - y^3", "x*y + x*y^2", "y/x + y^2",
        // homogeneous degree 0
        "(x+y)/x", "(x^2+y^2)/(x*y)", "y/x + 1", "1 + y/x",
        // rational in both
        "(x+y)/(x-y)", "(2*x+y)/(x+2*y)", "x/(y+1)", "(y^2)/(x^2)",
        // forms the classifier is documented to decline
        "y^2 + y", "y*y", "sin(y)", "exp(y)", "y*log(y)", "sqrt(y)", "tan(x)*y",
    };

    int solved = 0;
    int declined = 0;
    for (const auto& text : rhs_table) {
        const SymExpr rhs = parse(text);
        const SymExpr result = sym_dsolve_ode(clone_expr(rhs), "x", "y");
        if (is_sentinel(result)) {
            ++declined;
            continue;
        }
        ++solved;
        residual_is_zero(rhs, result, text);
    }
    // The table is meant to cover both outcomes; if either count collapsed the sweep would
    // silently stop testing one of them.
    EXPECT_GT(solved, 20) << "the classifier solved far fewer cases than expected";
    EXPECT_GT(declined, 0) << "nothing was declined, so the sentinel path went untested";
}

TEST(SymDsolveSweep, FrozenSeparableTableAgreesWithTheGeneralClassifier) {
    // sym_dsolve is documented as a frozen subset: every rhs it solves, sym_dsolve_ode must
    // solve identically.
    const std::vector<std::string> table{"1", "x", "y", "2*y", "x*y", "exp(x)", "sin(x)",
                                         "x^2", "1/x", "y/x"};
    for (const auto& text : table) {
        const SymExpr rhs = parse(text);
        const SymExpr frozen = sym_dsolve(clone_expr(rhs), "x", "y");
        if (frozen.op == SymOp::Deriv) continue;  // declined by the frozen table
        const SymExpr general = sym_dsolve_ode(clone_expr(rhs), "x", "y");
        EXPECT_EQ(sym_to_string(general), sym_to_string(frozen)) << text;
    }
}

TEST(SymDsolveSweep, ExactEquationsEitherSolveOrReportTheSentinel) {
    // m dx + n dy = 0 is exact iff dm/dy == dn/dx. Both an exact and a non-exact pair must
    // be handled without emitting a wrong potential.
    struct Case { const char* m; const char* n; bool exact; };
    const std::vector<Case> cases{
        {"2*x + y", "x + 2*y", true},          // F = x^2 + x y + y^2
        {"y", "x", true},                      // F = x y
        {"2*x*y", "x^2", true},                // F = x^2 y
        {"3*x^2 + y", "x + 3*y^2", true},      // F = x^3 + x y + y^3
        {"y", "2*x", false},                   // dm/dy = 1, dn/dx = 2
        {"sin(y)", "x", false},
    };
    for (const auto& c : cases) {
        const SymExpr m = parse(c.m);
        const SymExpr n = parse(c.n);
        const SymExpr f = sym_dsolve_exact(clone_expr(m), clone_expr(n), "x", "y");
        if (f.op == SymOp::Deriv) {
            EXPECT_FALSE(c.exact) << "declined an exact pair: " << c.m << " , " << c.n;
            continue;
        }
        ASSERT_TRUE(c.exact) << "returned a potential for a non-exact pair: " << c.m;
        // F_x == m and F_y == n at sample points.
        const SymExpr fx = sym_simplify(sym_diff(clone_expr(f), "x"));
        const SymExpr fy = sym_simplify(sym_diff(clone_expr(f), "y"));
        for (const double xv : {0.6, 1.7}) {
            for (const double yv : {0.9, -1.2}) {
                const std::map<std::string, double> env{{"x", xv}, {"y", yv}, {"C", 0.0}};
                EXPECT_NEAR(sym_eval(fx, env), sym_eval(m, env), 1e-7)
                    << c.m << " , " << c.n << " at (" << xv << "," << yv << ")";
                EXPECT_NEAR(sym_eval(fy, env), sym_eval(n, env), 1e-7)
                    << c.m << " , " << c.n << " at (" << xv << "," << yv << ")";
            }
        }
    }
}

TEST(SymDsolveSweep, InitialValueProblemsPinTheConstantOrReportIt) {
    // sym_dsolve_ivp must either reproduce y(x0) = y0 exactly or return the sentinel; it
    // must never return a solution that misses its own initial condition.
    const std::vector<std::string> table{"1", "x", "y", "2*y", "x*y", "exp(x)", "1/x",
                                         "y + x", "y/x", "x^2", "sin(x)"};
    int pinned = 0;
    for (const auto& text : table) {
        const SymExpr rhs = parse(text);
        const SymExpr general = sym_dsolve_ode(clone_expr(rhs), "x", "y");
        if (general.op == SymOp::Deriv) continue;
        for (const double y0 : {1.0, 2.5, -0.5}) {
            const SymExpr particular = sym_dsolve_ivp(clone_expr(general), "x", 1.0, y0);
            if (particular.op == SymOp::Deriv) continue;  // no C reproduces this condition
            ++pinned;
            const std::map<std::string, double> at_x0{{"x", 1.0}};
            const double got = sym_eval(particular, at_x0);
            if (!std::isfinite(got)) continue;
            EXPECT_NEAR(got, y0, 1e-6 * (1.0 + std::abs(y0)))
                << text << " with y(1)=" << y0 << " -> " << sym_to_string(particular);
            // A pinned solution must have no free C left.
            const std::map<std::string, double> other_c{{"x", 1.0}, {"C", 99.0}};
            EXPECT_NEAR(sym_eval(particular, other_c), got, 1e-9)
                << "the pinned solution still depends on C: " << sym_to_string(particular);
        }
    }
    EXPECT_GT(pinned, 5) << "no initial-value problem was pinned, so the search went untested";
}

TEST(SymDsolveSweep, SecondOrderConstantCoefficientForcings) {
    // a y'' + b y' + c y = forcing, over the three characteristic-root regimes and the
    // forcing shapes the table covers, including the resonant ones.
    struct Case { double a, b, c; const char* forcing; };
    const std::vector<Case> cases{
        {1, -3, 2, "0"},        // distinct real roots 1, 2
        {1, -2, 1, "0"},        // repeated root 1
        {1, 0, 1, "0"},         // complex roots +-i
        {1, -3, 2, "1"},        // constant forcing
        {1, -3, 2, "x"},        // polynomial forcing
        {1, -3, 2, "x^2 + 1"},
        {1, 0, 1, "sin(2*x)"},  // non-resonant sinusoid
        {1, 0, 1, "exp(x)"},
        {1, -3, 2, "exp(3*x)"},
        {1, -3, 2, "exp(x)"},   // RESONANT: exp(x) is a homogeneous solution
        {1, 0, 1, "sin(x)"},    // RESONANT sinusoid
        {1, 0, 4, "cos(2*x)"},  // RESONANT sinusoid
    };
    for (const auto& c : cases) {
        const SymExpr forcing = parse(c.forcing);
        const SymExpr sol = sym_dsolve_linear2(c.a, c.b, c.c, clone_expr(forcing), "x");
        if (sol.op == SymOp::Deriv) continue;  // outside the undetermined-coefficients table
        const SymExpr d1 = sym_simplify(sym_diff(clone_expr(sol), "x"));
        const SymExpr d2 = sym_simplify(sym_diff(clone_expr(d1), "x"));
        for (const double xv : {0.5, 1.4}) {
            const std::map<std::string, double> env{{"x", xv}, {"C1", 1.0}, {"C2", -0.5}};
            const double residual = c.a * sym_eval(d2, env) + c.b * sym_eval(d1, env) +
                                    c.c * sym_eval(sol, env) - sym_eval(forcing, env);
            if (!std::isfinite(residual)) continue;
            EXPECT_NEAR(residual, 0.0, 1e-6) << c.forcing << " at x=" << xv
                                             << " -> " << sym_to_string(sol);
        }
    }
}

TEST(SymDsolveSweep, IntegrationOfRationalFunctions) {
    // The quadrature stage integrates a rational integrand when the denominator is a linear
    // factor or a power of one -- 1/(x+a) -> log(x+a), 1/(x+a)^2 -> -(x+a)^-1. Partial
    // fractions over distinct or irreducible-quadratic denominators are not implemented, and
    // the module's contract is that it declines those with the sentinel rather than emitting
    // an unevaluated integral. This pins both halves of that behaviour.
    const std::vector<std::string> integrable{"1/(x+2)", "1/((x+1)^2)"};
    for (const auto& text : integrable) {
        const SymExpr rhs = parse(text);
        const SymExpr sol = sym_dsolve_ode(clone_expr(rhs), "x", "y");
        ASSERT_NE(sol.op, SymOp::Deriv) << "no longer integrates " << text;
        EXPECT_GT(residual_is_zero(rhs, sol, text), 0);
    }

    // Everything here is declined today. If a future partial-fraction stage starts solving
    // one, the residual check below is what keeps the new answer honest.
    const std::vector<std::string> harder{
        "1/(x^2-4)", "1/(x^2+4)", "1/(x^2+2*x+2)", "x/(x^2+1)", "x/(x+1)",
        "(x+1)/(x^2+1)", "(x^2+1)/(x+1)", "1/(x^3+x)", "(3*x+2)/(x^2+3*x+2)",
    };
    for (const auto& text : harder) {
        const SymExpr rhs = parse(text);
        const SymExpr sol = sym_dsolve_ode(clone_expr(rhs), "x", "y");
        if (sol.op == SymOp::Deriv) continue;
        residual_is_zero(rhs, sol, text);
    }
}
