// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include <cstddef>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ms {

enum class SymOp {
    Add,
    Mul,
    Var,
    Const,
    Deriv,
    Sin,
    Cos,
    Exp,
    Log,
    Pow,
    Sub,
    Div,
    Neg,
    Tan,
    Sqrt
};

struct SymExpr {
    SymOp op = SymOp::Const;
    double value = 0.0;
    std::string name;
    std::unique_ptr<SymExpr> left;
    std::unique_ptr<SymExpr> right;
};

SymExpr sym_const(double value);
SymExpr sym_var(const std::string& name);
SymExpr sym_add(SymExpr a, SymExpr b);
SymExpr sym_mul(SymExpr a, SymExpr b);
SymExpr sym_sub(SymExpr a, SymExpr b);
SymExpr sym_div(SymExpr a, SymExpr b);
SymExpr sym_neg(SymExpr a);
SymExpr sym_sin(SymExpr arg);
SymExpr sym_cos(SymExpr arg);
SymExpr sym_tan(SymExpr arg);
SymExpr sym_exp(SymExpr arg);
SymExpr sym_log(SymExpr arg);
SymExpr sym_sqrt(SymExpr arg);
SymExpr sym_pow(SymExpr base, SymExpr exponent);
SymExpr sym_deriv(SymExpr expr, const std::string& var);
SymExpr sym_diff(SymExpr expr, const std::string& var);
SymExpr sym_simplify(SymExpr expr);
SymExpr sym_expand(SymExpr expr);
SymExpr sym_collect(const SymExpr& expr, const std::string& var);
SymExpr sym_integrate(const SymExpr& expr, const std::string& var);

/// Structural equality: same operator, value, name and children throughout.
/// Not mathematical equality -- `x + 0` and `x` are different expressions here.
bool sym_equal(const SymExpr& a, const SymExpr& b);

/// True when `result` carries the unsupported-sentinel for `var`.
///
/// The functions below signal "no closed form" by returning sym_deriv(input, var).
/// That sentinel is not inert: sym_eval evaluates a Deriv node by differentiating
/// it, so a caller who integrates and then evaluates gets the derivative's value
/// where the integral was asked for, with nothing anywhere reporting a failure.
/// sym_integrate(1/x, "x") followed by sym_eval at x=2 returns -0.25; the integral
/// is log(2) = 0.693.
///
/// Callers that cannot tolerate that must check before using a result:
///
///     const auto r = sym_integrate(f, "x");
///     if (sym_is_unsupported(r, "x")) { ... no closed form ... }
///
/// The scan is recursive, and it has to be. The linearity rules recurse into each
/// operand and reassemble whatever comes back, so one unsupported term inside an
/// otherwise-supported expression yields a tree where the sentinel is a *subterm*
/// rather than the root. Comparing only the root against the input -- which is what
/// this predicate used to do -- reports such a result as a success and hands back a
/// mixture of a transform and a derivative:
///
///     sym_laplace("t + t*exp(2*t)")  ->  (1/s^2) + d/dt(t*exp(2*t))
///
/// The transforms below also propagate failure outward at every linearity site, so
/// in practice the sentinel arrives at the root anyway; this predicate is the
/// backstop that does not depend on every one of those sites being right.
///
/// The scan is exact rather than heuristic: SymOp::Deriv is constructed in exactly
/// one translation unit, only ever as this sentinel, and no parser or public
/// constructor can produce one -- so a Deriv node bearing `var` in a transform
/// result cannot be anything else.
bool sym_is_unsupported(const SymExpr& result, const std::string& var);

// Forward/inverse transforms. Unsupported forms return sym_deriv(expr, var)
// as an explicit sentinel (same convention as sym_integrate); see
// sym_is_unsupported above for why that sentinel needs checking rather than
// passing on.
SymExpr sym_laplace(const SymExpr& expr, const std::string& t, const std::string& s);
SymExpr sym_ilaplace(const SymExpr& expr, const std::string& s, const std::string& t);
SymExpr sym_mellin(const SymExpr& expr, const std::string& t, const std::string& s);
SymExpr sym_imellin(const SymExpr& expr, const std::string& s, const std::string& t);
SymExpr sym_hankel(const SymExpr& expr, const std::string& r, const std::string& k);
SymExpr sym_ihankel(const SymExpr& expr, const std::string& k, const std::string& r);
SymExpr sym_fourier(const SymExpr& expr, const std::string& t_var, const std::string& omega_var);
SymExpr sym_ifourier(const SymExpr& expr, const std::string& omega_var, const std::string& t_var);
SymExpr sym_ztransform(const SymExpr& expr, const std::string& n_var, const std::string& z_var);
SymExpr sym_iztransform(const SymExpr& expr, const std::string& z_var, const std::string& n_var);
// First-order ODE dy/d(indep)=rhs. Table-driven separable MVP; unsupported -> sym_deriv(rhs, indep).
// FROZEN v1 CONTRACT: this entry point keeps the exact separable-only behaviour it shipped
// with, including every rhs shape it declines (y + x, y^2 + y, y*y, y^1, sin(2*x), ...).
// There is no mathematical predicate separating those declined shapes from the ones the
// general classifier must solve, so widening this function would have to be done with a
// hard-coded shape list. Callers that want the general classifier (linear, Bernoulli,
// exact, homogeneous) call sym_dsolve_ode instead: it runs this table first and therefore
// returns an identical expression for every rhs this one solves.
SymExpr sym_dsolve(const SymExpr& rhs, const std::string& indep_var, const std::string& dep_var);

// General first-order ODE solver for dy/d(indep_var) = rhs, where rhs may depend on both
// indep_var and dep_var. Classification pipeline, tried in order:
//   1. separable          - the frozen sym_dsolve table (identical results)
//   2. quadrature         - rhs free of dep_var: y = integral(rhs) + C
//   3. linear             - y' = A(x) y + B(x): mu = exp(-integral A), y = (integral mu*B + C)/mu
//   4. Bernoulli          - y' = A(x) y + B(x) y^n, n != 0,1: v = y^(1-n) reduces to linear
//   5. exact              - rhs = P/Q with (-P) dx + Q dy = 0 exact and F solvable for y
//   6. homogeneous deg 0  - rhs = F(y/x): v = y/x separates, y = x*v(x)
// Every candidate produced by stages 2-6 is verified numerically: the candidate is
// differentiated symbolically with sym_diff and substituted back into the ODE at nine
// sample points, and a candidate whose residual does not vanish is discarded and the
// pipeline continues. The arbitrary constant is a variable named "C".
// Limitations: no Riccati, no Clairaut, no integrating-factor search for non-exact
// equations, no reduction of order; the homogeneous stage cannot invert relations that
// need an arctangent because the AST has no Atan node; polynomial degrees are capped at 8.
// Unsupported forms, and any needed integral the module cannot evaluate in closed form,
// return sym_deriv(rhs, indep_var) as an explicit sentinel - a wrong closed form is
// never emitted, and no result ever contains an unevaluated integral.
SymExpr sym_dsolve_ode(const SymExpr& rhs, const std::string& indep_var, const std::string& dep_var);

// Exact first-order equation m(x,y) dx + n(x,y) dy = 0. Tests d m/d y == d n/d x, then
// recovers the potential F with F_x = m and F_y = n by partial integration, and returns the
// implicit solution F(x,y) - C, whose zero set is the solution family (i.e. F(x,y) = C).
// Unlike sym_dsolve_ode the result is an implicit relation in both variables, not y(x).
// No integrating factor is searched for when the pair is not already exact.
// A non-exact pair, an integral the module cannot evaluate, or a recovered potential that
// fails its F_x/F_y check returns sym_deriv(m, indep_var) as an explicit sentinel.
SymExpr sym_dsolve_exact(const SymExpr& m, const SymExpr& n, const std::string& indep_var,
                         const std::string& dep_var);

// Second-order linear constant-coefficient ODE a y'' + b y' + c y = forcing(indep_var).
// The homogeneous part comes from the characteristic roots of a r^2 + b r + c:
//   distinct real r1,r2  -> C1 exp(r1 x) + C2 exp(r2 x)
//   repeated real r      -> (C1 + C2 x) exp(r x)
//   complex alpha+i*beta -> exp(alpha x) (C1 cos(beta x) + C2 sin(beta x))
// A particular solution is added by undetermined coefficients for forcing terms that are
// polynomials (degree <= 8), K exp(g x + d), and K sin/cos(w x + p), including the resonant
// cases, summed termwise. The two arbitrary constants are variables named "C1" and "C2";
// the dependent variable never appears in the result, so it is not a parameter.
// Variable-coefficient equations (Euler-Cauchy included) are out of scope: a == 0 (not
// second order), or a forcing term outside that table, returns sym_deriv(forcing,
// indep_var) as an explicit sentinel.
SymExpr sym_dsolve_linear2(double a, double b, double c, const SymExpr& forcing,
                           const std::string& indep_var);

// Pins the single arbitrary constant "C" of a general solution from y(x0) = y0 and returns
// the particular solution. Solves for C exactly when the solution is affine in C, otherwise
// by bracketing and bisecting on a deterministic ladder of trial values. A solution with no
// "C", or an initial condition no C reproduces, returns sym_deriv(general_solution,
// indep_var) as an explicit sentinel; a sentinel passed in is propagated unchanged.
SymExpr sym_dsolve_ivp(const SymExpr& general_solution, const std::string& indep_var,
                       double x0, double y0);

// Pins the two arbitrary constants "C1" and "C2" of a second-order general solution from
// y(x0) = y0 and y'(x0) = yp0 by solving the 2x2 linear system the constants enter through
// (every form sym_dsolve_linear2 emits is affine in C1 and C2, so no iteration is needed).
// A singular system, or an initial condition the pinned solution fails to reproduce,
// returns sym_deriv(general_solution, indep_var) as an explicit sentinel; a sentinel
// passed in is propagated unchanged.
SymExpr sym_dsolve_ivp2(const SymExpr& general_solution, const std::string& indep_var,
                        double x0, double y0, double yp0);
SymExpr sym_substitute(const SymExpr& expr, const std::string& var, const SymExpr& replacement);
double sym_eval(const SymExpr& expr, const std::map<std::string, double>& env);
std::string sym_to_string(const SymExpr& expr);

struct SymParseError {
    std::string message;
    size_t position = 0;
};

std::expected<SymExpr, SymParseError> sym_parse(const std::string& text);

double sym_limit(const SymExpr& expr, const std::string& var, double point);
SymExpr sym_series(const SymExpr& expr, const std::string& var, double point, int order);

struct SymSolveError {
    std::string message;
};

std::expected<std::map<std::string, SymExpr>, SymSolveError> sym_solve_linear(
    const std::vector<SymExpr>& equations, const std::vector<std::string>& vars);

} // namespace ms
