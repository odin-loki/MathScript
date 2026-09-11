// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MathScript Integration Tests: REPL symbolic bindings pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

ms::Result<std::string> run(Interpreter& interp, const std::string& cmd) {
    return interp.execute(cmd);
}

double parse_scalar_output(const std::string& text) {
    return std::stod(ms::interp::Interpreter::trim(text));
}

} // namespace

TEST(SymbolicReplPipeline, SymbolicBindingsPipeline) {
    Interpreter interp;

    // sym_diff: differentiate sin(x)*x^2 w.r.t. x
    const auto diff = run(interp, "sym_diff(\"sin(x)*x^2\", \"x\")");
    ASSERT_TRUE(diff.has_value());
    EXPECT_NE(diff->find("x"), std::string::npos);
    expect_error(interp, "sym_diff(\"sin(x)*x^2\")");
    expect_error(interp, "sym_diff(\"bad(@)\", \"x\")");

    // sym_simplify: fold constants
    const auto simp = run(interp, "sym_simplify(\"2+3\")");
    ASSERT_TRUE(simp.has_value());
    EXPECT_NE(simp->find('5'), std::string::npos);
    expect_error(interp, "sym_simplify()");
    expect_error(interp, "sym_simplify(\"1+\")");

    // sym_integrate: power rule on x^2
    const auto integ = run(interp, "sym_integrate(\"x^2\", \"x\")");
    ASSERT_TRUE(integ.has_value());
    EXPECT_NE(integ->find("x"), std::string::npos);
    expect_error(interp, "sym_integrate(\"x^2\")");

    // sin(2*x) integrates now, to -cos(2*x)/2. This assertion used to require it to
    // fail, with a comment saying the failure was a gap in the table rather than a
    // property of the input.
    const auto linear_arg = run(interp, "sym_integrate(\"sin(2*x)\", \"x\")");
    ASSERT_TRUE(linear_arg.has_value());
    EXPECT_NE(linear_arg->find("cos"), std::string::npos);

    // An expression sym_integrate genuinely cannot handle is reported rather than
    // answered. It used to return the unsupported sentinel -- sym_deriv(expr, var) --
    // which printed as "d/dx(...)", and this test pinned that string.
    //
    // The sentinel is not inert: sym_eval evaluates a Deriv node by differentiating
    // it, so integrate-then-evaluate returned the derivative's value where the
    // integral was asked for, with nothing reporting a failure anywhere.
    expect_error(interp, "sym_integrate(\"exp(x^2)\", \"x\")");

    // sym_eval: numeric evaluation
    const auto eval = run(interp, "sym_eval(\"x^2+1\", \"x=3\")");
    ASSERT_TRUE(eval.has_value());
    EXPECT_NEAR(parse_scalar_output(*eval), 10.0, 1e-12);
    expect_error(interp, "sym_eval(\"x^2+1\", \"x\")");
    expect_error(interp, "sym_eval(\"x^2+1\", \"x=bad\")");

    // sym_expand: distribute multiplication
    const auto expanded = run(interp, "sym_expand(\"(x+1)*(x+2)\")");
    ASSERT_TRUE(expanded.has_value());
    EXPECT_NE(expanded->find("x"), std::string::npos);
    expect_error(interp, "sym_expand()");

    // sym_collect: combine like terms
    const auto collected = run(interp, "sym_collect(\"x+x+1\", \"x\")");
    ASSERT_TRUE(collected.has_value());
    EXPECT_NE(collected->find("x"), std::string::npos);
    expect_error(interp, "sym_collect(\"x+x+1\")");

    // sym_substitute: replace variable
    const auto substituted = run(interp, "sym_substitute(\"x^2\", \"x\", \"t\")");
    ASSERT_TRUE(substituted.has_value());
    EXPECT_NE(substituted->find("t"), std::string::npos);
    expect_error(interp, "sym_substitute(\"x^2\", \"x\")");

    // sym_limit: sin(x)/x at 0
    const auto limit = run(interp, "sym_limit(\"sin(x)/x\", \"x\", 0)");
    ASSERT_TRUE(limit.has_value());
    EXPECT_NEAR(parse_scalar_output(*limit), 1.0, 1e-5);
    expect_error(interp, "sym_limit(\"sin(x)/x\", \"x\")");

    // sym_series: Taylor expansion at 0
    const auto series = run(interp, "sym_series(\"exp(x)\", \"x\", 0, 3)");
    ASSERT_TRUE(series.has_value());
    EXPECT_NE(series->find("x"), std::string::npos);
    expect_error(interp, "sym_series(\"exp(x)\", \"x\", 0)");

    // sym_solve_linear: 2x + 4 = 0
    const auto solved = run(interp, "sym_solve_linear(\"2*x+4\", \"x\")");
    ASSERT_TRUE(solved.has_value());
    EXPECT_NE(solved->find("x"), std::string::npos);
    expect_error(interp, "sym_solve_linear(\"2*x+4\")");

    // sym_laplace: exponential decay
    const auto laplace = run(interp, "sym_laplace(\"exp(2*t)\", \"t\", \"s\")");
    ASSERT_TRUE(laplace.has_value());
    EXPECT_NE(laplace->find("s"), std::string::npos);
    expect_error(interp, "sym_laplace(\"exp(2*t)\", \"t\")");

    // sym_fourier: decaying exponential
    const auto fourier = run(interp, "sym_fourier(\"exp(-2*t)\", \"t\", \"omega\")");
    ASSERT_TRUE(fourier.has_value());
    EXPECT_NE(fourier->find("omega"), std::string::npos);
    expect_error(interp, "sym_fourier(\"exp(-2*t)\", \"t\")");

    // sym_ztransform: geometric sequence
    const auto ztransform = run(interp, "sym_ztransform(\"0.5^n\", \"n\", \"z\")");
    ASSERT_TRUE(ztransform.has_value());
    EXPECT_NE(ztransform->find("z"), std::string::npos);
    expect_error(interp, "sym_ztransform(\"0.5^n\", \"n\")");
}
