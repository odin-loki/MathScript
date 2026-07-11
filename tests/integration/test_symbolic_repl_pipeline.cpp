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

    // sym_integrate unsupported form returns deriv sentinel string
    const auto unsupported = run(interp, "sym_integrate(\"sin(2*x)\", \"x\")");
    ASSERT_TRUE(unsupported.has_value());
    EXPECT_NE(unsupported->find("d/d"), std::string::npos);

    // sym_eval: numeric evaluation
    const auto eval = run(interp, "sym_eval(\"x^2+1\", \"x=3\")");
    ASSERT_TRUE(eval.has_value());
    EXPECT_NEAR(parse_scalar_output(*eval), 10.0, 1e-12);
    expect_error(interp, "sym_eval(\"x^2+1\", \"x\")");
    expect_error(interp, "sym_eval(\"x^2+1\", \"x=bad\")");
}
