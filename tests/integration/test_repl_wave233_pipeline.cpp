// MathScript Integration Tests: REPL Interpreter – Wave 233 Optim Bindings Pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

void expect_near_in_output(Interpreter& interp, const std::string& cmd, double expected,
                           double tol) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    const std::size_t start = result->find('[');
    ASSERT_NE(start, std::string::npos) << *result;
    const std::size_t end = result->find(']', start);
    ASSERT_NE(end, std::string::npos) << *result;
    const double x_opt = std::stod(result->substr(start + 1, end - start - 1));
    EXPECT_NEAR(x_opt, expected, tol) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave233Pipeline, BfgsMinimizesQuadratic) {
    Interpreter interp;
    expect_near_in_output(interp, R"cmd(bfgs("(x0-3)*(x0-3)", [0]))cmd", 3.0, 1e-3);
}

TEST(ReplWave233Pipeline, AdvancedOptimBindings) {
    Interpreter interp;

    expect_near_in_output(interp, R"cmd(nelder_mead("(x0-3)*(x0-3)", [8]))cmd", 3.0, 1e-2);
    expect_near_in_output(interp, R"cmd(lbfgs("(x0-3)*(x0-3)", [0]))cmd", 3.0, 1e-3);
    expect_contains(interp, R"cmd(golden_section("(x0-3)*(x0-3)", 0, 10))cmd", "3.");
    expect_ok(interp, R"cmd(levenberg_marquardt("x0-3", [0]))cmd");
    expect_ok(interp, R"cmd(adam("(x0-3)*(x0-3)", [0], 0.05, 500))cmd");

    expect_contains(interp, "help", "bfgs");
    expect_contains(interp, "help", "nelder_mead");
    expect_contains(interp, "help", "lbfgs");
    expect_contains(interp, "help", "golden_section");
    expect_contains(interp, "help", "levenberg_marquardt");
    expect_contains(interp, "help", "adam");
}
