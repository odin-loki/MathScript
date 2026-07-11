// MathScript Integration Tests: REPL ODE bindings pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
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

std::pair<double, double> parse_last_traj_row(const std::string& text) {
    std::pair<double, double> last{0.0, 0.0};
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed.size() < 5 || trimmed.front() != '[' || trimmed.back() != ']') {
            continue;
        }
        const std::string inner = trimmed.substr(1, trimmed.size() - 2);
        const auto comma = inner.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        last.first = std::stod(Interpreter::trim(inner.substr(0, comma)));
        last.second = std::stod(Interpreter::trim(inner.substr(comma + 1)));
    }
    return last;
}

void expect_exponential_growth(const std::string& text, double t_end, double tolerance) {
    const auto last = parse_last_traj_row(text);
    EXPECT_NEAR(last.first, t_end, 1e-9);
    EXPECT_NEAR(last.second, std::exp(t_end), tolerance);
    EXPECT_TRUE(std::isfinite(last.second));
}

} // namespace

TEST(OdeReplPipeline, OdeBindingsPipeline) {
    Interpreter interp;

    const auto euler = run(interp, "ode_euler(\"y\", 0, 1, 1, 100)");
    ASSERT_TRUE(euler.has_value());
    expect_exponential_growth(*euler, 1.0, 0.05);
    expect_error(interp, "ode_euler(\"(y\", 0, 1, 1, 100)");

    const auto rk4 = run(interp, "ode_rk4(\"y\", 0, 1, 1, 100)");
    ASSERT_TRUE(rk4.has_value());
    expect_exponential_growth(*rk4, 1.0, 0.02);
    expect_error(interp, "ode_rk4(\"bad(@)\", 0, 1, 1, 100)");

    const auto midpoint = run(interp, "ode_midpoint(\"y\", 0, 1, 1, 100)");
    ASSERT_TRUE(midpoint.has_value());
    expect_exponential_growth(*midpoint, 1.0, 0.03);
    expect_error(interp, "ode_midpoint(\"1+\", 0, 1, 1, 100)");

    const auto rk45 = run(interp, "ode_rk45(\"y\", 0, 1, 1, 1e-6, 1e-9)");
    ASSERT_TRUE(rk45.has_value());
    expect_exponential_growth(*rk45, 1.0, 0.01);
    expect_error(interp, "ode_rk45(\"y +\", 0, 1, 1, 1e-6, 1e-9)");

    const auto backward = run(interp, "ode_backward_euler(\"y\", 0, 1, 1, 100)");
    ASSERT_TRUE(backward.has_value());
    expect_exponential_growth(*backward, 1.0, 0.08);
    expect_error(interp, "ode_backward_euler(\"sin(\", 0, 1, 1, 100)");
}
