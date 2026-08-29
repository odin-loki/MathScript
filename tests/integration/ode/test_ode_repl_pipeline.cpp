// MathScript Integration Tests: REPL ODE bindings pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

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

std::vector<double> parse_last_traj_row_all(const std::string& text) {
    std::vector<double> last;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed.size() < 5 || trimmed.front() != '[' || trimmed.back() != ']') {
            continue;
        }
        const std::string inner = trimmed.substr(1, trimmed.size() - 2);
        std::vector<double> row;
        std::stringstream cell_stream(inner);
        std::string cell;
        while (std::getline(cell_stream, cell, ',')) {
            row.push_back(std::stod(Interpreter::trim(cell)));
        }
        if (!row.empty()) {
            last = std::move(row);
        }
    }
    return last;
}

void expect_exponential_growth(const std::string& text, double t_end, double tolerance) {
    const auto last = parse_last_traj_row(text);
    EXPECT_NEAR(last.first, t_end, 1e-9);
    EXPECT_NEAR(last.second, std::exp(t_end), tolerance);
    EXPECT_TRUE(std::isfinite(last.second));
}

void expect_harmonic_oscillator(const std::string& text, double t_end, double y0_tol,
                                double y1_tol) {
    const auto last = parse_last_traj_row_all(text);
    ASSERT_GE(last.size(), 3u);
    EXPECT_NEAR(last[0], t_end, 1e-6);
    EXPECT_NEAR(last[1], std::cos(t_end), y0_tol);
    EXPECT_NEAR(last[2], -std::sin(t_end), y1_tol);
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

    const auto bdf2 = run(interp, "ode_bdf2(\"-10*y\", 0, 1, 1, 200)");
    ASSERT_TRUE(bdf2.has_value());
    const auto bdf2_last = parse_last_traj_row(*bdf2);
    EXPECT_NEAR(bdf2_last.first, 1.0, 1e-9);
    EXPECT_NEAR(bdf2_last.second, std::exp(-10.0), 0.05);
    expect_error(interp, "ode_bdf2(\"bad(@)\", 0, 1, 1, 100)");

    const auto verlet = run(interp, "ode_verlet(\"-9.8\", 0, 0, 0, 1, 1000)");
    ASSERT_TRUE(verlet.has_value());
    const auto verlet_last = parse_last_traj_row_all(*verlet);
    ASSERT_EQ(verlet_last.size(), 3u);
    EXPECT_NEAR(verlet_last[0], 1.0, 1e-9);
    EXPECT_NEAR(verlet_last[1], -4.9, 0.05);
    EXPECT_NEAR(verlet_last[2], -9.8, 0.1);
    expect_error(interp, "ode_verlet(\"1+\", 0, 0, 0, 1, 100)");

    constexpr double t_end = 1.0;

    const auto euler_vec = run(interp, "ode_euler_vec(\"y1; -y0\", 0, [1, 0], 1, 1000)");
    ASSERT_TRUE(euler_vec.has_value());
    expect_harmonic_oscillator(*euler_vec, t_end, 0.08, 0.08);
    expect_error(interp, "ode_euler_vec(\"y1\", 0, [1, 0], 1, 100)");

    const auto rk4_vec = run(interp, "ode_rk4_vec(\"y1; -y0\", 0, [1, 0], 1, 1000)");
    ASSERT_TRUE(rk4_vec.has_value());
    expect_harmonic_oscillator(*rk4_vec, t_end, 0.02, 0.02);
    expect_error(interp, "ode_rk4_vec(\"y1; 1+\", 0, [1, 0], 1, 100)");

    const auto rk45_vec =
        run(interp, "ode_rk45_vec(\"y1; -y0\", 0, [1, 0], 1, 1e-6, 1e-9)");
    ASSERT_TRUE(rk45_vec.has_value());
    expect_harmonic_oscillator(*rk45_vec, t_end, 0.01, 0.01);
    expect_error(interp, "ode_rk45_vec(\"y1; sin(\", 0, [1, 0], 1, 1e-6, 1e-9)");

    const auto verlet_vec =
        run(interp, "ode_verlet_vec(\"-9.8; 0\", 0, [0, 0], [0, 5], 1, 1000)");
    ASSERT_TRUE(verlet_vec.has_value());
    const auto verlet_vec_last = parse_last_traj_row_all(*verlet_vec);
    ASSERT_EQ(verlet_vec_last.size(), 5u);
    EXPECT_NEAR(verlet_vec_last[0], 1.0, 1e-9);
    EXPECT_NEAR(verlet_vec_last[1], -4.9, 0.05);
    EXPECT_NEAR(verlet_vec_last[2], 5.0, 0.05);
    EXPECT_NEAR(verlet_vec_last[3], -9.8, 0.1);
    EXPECT_NEAR(verlet_vec_last[4], 5.0, 0.05);
    expect_error(interp, "ode_verlet_vec(\"-9.8\", 0, [0, 0], [0, 5], 1, 100)");
}
