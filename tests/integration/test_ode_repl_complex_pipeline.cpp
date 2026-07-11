// MathScript Integration Tests: REPL complex ODE formula-bridge pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "ms/interp/repl_engine.hpp"
#include "ms/ode/ode.hpp"

using namespace ms;
using namespace ms::interp;

namespace {

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

std::vector<double> parse_last_section_row_all(const std::string& text,
                                               const std::string& section) {
    std::vector<double> last;
    bool in_section = false;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed == section + " =") {
            in_section = true;
            continue;
        }
        if (in_section && trimmed.find('=') != std::string::npos && trimmed.back() == '=') {
            break;
        }
        if (!in_section) {
            continue;
        }
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

bool parse_converged_flag(const std::string& text) {
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed == "converged = true") {
            return true;
        }
        if (trimmed == "converged = false") {
            return false;
        }
    }
    return false;
}

size_t parse_event_count(const std::string& text) {
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed.rfind("event_count = ", 0) == 0) {
            return static_cast<size_t>(std::stoul(trimmed.substr(14)));
        }
    }
    return 0;
}

std::vector<double> parse_first_event_row(const std::string& text) {
    bool in_events = false;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed == "events =") {
            in_events = true;
            continue;
        }
        if (in_events && trimmed.size() >= 5 && trimmed.front() == '[' && trimmed.back() == ']') {
            const std::string inner = trimmed.substr(1, trimmed.size() - 2);
            std::vector<double> row;
            std::stringstream cell_stream(inner);
            std::string cell;
            while (std::getline(cell_stream, cell, ',')) {
                row.push_back(std::stod(Interpreter::trim(cell)));
            }
            return row;
        }
    }
    return {};
}

std::vector<double> parse_first_traj_row_all(const std::string& text) {
    bool in_traj = false;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed == "traj =") {
            in_traj = true;
            continue;
        }
        if (!in_traj) {
            continue;
        }
        if (trimmed.empty() || trimmed.front() != '[') {
            break;
        }
        const std::string inner = trimmed.substr(1, trimmed.size() - 2);
        std::vector<double> row;
        std::stringstream cell_stream(inner);
        std::string cell;
        while (std::getline(cell_stream, cell, ',')) {
            row.push_back(std::stod(Interpreter::trim(cell)));
        }
        return row;
    }
    return {};
}

std::pair<double, double> parse_last_traj_section_row(const std::string& text) {
    std::pair<double, double> last{0.0, 0.0};
    bool in_traj = false;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = Interpreter::trim(line);
        if (trimmed == "traj =") {
            in_traj = true;
            continue;
        }
        if (!in_traj) {
            continue;
        }
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed.front() != '[') {
            break;
        }
        if (trimmed.size() < 5 || trimmed.back() != ']') {
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

} // namespace

TEST(OdeReplComplexPipeline, BackwardEulerVec_ExponentialDecay) {
    Interpreter interp;
    const auto result = run(interp, "ode_backward_euler_vec(\"-y0\", 0, [1], 1, 200)");
    ASSERT_TRUE(result.has_value());
    const auto last = parse_last_traj_row_all(*result);
    ASSERT_GE(last.size(), 2u);
    EXPECT_NEAR(last[0], 1.0, 1e-9);
    EXPECT_NEAR(last[1], std::exp(-1.0), 0.08);
}

TEST(OdeReplComplexPipeline, BackwardEulerVec_HarmonicOscillator) {
    Interpreter interp;
    const auto result =
        run(interp, "ode_backward_euler_vec(\"y1; -y0\", 0, [0, 1], 1.570796, 400)");
    ASSERT_TRUE(result.has_value());
    const auto last = parse_last_traj_row_all(*result);
    ASSERT_GE(last.size(), 3u);
    EXPECT_NEAR(last[1], std::sin(1.570796), 0.08);
    EXPECT_NEAR(last[2], std::cos(1.570796), 0.08);
}

TEST(OdeReplComplexPipeline, BackwardEulerVec_MalformedFormulaErrors) {
    Interpreter interp;
    expect_error(interp, "ode_backward_euler_vec(\"y1; sin(\", 0, [1, 0], 1, 100)");
    expect_error(interp, "ode_backward_euler_vec(\"y1\", 0, [1, 0], 1, 100)");
}

TEST(OdeReplComplexPipeline, DaeIndex1_ExpDecayWithConstraint) {
    Interpreter interp;
    const auto result =
        run(interp, "ode_dae_index1(\"-y0\", \"z0 - 2*y0\", 0, [1], [2], 1, 200)");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(parse_converged_flag(*result));
    const auto y_last = parse_last_section_row_all(*result, "y_traj");
    const auto z_last = parse_last_section_row_all(*result, "z_traj");
    ASSERT_GE(y_last.size(), 2u);
    ASSERT_GE(z_last.size(), 2u);
    EXPECT_NEAR(y_last[1], std::exp(-1.0), 0.08);
    EXPECT_NEAR(z_last[1], 2.0 * std::exp(-1.0), 0.08);
}

TEST(OdeReplComplexPipeline, DaeIndex1_SinCosSolution) {
    Interpreter interp;
    const auto result =
        run(interp, "ode_dae_index1(\"z0\", \"z0 - cos(t)\", 0, [0], [1], 3.141593, 400)");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(parse_converged_flag(*result));
    const auto y_last = parse_last_section_row_all(*result, "y_traj");
    const auto z_last = parse_last_section_row_all(*result, "z_traj");
    ASSERT_GE(y_last.size(), 2u);
    ASSERT_GE(z_last.size(), 2u);
    EXPECT_NEAR(y_last[1], std::sin(3.141593), 0.08);
    EXPECT_NEAR(z_last[1], std::cos(3.141593), 0.08);
}

TEST(OdeReplComplexPipeline, DaeIndex1_MidTrajectory) {
    Interpreter interp;
    const auto result =
        run(interp, "ode_dae_index1(\"z0\", \"z0 - cos(t)\", 0, [0], [1], 1.570796, 300)");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(parse_converged_flag(*result));
    const auto y_last = parse_last_section_row_all(*result, "y_traj");
    ASSERT_GE(y_last.size(), 2u);
    EXPECT_NEAR(y_last[1], std::sin(1.570796), 0.08);
}

TEST(OdeReplComplexPipeline, DaeIndex1_MalformedFormulaErrors) {
    Interpreter interp;
    expect_error(interp, "ode_dae_index1(\"z0\", \"z0 - cos(\", 0, [0], [1], 1, 100)");
    expect_error(interp, "ode_dae_index1(\"z0\", \"z0 - cos(t)\", 0, [0], [1, 2], 1, 100)");
}

TEST(OdeReplComplexPipeline, BvpShooting_SinSolution) {
    Interpreter interp;
    const auto result =
        run(interp, "ode_bvp_shooting(\"-y\", 0, 0, 1.570796, 1, 400)");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(parse_converged_flag(*result));
    const auto last = parse_last_traj_row_all(*result);
    ASSERT_GE(last.size(), 3u);
    EXPECT_NEAR(last[0], 1.570796, 1e-5);
    EXPECT_NEAR(last[1], 1.0, 0.02);
    EXPECT_NEAR(last[2], 0.0, 0.05);
}

TEST(OdeReplComplexPipeline, BvpShooting_MidpointMatchesSin) {
    Interpreter interp;
    const auto result =
        run(interp, "ode_bvp_shooting(\"-y\", 0, 0, 1.570796, 1, 400)");
    ASSERT_TRUE(result.has_value());
    std::istringstream ss(*result);
    std::string line;
    size_t row_count = 0;
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
        if (row.size() >= 2) {
            EXPECT_NEAR(row[1], std::sin(row[0]), 0.03);
            ++row_count;
        }
    }
    EXPECT_GT(row_count, 3u);
}

TEST(OdeReplComplexPipeline, BvpShooting_LinearSolution) {
    Interpreter interp;
    const auto result = run(interp, "ode_bvp_shooting(\"0\", 0, 0, 1, 1, 100)");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(parse_converged_flag(*result));
    const auto last = parse_last_traj_row_all(*result);
    ASSERT_GE(last.size(), 3u);
    EXPECT_NEAR(last[1], 1.0, 0.01);
    EXPECT_NEAR(last[2], 1.0, 0.02);
}

TEST(OdeReplComplexPipeline, BvpShooting_MalformedFormulaErrors) {
    Interpreter interp;
    expect_error(interp, "ode_bvp_shooting(\"1+\", 0, 0, 1, 1, 100)");
}

TEST(OdeReplComplexPipeline, DdeFixedStep_MatchesDirectCppCall) {
    Interpreter interp;
    const std::string cmd = "ode_dde_fixed_step(\"-y + ydelay\", \"3\", 0, 5, 1000, 200)";
    const auto repl_result = run(interp, cmd);
    ASSERT_TRUE(repl_result.has_value());

    const auto f = [](double, double y, double yd) { return -y + yd; };
    const auto hist = [](double) { return 3.0; };
    const auto direct = ode_dde_fixed_step(f, hist, 0.0, 5.0, 1000.0, 200);
    ASSERT_FALSE(direct.y.empty());

    const auto repl_last = parse_last_traj_row(*repl_result);
    EXPECT_NEAR(repl_last.first, direct.t.back(), 1e-9);
    EXPECT_NEAR(repl_last.second, direct.y.back(), 1e-9);
    EXPECT_NEAR(repl_last.second, 3.0, 0.1);
}

TEST(OdeReplComplexPipeline, DdeFixedStep_HistoryUsedAtStart) {
    Interpreter interp;
    const auto result = run(interp, "ode_dde_fixed_step(\"ydelay\", \"2\", 0, 0.5, 0.2, 10)");
    ASSERT_TRUE(result.has_value());
    const auto first = parse_first_traj_row_all(*result);
    ASSERT_GE(first.size(), 2u);
    EXPECT_NEAR(first[0], 0.0, 1e-12);
    EXPECT_NEAR(first[1], 2.0, 1e-12);
}

TEST(OdeReplComplexPipeline, DdeFixedStep_MalformedFormulaErrors) {
    Interpreter interp;
    expect_error(interp, "ode_dde_fixed_step(\"y +\", \"2\", 0, 1, 0.5, 10)");
    expect_error(interp, "ode_dde_fixed_step(\"y\", \"sin(\", 0, 1, 0.5, 10)");
}

TEST(OdeReplComplexPipeline, EventDetect_ZeroCrossingAtFive) {
    Interpreter interp;
    const auto result = run(interp, "ode_event_detect(\"1\", \"y\", 0, -5, 10, 100)");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(parse_event_count(*result), 1u);
    const auto event = parse_first_event_row(*result);
    ASSERT_EQ(event.size(), 2u);
    EXPECT_NEAR(event[0], 5.0, 1e-2);
    EXPECT_NEAR(event[1], 0.0, 1e-2);
}

TEST(OdeReplComplexPipeline, EventDetect_TrajectoryPreserved) {
    Interpreter interp;
    const auto result = run(interp, "ode_event_detect(\"1\", \"y\", 0, -2, 4, 40)");
    ASSERT_TRUE(result.has_value());
    const auto last = parse_last_traj_section_row(*result);
    EXPECT_NEAR(last.first, 4.0, 1e-9);
    EXPECT_NEAR(last.second, 2.0, 1e-9);
}

TEST(OdeReplComplexPipeline, EventDetect_NoEventWhenAlwaysPositive) {
    Interpreter interp;
    const auto result = run(interp, "ode_event_detect(\"y\", \"y\", 0, 1, 1, 50)");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(parse_event_count(*result), 0u);
}

TEST(OdeReplComplexPipeline, EventDetect_MalformedFormulaErrors) {
    Interpreter interp;
    expect_error(interp, "ode_event_detect(\"1+\", \"y\", 0, -5, 10, 100)");
    expect_error(interp, "ode_event_detect(\"1\", \"y +\", 0, -5, 10, 100)");
}
