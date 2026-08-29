#pragma once

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"

namespace {

inline void expect_ok(ms::interp::Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

inline void expect_contains(ms::interp::Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

inline void expect_error_contains(ms::interp::Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_FALSE(result.has_value()) << cmd;
    const std::string message = ms::format_error(result.error());
    EXPECT_NE(message.find(needle), std::string::npos) << cmd << " error: " << message;
}

inline void expect_error(ms::interp::Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

inline double packed_tf_dcgain(const ms::Matrix<double>& packed) {
    std::vector<double> num(packed.cols());
    std::vector<double> den(packed.cols());
    for (size_t j = 0; j < packed.cols(); ++j) {
        num[j] = packed(0, j);
        den[j] = packed(1, j);
    }
    while (num.size() > 1 && std::abs(num.back()) < 1e-15) {
        num.pop_back();
    }
    while (den.size() > 1 && std::abs(den.back()) < 1e-15) {
        den.pop_back();
    }
    return ms::control::dcgain(ms::control::tf(std::move(num), std::move(den)));
}

inline std::pair<double, double> parse_last_ode_traj_row(const std::string& text) {
    std::pair<double, double> last{0.0, 0.0};
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = ms::interp::Interpreter::trim(line);
        if (trimmed.size() < 5 || trimmed.front() != '[' || trimmed.back() != ']') {
            continue;
        }
        const std::string inner = trimmed.substr(1, trimmed.size() - 2);
        const auto comma = inner.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        last.first = std::stod(ms::interp::Interpreter::trim(inner.substr(0, comma)));
        last.second = std::stod(ms::interp::Interpreter::trim(inner.substr(comma + 1)));
    }
    return last;
}

inline std::vector<double> parse_last_ode_traj_row_all(const std::string& text) {
    std::vector<double> last;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string trimmed = ms::interp::Interpreter::trim(line);
        if (trimmed.size() < 5 || trimmed.front() != '[' || trimmed.back() != ']') {
            continue;
        }
        const std::string inner = trimmed.substr(1, trimmed.size() - 2);
        std::vector<double> row;
        std::stringstream cell_stream(inner);
        std::string cell;
        while (std::getline(cell_stream, cell, ',')) {
            row.push_back(std::stod(ms::interp::Interpreter::trim(cell)));
        }
        if (!row.empty()) {
            last = std::move(row);
        }
    }
    return last;
}

inline double parse_scalar_x_opt(const std::string& text) {
    const auto pos = text.find("x_opt = ");
    if (pos == std::string::npos) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(text.substr(pos + 8));
}

inline double parse_optim_f_val(const std::string& text) {
    const auto pos = text.find("f_val = ");
    if (pos == std::string::npos) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(text.substr(pos + 8));
}

}  // namespace
