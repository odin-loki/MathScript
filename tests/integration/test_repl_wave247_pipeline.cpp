// MathScript Integration Tests: REPL Interpreter – Wave 247 Bindings Pipeline
//
// Wired: signal_unwrap.

#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>

#include "ms/interp/repl_engine.hpp"
#include "ms/signal/signal.hpp"

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

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave247Pipeline, SignalUnwrapBinding) {
    Interpreter interp;

    expect_ok(interp, "ph = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(ph)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    const auto& col = interp.state().matrices.at("u");
    EXPECT_EQ(col.cols(), 1u);
    EXPECT_EQ(col.rows(), 5u);

    constexpr double pi = 3.14159265358979323846;
    const std::vector<double> wrapped{0.0, pi / 2.0, pi, -pi / 2.0, 0.0};
    const auto expected = ms::unwrap(wrapped);
    ASSERT_EQ(expected.size(), col.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(col(i, 0), expected[i], 1e-10);
    }

    expect_error(interp, "signal_unwrap()");
    expect_contains(interp, "help", "signal_unwrap");
}
