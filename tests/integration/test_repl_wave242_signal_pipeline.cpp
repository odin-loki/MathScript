// MathScript Integration Tests: REPL Interpreter – Wave 242 Signal Hilbert Bindings
//
// Wired: signal_envelope, signal_hilbert, signal_instantaneous_freq.

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

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave242Pipeline, SignalEnvelopeBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    const auto& env = interp.state().matrices.at("env");
    EXPECT_EQ(env.cols(), 1u);
    EXPECT_EQ(env.rows(), 8u);
    for (size_t i = 0; i < env.rows(); ++i) {
        EXPECT_TRUE(std::isfinite(env(i, 0)));
        EXPECT_GE(env(i, 0), 0.0);
    }

    expect_error(interp, "signal_envelope()");
    expect_contains(interp, "help", "signal_envelope");
}

TEST(ReplWave242Pipeline, SignalHilbertBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "z = signal_hilbert(x)");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("z"), 0u);
    const auto& z = interp.state().matrices.at("z");
    EXPECT_EQ(z.cols(), 2u);
    EXPECT_EQ(z.rows(), 8u);
    for (size_t i = 0; i < z.rows(); ++i) {
        EXPECT_TRUE(std::isfinite(z(i, 0)));
        EXPECT_TRUE(std::isfinite(z(i, 1)));
    }

    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    const auto& env = interp.state().matrices.at("env");
    for (size_t i = 0; i < z.rows(); ++i) {
        const double mag = std::hypot(z(i, 0), z(i, 1));
        EXPECT_NEAR(env(i, 0), mag, 1e-10);
    }

    expect_error(interp, "signal_hilbert()");
    expect_contains(interp, "help", "signal_hilbert");
}

TEST(ReplWave242Pipeline, SignalInstantaneousFreqBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "f = signal_instantaneous_freq(x, 8.0)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freq = interp.state().matrices.at("f");
    EXPECT_EQ(freq.cols(), 1u);
    EXPECT_EQ(freq.rows(), 8u);
    for (size_t i = 0; i < freq.rows(); ++i) {
        EXPECT_TRUE(std::isfinite(freq(i, 0)));
    }

    expect_error(interp, "signal_instantaneous_freq(x)");
    expect_contains(interp, "help", "signal_instantaneous_freq");
}
