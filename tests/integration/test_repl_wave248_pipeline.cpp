// MathScript Integration Tests: REPL Interpreter – Wave 248 Pipeline
//
// Wired: signal_upsample, signal_downsample.

#include <gtest/gtest.h>
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

TEST(ReplWave248Pipeline, SignalUpsampleBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2]");
    expect_ok(interp, "u = signal_upsample(x, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    const auto& u = interp.state().matrices.at("u");
    EXPECT_EQ(u.cols(), 1u);
    EXPECT_EQ(u.rows(), 6u);

    const auto expected = ms::upsample({1.0, 2.0}, 3);
    ASSERT_EQ(expected.size(), u.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(u(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "signal_upsample(x, 2)");
    expect_error(interp, "signal_upsample(x)");
    expect_error(interp, "signal_upsample(x, 0)");
    expect_contains(interp, "help", "signal_upsample(x,n)");
}

TEST(ReplWave248Pipeline, SignalDownsampleBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6]");
    expect_ok(interp, "d = signal_downsample(x, 2)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    const auto& d = interp.state().matrices.at("d");
    EXPECT_EQ(d.cols(), 1u);
    EXPECT_EQ(d.rows(), 3u);

    const auto expected = ms::downsample({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2);
    ASSERT_EQ(expected.size(), d.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(d(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "signal_downsample(x, 3)");
    expect_error(interp, "signal_downsample(x)");
    expect_error(interp, "signal_downsample(x, 0)");
    expect_contains(interp, "help", "signal_downsample(x,n)");
}

TEST(ReplWave248Pipeline, UpsampleDownsampleRoundTrip) {
    Interpreter interp;

    expect_ok(interp, "sig = [1; 2; 3; 4]");
    expect_ok(interp, "up = signal_upsample(sig, 2)");
    expect_ok(interp, "back = signal_downsample(up, 2)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    const auto& back = interp.state().matrices.at("back");
    ASSERT_EQ(back.rows(), 4u);
    EXPECT_NEAR(back(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(back(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(back(2, 0), 3.0, 1e-12);
    EXPECT_NEAR(back(3, 0), 4.0, 1e-12);
}
