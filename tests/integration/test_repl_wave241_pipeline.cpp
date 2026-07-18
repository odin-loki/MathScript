// MathScript Integration Tests: REPL Interpreter – Wave 241 Signal Bindings
//
// Wired: signal_cheby2, signal_periodogram, signal_welch_psd.

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

TEST(ReplWave241Pipeline, SignalCheby2Binding) {
    Interpreter interp;

    // scipy.signal.cheby2(2, 40.0, 0.25, fs=2.0, btype='low')
    expect_ok(interp, "ba = signal_cheby2(2, 40.0, 0.25, 2.0)");
    ASSERT_GT(interp.state().matrices.count("ba"), 0u);
    const auto& ba = interp.state().matrices.at("ba");
    EXPECT_EQ(ba.rows(), 2u);
    EXPECT_EQ(ba.cols(), 3u);
    EXPECT_NEAR(ba(0, 0), 0.01236943, 1e-6);
    EXPECT_NEAR(ba(0, 1), -0.01209834, 1e-6);
    EXPECT_NEAR(ba(0, 2), 0.01236943, 1e-6);
    EXPECT_NEAR(ba(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ba(1, 1), -1.83553964, 1e-6);
    EXPECT_NEAR(ba(1, 2), 0.84818017, 1e-6);

    expect_ok(interp, "bah = signal_cheby2(2, 40.0, 0.25, 2.0, 1)");
    ASSERT_GT(interp.state().matrices.count("bah"), 0u);
    const auto& bah = interp.state().matrices.at("bah");
    EXPECT_EQ(bah.rows(), 2u);
    EXPECT_EQ(bah.cols(), 3u);
    EXPECT_NEAR(bah(1, 0), 1.0, 1e-12);

    expect_error(interp, "signal_cheby2(2, 40.0, 0.25)");
    expect_contains(interp, "help", "signal_cheby2");
}

TEST(ReplWave241Pipeline, SignalPeriodogramBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "psd = signal_periodogram(x, 8.0)");
    ASSERT_GT(interp.state().matrices.count("psd"), 0u);
    const auto& psd = interp.state().matrices.at("psd");
    EXPECT_EQ(psd.cols(), 2u);
    EXPECT_GT(psd.rows(), 0u);
    EXPECT_NEAR(psd(0, 0), 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(psd(0, 1)));

    expect_error(interp, "signal_periodogram(x)");
    expect_contains(interp, "help", "signal_periodogram");
}

TEST(ReplWave241Pipeline, SignalWelchPsdBinding) {
    Interpreter interp;

    expect_ok(interp,
              "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "w = signal_welch_psd(x, 8.0, 8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.cols(), 2u);
    EXPECT_GT(w.rows(), 0u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(w(0, 1)));

    expect_error(interp, "signal_welch_psd(x, 8.0)");
    expect_contains(interp, "help", "signal_welch_psd");
}
