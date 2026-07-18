// MathScript Integration Tests: REPL Interpreter – Wave 244 Bindings Pipeline
//
// Wired: tfqmr, lsmr, signal_spectrogram.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/linalg/linalg.hpp"
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

TEST(ReplWave244Pipeline, TfqmrBinding) {
    Interpreter interp;

    // Diagonal SPD system — TFQMR converges reliably here.
    expect_ok(interp, "A = [4, 0, 0; 0, 5, 0; 0, 0, 6]");
    expect_ok(interp, "b = [4; 5; 6]");
    expect_ok(interp, "x = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 1.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("x")(2, 0), 1.0, 1e-5);

    expect_error(interp, "tfqmr(A)");
    expect_contains(interp, "help", "tfqmr");
}

TEST(ReplWave244Pipeline, LsmrBinding) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 0, 0; 0, 5, 0; 0, 0, 6]");
    expect_ok(interp, "b = [4; 5; 6]");
    expect_ok(interp, "x = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(2, 0), 1.0, 1e-6);

    const ms::Matrix<double> A{{4.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}};
    const ms::Matrix<double> b{{4.0}, {5.0}, {6.0}};
    const auto expected = ms::lsmr(A, b);
    ASSERT_TRUE(expected.has_value());
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), (*expected)(0, 0), 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), (*expected)(1, 0), 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(2, 0), (*expected)(2, 0), 1e-6);

    expect_error(interp, "lsmr(A)");
    expect_contains(interp, "help", "lsmr");
}

TEST(ReplWave244Pipeline, SignalSpectrogramBinding) {
    Interpreter interp;

    // 512-sample sinusoid so default segment_len=256 yields multiple time rows.
    std::string x_cmd = "x = [";
    for (int i = 0; i < 512; ++i) {
        if (i > 0) {
            x_cmd += "; ";
        }
        const double t = static_cast<double>(i) / 1000.0;
        x_cmd += std::to_string(std::sin(2.0 * 3.141592653589793 * 30.0 * t));
    }
    x_cmd += "]";
    expect_ok(interp, x_cmd);
    expect_ok(interp, "S = signal_spectrogram(x, 1000)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    const auto& S = interp.state().matrices.at("S");
    EXPECT_GT(S.rows(), 1u);
    EXPECT_GT(S.cols(), 1u);
    for (size_t i = 0; i < S.rows(); ++i) {
        for (size_t j = 0; j < S.cols(); ++j) {
            EXPECT_TRUE(std::isfinite(S(i, j)));
            EXPECT_GE(S(i, j), 0.0);
        }
    }

    expect_error(interp, "signal_spectrogram(x)");
    expect_contains(interp, "help", "signal_spectrogram");
}
