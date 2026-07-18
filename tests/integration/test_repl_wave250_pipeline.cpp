// MathScript Integration Tests: REPL Interpreter – Wave 250 Pipeline
//
// Wired: signal_filtfilt,
// plus Wave 249 smoke (crypto_sha256, cuda_reduce).

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
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

TEST(ReplWave250Pipeline, SignalFiltfiltBinding) {
    Interpreter interp;

    // Simple moving-average FIR: b = [1/3,1/3,1/3], a = [1]
    expect_ok(interp, "b = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]");
    expect_ok(interp, "a = [1]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "y = signal_filtfilt(b, a, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    const auto& y = interp.state().matrices.at("y");
    EXPECT_EQ(y.cols(), 1u);
    EXPECT_EQ(y.rows(), 8u);

    const std::vector<double> b_vec{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    const std::vector<double> a_vec{1.0};
    const std::vector<double> x_vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto expected = ms::filtfilt(b_vec, a_vec, x_vec);
    ASSERT_EQ(expected.size(), y.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y(i, 0), expected[i], 1e-12);
    }

    // Also accept Nx1 coefficient columns (same coercion as signal_convolve)
    expect_ok(interp, "bc = [0.25; 0.25; 0.25; 0.25]");
    expect_ok(interp, "ac = [1]");
    expect_ok(interp, "yc = signal_filtfilt(bc, ac, x)");
    ASSERT_GT(interp.state().matrices.count("yc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("yc").rows(), 8u);

    expect_ok(interp, "signal_filtfilt(b, a, x)");
    expect_error(interp, "signal_filtfilt(b, a)");
    expect_error(interp, "signal_filtfilt(b)");
    expect_contains(interp, "help", "signal_filtfilt(b,a,x)");
}

TEST(ReplWave250Pipeline, SignalFiltfiltViaCheby2) {
    Interpreter interp;

    expect_ok(interp, "ba = signal_cheby2(2, 40.0, 0.25, 2.0)");
    ASSERT_GT(interp.state().matrices.count("ba"), 0u);
    const auto& ba = interp.state().matrices.at("ba");
    ASSERT_EQ(ba.rows(), 2u);
    ASSERT_GE(ba.cols(), 1u);

    // Split cheby2 2×N [b;a] into separate row vectors for filtfilt
    std::ostringstream b_cmd;
    b_cmd << "b = [";
    for (size_t j = 0; j < ba.cols(); ++j) {
        if (j > 0) {
            b_cmd << ", ";
        }
        b_cmd << ba(0, j);
    }
    b_cmd << "]";
    std::ostringstream a_cmd;
    a_cmd << "a = [";
    for (size_t j = 0; j < ba.cols(); ++j) {
        if (j > 0) {
            a_cmd << ", ";
        }
        a_cmd << ba(1, j);
    }
    a_cmd << "]";
    expect_ok(interp, b_cmd.str());
    expect_ok(interp, a_cmd.str());
    expect_ok(interp, "x = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "y = signal_filtfilt(b, a, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("y").cols(), 1u);
}

TEST(ReplWave250Pipeline, Wave249CryptoSha256Smoke) {
    Interpreter interp;

    expect_contains(interp, "crypto_sha256(\"\")",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect_contains(interp, "help", "crypto_sha256(hex_data)");
}

TEST(ReplWave250Pipeline, Wave249CudaReduceSmoke) {
    Interpreter interp;

    expect_ok(interp, "x = 3.5");
    expect_ok(interp, "y = cuda_reduce(x)");
    ASSERT_GT(interp.state().scalars.count("y"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("y"), 3.5, 1e-12);
    expect_contains(interp, "help", "cuda_reduce(x)");
}
