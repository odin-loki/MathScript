// MathScript Integration Tests: REPL Interpreter – Wave 254 Pipeline
//
// Wired: signal_czt / signal_czt_zoom (Chirp Z-Transform via ms::czt / czt_zoom_fft),
// plus Wave 253 smoke APIs already on main:
//   signal_conv2 OR signal_median_filter, crypto_aes128_decrypt_block, cuda_allgather.
// Deterministic so this pipeline can land before other Wave 254 merges.

#include <gtest/gtest.h>
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

TEST(ReplWave254Pipeline, SignalCztDftContour) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "Z = signal_czt(x, 4, 0, -1, 1, 0)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    const auto& Z = interp.state().matrices.at("Z");
    EXPECT_EQ(Z.rows(), 4u);
    EXPECT_EQ(Z.cols(), 2u);
    EXPECT_NEAR(Z(0, 0), 10.0, 1e-9);
    EXPECT_NEAR(Z(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(Z(2, 0), -2.0, 1e-9);
    EXPECT_NEAR(Z(2, 1), 0.0, 1e-9);

    expect_contains(interp, "help", "signal_czt(x,m,w_re,w_im,a_re,a_im)");
    expect_error(interp, "signal_czt()");
}

TEST(ReplWave254Pipeline, SignalCztZoomSmoke) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "zoom = signal_czt_zoom(x, 0.5, 1.5, 16, 8.0)");
    ASSERT_GT(interp.state().matrices.count("zoom"), 0u);
    const auto& zoom = interp.state().matrices.at("zoom");
    EXPECT_EQ(zoom.rows(), 16u);
    EXPECT_EQ(zoom.cols(), 2u);
    EXPECT_TRUE(std::isfinite(zoom(0, 0)));
    EXPECT_TRUE(std::isfinite(zoom(0, 1)));
    expect_contains(interp, "help", "signal_czt_zoom(x,f_start,f_stop,m,fs)");
}

TEST(ReplWave254Pipeline, Wave253SignalConv2OrMedianSmoke) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 1.0, 1e-12);

    expect_ok(interp, "x = [5; 1; 3; 2; 4]");
    expect_ok(interp, "mf = signal_median_filter(x, 3)");
    ASSERT_GT(interp.state().matrices.count("mf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mf").rows(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("mf")(1, 0), 3.0, 1e-12);
}

TEST(ReplWave254Pipeline, Wave253CryptoAes128DecryptSmoke) {
    Interpreter interp;

    expect_contains(
        interp,
        R"cmd(crypto_aes128_decrypt_block("2b7e151628aed2a6abf7158809cf4f3c", "3925841d02dc09fbdc118597196a0b32"))cmd",
        "3243f6a8885a308d313198a2e0370734");
    expect_contains(interp, "help", "crypto_aes128_decrypt_block");
}

TEST(ReplWave254Pipeline, Wave253CudaAllgatherAssignmentSmoke) {
    Interpreter interp;

    expect_ok(interp, "y = cuda_allgather(3.5)");
    ASSERT_GT(interp.state().scalars.count("y"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("y"), 3.5, 1e-12);
    expect_contains(interp, "help", "cuda_allgather(x)");
}
