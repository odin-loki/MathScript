// MathScript Integration Tests: REPL Interpreter – Wave 247 Pipeline
//
// Covers Wave 246 APIs: dist_lsmr, graph_maximum_matching, cuda_allreduce_prod,
// and crypto_aes256_gcm encrypt/decrypt.

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

TEST(ReplWave247Pipeline, DistLsmrSquareSystem) {
    Interpreter interp;

    // Identity square system: dist_lsmr gathers + lsmr.
    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "b = [3; 5]");
    expect_ok(interp, "x = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 2u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(x(1, 0), 5.0, 1e-8);

    expect_error(interp, "dist_lsmr(A)");
    expect_contains(interp, "help", "dist_lsmr");
}

TEST(ReplWave247Pipeline, GraphMaximumMatchingBinding) {
    Interpreter interp;

    // Single edge 0-1: maximum matching is one edge [0, 1].
    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    ASSERT_GT(interp.state().matrices.count("M"), 0u);
    const auto& M = interp.state().matrices.at("M");
    EXPECT_EQ(M.rows(), 1u);
    EXPECT_EQ(M.cols(), 2u);
    EXPECT_NEAR(M(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(M(0, 1), 1.0, 1e-9);

    expect_error(interp, "graph_maximum_matching(A, 0)");
    expect_contains(interp, "help", "graph_maximum_matching");
}

TEST(ReplWave247Pipeline, CudaAllreduceProdStub) {
    Interpreter interp;

    // Stub NCCL: allreduce_prod is identity when MS_HAS_NCCL=0 / comm_size=1.
    expect_ok(interp, "p = cuda_allreduce_prod(2)");
    ASSERT_GT(interp.state().scalars.count("p"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("p"), 2.0, 1e-9);
    expect_contains(interp, "cuda_allreduce_prod(2)", "2");

    expect_contains(interp, "help", "cuda_allreduce_prod");
}

TEST(ReplWave247Pipeline, CryptoAes256GcmRoundTrip) {
    Interpreter interp;

    // NIST SP 800-38D Test Case 14 (single block, empty AAD).
    constexpr const char* key =
        "0000000000000000000000000000000000000000000000000000000000000000";
    constexpr const char* iv = "000000000000000000000000";
    constexpr const char* plain = "00000000000000000000000000000000";

    expect_contains(
        interp,
        std::string("crypto_aes256_gcm_encrypt(\"") + key + "\", \"" + iv +
            "\", \"\", \"" + plain + "\")",
        "cea7403d4d606b6e074ec5d3baf39d18");
    expect_contains(
        interp,
        std::string("crypto_aes256_gcm_encrypt(\"") + key + "\", \"" + iv +
            "\", \"\", \"" + plain + "\")",
        "d0d1c8a799996bf0265b98b5d48ab919");

    expect_contains(
        interp,
        std::string("crypto_aes256_gcm_decrypt(\"") + key + "\", \"" + iv +
            "\", \"\", \"cea7403d4d606b6e074ec5d3baf39d18\", "
            "\"d0d1c8a799996bf0265b98b5d48ab919\")",
        plain);

    expect_contains(interp, "help", "crypto_aes256_gcm_encrypt");
    expect_contains(interp, "help", "crypto_aes256_gcm_decrypt");
}
