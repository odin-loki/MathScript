// MathScript Integration Tests: REPL Interpreter – Wave 247 Pipeline
//
// Wired: geo_minkowski_sum, geo_min_bounding_rect, signal_unwrap,
// dist_lsmr, graph_maximum_matching, cuda_allreduce_prod, crypto_aes256_gcm.

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

TEST(ReplWave247Pipeline, GeoMinkowskiSumBinding) {
    Interpreter interp;

    // Unit square + unit square → 2x2 square (area 4).
    expect_ok(interp, "A = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "B = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "S = geo_minkowski_sum(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    const auto& S = interp.state().matrices.at("S");
    EXPECT_EQ(S.cols(), 2u);
    EXPECT_EQ(S.rows(), 4u);
    expect_ok(interp, "area = geo_polygon_area(S)");
    ASSERT_GT(interp.state().scalars.count("area"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("area"), 4.0, 1e-9);

    expect_ok(interp, "geo_minkowski_sum(A, B)");
    expect_error(interp, "geo_minkowski_sum(A)");
    expect_contains(interp, "help", "geo_minkowski_sum(A,B)");
}

TEST(ReplWave247Pipeline, GeoMinBoundingRectBinding) {
    Interpreter interp;

    // Axis-aligned unit square → center (0.5,0.5), width/height 1, angle ~0 (mod pi/2).
    expect_ok(interp, "P = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "R = geo_min_bounding_rect(P)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    const auto& R = interp.state().matrices.at("R");
    ASSERT_EQ(R.rows(), 5u);
    ASSERT_EQ(R.cols(), 1u);
    EXPECT_NEAR(R(0, 0), 0.5, 1e-9);
    EXPECT_NEAR(R(1, 0), 0.5, 1e-9);
    // width/height may swap with angle offset by pi/2
    const double w = R(2, 0);
    const double h = R(3, 0);
    EXPECT_NEAR(w * h, 1.0, 1e-9);
    EXPECT_TRUE((std::abs(w - 1.0) < 1e-6 && std::abs(h - 1.0) < 1e-6) ||
                (std::abs(w - 1.0) < 1e-6 || std::abs(h - 1.0) < 1e-6));

    expect_ok(interp, "geo_min_bounding_rect(P)");
    expect_error(interp, "geo_min_bounding_rect(P, 0)");
    expect_contains(interp, "help", "geo_min_bounding_rect(P)");
    expect_contains(interp, "help", "[cx;cy;width;height;angle_rad]");
}

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
    expect_ok(interp, "q = cuda_allreduce_avg(2)");
    ASSERT_GT(interp.state().scalars.count("q"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("q"), 2.0, 1e-9);

    expect_contains(interp, "help", "cuda_allreduce_prod");
    expect_contains(interp, "help", "cuda_allreduce_avg");
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
