// MathScript Integration Tests: REPL Interpreter – Wave 249 Pipeline
//
// Wired: signal_resample, signal_decimate, signal_interpolate,
// plus Wave 248 smoke (crypto_constant_time_eq, cuda_broadcast, geo_clip_polygon).

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

TEST(ReplWave249Pipeline, SignalDecimateBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6]");
    expect_ok(interp, "d = signal_decimate(x, 2)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    const auto& d = interp.state().matrices.at("d");
    EXPECT_EQ(d.cols(), 1u);
    EXPECT_EQ(d.rows(), 3u);

    const auto expected = ms::decimate({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2);
    ASSERT_EQ(expected.size(), d.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(d(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "signal_decimate(x, 3)");
    expect_error(interp, "signal_decimate(x)");
    expect_error(interp, "signal_decimate(x, 0)");
    expect_contains(interp, "help", "signal_decimate(x,q)");
}

TEST(ReplWave249Pipeline, SignalInterpolateBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2]");
    expect_ok(interp, "u = signal_interpolate(x, 2)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    const auto& u = interp.state().matrices.at("u");
    EXPECT_EQ(u.cols(), 1u);
    EXPECT_EQ(u.rows(), 4u);

    const auto expected = ms::interpolate({1.0, 2.0}, 2);
    ASSERT_EQ(expected.size(), u.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(u(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "signal_interpolate(x, 3)");
    expect_error(interp, "signal_interpolate(x)");
    expect_error(interp, "signal_interpolate(x, 0)");
    expect_contains(interp, "help", "signal_interpolate(x,p)");
}

TEST(ReplWave249Pipeline, SignalResampleBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "r = signal_resample(x, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    const auto& r = interp.state().matrices.at("r");
    EXPECT_EQ(r.cols(), 1u);
    // interpolate by 2 then decimate by 2 → same length as input
    EXPECT_EQ(r.rows(), 4u);

    const auto expected = ms::resample({1.0, 2.0, 3.0, 4.0}, 2, 2);
    ASSERT_EQ(expected.size(), r.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(r(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "r32 = signal_resample(x, 3, 2)");
    ASSERT_GT(interp.state().matrices.count("r32"), 0u);
    // length ≈ ceil(N*p/q) after interpolate(N*p) then decimate(/q)
    EXPECT_EQ(interp.state().matrices.at("r32").rows(),
              ms::resample({1.0, 2.0, 3.0, 4.0}, 3, 2).size());

    expect_ok(interp, "signal_resample(x, 2, 1)");
    expect_error(interp, "signal_resample(x)");
    expect_error(interp, "signal_resample(x, 2)");
    expect_error(interp, "signal_resample(x, 0, 1)");
    expect_contains(interp, "help", "signal_resample(x,p,q)");
}

TEST(ReplWave249Pipeline, Wave248CryptoConstantTimeEqSmoke) {
    Interpreter interp;

    expect_contains(interp, "crypto_constant_time_eq(\"00ff\", \"00ff\")", "1");
    expect_contains(interp, "crypto_constant_time_eq(\"00ff\", \"00fe\")", "0");
    expect_contains(interp, "help", "crypto_constant_time_eq(hex_a,hex_b)");
}

TEST(ReplWave249Pipeline, Wave248CudaBroadcastSmoke) {
    Interpreter interp;

    expect_ok(interp, "x = 3.5");
    expect_ok(interp, "y = cuda_broadcast(x)");
    ASSERT_GT(interp.state().scalars.count("y"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("y"), 3.5, 1e-12);
    expect_contains(interp, "help", "cuda_broadcast(x)");
}

TEST(ReplWave249Pipeline, Wave248GeoClipPolygonSmoke) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 0; 4, 0; 4, 4; 0, 4]");
    expect_ok(interp, "B = [2, 2; 6, 2; 6, 6; 2, 6]");
    expect_ok(interp, "C = geo_clip_polygon(A, B)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 2u);
    EXPECT_GE(interp.state().matrices.at("C").rows(), 3u);
}
