// MathScript Integration Tests: REPL Interpreter – Wave 247 Geo Bindings
//
// Wired: geo_minkowski_sum, geo_min_bounding_rect.

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
