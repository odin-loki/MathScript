// MathScript Integration Tests: REPL Interpreter – Wave 248 Pipeline
//
// Wired: geo_clip_polygon.

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

TEST(ReplWave248Pipeline, GeoClipPolygonBinding) {
    Interpreter interp;

    // Subject [0,4]x[0,4] clipped by window [2,6]x[2,6] → 2x2 overlap, area 4.
    expect_ok(interp, "A = [0, 0; 4, 0; 4, 4; 0, 4]");
    expect_ok(interp, "B = [2, 2; 6, 2; 6, 6; 2, 6]");
    expect_ok(interp, "C = geo_clip_polygon(A, B)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.cols(), 2u);
    EXPECT_GE(C.rows(), 3u);
    expect_ok(interp, "area = geo_polygon_area(C)");
    ASSERT_GT(interp.state().scalars.count("area"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("area"), 4.0, 1e-9);

    expect_ok(interp, "geo_clip_polygon(A, B)");
    expect_error(interp, "geo_clip_polygon(A)");
    expect_contains(interp, "help", "geo_clip_polygon(A,B)");
}
