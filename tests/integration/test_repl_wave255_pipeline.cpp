// MathScript Integration Tests: REPL Interpreter – Wave 255 Pipeline
//
// Smoke: Wave 254 APIs already on main (mpi_bcast, geo AABB, signal_czt/deconv).
// Deterministic; optional Wave 255 APIs belong in separate tests once merged.

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

} // namespace

TEST(ReplWave255Pipeline, MpiBcastIdentityAssignment) {
    Interpreter interp;

    expect_ok(interp, "bx = mpi_bcast(3.5)");
    ASSERT_GT(interp.state().scalars.count("bx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bx"), 3.5, 1e-12);
    expect_contains(interp, "help", "mpi_bcast(x)");
}

TEST(ReplWave255Pipeline, GeoPointInAabbSmoke) {
    Interpreter interp;

    expect_ok(interp, "inside = geo_point_in_aabb(1, 1, 0, 0, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("inside"), 1.0, 1e-9);
    expect_ok(interp, "outside = geo_point_in_aabb(3, 1, 0, 0, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("outside"), 0.0, 1e-9);
    expect_contains(interp, "help", "geo_point_in_aabb(px,py,minx,miny,maxx,maxy)");
}

TEST(ReplWave255Pipeline, GeoOverlapAabbSmoke) {
    Interpreter interp;

    expect_ok(interp, "hit = geo_overlap_aabb(0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = geo_overlap_aabb(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(
        interp,
        "help",
        "geo_overlap_aabb(aminx,aminy,aminz,amaxx,amaxy,amaxz,bminx,bminy,bminz,bmaxx,bmaxy,bmaxz)");
}

TEST(ReplWave255Pipeline, SignalCztDftContour) {
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
}

TEST(ReplWave255Pipeline, SignalDeconvSmoke) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);
    expect_contains(interp, "help", "signal_deconv(y,b)");
}
