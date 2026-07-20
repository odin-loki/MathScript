// MathScript Integration Tests: REPL Interpreter – Wave 257 Pipeline
//
// Smoke: Wave 256 APIs already on main (stats regression/bootstrap, graph
// modularity/eccentricity, geo 3D nearest/plane distance, imflip).
// Deterministic; Wave 257 features belong in separate tests once merged.

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

} // namespace

TEST(ReplWave257Pipeline, StatsLinearRegression) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    ASSERT_GT(interp.state().matrices.count("lr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lr").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("lr").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 2), 1.0, 1e-9);
    expect_contains(interp, "help", "stats_linear_regression(x,y)");
}

TEST(ReplWave257Pipeline, StatsBootstrapCi) {
    Interpreter interp;

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    ASSERT_GT(interp.state().matrices.count("ci"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ci").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("ci")(0, 0), 5.5, 1e-9);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("ci")(0, 1)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("ci")(0, 2)));
    EXPECT_GT(interp.state().matrices.at("ci")(0, 3), 0.0);
    expect_contains(interp, "help", "stats_bootstrap_ci(x)");
}

TEST(ReplWave257Pipeline, GraphModularity) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0; 1,0,1; 0,1,0]");
    expect_ok(interp, "C = [0, 1, -1; 2, -1, -1]");
    expect_ok(interp, "Q = graph_modularity(A, C)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("Q")));
    expect_contains(interp, "help", "graph_modularity(A,C)");
}

TEST(ReplWave257Pipeline, GraphEccentricity) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    ASSERT_GT(interp.state().matrices.count("ecc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("ecc").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("ecc")(0, 0), 3.0, 1e-9);
    expect_contains(interp, "help", "graph_eccentricity(A)");
}

TEST(ReplWave257Pipeline, GeoKdtree3dNearest) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0, 0; 1, 0, 0; 2, 0, 0]");
    expect_ok(interp, "idx = geo_kdtree_3d_nearest(P, 0.9, 0, 0)");
    ASSERT_GT(interp.state().scalars.count("idx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("idx"), 1.0, 1e-12);
    expect_contains(interp, "help", "geo_kdtree_3d_nearest(P,x,y,z)");
}

TEST(ReplWave257Pipeline, GeoDistPointPlane) {
    Interpreter interp;

    expect_ok(interp, "d = geo_dist_point_plane(0, 0, 5, 0, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 5.0, 1e-12);
    expect_contains(interp, "help", "geo_dist_point_plane(px,py,pz,nx,ny,nz,d)");
}

TEST(ReplWave257Pipeline, ImflipHorizontal) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("F").cols(), 2u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);
    expect_contains(interp, "help", "imflip(M,horizontal)");
}
