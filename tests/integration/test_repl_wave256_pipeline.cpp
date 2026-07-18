// MathScript Integration Tests: REPL Interpreter – Wave 256 Pipeline
//
// Wave 256: 3D geo KD-tree nearest, ray–triangle intersect, point–plane /
// point–segment3 distances.

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

TEST(ReplWave256Pipeline, GeoKdtree3dNearest) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0, 0; 1, 0, 0; 2, 0, 0]");
    expect_ok(interp, "idx = geo_kdtree_3d_nearest(P, 0.9, 0, 0)");
    ASSERT_GT(interp.state().scalars.count("idx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("idx"), 1.0, 1e-12);
    expect_contains(interp, "help", "geo_kdtree_3d_nearest(P,x,y,z)");
}

TEST(ReplWave256Pipeline, GeoIntersectRayTri) {
    Interpreter interp;

    expect_ok(interp, "hit = geo_intersect_ray_tri(0, 0, -1, 0, 0, 1, -1, -1, 0, 1, -1, 0, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = geo_intersect_ray_tri(0, 0, -1, 0, 0, -1, -1, -1, 0, 1, -1, 0, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(
        interp,
        "help",
        "geo_intersect_ray_tri(ox,oy,oz,dx,dy,dz,ax,ay,az,bx,by,bz,cx,cy,cz)");
}

TEST(ReplWave256Pipeline, GeoDistPointPlane) {
    Interpreter interp;

    // Plane z = 0 => n=(0,0,1), d=0; point (0,0,5) distance 5
    expect_ok(interp, "d = geo_dist_point_plane(0, 0, 5, 0, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 5.0, 1e-12);
    expect_contains(interp, "help", "geo_dist_point_plane(px,py,pz,nx,ny,nz,d)");
}

TEST(ReplWave256Pipeline, GeoDistPointSeg3d) {
    Interpreter interp;

    // Point (2,3,0) to segment (0,0,0)-(4,0,0) => distance 3
    expect_ok(interp, "dps = geo_dist_point_seg3d(2, 3, 0, 0, 0, 0, 4, 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("dps"), 3.0, 1e-12);
    expect_contains(interp, "help", "geo_dist_point_seg3d(px,py,pz,x1,y1,z1,x2,y2,z2)");
}
