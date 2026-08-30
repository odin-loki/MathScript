#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, graph_geo_combo_numthy) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_pagerank(A)");
    expect_contains(interp, "help", "geo_dist2d(x1,y1,x2,y2)");
    expect_contains(interp, "help", "combo_nchoosek(n,k)");
    expect_contains(interp, "help", "numthy_gcd(a,b)");

    expect_contains(interp, "geo_dist2d(0, 0, 3, 4)", "5");
    expect_contains(interp, "combo_nchoosek(5, 2)", "10");
    expect_contains(interp, "numthy_gcd(48, 18)", "6");

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_contains(interp, "graph_dijkstra_dist(A, 0, 4)", "7");
    expect_ok(interp, "pr = graph_pagerank(A)");
    ASSERT_TRUE(interp.state().matrices.count("pr") > 0);
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "area = geo_convex_hull_area(sq)");
    EXPECT_NEAR(interp.state().scalars.at("area"), 1.0, 1e-9);
    expect_contains(interp, "geo_convex_hull_area(sq)", "1");

    expect_contains(interp, "help", "geo_convex_hull(P)");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    ASSERT_TRUE(interp.state().matrices.count("hull") > 0);
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("hull").cols(), 2u);
    expect_ok(interp, "hull_area = geo_convex_hull_area(hull)");
    EXPECT_NEAR(interp.state().scalars.at("hull_area"), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_polygon_area) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_polygon_area(P)");

    expect_ok(interp, "P = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "a = geo_polygon_area(P)");
    EXPECT_NEAR(interp.state().scalars.at("a"), 1.0, 1e-9);
    expect_contains(interp, "geo_polygon_area(P)", "1");

    expect_ok(interp, "T = [0, 0; 2, 0; 1, 2]");
    expect_ok(interp, "ta = geo_polygon_area(T)");
    EXPECT_NEAR(interp.state().scalars.at("ta"), 2.0, 1e-9);
    expect_contains(interp, "geo_polygon_area([0, 0; 2, 0; 1, 2])", "2");
}

TEST(ReplCommandsTest, geo_dist3d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist3d(x1,y1,z1,x2,y2,z2)");

    expect_ok(interp, "d3 = geo_dist3d(0, 0, 0, 3, 4, 12)");
    EXPECT_NEAR(interp.state().scalars.at("d3"), 13.0, 1e-9);

    expect_contains(interp, "geo_dist3d(0, 0, 0, 3, 4, 12)", "13");
}

TEST(ReplCommandsTest, geo_polygon_perimeter) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_polygon_perimeter(P)");

    expect_ok(interp, "per = geo_polygon_perimeter([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("per"), 16.0, 1e-9);

    expect_contains(interp, "geo_polygon_perimeter([0, 0; 4, 0; 4, 4; 0, 4])", "16");
}

TEST(ReplCommandsTest, geo_triangle_area) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_triangle_area(x1,y1,x2,y2,x3,y3)");

    expect_ok(interp, "tri = geo_triangle_area(0, 0, 4, 0, 0, 3)");
    EXPECT_NEAR(interp.state().scalars.at("tri"), 6.0, 1e-9);

    expect_contains(interp, "geo_triangle_area(0, 0, 4, 0, 0, 3)", "6");
}

TEST(ReplCommandsTest, geo_dist_sq2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_sq2d(x1,y1,x2,y2)");

    expect_ok(interp, "dsq = geo_dist_sq2d(0, 0, 3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("dsq"), 25.0, 1e-9);

    expect_contains(interp, "geo_dist_sq2d(0, 0, 3, 4)", "25");
}

TEST(ReplCommandsTest, geo_vec2d_length) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_vec2d_length(x,y)");

    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-9);

    expect_contains(interp, "geo_vec2d_length(3, 4)", "5");
}

TEST(ReplCommandsTest, geo_cross2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_cross2d(x1,y1,x2,y2)");

    expect_ok(interp, "cr = geo_cross2d(1, 2, 3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("cr"), -2.0, 1e-9);

    expect_contains(interp, "geo_cross2d(1, 2, 3, 4)", "-2");
}

TEST(ReplCommandsTest, geo_centroid_x) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_centroid_x(P)");

    expect_ok(interp, "cx = geo_centroid_x([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("cx"), 2.0, 1e-9);

    expect_contains(interp, "geo_centroid_x([0, 0; 4, 0; 4, 4; 0, 4])", "2");
}

TEST(ReplCommandsTest, geo_centroid_y) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_centroid_y(P)");

    expect_ok(interp, "cy = geo_centroid_y([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("cy"), 2.0, 1e-9);

    expect_contains(interp, "geo_centroid_y([0, 0; 4, 0; 4, 4; 0, 4])", "2");
}

TEST(ReplCommandsTest, geo_dist_point_line2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_point_line2d(px,py,a,b,c)");

    expect_ok(interp, "d = geo_dist_point_line2d(0, 0, 1, 1, -1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), std::sqrt(2.0) / 2.0, 1e-5);

    expect_contains(interp, "geo_dist_point_line2d(0, 0, 1, 1, -1)", "0.707");
}

TEST(ReplCommandsTest, geo_volume_tetrahedron) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_volume_tetrahedron");

    expect_ok(interp, "v = geo_volume_tetrahedron(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("v"), 1.0 / 6.0, 1e-5);

    expect_contains(interp, "geo_volume_tetrahedron(0,0,0,1,0,0,0,1,0,0,0,1)", "0.166667");
}

TEST(ReplCommandsTest, geo_delaunay_2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_delaunay_2d(P)");

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T3").rows(), 1u);

    expect_ok(interp, "Tsq = geo_delaunay_2d([0, 0; 1, 0; 0, 1; 1, 1])");
    ASSERT_GT(interp.state().matrices.count("Tsq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Tsq").rows(), 2u);
}

TEST(ReplCommandsTest, geo_kdtree_nearest) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_kdtree_nearest(P,x,y)");

    expect_ok(interp, "idx = geo_kdtree_nearest([0, 0; 1, 0; 2, 0; 3, 0; 4, 0], 1.1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("idx"), 1.0, 1e-9);

    expect_contains(interp, "geo_kdtree_nearest([0, 0; 1, 0; 2, 0; 3, 0; 4, 0], 1.1, 0)", "1");
}

TEST(ReplCommandsTest, geo_kdtree_query) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_kdtree_knn(P,x,y,k)");
    expect_contains(interp, "help", "geo_kdtree_range(P,x,y,r)");

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("n").cols(), 1u);

    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("r").cols(), 1u);
}

TEST(ReplCommandsTest, geo_voronoi) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_voronoi(P)");

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
}

TEST(ReplCommandsTest, geo_hermite_x) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_hermite_x(");

    expect_ok(interp, "hx = geo_hermite_x(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hx"), 0.5, 1e-9);
}

TEST(ReplCommandsTest, geo_signed_area) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_signed_area(P)");

    expect_ok(interp, "sa = geo_signed_area([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("sa"), 16.0, 1e-9);

    expect_contains(interp, "geo_signed_area([0, 0; 4, 0; 4, 4; 0, 4])", "16");
}

TEST(ReplCommandsTest, geo_moment_of_inertia) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_moment_of_inertia(P)");

    expect_ok(interp, "mi = geo_moment_of_inertia([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("mi"), 128.0 / 3.0, 1e-5);

    expect_contains(interp, "geo_moment_of_inertia([0, 0; 4, 0; 4, 4; 0, 4])", "42.6667");
}

TEST(ReplCommandsTest, geo_dist_point_seg2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_point_seg2d(px,py,x1,y1,x2,y2)");

    expect_ok(interp, "dps = geo_dist_point_seg2d(2, 3, 0, 0, 4, 0)");
    EXPECT_NEAR(interp.state().scalars.at("dps"), 3.0, 1e-9);

    expect_contains(interp, "geo_dist_point_seg2d(2, 3, 0, 0, 4, 0)", "3");
}

TEST(ReplCommandsTest, geo_point_in_polygon) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_point_in_polygon(px,py,P)");

    expect_ok(interp, "pip = geo_point_in_polygon(2, 2, [0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("pip"), 1.0, 1e-9);

    expect_contains(interp, "geo_point_in_polygon(2, 2, [0, 0; 4, 0; 4, 4; 0, 4])", "1");
}

TEST(ReplCommandsTest, geo_overlap_circles) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_overlap_circles(x1,y1,r1,x2,y2,r2)");

    expect_ok(interp, "ov = geo_overlap_circles(0, 0, 1, 0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ov"), 1.0, 1e-9);
    expect_ok(interp, "sep = geo_overlap_circles(0, 0, 1, 3, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sep"), 0.0, 1e-9);

    expect_contains(interp, "geo_overlap_circles(0, 0, 1, 0, 0, 1)", "1");
    expect_contains(interp, "geo_overlap_circles(0, 0, 1, 3, 0, 1)", "0");
}

TEST(ReplCommandsTest, geo_aabb) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_point_in_aabb(px,py,minx,miny,maxx,maxy)");
    expect_contains(interp, "help",
                    "geo_overlap_aabb(aminx,aminy,aminz,amaxx,amaxy,amaxz,bminx,bminy,bminz,bmaxx,bmaxy,bmaxz)");

    expect_ok(interp, "inside = geo_point_in_aabb(1, 1, 0, 0, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("inside"), 1.0, 1e-9);
    expect_ok(interp, "outside = geo_point_in_aabb(3, 1, 0, 0, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("outside"), 0.0, 1e-9);
    expect_contains(interp, "geo_point_in_aabb(1, 1, 0, 0, 2, 2)", "1");

    expect_ok(interp, "hit = geo_overlap_aabb(0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = geo_overlap_aabb(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(interp, "geo_overlap_aabb(0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5)", "1");
}

TEST(ReplCommandsTest, geo_ray_intersect) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_intersect_seg_seg(x1,y1,x2,y2,x3,y3,x4,y4)");
    expect_contains(interp, "help", "geo_intersect_ray_sphere(ox,oy,oz,dx,dy,dz,cx,cy,cz,r)");
    expect_contains(interp, "help",
                    "geo_intersect_ray_aabb(ox,oy,oz,dx,dy,dz,minx,miny,minz,maxx,maxy,maxz)");

    expect_ok(interp, "hit = geo_intersect_seg_seg(0, 0, 2, 2, 0, 2, 2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = geo_intersect_seg_seg(0, 0, 1, 0, 0, 1, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(interp, "geo_intersect_seg_seg(0, 0, 2, 2, 0, 2, 2, 0)", "1");
    expect_contains(interp, "geo_intersect_seg_seg(0, 0, 1, 0, 0, 1, 1, 1)", "0");

    expect_ok(interp, "rs = geo_intersect_ray_sphere(0, 0, 0, 1, 0, 0, 2, 0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rs"), 1.0, 1e-9);

    expect_ok(interp, "ra = geo_intersect_ray_aabb(-5, 0, 0, 1, 0, 0, -1, -1, -1, 1, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ra"), 1.0, 1e-9);
    expect_contains(interp, "geo_intersect_ray_aabb(-5, 0, 0, 1, 0, 0, -1, -1, -1, 1, 1, 1)", "1");
}

TEST(ReplCommandsTest, geo_triangulate_hull3d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_triangulate_polygon(P)");
    expect_contains(interp, "help", "geo_convex_hull_3d(P)");

    expect_ok(interp, "P = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "T = geo_triangulate_polygon(P)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
}

TEST(ReplCommandsTest, geo_bezier_eval_x) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_eval_x(P,t)");

    expect_ok(interp, "bx = geo_bezier_eval_x([0, 0; 1, 2; 2, 0], 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bx"), 1.0, 1e-9);

    expect_contains(interp, "geo_bezier_eval_x([0, 0; 1, 2; 2, 0], 0.5)", "1");
}

TEST(ReplCommandsTest, geo_bezier_eval_y) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_eval_y(P,t)");

    expect_ok(interp, "by = geo_bezier_eval_y([0, 0; 1, 2; 2, 0], 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by"), 1.0, 1e-9);

    expect_contains(interp, "geo_bezier_eval_y([0, 0; 1, 2; 2, 0], 0.5)", "1");
}

TEST(ReplCommandsTest, geo_bezier_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_eval(P,t)");

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_bezier_deriv) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_deriv(P,t)");

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("d").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_hermite_curve(p0x,p0y,m0x,m0y,p1x,p1y,m1x,m1y,t)");

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 0.25, 1e-9);
}

TEST(ReplCommandsTest, geo_catmull_rom) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_catmull_rom(P,t)");

    expect_ok(interp, "ctrl = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "pt = geo_catmull_rom(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.5, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, geo_bspline_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bspline_eval(P,knots,degree,t)");

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "pt = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hull_bezier_kdtree3d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_upper_hull(P)");
    expect_contains(interp, "help", "geo_lower_hull(P)");
    expect_contains(interp, "help", "geo_bezier_subdivide(P,t)");
    expect_contains(interp, "help", "geo_kdtree_3d_knn(P,x,y,z,k)");
    expect_contains(interp, "help", "geo_kdtree_3d_range(P,x,y,z,r)");

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("uh").cols(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("lh").cols(), 2u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("sub").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("sub")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sub")(5, 0), 2.0, 1e-9);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("n").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("n")(0, 0), 1.0, 1e-9);

    expect_ok(interp, "r = geo_kdtree_3d_range(P, 1.0, 0, 0, 1.5)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_GE(interp.state().matrices.at("r").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("r").cols(), 1u);
}

TEST(ReplCommandsTest, geo) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);

    expect_ok(interp, "r = geo_kdtree_3d_range(P, 1.0, 0, 0, 1.5)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_GE(interp.state().matrices.at("r").rows(), 2u);
}

TEST(ReplCommandsTest, geo_intersect_ray_tri) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_intersect_ray_tri(ox,oy,oz,dx,dy,dz,ax,ay,az,bx,by,bz,cx,cy,cz)");

    expect_ok(interp, "hit = geo_intersect_ray_tri(0, 0, -1, 0, 0, 1, 1, 0, 0, -1, 1, 0, -1, -1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = geo_intersect_ray_tri(0, 0, -1, 0, 0, -1, 1, 0, 0, -1, 1, 0, -1, -1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(interp, "geo_intersect_ray_tri(0, 0, -1, 0, 0, 1, 1, 0, 0, -1, 1, 0, -1, -1, 0)", "1");

    expect_error_contains(interp, "geo_intersect_ray_tri(0, 0, 0)", "geo_intersect_ray_tri");
}

TEST(ReplCommandsTest, geo_dist_point_plane) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_point_plane(px,py,pz,nx,ny,nz,d)");

    expect_ok(interp, "d = geo_dist_point_plane(0, 0, 1, 0, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 1.0, 1e-9);
    expect_ok(interp, "on = geo_dist_point_plane(0, 0, 0, 0, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("on"), 0.0, 1e-9);
    expect_contains(interp, "geo_dist_point_plane(0, 0, 1, 0, 0, 1, 0)", "1");

    expect_error_contains(interp, "geo_dist_point_plane(0, 0, 1, 0, 0, 1, not_a_number)",
                         "geo_dist_point_plane");
}

TEST(ReplCommandsTest, geo_dist_point_seg3d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_point_seg3d(px,py,pz,x1,y1,z1,x2,y2,z2)");

    expect_ok(interp, "d = geo_dist_point_seg3d(0, 0, 0, 1, 0, 0, 2, 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 1.0, 1e-9);
    expect_ok(interp, "on = geo_dist_point_seg3d(1.5, 0, 0, 1, 0, 0, 2, 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("on"), 0.0, 1e-9);
    expect_contains(interp, "geo_dist_point_seg3d(0, 0, 0, 1, 0, 0, 2, 0, 0)", "1");

    expect_error_contains(interp, "geo_dist_point_seg3d(0, 0, 0, 1, 0, 0, 2, 0, missing)",
                         "geo_dist_point_seg3d");
}

TEST(ReplCommandsTest, geo_kdtree_3d_nearest) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_kdtree_3d_nearest(P,x,y,z)");

    expect_ok(interp, "P = [0, 0, 0; 1, 0, 0; 2, 0, 0]");
    expect_ok(interp, "idx = geo_kdtree_3d_nearest(P, 0.9, 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("idx"), 1.0, 1e-9);
    expect_contains(interp, "geo_kdtree_3d_nearest(P, 0.9, 0, 0)", "1");

    expect_ok(interp, "P2 = [0, 0; 1, 0]");
    expect_error_contains(interp, "bad = geo_kdtree_3d_nearest(P2, 0, 0, 0)", "Nx3");
    expect_error_contains(interp, "bad = geo_kdtree_3d_nearest(missing, 0, 0, 0)",
                         "unknown matrix");
}

TEST(ReplCommandsTest, geo_poly_boolean) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_poly_union(A,B)");
    expect_contains(interp, "help", "geo_poly_intersect(A,B)");
    expect_contains(interp, "help", "geo_poly_diff(A,B)");

    expect_ok(interp, "A = [0, 0; 2, 0; 2, 2; 0, 2]");
    expect_ok(interp, "B = [1, 1; 3, 1; 3, 3; 1, 3]");
    expect_ok(interp, "U = geo_poly_union(A, B)");
    ASSERT_GT(interp.state().matrices.count("U"), 0u);
    EXPECT_EQ(interp.state().matrices.at("U").cols(), 2u);
    EXPECT_GE(interp.state().matrices.at("U").rows(), 4u);

    expect_ok(interp, "I = geo_poly_intersect(A, B)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_EQ(interp.state().matrices.at("I").cols(), 2u);
    EXPECT_GE(interp.state().matrices.at("I").rows(), 3u);

    expect_ok(interp, "D = geo_poly_diff(A, B)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 2u);
    EXPECT_GE(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "bad = [0, 0, 0; 1, 0, 0]");
    expect_error_contains(interp, "Ubad = geo_poly_union(bad, A)", "Nx2");
    expect_error_contains(interp, "Ibad = geo_poly_intersect(missing, A)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_minkowski_sum) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_minkowski_sum(A,B)");

    expect_ok(interp, "A = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "B = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "S = geo_minkowski_sum(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 2u);
    EXPECT_GE(interp.state().matrices.at("S").rows(), 4u);

    expect_ok(interp, "bad = [0; 1; 2]");
    expect_error_contains(interp, "geo_minkowski_sum(bad, A)", "Nx2");
}

TEST(ReplCommandsTest, geo_clip_polygon) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_clip_polygon(A,B)");

    expect_ok(interp, "subj = [0, 0; 2, 0; 2, 2; 0, 2]");
    expect_ok(interp, "win = [0.5, 0.5; 1.5, 0.5; 1.5, 1.5; 0.5, 1.5]");
    expect_ok(interp, "C = geo_clip_polygon(subj, win)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 2u);
    EXPECT_GE(interp.state().matrices.at("C").rows(), 3u);

    expect_ok(interp, "bad = [0, 0, 0; 1, 0, 0]");
    expect_error_contains(interp, "Cbad = geo_clip_polygon(bad, win)", "Nx2");
    expect_error_contains(interp, "Cmiss = geo_clip_polygon(missing, win)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, delaunay_hull) {
    Interpreter interp;

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);
}

TEST(ReplCommandsTest, voronoi_minrect) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_gauss) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
}

TEST(ReplCommandsTest, schur_bezier) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, hermite_bspline) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, hulls) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
}

TEST(ReplCommandsTest, bezier_kdtree) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_2) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_gauss_2) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
}

TEST(ReplCommandsTest, schur_bezier_2) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, hermite_bspline_2) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, hulls_2) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
}

TEST(ReplCommandsTest, bezier_kdtree_2) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_3) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_4) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_2) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_2) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_2) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_2) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_2) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_2) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_5) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_3) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_3) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_3) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_3) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_3) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_3) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_6) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_4) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_4) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_4) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_4) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_4) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_4) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_7) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_5) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_5) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_5) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_5) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_5) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_5) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_8) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_6) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_6) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_6) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_6) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_6) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_6) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_9) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_7) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_7) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_7) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_7) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_7) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_7) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_10) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_8) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_8) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_8) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_8) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_8) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_8) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_11) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_9) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_9) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_9) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_9) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_9) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_9) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_12) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_10) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_10) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_10) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_10) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_10) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_10) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_13) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_11) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_11) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_11) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_11) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_11) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_11) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_14) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_12) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_12) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_12) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_12) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_12) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_12) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_15) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_13) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_13) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_13) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_13) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_13) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_13) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_16) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_14) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_14) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_14) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_14) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_14) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_14) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_17) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_15) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_15) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_15) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_15) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_15) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_15) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_18) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_16) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_16) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_16) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_16) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_16) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_16) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_19) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_17) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_17) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_17) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_17) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_17) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_17) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_20) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_18) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_18) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_18) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_18) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_18) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_18) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_21) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_19) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_19) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_19) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_19) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_19) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_19) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_22) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_20) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_20) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_20) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_20) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_20) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_20) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_23) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_21) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_21) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_21) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_21) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_21) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_21) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_24) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_22) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_22) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_22) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_22) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_22) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_22) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_25) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_23) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_23) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_23) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_23) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_23) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_23) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_26) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_24) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_24) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_24) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_24) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_24) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_24) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_27) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_25) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_25) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_25) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_25) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_25) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_25) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_28) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_26) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_26) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_26) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_26) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_26) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_26) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_29) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_27) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_27) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_27) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_27) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_27) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_27) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_30) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_28) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_28) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_28) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_28) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_28) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_28) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_31) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_29) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_29) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_29) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_29) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_29) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_29) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_32) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_30) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_30) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_30) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_30) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_upper_hull_geo_lower_hull_30) {
    Interpreter interp;

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "lh = geo_lower_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("lh"), 0u);
    EXPECT_GE(interp.state().matrices.at("lh").rows(), 2u);
}

TEST(ReplCommandsTest, geo_bezier_subdivide_geo_kdtree_3d_knn_30) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("sub"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, geo_vec2d_length_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}

TEST(ReplCommandsTest, voronoi_minrect_33) {
    Interpreter interp;

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(ReplCommandsTest, kdtree_knn_range_31) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(ReplCommandsTest, triangulate_hull3d_31) {
    Interpreter interp;

    expect_ok(interp, "T = geo_triangulate_polygon([0,0; 1,0; 1,1; 0,1])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
}

TEST(ReplCommandsTest, schur_geo_bezier_eval_geo_bezier_deriv_31) {
    Interpreter interp;

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, geo_hermite_curve_geo_bspline_eval_31) {
    Interpreter interp;

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, geo_kdtree_nearest_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "geo_kdtree_nearest([0, 0; 1, 0; 2, 0; 3, 0; 4, 0], 1.1, 0)", "1");
    expect_error_contains(interp, "geo_kdtree_nearest([0; 1; 2], 0, 0)", "Nx2");
}

TEST(ReplCommandsTest, geo_bezier_eval_x_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "geo_bezier_eval_x([0, 0; 1, 2; 2, 0], 0.5)", "1");
    expect_error_contains(interp, "geo_bezier_eval_x([0, 0; 1, 1], 0.5)",
                          "at least 3 control points");
}

TEST(ReplCommandsTest, geo_kdtree_3d_nearest_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "P = [0, 0, 0; 1, 0, 0; 0, 1, 0]");
    expect_contains(interp, "geo_kdtree_3d_nearest(P, 0.9, 0, 0)", "1");
    expect_ok(interp, "P2 = [0, 0; 1, 0]");
    expect_error_contains(interp, "geo_kdtree_3d_nearest(P2, 0, 0, 0)", "Nx3");
}

TEST(ReplCommandsTest, geo_delaunay_2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])", "triangles");
    expect_error_contains(interp, "geo_delaunay_2d([0, 0; 1, 0])", "at least 3 points");
}

TEST(ReplCommandsTest, geo_voronoi_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])", "voronoi");
    expect_error_contains(interp, "geo_voronoi([0, 0; 1, 0])", "at least 3 points");
}

TEST(ReplCommandsTest, geo_convex_hull_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_convex_hull([0, 0; 1, 0; 0, 1])", "hull");
    expect_error_contains(interp, "geo_convex_hull([0, 0; 1, 0])", "at least 3 points");
}

TEST(ReplCommandsTest, geo_triangulate_polygon_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_triangulate_polygon([0, 0; 1, 0; 1, 1; 0, 1])", "triangles");
    expect_error_contains(interp, "geo_triangulate_polygon([0, 0; 1, 0])", "at least 3 points");
}

TEST(ReplCommandsTest, geo_convex_hull_3d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_convex_hull_3d([1, 1, 1; 1, -1, -1; -1, 1, -1; -1, -1, 1])",
                    "faces");
    expect_error_contains(interp, "geo_convex_hull_3d([1, 1, 1; 0, 0, 0; 2, 2, 2])",
                          "at least 4 points");
}

TEST(ReplCommandsTest, geo_min_bounding_rect_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])", "rect");
    expect_error_contains(interp, "geo_min_bounding_rect(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_polygon_area_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_polygon_area([0, 0; 1, 0; 1, 1; 0, 1])", "1");
    expect_error_contains(interp, "geo_polygon_area(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_polygon_perimeter_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_polygon_perimeter([0, 0; 4, 0; 4, 4; 0, 4])", "16");
    expect_error_contains(interp, "geo_polygon_perimeter(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_signed_area_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_signed_area([0, 0; 4, 0; 4, 4; 0, 4])", "16");
    expect_error_contains(interp, "geo_signed_area(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_centroid_x_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_centroid_x([0, 0; 4, 0; 4, 4; 0, 4])", "2");
    expect_error_contains(interp, "geo_centroid_x(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_centroid_y_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_centroid_y([0, 0; 4, 0; 4, 4; 0, 4])", "2");
    expect_error_contains(interp, "geo_centroid_y(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_convex_hull_area_noassign) {
    Interpreter interp;
    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_contains(interp, "geo_convex_hull_area(sq)", "1");
    expect_error_contains(interp, "geo_convex_hull_area([0, 0; 1, 0])", "at least 3 points");
}

TEST(ReplCommandsTest, geo_moment_of_inertia_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_moment_of_inertia([0, 0; 4, 0; 4, 4; 0, 4])", "42.6667");
    expect_error_contains(interp, "geo_moment_of_inertia(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_bezier_eval_x_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_bezier_eval_x([0, 0; 1, 2; 2, 0], 0.5)", "1");
    expect_error_contains(interp, "geo_bezier_eval_x([0, 0; 1, 1], 0.5)",
                          "at least 3 control points");
}

TEST(ReplCommandsTest, geo_bezier_eval_y_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_bezier_eval_y([0, 0; 1, 2; 2, 0], 0.5)", "1");
    expect_error_contains(interp, "geo_bezier_eval_y([0, 0; 1, 1], 0.5)",
                          "at least 3 control points");
}

TEST(ReplCommandsTest, geo_intersect_seg_seg_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_intersect_seg_seg(0, 0, 2, 2, 0, 2, 2, 0)", "1");
    expect_contains(interp, "geo_intersect_seg_seg(0, 0, 1, 0, 0, 1, 1, 1)", "0");
    expect_error_contains(interp, "geo_intersect_seg_seg(0, 0, 2, 2, 0, 2, 2, missing)",
                         "geo_intersect_seg_seg");
}

TEST(ReplCommandsTest, geo_dist_point_seg3d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_dist_point_seg3d(0, 0, 0, 1, 0, 0, 2, 0, 0)", "1");
    expect_error_contains(interp, "geo_dist_point_seg3d(0, 0, 0, 1, 0, 0, 2, 0, missing)",
                         "geo_dist_point_seg3d");
}

TEST(ReplCommandsTest, geo_dist_point_plane_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_dist_point_plane(0, 0, 1, 0, 0, 1, 0)", "1");
    expect_error_contains(interp, "geo_dist_point_plane(0, 0, 1, 0, 0, 1, not_a_number)",
                         "geo_dist_point_plane");
}

TEST(ReplCommandsTest, geo_dist_point_line2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_dist_point_line2d(0, 0, 1, 1, -1)", "0.707");
    expect_error_contains(interp, "geo_dist_point_line2d(0, 0, 1, 1, missing)",
                         "geo_dist_point_line2d");
}

TEST(ReplCommandsTest, geo_dist3d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_dist3d(0, 0, 0, 3, 4, 12)", "13");
    expect_error_contains(interp, "geo_dist3d(0, 0, 0, 3, 4, missing)", "geo_dist3d");
}

TEST(ReplCommandsTest, geo_triangle_area_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_triangle_area(0, 0, 4, 0, 0, 3)", "6");
    expect_error_contains(interp, "geo_triangle_area(0, 0, 4, 0, 0, missing)",
                         "geo_triangle_area");
}

TEST(ReplCommandsTest, geo_dist_point_seg2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_dist_point_seg2d(2, 3, 0, 0, 4, 0)", "3");
    expect_error_contains(interp, "geo_dist_point_seg2d(2, 3, 0, 0, 4, missing)",
                         "geo_dist_point_seg2d");
}

TEST(ReplCommandsTest, geo_overlap_circles_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_overlap_circles(0, 0, 1, 0, 0, 1)", "1");
    expect_contains(interp, "geo_overlap_circles(0, 0, 1, 3, 0, 1)", "0");
    expect_error_contains(interp, "geo_overlap_circles(0, 0, 1, 3, 0, missing)",
                         "geo_overlap_circles");
}

TEST(ReplCommandsTest, geo_point_in_aabb_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_point_in_aabb(1, 1, 0, 0, 2, 2)", "1");
    expect_error_contains(interp, "geo_point_in_aabb(1, 1, 0, 0, 2, missing)",
                         "geo_point_in_aabb");
}

TEST(ReplCommandsTest, geo_poly_union_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [0, 0; 2, 0; 2, 2; 0, 2]");
    expect_ok(interp, "B = [1, 1; 3, 1; 3, 3; 1, 3]");
    expect_contains(interp, "geo_poly_union(A, B)", "poly =");
    expect_ok(interp, "bad = [0, 0, 0; 1, 0, 0]");
    expect_error_contains(interp, "geo_poly_union(bad, A)", "Nx2");
}

TEST(ReplCommandsTest, geo_poly_intersect_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [0, 0; 2, 0; 2, 2; 0, 2]");
    expect_ok(interp, "B = [1, 1; 3, 1; 3, 3; 1, 3]");
    expect_contains(interp, "geo_poly_intersect(A, B)", "poly =");
    expect_error_contains(interp, "geo_poly_intersect(missing, A)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_poly_diff_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [0, 0; 2, 0; 2, 2; 0, 2]");
    expect_ok(interp, "B = [1, 1; 3, 1; 3, 3; 1, 3]");
    expect_contains(interp, "geo_poly_diff(A, B)", "poly =");
    expect_error_contains(interp, "geo_poly_diff(missing, A)", "unknown matrix");
}

TEST(ReplCommandsTest, geo_minkowski_sum_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "B = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_contains(interp, "geo_minkowski_sum(A, B)", "sum =");
    expect_ok(interp, "bad = [0; 1; 2]");
    expect_error_contains(interp, "geo_minkowski_sum(bad, A)", "Nx2");
}

TEST(ReplCommandsTest, geo_clip_polygon_noassign) {
    Interpreter interp;
    expect_ok(interp, "subj = [0, 0; 2, 0; 2, 2; 0, 2]");
    expect_ok(interp, "win = [0.5, 0.5; 1.5, 0.5; 1.5, 1.5; 0.5, 1.5]");
    expect_contains(interp, "geo_clip_polygon(subj, win)", "clipped =");
    expect_ok(interp, "bad = [0, 0, 0; 1, 0, 0]");
    expect_error_contains(interp, "geo_clip_polygon(bad, win)", "Nx2");
}

TEST(ReplCommandsTest, geo_dist2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_dist2d(0, 0, 3, 4)", "5");
    expect_error_contains(interp, "geo_dist2d(0, 0, 3, missing)", "geo_dist2d");
}

TEST(ReplCommandsTest, geo_dist_sq2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_dist_sq2d(0, 0, 3, 4)", "25");
    expect_error_contains(interp, "geo_dist_sq2d(0, 0, 3, missing)",
                          "expected geo_dist_sq2d(x1,y1,x2,y2)");
}

TEST(ReplCommandsTest, geo_cross2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "geo_cross2d(1, 0, 0, 1)", "1");
    expect_error_contains(interp, "geo_cross2d(1, 0, 0, missing)",
                          "expected geo_cross2d(x1,y1,x2,y2)");
}
