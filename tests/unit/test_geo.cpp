#define _USE_MATH_DEFINES
#include "ms/geo/geo.hpp"
#include <cmath>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::geo;

// ---- Vec ops ----

TEST(GeoVec, Basic2D) {
    Vec2D a{1,2}, b{3,4};
    auto s = a + b;
    EXPECT_DOUBLE_EQ(s.x, 4.0);
    EXPECT_DOUBLE_EQ(s.y, 6.0);
    EXPECT_NEAR(length(a), std::sqrt(5.0), 1e-12);
    EXPECT_NEAR(cross2d(a,b), -2.0, 1e-12);  // 1*4 - 2*3
}

TEST(GeoVec, Basic3D) {
    Vec3D a{1,0,0}, b{0,1,0};
    auto c = cross(a, b);
    EXPECT_NEAR(c.x, 0.0, 1e-12);
    EXPECT_NEAR(c.y, 0.0, 1e-12);
    EXPECT_NEAR(c.z, 1.0, 1e-12);
}

TEST(GeoDist, Points) {
    Point2D p{0,0}, q{3,4};
    EXPECT_NEAR(dist(p,q), 5.0, 1e-12);
    Point3D a{0,0,0}, b{1,1,1};
    EXPECT_NEAR(dist(a,b), std::sqrt(3.0), 1e-12);
}

// ---- Convex Hull ----

TEST(GeoHull, SquarePlusCenter) {
    std::vector<Point2D> pts = {{0,0},{1,0},{1,1},{0,1},{0.5,0.5}};
    auto hull = convex_hull_2d(pts);
    EXPECT_EQ(hull.size(), 4u);  // center not on hull
}

TEST(GeoHull, Collinear) {
    std::vector<Point2D> pts = {{0,0},{1,0},{2,0},{3,0}};
    auto hull = convex_hull_2d(pts);
    EXPECT_GE(hull.size(), 2u);
}

// ---- Delaunay ----

TEST(GeoDelaunay, ThreePoints) {
    std::vector<Point2D> pts = {{0,0},{1,0},{0,1}};
    auto tris = delaunay_2d(pts);
    EXPECT_EQ(tris.size(), 1u);
}

TEST(GeoDelaunay, FourPoints) {
    std::vector<Point2D> pts = {{0,0},{1,0},{1,1},{0,1}};
    auto tris = delaunay_2d(pts);
    EXPECT_EQ(tris.size(), 2u);  // square triangulated into 2 triangles
}

TEST(GeoVoronoi, FourPoints) {
    std::vector<Point2D> pts = {{0,0},{1,0},{1,1},{0,1}};
    auto centers = voronoi(pts);
    EXPECT_EQ(centers.size(), 2u);
}

// ---- KD-Tree 2D ----

TEST(GeoKDTree2D, Nearest) {
    std::vector<Point2D> pts = {{0,0},{1,0},{2,0},{3,0},{4,0}};
    KDTree2D kd(pts);
    int idx = kd.nearest({1.1, 0.0});
    EXPECT_EQ(idx, 1);
}

TEST(GeoKDTree2D, KNN) {
    std::vector<Point2D> pts = {{0,0},{1,0},{2,0},{3,0},{4,0}};
    KDTree2D kd(pts);
    auto nbrs = kd.knn({1.0, 0.0}, 2);
    EXPECT_EQ(nbrs.size(), 2u);
}

TEST(GeoKDTree2D, Range) {
    std::vector<Point2D> pts = {{0,0},{1,0},{2,0},{3,0},{4,0}};
    KDTree2D kd(pts);
    auto r = kd.range({2.0, 0.0}, 1.5);
    EXPECT_GE(r.size(), 3u);  // 1, 2, 3
}

// ---- KD-Tree 3D ----

TEST(GeoKDTree3D, Nearest) {
    std::vector<Point3D> pts = {{0,0,0},{1,0,0},{2,0,0}};
    KDTree3D kd(pts);
    int idx = kd.nearest({0.9, 0, 0});
    EXPECT_EQ(idx, 1);
}

// ---- Intersections ----

TEST(GeoIntersect, SegSeg_Cross) {
    Segment2D s1{{0,0},{2,2}}, s2{{0,2},{2,0}};
    Point2D out;
    EXPECT_TRUE(intersect_seg_seg(s1, s2, &out));
    EXPECT_NEAR(out.x, 1.0, 1e-10);
    EXPECT_NEAR(out.y, 1.0, 1e-10);
}

TEST(GeoIntersect, SegSeg_NoIntersect) {
    Segment2D s1{{0,0},{1,0}}, s2{{0,1},{1,1}};
    EXPECT_FALSE(intersect_seg_seg(s1, s2));
}

TEST(GeoIntersect, RaySphere) {
    Ray3D ray{{0,0,-5}, {0,0,1}};
    Sphere3D sphere{{0,0,0}, 1.0};
    double t;
    EXPECT_TRUE(intersect_ray_sphere(ray, sphere, &t));
    EXPECT_NEAR(t, 4.0, 1e-10);
}

TEST(GeoIntersect, RayAABB) {
    Ray3D ray{{-5,0,0}, {1,0,0}};
    AABB3D box{{-1,-1,-1},{1,1,1}};
    double t;
    EXPECT_TRUE(intersect_ray_aabb(ray, box, &t));
    EXPECT_NEAR(t, 4.0, 1e-10);
}

TEST(GeoIntersect, PointInPolygon) {
    Polygon2D poly = {{0,0},{4,0},{4,4},{0,4}};
    EXPECT_TRUE(point_in_polygon({2,2}, poly));
    EXPECT_FALSE(point_in_polygon({5,5}, poly));
}

// ---- Distances ----

TEST(GeoDist, PointToSegment) {
    Segment2D s{{0,0},{4,0}};
    EXPECT_NEAR(dist_point_segment({2, 3}, s), 3.0, 1e-10);
    EXPECT_NEAR(dist_point_segment({-1, 0}, s), 1.0, 1e-10);
}

// ---- Bezier ----

TEST(GeoBezier, Eval) {
    std::vector<Point2D> ctrl = {{0,0},{1,2},{2,0}};
    Point2D p = bezier_eval(ctrl, 0.5);
    EXPECT_NEAR(p.x, 1.0, 1e-10);
    EXPECT_NEAR(p.y, 1.0, 1e-10);
}

TEST(GeoBezier, Endpoints) {
    std::vector<Point2D> ctrl = {{0,0},{1,2},{3,1}};
    auto p0 = bezier_eval(ctrl, 0.0);
    auto p1 = bezier_eval(ctrl, 1.0);
    EXPECT_NEAR(p0.x, 0.0, 1e-10);
    EXPECT_NEAR(p1.x, 3.0, 1e-10);
}

TEST(GeoBezier, Subdivide) {
    std::vector<Point2D> ctrl = {{0,0},{1,2},{2,0}};
    auto [left, right] = bezier_subdivide(ctrl, 0.5);
    EXPECT_EQ(left.size(), 3u);
    EXPECT_EQ(right.size(), 3u);
    // Left starts at p0, right ends at p2
    EXPECT_NEAR(left[0].x, 0.0, 1e-10);
    EXPECT_NEAR(right.back().x, 2.0, 1e-10);
}

TEST(GeoBezier, HermiteCurve) {
    Point2D p0{0,0}, p1{1,0};
    Vec2D m0{0,1}, m1{0,-1};
    auto mid = hermite_curve(p0, m0, p1, m1, 0.5);
    EXPECT_NEAR(mid.x, 0.5, 1e-10);
}

// ---- Measurements ----

TEST(GeoMeasure, SquareArea) {
    Polygon2D sq = {{0,0},{4,0},{4,4},{0,4}};
    EXPECT_NEAR(area(sq), 16.0, 1e-10);
}

TEST(GeoMeasure, Perimeter) {
    Polygon2D sq = {{0,0},{4,0},{4,4},{0,4}};
    EXPECT_NEAR(perimeter(sq), 16.0, 1e-10);
}

TEST(GeoMeasure, Centroid) {
    Polygon2D sq = {{0,0},{4,0},{4,4},{0,4}};
    auto c = centroid(sq);
    EXPECT_NEAR(c.x, 2.0, 1e-10);
    EXPECT_NEAR(c.y, 2.0, 1e-10);
}

TEST(GeoMeasure, TriangleArea2D) {
    EXPECT_NEAR(area({0,0},{1,0},{0,1}), 0.5, 1e-10);
}

TEST(GeoMeasure, TetrahedronVolume) {
    double vol = volume_tetrahedron({0,0,0},{1,0,0},{0,1,0},{0,0,1});
    EXPECT_NEAR(vol, 1.0/6.0, 1e-10);
}
