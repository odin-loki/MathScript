#define _USE_MATH_DEFINES
#include "ms/geo/geo.hpp"
#include <cmath>
#include <set>
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

// ---- Convex Hull 3D ----

TEST(GeoHull3D, Tetrahedron) {
    std::vector<Point3D> pts = {{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}};
    auto hull = convex_hull_3d(pts);
    EXPECT_EQ(hull.size(), 4u);

    int count[4] = {0,0,0,0};
    for (auto& f : hull) { count[f.a]++; count[f.b]++; count[f.c]++; }
    for (int i = 0; i < 4; ++i) EXPECT_EQ(count[i], 3);
}

TEST(GeoHull3D, Cube) {
    std::vector<Point3D> pts;
    for (double x : {0.0, 1.0})
        for (double y : {0.0, 1.0})
            for (double z : {0.0, 1.0})
                pts.push_back({x, y, z});
    auto hull = convex_hull_3d(pts);
    EXPECT_EQ(hull.size(), 12u);
}

TEST(GeoHull3D, Octahedron) {
    std::vector<Point3D> pts = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    auto hull = convex_hull_3d(pts);
    EXPECT_EQ(hull.size(), 8u);
}

TEST(GeoHull3D, InteriorPointExcluded) {
    std::vector<Point3D> pts = {{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1},{0,0,0}};
    // Index 4 (the tetrahedron's centroid) sits strictly inside the hull.
    auto hull = convex_hull_3d(pts);
    for (auto& f : hull) {
        EXPECT_NE(f.a, 4);
        EXPECT_NE(f.b, 4);
        EXPECT_NE(f.c, 4);
    }
    EXPECT_EQ(hull.size(), 4u);  // hull is unaffected by the interior point
}

TEST(GeoHull3D, OutwardFacingAndOneSided) {
    std::vector<Point3D> pts;
    for (double x : {0.0, 1.0})
        for (double y : {0.0, 1.0})
            for (double z : {0.0, 1.0})
                pts.push_back({x, y, z});
    auto hull = convex_hull_3d(pts);
    ASSERT_EQ(hull.size(), 12u);

    Point3D centroid{0.5, 0.5, 0.5};
    for (auto& f : hull) {
        Vec3D nrm = cross(vec3(pts[f.a], pts[f.b]), vec3(pts[f.a], pts[f.c]));
        double nlen = length(nrm);
        ASSERT_GT(nlen, 1e-12);
        double eps = 1e-9 * nlen;

        // Outward-pointing check: the face should point away from the cloud's centroid.
        Point3D face_centroid{(pts[f.a].x+pts[f.b].x+pts[f.c].x)/3.0,
                               (pts[f.a].y+pts[f.b].y+pts[f.c].y)/3.0,
                               (pts[f.a].z+pts[f.b].z+pts[f.c].z)/3.0};
        Vec3D to_face = vec3(centroid, face_centroid);
        EXPECT_GT(dot(nrm, to_face), 0.0);

        // Re-derive the "all other points on one side" property on the output.
        for (size_t p = 0; p < pts.size(); ++p) {
            double d = dot(nrm, vec3(pts[f.a], pts[p]));
            EXPECT_LE(d, eps);
        }
    }
}

TEST(GeoHull3D, FewerThanFourPointsIsEmpty) {
    EXPECT_TRUE(convex_hull_3d({}).empty());
    EXPECT_TRUE(convex_hull_3d({{0,0,0}}).empty());
    EXPECT_TRUE(convex_hull_3d({{0,0,0},{1,0,0}}).empty());
    EXPECT_TRUE(convex_hull_3d({{0,0,0},{1,0,0},{0,1,0}}).empty());
}

TEST(GeoHull3D, CoplanarSquareIsEmpty) {
    std::vector<Point3D> pts = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
    auto hull = convex_hull_3d(pts);
    EXPECT_TRUE(hull.empty());
}

TEST(GeoHull3D, CoplanarManyPointsIsEmpty) {
    std::vector<Point3D> pts = {{0,0,0},{1,0,0},{2,0,0},{2,2,0},{1,1,0},{0,2,0},{0.5,0.5,0}};
    auto hull = convex_hull_3d(pts);
    EXPECT_TRUE(hull.empty());
}

TEST(GeoHull3D, DuplicatePointsNoCrash) {
    std::vector<Point3D> pts = {{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1},{1,1,1}};  // last duplicates first
    std::vector<Triangle3Di> hull;
    ASSERT_NO_THROW(hull = convex_hull_3d(pts));
    EXPECT_LE(hull.size(), 10u);  // sanity bound: at most C(5,3) candidate triples
}

TEST(GeoHull3D, EulerCharacteristic) {
    // 12 hand-chosen points: the vertices of a regular icosahedron (golden
    // ratio construction). Every face is a genuine 3-point triangle (no
    // coincidental >3-point coplanarity), so this exercises the "regular"
    // face path with a known, non-trivial (V=12, E=30, F=20) structure.
    const double phi = 1.6180339887498949;
    std::vector<Point3D> pts = {
        { 1, phi, 0}, {-1, phi, 0}, { 1,-phi, 0}, {-1,-phi, 0},
        { 0, 1, phi}, { 0,-1, phi}, { 0, 1,-phi}, { 0,-1,-phi},
        { phi, 0, 1}, {-phi, 0, 1}, { phi, 0,-1}, {-phi, 0,-1},
    };
    auto hull = convex_hull_3d(pts);
    EXPECT_EQ(hull.size(), 20u);

    std::set<int> vertices;
    for (auto& f : hull) { vertices.insert(f.a); vertices.insert(f.b); vertices.insert(f.c); }
    int V = static_cast<int>(vertices.size());
    int F = static_cast<int>(hull.size());
    ASSERT_EQ(F % 2, 0);
    int E = 3 * F / 2;
    EXPECT_EQ(V - E + F, 2);
}

TEST(GeoHull3D, EulerCharacteristicTetrahedron) {
    std::vector<Point3D> pts = {{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}};
    auto hull = convex_hull_3d(pts);
    std::set<int> vertices;
    for (auto& f : hull) { vertices.insert(f.a); vertices.insert(f.b); vertices.insert(f.c); }
    int V = static_cast<int>(vertices.size());
    int F = static_cast<int>(hull.size());
    int E = 3 * F / 2;
    EXPECT_EQ(V - E + F, 2);
}

TEST(GeoHull3D, AllFacesHaveDistinctIndices) {
    std::vector<Point3D> pts;
    for (double x : {0.0, 1.0})
        for (double y : {0.0, 1.0})
            for (double z : {0.0, 1.0})
                pts.push_back({x, y, z});
    auto hull = convex_hull_3d(pts);
    for (auto& f : hull) {
        EXPECT_NE(f.a, f.b);
        EXPECT_NE(f.b, f.c);
        EXPECT_NE(f.a, f.c);
    }
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
