#define _USE_MATH_DEFINES
#include "ms/geo/geo.hpp"
#include <cmath>
#include <limits>
#include <random>
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

// ---- Minimum Bounding Rectangle ----

namespace {

// Normalises an angle difference so it lies in (-pi/4, pi/4], accounting for the
// mod-pi/2 ambiguity of which rectangle side is called "width" vs "height".
double angle_diff_mod_halfpi(double a, double b) {
    double d = std::fmod(a - b, M_PI / 2.0);
    if (d > M_PI / 4.0) d -= M_PI / 2.0;
    if (d < -M_PI / 4.0) d += M_PI / 2.0;
    return d;
}

// Checks every point of `pts` lies within `r` (with tolerance `eps`), by rotating each
// point into the rectangle's local frame and comparing against the half-extents.
void expect_rect_contains_all(const MinBoundingRect& r, const std::vector<Point2D>& pts,
                               double eps = 1e-9) {
    double c = std::cos(r.angle), s = std::sin(r.angle);
    for (const auto& p : pts) {
        double dx = p.x - r.center.x, dy = p.y - r.center.y;
        double lx =  c*dx + s*dy;   // rotate by -angle into local frame
        double ly = -s*dx + c*dy;
        EXPECT_LE(std::abs(lx), r.width * 0.5 + eps);
        EXPECT_LE(std::abs(ly), r.height * 0.5 + eps);
    }
}

} // namespace

TEST(GeoMinBoundingRect, AxisAlignedRectangle) {
    // 4x2 rectangle corners, plus a few interior/edge-redundant points.
    std::vector<Point2D> pts = {
        {0,0}, {4,0}, {4,2}, {0,2},
        {2,0}, {2,2}, {2,1},   // edge midpoints / centroid: hull-redundant
    };
    auto r = min_bounding_rect(pts);
    // Width/height axis convention may swap depending on which hull edge wins ties.
    double lo = std::min(r.width, r.height), hi = std::max(r.width, r.height);
    EXPECT_NEAR(lo, 2.0, 1e-9);
    EXPECT_NEAR(hi, 4.0, 1e-9);
    EXPECT_NEAR(r.area(), 8.0, 1e-9);
    EXPECT_NEAR(angle_diff_mod_halfpi(r.angle, 0.0), 0.0, 1e-9);
}

TEST(GeoMinBoundingRect, RotatedRectangleRecoversAngleAndDims) {
    const double rot = 30.0 * M_PI / 180.0;
    double c = std::cos(rot), s = std::sin(rot);
    // Original axis-aligned 4x2 rectangle corners, rotated by 30 degrees about the origin.
    std::vector<Point2D> base = {{-2,-1}, {2,-1}, {2,1}, {-2,1}};
    std::vector<Point2D> pts;
    for (auto& p : base) pts.push_back({c*p.x - s*p.y, s*p.x + c*p.y});

    auto r = min_bounding_rect(pts);
    double lo = std::min(r.width, r.height), hi = std::max(r.width, r.height);
    EXPECT_NEAR(lo, 2.0, 1e-9);
    EXPECT_NEAR(hi, 4.0, 1e-9);
    EXPECT_NEAR(r.area(), 8.0, 1e-9);
    EXPECT_NEAR(angle_diff_mod_halfpi(r.angle, rot), 0.0, 1e-9);
}

TEST(GeoMinBoundingRect, RotatedSquareDiamond) {
    // A square rotated 45 degrees ("diamond"): vertices at distance 1 along each axis.
    std::vector<Point2D> pts = {{1,0}, {0,1}, {-1,0}, {0,-1}};
    auto r = min_bounding_rect(pts);
    // Side length sqrt(2), area 2, exactly recovered by the calipers (no looser AABB=area 2 too,
    // coincidentally, since a diamond's AABB is also a 2x2 square of area 4 -- but the true
    // minimum oriented rectangle here is the diamond itself, area 2).
    EXPECT_NEAR(r.area(), 2.0, 1e-9);
    double lo = std::min(r.width, r.height), hi = std::max(r.width, r.height);
    EXPECT_NEAR(lo, std::sqrt(2.0), 1e-9);
    EXPECT_NEAR(hi, std::sqrt(2.0), 1e-9);
    EXPECT_NEAR(angle_diff_mod_halfpi(r.angle, M_PI / 4.0), 0.0, 1e-9);
}

TEST(GeoMinBoundingRect, ScatteredCloudMatchesBruteForceMinimum) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ang_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> rad_dist(0.5, 3.0);
    std::vector<Point2D> pts;
    for (int i = 0; i < 40; ++i) {
        double a = ang_dist(rng), rr = rad_dist(rng);
        pts.push_back({rr * std::cos(a), rr * std::sin(a) * 0.4});  // squashed ellipse-ish cloud
    }

    auto r = min_bounding_rect(pts);

    // Brute-force scan of candidate orientations; the calipers result must be at least as
    // good (smaller or equal area) as every sampled orientation's axis-aligned bbox area.
    double brute_min_area = std::numeric_limits<double>::infinity();
    const int kSteps = 4000;
    for (int i = 0; i < kSteps; ++i) {
        double theta = M_PI * static_cast<double>(i) / kSteps;
        double c = std::cos(theta), s = std::sin(theta);
        double minx =  std::numeric_limits<double>::infinity(), maxx = -minx;
        double miny =  std::numeric_limits<double>::infinity(), maxy = -miny;
        for (auto& p : pts) {
            double rx =  c*p.x + s*p.y;
            double ry = -s*p.x + c*p.y;
            minx = std::min(minx, rx); maxx = std::max(maxx, rx);
            miny = std::min(miny, ry); maxy = std::max(maxy, ry);
        }
        brute_min_area = std::min(brute_min_area, (maxx-minx) * (maxy-miny));
    }

    EXPECT_LE(r.area(), brute_min_area + 1e-6);
}

TEST(GeoMinBoundingRect, AreaMatchesWidthTimesHeight) {
    MinBoundingRect r;
    r.width = 3.5; r.height = 2.25;
    EXPECT_DOUBLE_EQ(r.area(), 3.5 * 2.25);
}

TEST(GeoMinBoundingRect, CornersRoundTripReproducesDimensions) {
    std::vector<Point2D> pts = {{-2,-1}, {2,-1}, {2,1}, {-2,1}, {0,0}};
    auto r = min_bounding_rect(pts);
    auto corners = r.corners();
    ASSERT_EQ(corners.size(), 4u);

    double c = std::cos(r.angle), s = std::sin(r.angle);
    double minx =  std::numeric_limits<double>::infinity(), maxx = -minx;
    double miny =  std::numeric_limits<double>::infinity(), maxy = -miny;
    for (auto& corner : corners) {
        double dx = corner.x - r.center.x, dy = corner.y - r.center.y;
        double lx =  c*dx + s*dy;
        double ly = -s*dx + c*dy;
        minx = std::min(minx, lx); maxx = std::max(maxx, lx);
        miny = std::min(miny, ly); maxy = std::max(maxy, ly);
    }
    EXPECT_NEAR(maxx - minx, r.width, 1e-9);
    EXPECT_NEAR(maxy - miny, r.height, 1e-9);
}

TEST(GeoMinBoundingRect, ZeroPointsIsSensible) {
    std::vector<Point2D> pts;
    auto r = min_bounding_rect(pts);
    EXPECT_NEAR(r.width, 0.0, 1e-12);
    EXPECT_NEAR(r.height, 0.0, 1e-12);
    EXPECT_NEAR(r.area(), 0.0, 1e-12);
}

TEST(GeoMinBoundingRect, OnePointIsSensible) {
    std::vector<Point2D> pts = {{3, 5}};
    auto r = min_bounding_rect(pts);
    EXPECT_NEAR(r.width, 0.0, 1e-12);
    EXPECT_NEAR(r.height, 0.0, 1e-12);
    EXPECT_NEAR(r.center.x, 3.0, 1e-12);
    EXPECT_NEAR(r.center.y, 5.0, 1e-12);
}

TEST(GeoMinBoundingRect, TwoPointsFallsBackToAABB) {
    std::vector<Point2D> pts = {{0, 0}, {4, 3}};
    auto r = min_bounding_rect(pts);
    // Documented fallback: axis-aligned bbox of the raw points.
    EXPECT_NEAR(r.width, 4.0, 1e-12);
    EXPECT_NEAR(r.height, 3.0, 1e-12);
    EXPECT_NEAR(r.center.x, 2.0, 1e-12);
    EXPECT_NEAR(r.center.y, 1.5, 1e-12);
}

TEST(GeoMinBoundingRect, ThreeCollinearPointsNoCrash) {
    std::vector<Point2D> pts = {{0,0}, {1,0}, {2,0}};
    auto r = min_bounding_rect(pts);
    // Degenerate hull (<3 verts) triggers the axis-aligned bbox fallback.
    EXPECT_NEAR(r.width, 2.0, 1e-9);
    EXPECT_NEAR(r.height, 0.0, 1e-9);
    EXPECT_GE(r.area(), 0.0);
}

TEST(GeoMinBoundingRect, AllIdenticalPointsNoCrash) {
    std::vector<Point2D> pts = {{2,2}, {2,2}, {2,2}, {2,2}};
    auto r = min_bounding_rect(pts);
    EXPECT_NEAR(r.width, 0.0, 1e-12);
    EXPECT_NEAR(r.height, 0.0, 1e-12);
}

TEST(GeoMinBoundingRect, ContainsAllPointsRotatedCluster) {
    const double rot = 17.0 * M_PI / 180.0;
    double c = std::cos(rot), s = std::sin(rot);
    std::vector<Point2D> base = {{-3,-1}, {3,-1}, {3,1}, {-3,1}, {1,0.5}, {-1,-0.7}, {0,0}};
    std::vector<Point2D> pts;
    for (auto& p : base) pts.push_back({c*p.x - s*p.y, s*p.x + c*p.y});

    auto r = min_bounding_rect(pts);
    expect_rect_contains_all(r, pts);
}

TEST(GeoMinBoundingRect, ContainsAllPointsScatteredCloud) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> d(-5.0, 5.0);
    std::vector<Point2D> pts;
    for (int i = 0; i < 30; ++i) pts.push_back({d(rng), d(rng)});

    auto r = min_bounding_rect(pts);
    expect_rect_contains_all(r, pts, 1e-7);
}

TEST(GeoMinBoundingRect, ContainsAllPointsRegularHexagon) {
    std::vector<Point2D> pts;
    for (int i = 0; i < 6; ++i) {
        double a = i * M_PI / 3.0;
        pts.push_back({std::cos(a), std::sin(a)});
    }
    auto r = min_bounding_rect(pts);
    expect_rect_contains_all(r, pts);
}

TEST(GeoMinBoundingRect, CornersFormARectangle) {
    std::vector<Point2D> pts = {{-2,-1}, {2,-1}, {2,1}, {-2,1}, {0.3, 0.1}};
    auto r = min_bounding_rect(pts);
    auto k = r.corners();

    double side01 = dist(k[0], k[1]);
    double side12 = dist(k[1], k[2]);
    double side23 = dist(k[2], k[3]);
    double side30 = dist(k[3], k[0]);
    EXPECT_NEAR(side01, side23, 1e-9);   // opposite sides equal
    EXPECT_NEAR(side12, side30, 1e-9);
    EXPECT_NEAR(side01, r.width, 1e-9);
    EXPECT_NEAR(side12, r.height, 1e-9);

    double diag02 = dist(k[0], k[2]);
    double diag13 = dist(k[1], k[3]);
    EXPECT_NEAR(diag02, diag13, 1e-9);   // diagonals equal length (rectangle property)
}

// ---- Minkowski Sum ----

namespace {

// Verifies `poly` is convex and CCW-wound: every consecutive edge triple must turn left
// (non-negative cross product), which for a CCW polygon also implies non-negative signed area.
void expect_convex_ccw(const Polygon2D& poly, double eps = 1e-9) {
    int n = static_cast<int>(poly.size());
    if (n < 3) return;  // degenerate (point/segment) is trivially "convex"
    for (int i = 0; i < n; ++i) {
        Vec2D e1 = vec2(poly[i], poly[(i + 1) % n]);
        Vec2D e2 = vec2(poly[(i + 1) % n], poly[(i + 2) % n]);
        EXPECT_GE(cross2d(e1, e2), -eps) << "reflex/CW turn at vertex " << i;
    }
    EXPECT_GE(signed_area(poly), -eps) << "polygon is not CCW-wound";
}

// Compares two polygons as sets of (x,y) points (ignoring winding start / traversal order),
// each matched at most once, within tolerance `eps`.
void expect_same_point_set(const Polygon2D& p, const Polygon2D& q, double eps = 1e-6) {
    ASSERT_EQ(p.size(), q.size()) << "point sets differ in size";
    std::vector<bool> used(q.size(), false);
    for (const auto& pt : p) {
        bool found = false;
        for (size_t i = 0; i < q.size(); ++i) {
            if (used[i]) continue;
            if (std::abs(pt.x - q[i].x) < eps && std::abs(pt.y - q[i].y) < eps) {
                used[i] = true;
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "point (" << pt.x << ", " << pt.y << ") had no match";
    }
}

// Brute-force reference: pairwise vertex sums, then convex_hull_2d -- valid for any convex
// a/b, used here purely as an independent cross-check of the merge-by-angle production code.
Polygon2D minkowski_bruteforce_reference(const Polygon2D& a, const Polygon2D& b) {
    std::vector<Point2D> sums;
    for (const auto& pa : a)
        for (const auto& pb : b)
            sums.push_back({pa.x + pb.x, pa.y + pb.y});
    return convex_hull_2d(sums);
}

// `phase` rotates the whole polygon's vertex placement. A nonzero phase matters for
// cross-checks against an axis-aligned square: at phase 0, an odd-gon (e.g. a pentagon) has
// one edge that is exactly vertical by construction, which can coincide (up to the sub-ULP
// floating-point noise between cos() of two "symmetric" angles that are mathematically but not
// bit-for-bit identical) with the square's exactly-vertical edge. The production merge-by-angle
// code correctly treats that near-zero cross product as a tie and merges the two collinear
// edges, while the epsilon-free brute-force convex_hull_2d reference can instead keep it as a
// spurious extra near-duplicate vertex -- a genuine floating-point artifact, not a functional
// disagreement. An arbitrary non-axis-aligned phase avoids the coincidence entirely.
Polygon2D regular_ngon(int n, double radius, Point2D center = {0, 0}, double phase = 0.0) {
    Polygon2D poly;
    for (int i = 0; i < n; ++i) {
        double a = phase + i * 2.0 * M_PI / n;
        poly.push_back({center.x + radius * std::cos(a), center.y + radius * std::sin(a)});
    }
    return poly;
}

} // namespace

TEST(GeoMinkowski, TwoIdenticalUnitSquaresGiveTwoByTwoSquare) {
    Polygon2D a = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    Polygon2D b = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    auto sum = minkowski_sum_convex(a, b);

    ASSERT_EQ(sum.size(), 4u);
    Polygon2D expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
    expect_same_point_set(sum, expected);
    EXPECT_NEAR(area(sum), 4.0, 1e-9);
    expect_convex_ccw(sum);
}

TEST(GeoMinkowski, TranslatedSquareShiftsResultButKeepsSize) {
    Polygon2D a = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    Polygon2D b = {{1, 0}, {2, 0}, {2, 1}, {1, 1}};  // same unit square, translated by (1,0)
    auto sum = minkowski_sum_convex(a, b);

    ASSERT_EQ(sum.size(), 4u);
    Polygon2D expected = {{1, 0}, {3, 0}, {3, 2}, {1, 2}};
    expect_same_point_set(sum, expected);
    EXPECT_NEAR(area(sum), 4.0, 1e-9);
    expect_convex_ccw(sum);
}

TEST(GeoMinkowski, SquarePlusUnitSegmentGivesTwoByOneRectangle) {
    Polygon2D a = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    Polygon2D b = {{0, 0}, {1, 0}};  // degenerate "polygon": a horizontal unit segment
    auto sum = minkowski_sum_convex(a, b);

    Polygon2D expected = {{0, 0}, {2, 0}, {2, 1}, {0, 1}};
    ASSERT_EQ(sum.size(), 4u);
    expect_same_point_set(sum, expected);
    EXPECT_NEAR(area(sum), 2.0, 1e-9);
    expect_convex_ccw(sum);
}

TEST(GeoMinkowski, SinglePointTranslatesPolygonExactly) {
    Polygon2D a = {{0, 0}, {2, 0}, {2, 1}, {0, 1}};
    Polygon2D b = {{3, 5}};  // single point: trivially convex, degenerate <3-vertex input
    auto sum = minkowski_sum_convex(a, b);

    Polygon2D expected = {{3, 5}, {5, 5}, {5, 6}, {3, 6}};
    ASSERT_EQ(sum.size(), 4u);
    expect_same_point_set(sum, expected);
    expect_convex_ccw(sum);
}

TEST(GeoMinkowski, BothEmptyGivesEmpty) {
    Polygon2D a, b;
    auto sum = minkowski_sum_convex(a, b);
    EXPECT_TRUE(sum.empty());
}

TEST(GeoMinkowski, OneEmptyGivesEmpty) {
    Polygon2D a = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    Polygon2D b;
    EXPECT_TRUE(minkowski_sum_convex(a, b).empty());
    EXPECT_TRUE(minkowski_sum_convex(b, a).empty());
}

TEST(GeoMinkowski, TwoSinglePointsSumsToOnePoint) {
    Polygon2D a = {{2, 3}};
    Polygon2D b = {{10, -1}};
    auto sum = minkowski_sum_convex(a, b);
    ASSERT_EQ(sum.size(), 1u);
    EXPECT_NEAR(sum[0].x, 12.0, 1e-12);
    EXPECT_NEAR(sum[0].y, 2.0, 1e-12);
}

TEST(GeoMinkowski, TriangleAndSquareMatchesBruteForceReference) {
    Polygon2D tri = {{0, 0}, {2, 0}, {1, 2}};
    Polygon2D sq  = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    auto fast = minkowski_sum_convex(tri, sq);
    auto brute = minkowski_bruteforce_reference(tri, sq);

    expect_same_point_set(fast, brute);
    expect_convex_ccw(fast);
    EXPECT_LE(fast.size(), tri.size() + sq.size());
}

TEST(GeoMinkowski, ResultIsConvexForAsymmetricTriangles) {
    Polygon2D t1 = {{0, 0}, {3, 0}, {0, 1}};
    Polygon2D t2 = {{0, 0}, {1, 0}, {0, 4}};
    auto sum = minkowski_sum_convex(t1, t2);
    expect_convex_ccw(sum);
    EXPECT_LE(sum.size(), t1.size() + t2.size());
    auto brute = minkowski_bruteforce_reference(t1, t2);
    expect_same_point_set(sum, brute);
}

TEST(GeoMinkowski, VertexCountNeverExceedsSumOfInputs) {
    Polygon2D a = {{0, 0}, {2, 0}, {2, 1}, {0, 1}};                 // rectangle, 4 verts
    Polygon2D b = {{0, 0}, {1, 0}, {1.5, 1}, {0.5, 1.5}, {-0.5, 1}}; // convex pentagon, 5 verts
    auto sum = minkowski_sum_convex(a, b);
    EXPECT_LE(sum.size(), a.size() + b.size());
    expect_convex_ccw(sum);
}

TEST(GeoMinkowski, CommutativeForTriangleAndSquare) {
    Polygon2D tri = {{0, 0}, {2, 0}, {1, 2}};
    Polygon2D sq  = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    auto ab = minkowski_sum_convex(tri, sq);
    auto ba = minkowski_sum_convex(sq, tri);
    expect_same_point_set(ab, ba);
    expect_convex_ccw(ab);
    expect_convex_ccw(ba);
}

TEST(GeoMinkowski, CommutativeForAsymmetricConvexPentagons) {
    Polygon2D p1 = {{0, 0}, {2, 0}, {3, 1}, {1, 3}, {-1, 1}};
    Polygon2D p2 = {{0, 0}, {1, 0}, {1.5, 0.5}, {0.5, 1.5}, {-0.5, 0.5}};

    auto ab = minkowski_sum_convex(p1, p2);
    auto ba = minkowski_sum_convex(p2, p1);
    expect_same_point_set(ab, ba);
}

TEST(GeoMinkowski, RegularPentagonPlusSmallSquareMatchesBruteForce) {
    Polygon2D pentagon = regular_ngon(5, 2.0, {0, 0}, 0.3);  // phase avoids an on-axis edge
    Polygon2D square = {{-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1}};

    auto fast = minkowski_sum_convex(pentagon, square);
    auto brute = minkowski_bruteforce_reference(pentagon, square);

    expect_same_point_set(fast, brute);
    expect_convex_ccw(fast);
    EXPECT_LE(fast.size(), pentagon.size() + square.size());
}

TEST(GeoMinkowski, RegularHexagonPlusSmallSquareMatchesBruteForce) {
    Polygon2D hexagon = regular_ngon(6, 1.5);
    Polygon2D square = {{-0.2, -0.2}, {0.2, -0.2}, {0.2, 0.2}, {-0.2, 0.2}};

    auto fast = minkowski_sum_convex(hexagon, square);
    auto brute = minkowski_bruteforce_reference(hexagon, square);

    expect_same_point_set(fast, brute);
    expect_convex_ccw(fast);
    EXPECT_LE(fast.size(), hexagon.size() + square.size());
}

TEST(GeoMinkowski, TwoRegularPolygonsOfDifferentSizeMatchesBruteForce) {
    Polygon2D pentagon = regular_ngon(5, 1.0);
    Polygon2D hexagon = regular_ngon(6, 0.5, {0.3, -0.2});

    auto fast = minkowski_sum_convex(pentagon, hexagon);
    auto brute = minkowski_bruteforce_reference(pentagon, hexagon);

    expect_same_point_set(fast, brute);
    expect_convex_ccw(fast);
    EXPECT_LE(fast.size(), pentagon.size() + hexagon.size());
}

TEST(GeoMinkowski, StartsFromBottomMostSummedVertex) {
    // The lowest point of the sum must be exactly a's bottom-most vertex plus b's bottom-most
    // vertex (both polygons' angular walk starts there by construction).
    Polygon2D a = {{0, 0}, {2, 0}, {1, 3}};
    Polygon2D b = {{5, -4}, {6, -4}, {6, -3}, {5, -3}};
    auto sum = minkowski_sum_convex(a, b);
    ASSERT_FALSE(sum.empty());

    Point2D lowest = sum[0];
    for (const auto& p : sum)
        if (p.y < lowest.y || (p.y == lowest.y && p.x < lowest.x)) lowest = p;

    EXPECT_NEAR(lowest.x, 0.0 + 5.0, 1e-9);
    EXPECT_NEAR(lowest.y, 0.0 + -4.0, 1e-9);
}

TEST(GeoMinkowski, AreaGrowsMonotonicallyWithScaledSecondOperand) {
    Polygon2D a = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    Polygon2D small = {{-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1}};
    Polygon2D big   = {{-0.4, -0.4}, {0.4, -0.4}, {0.4, 0.4}, {-0.4, 0.4}};

    auto sum_small = minkowski_sum_convex(a, small);
    auto sum_big = minkowski_sum_convex(a, big);
    EXPECT_GT(area(sum_big), area(sum_small));
    expect_convex_ccw(sum_small);
    expect_convex_ccw(sum_big);
}

TEST(GeoMinkowski, ThreeVertexTrianglesProduceAtMostSixVertices) {
    Polygon2D t1 = {{0, 0}, {1, 0}, {0, 1}};
    Polygon2D t2 = {{0, 0}, {1, 0}, {1, 1}};
    auto sum = minkowski_sum_convex(t1, t2);
    EXPECT_LE(sum.size(), 6u);
    EXPECT_GE(sum.size(), 3u);
    expect_convex_ccw(sum);
    auto brute = minkowski_bruteforce_reference(t1, t2);
    expect_same_point_set(sum, brute);
}
