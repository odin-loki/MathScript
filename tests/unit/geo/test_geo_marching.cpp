// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#define _USE_MATH_DEFINES
#include "ms/geo/geo.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::geo;

namespace {

// Build an n x n x n sample grid of f over the cube [lo,hi]^3, x-fastest.
std::vector<double> grid3(int n, double lo, double hi, double (*f)(double, double, double)) {
    const double h = (hi - lo) / (n - 1);
    std::vector<double> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) *
                          static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                v[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
                  (static_cast<std::size_t>(j) + static_cast<std::size_t>(n) *
                   static_cast<std::size_t>(k))] = f(lo + i * h, lo + j * h, lo + k * h);
    return v;
}

std::vector<double> grid2(int n, double lo, double hi, double (*f)(double, double)) {
    const double h = (hi - lo) / (n - 1);
    std::vector<double> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
              static_cast<std::size_t>(j)] = f(lo + i * h, lo + j * h);
    return v;
}

double sphere_f(double x, double y, double z) { return x*x + y*y + z*z - 1.0; }
double plane_f(double, double, double z) { return z - 0.5; }
double circle_f(double x, double y) { return x*x + y*y - 1.0; }
double neg_circle_f(double x, double y) { return -(x*x + y*y - 1.0); }
double hline_f(double, double y) { return y - 0.5; }

// Signed area swept by the directed contour (positive = interior on the left).
double contour_signed_area(const std::vector<Segment2D>& segs) {
    double s = 0.0;
    for (const auto& g : segs) s += g.a.x * g.b.y - g.b.x * g.a.y;
    return 0.5 * s;
}

double contour_length(const std::vector<Segment2D>& segs) {
    double s = 0.0;
    for (const auto& g : segs) s += dist(g.a, g.b);
    return s;
}

// Every directed triangle edge appears exactly once and its reverse exactly once.
bool is_watertight(const TriMesh3D& m) {
    std::map<std::pair<int,int>, int> d;
    for (const auto& t : m.triangles) {
        d[{t.a, t.b}] += 1;
        d[{t.b, t.c}] += 1;
        d[{t.c, t.a}] += 1;
    }
    for (const auto& kv : d) {
        if (kv.second != 1) return false;
        auto it = d.find({kv.first.second, kv.first.first});
        if (it == d.end() || it->second != 1) return false;
    }
    return true;
}

bool all_finite(const std::vector<Triangle3D>& tris) {
    for (const auto& t : tris) {
        if (!std::isfinite(t.a.x) || !std::isfinite(t.a.y) || !std::isfinite(t.a.z)) return false;
        if (!std::isfinite(t.b.x) || !std::isfinite(t.b.y) || !std::isfinite(t.b.z)) return false;
        if (!std::isfinite(t.c.x) || !std::isfinite(t.c.y) || !std::isfinite(t.c.z)) return false;
    }
    return true;
}

}  // namespace

// ---- Marching cubes: single-cell exactness ----

TEST(GeoMarchingCubes, SingleCornerBelowIsoGivesOneExactTriangle) {
    std::vector<double> f(8, 1.0);
    f[0] = -1.0;                                  // corner 0 = grid node (0,0,0)
    auto tris = marching_cubes(f, 2, 2, 2, 0.0);
    ASSERT_EQ(tris.size(), 1u);
    // iso is midway between -1 and +1, so each cut edge is bisected.
    EXPECT_NEAR(tris[0].a.x, 0.5, 1e-15);
    EXPECT_NEAR(tris[0].a.y, 0.0, 1e-15);
    EXPECT_NEAR(tris[0].a.z, 0.0, 1e-15);
    EXPECT_NEAR(tris[0].b.x, 0.0, 1e-15);
    EXPECT_NEAR(tris[0].b.y, 0.5, 1e-15);
    EXPECT_NEAR(tris[0].b.z, 0.0, 1e-15);
    EXPECT_NEAR(tris[0].c.x, 0.0, 1e-15);
    EXPECT_NEAR(tris[0].c.y, 0.0, 1e-15);
    EXPECT_NEAR(tris[0].c.z, 0.5, 1e-15);
    // Equilateral triangle of side sqrt(0.5): area = sqrt(3)/8.
    EXPECT_NEAR(mesh_surface_area(tris), std::sqrt(3.0) / 8.0, 1e-14);
    EXPECT_NEAR(mesh_surface_area(tris), 0.21650635094610965, 1e-14);
    // Normal points AWAY from corner 0, i.e. out of the solid {f < iso}.
    const Vec3D n = cross(vec3(tris[0].a, tris[0].b), vec3(tris[0].a, tris[0].c));
    EXPECT_NEAR(n.x, 0.25, 1e-15);
    EXPECT_NEAR(n.y, 0.25, 1e-15);
    EXPECT_NEAR(n.z, 0.25, 1e-15);
}

TEST(GeoMarchingCubes, CellFullyInsideOrFullyOutsideGivesNoTriangles) {
    EXPECT_TRUE(marching_cubes(std::vector<double>(8, -1.0), 2, 2, 2, 0.0).empty());  // case 255
    EXPECT_TRUE(marching_cubes(std::vector<double>(8,  1.0), 2, 2, 2, 0.0).empty());  // case 0
}

TEST(GeoMarchingCubes, DegenerateGridsGiveEmpty) {
    EXPECT_TRUE(marching_cubes({}, 0, 0, 0, 0.0).empty());
    EXPECT_TRUE(marching_cubes(std::vector<double>(4, 1.0), 1, 2, 2, 0.0).empty());  // nx < 2
    EXPECT_TRUE(marching_cubes(std::vector<double>(4, 1.0), 2, 1, 2, 0.0).empty());  // ny < 2
    EXPECT_TRUE(marching_cubes(std::vector<double>(4, 1.0), 2, 2, 1, 0.0).empty());  // nz < 2
    EXPECT_TRUE(marching_cubes(std::vector<double>(7, 1.0), 2, 2, 2, 0.0).empty());  // size mismatch
    const auto m = marching_cubes_mesh(std::vector<double>(7, 1.0), 2, 2, 2, 0.0);
    EXPECT_TRUE(m.vertices.empty());
    EXPECT_TRUE(m.triangles.empty());
}

TEST(GeoMarchingCubes, IsoOutsideFieldRangeGivesEmpty) {
    const auto f = grid3(9, -1.5, 1.5, sphere_f);   // range [-1, 5.75]
    EXPECT_TRUE(marching_cubes(f, 9, 9, 9,  100.0).empty());  // above max: everything below iso
    EXPECT_TRUE(marching_cubes(f, 9, 9, 9, -100.0).empty());  // below min: nothing below iso
    // iso exactly at the minimum is still empty: the test is strict `<`.
    EXPECT_TRUE(marching_cubes(f, 9, 9, 9,   -1.0).empty());
}

TEST(GeoMarchingCubes, SampleExactlyOnIsoCountsAsOutside) {
    // Corner 0 sits exactly on the level and the rest are above: nothing is strictly below,
    // so the cell is case 0 and contributes no triangle at all.
    std::vector<double> f(8, 1.0);
    f[0] = 0.0;
    EXPECT_TRUE(marching_cubes(f, 2, 2, 2, 0.0).empty());

    // Corner 0 below, corner 1 exactly on the level: the crossing on edge 0-1 lands at t = 1,
    // i.e. exactly on grid node (1,0,0).
    f[0] = -1.0;
    f[1] = 0.0;
    const auto tris = marching_cubes(f, 2, 2, 2, 0.0);
    ASSERT_EQ(tris.size(), 1u);
    EXPECT_DOUBLE_EQ(tris[0].a.x, 1.0);
    EXPECT_DOUBLE_EQ(tris[0].a.y, 0.0);
    EXPECT_DOUBLE_EQ(tris[0].a.z, 0.0);

    // Corner 0 exactly on the level with every other corner below (case 254): all three cut
    // edges interpolate back to node (0,0,0), so the emitted triangle is kept but degenerate
    // and contributes nothing to either measure.
    std::vector<double> g(8, -1.0);
    g[0] = 0.0;
    const auto deg = marching_cubes(g, 2, 2, 2, 0.0);
    ASSERT_EQ(deg.size(), 1u);
    EXPECT_DOUBLE_EQ(mesh_surface_area(deg), 0.0);
    EXPECT_DOUBLE_EQ(mesh_volume(deg), 0.0);
}

TEST(GeoMarchingCubes, PlanarFieldGivesExactFlatSquare) {
    // f = z - 0.5 on a 5x5x5 grid over the unit cube: the isosurface is the unit square z=0.5.
    const int n = 5;
    const double h = 1.0 / (n - 1);
    const auto f = grid3(n, 0.0, 1.0, plane_f);
    const auto tris = marching_cubes(f, n, n, n, 0.0, {0.0, 0.0, 0.0}, {h, h, h});
    EXPECT_EQ(tris.size(), 32u);                       // 16 cells in the cut layer, 2 each
    EXPECT_NEAR(mesh_surface_area(tris), 1.0, 1e-14);  // exactly the unit square
    for (const auto& t : tris) {
        EXPECT_NEAR(t.a.z, 0.5, 1e-15);
        EXPECT_NEAR(t.b.z, 0.5, 1e-15);
        EXPECT_NEAR(t.c.z, 0.5, 1e-15);
    }
}

// ---- Marching cubes: convergence on an analytic sphere ----

TEST(GeoMarchingCubes, SphereSurfaceAreaApproachesFourPiRSquared) {
    const double exact = 4.0 * M_PI;               // r = 1
    {
        const auto f = grid3(21, -1.5, 1.5, sphere_f);
        const auto tris = marching_cubes(f, 21, 21, 21, 0.0,
                                         {-1.5,-1.5,-1.5}, {0.15,0.15,0.15});
        EXPECT_EQ(tris.size(), 1640u);
        const double a = mesh_surface_area(tris);
        EXPECT_NEAR(a, 12.447241, 1e-4);              // measured value
        EXPECT_LT(std::abs(a / exact - 1.0), 0.015);  // within 1.5%
    }
    {   // halving h roughly halves the error
        const auto f = grid3(41, -1.5, 1.5, sphere_f);
        const auto tris = marching_cubes(f, 41, 41, 41, 0.0,
                                         {-1.5,-1.5,-1.5}, {0.075,0.075,0.075});
        EXPECT_EQ(tris.size(), 6632u);
        const double a = mesh_surface_area(tris);
        EXPECT_NEAR(a, 12.537011, 1e-4);              // measured value
        EXPECT_LT(std::abs(a / exact - 1.0), 0.005);  // within 0.5%
    }
}

TEST(GeoMarchingCubes, SphereVolumeApproachesFourThirdsPiRCubed) {
    const double exact = 4.0 / 3.0 * M_PI;
    const auto f = grid3(41, -1.5, 1.5, sphere_f);
    const auto tris = marching_cubes(f, 41, 41, 41, 0.0, {-1.5,-1.5,-1.5}, {0.075,0.075,0.075});
    const double v = mesh_volume(tris);
    EXPECT_GT(v, 0.0);                             // outward orientation => positive volume
    EXPECT_NEAR(v, 4.171240, 1e-4);                // measured value
    EXPECT_LT(std::abs(v / exact - 1.0), 0.01);    // within 1%
    // The divergence-theorem sum is origin-independent for a closed mesh: translating the grid
    // must not change the answer.
    const auto shifted = marching_cubes(f, 41, 41, 41, 0.0,
                                        {98.5, -101.5, 7.5}, {0.075,0.075,0.075});
    EXPECT_NEAR(mesh_volume(shifted), v, 1e-8);
}

TEST(GeoMarchingCubes, IndexedSphereMeshIsWatertightAndGenusZero) {
    const auto f = grid3(21, -1.5, 1.5, sphere_f);
    const auto m = marching_cubes_mesh(f, 21, 21, 21, 0.0, {-1.5,-1.5,-1.5}, {0.15,0.15,0.15});
    EXPECT_EQ(m.vertices.size(), 822u);
    EXPECT_EQ(m.triangles.size(), 1640u);
    EXPECT_TRUE(is_watertight(m));
    // Closed triangle mesh: E = 3F/2, so V - E + F = 2 for a single genus-0 component.
    const int V = static_cast<int>(m.vertices.size());
    const int F = static_cast<int>(m.triangles.size());
    const int E = 3 * F / 2;
    EXPECT_EQ(V - E + F, 2);
    // Every vertex lies on the sphere: linear interpolation of a quadratic field over a cell of
    // size 0.15 keeps the worst radial deviation near 0.003.
    for (const auto& p : m.vertices)
        EXPECT_NEAR(std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z), 1.0, 0.01);
}

TEST(GeoMarchingCubes, IndexedMeshExpandsToTheSameSoup) {
    const auto f = grid3(9, -1.5, 1.5, sphere_f);
    const auto soup = marching_cubes(f, 9, 9, 9, 0.0, {-1.5,-1.5,-1.5}, {0.375,0.375,0.375});
    const auto m = marching_cubes_mesh(f, 9, 9, 9, 0.0, {-1.5,-1.5,-1.5}, {0.375,0.375,0.375});
    ASSERT_EQ(soup.size(), m.triangles.size());
    EXPECT_LT(m.vertices.size(), 3 * m.triangles.size());   // vertices really are shared
    for (std::size_t t = 0; t < soup.size(); ++t) {
        const auto& idx = m.triangles[t];
        EXPECT_DOUBLE_EQ(soup[t].a.x, m.vertices[static_cast<std::size_t>(idx.a)].x);
        EXPECT_DOUBLE_EQ(soup[t].a.y, m.vertices[static_cast<std::size_t>(idx.a)].y);
        EXPECT_DOUBLE_EQ(soup[t].b.y, m.vertices[static_cast<std::size_t>(idx.b)].y);
        EXPECT_DOUBLE_EQ(soup[t].b.z, m.vertices[static_cast<std::size_t>(idx.b)].z);
        EXPECT_DOUBLE_EQ(soup[t].c.x, m.vertices[static_cast<std::size_t>(idx.c)].x);
        EXPECT_DOUBLE_EQ(soup[t].c.z, m.vertices[static_cast<std::size_t>(idx.c)].z);
    }
    EXPECT_DOUBLE_EQ(mesh_surface_area(soup), mesh_surface_area(m));
    EXPECT_DOUBLE_EQ(mesh_volume(soup), mesh_volume(m));
}

TEST(GeoMarchingCubes, NonFiniteSamplesSkipTheirCell) {
    std::vector<double> f(8, 1.0);
    f[0] = -1.0;
    f[7] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(marching_cubes(f, 2, 2, 2, 0.0).empty());   // the only cell is skipped

    // In a larger grid only the poisoned cells vanish and everything stays finite. Node index
    // 7 + 9*(4 + 9*4) = 367 is grid node (7,4,4), a node adjacent to the sphere, so the eight
    // cells around it really are cut and really do disappear.
    auto g = grid3(9, -1.5, 1.5, sphere_f);
    const auto clean = marching_cubes(g, 9, 9, 9, 0.0, {-1.5,-1.5,-1.5}, {0.375,0.375,0.375});
    g[367] = std::numeric_limits<double>::infinity();
    const auto dirty = marching_cubes(g, 9, 9, 9, 0.0, {-1.5,-1.5,-1.5}, {0.375,0.375,0.375});
    EXPECT_EQ(clean.size(), 248u);
    EXPECT_EQ(dirty.size(), 240u);   // a genuine hole, not a silent no-op
    EXPECT_TRUE(all_finite(dirty));

    // A NaN far from the surface changes nothing, because its cells were uncut anyway.
    auto q = grid3(9, -1.5, 1.5, sphere_f);
    q[0] = std::numeric_limits<double>::quiet_NaN();
    const auto corner = marching_cubes(q, 9, 9, 9, 0.0, {-1.5,-1.5,-1.5}, {0.375,0.375,0.375});
    EXPECT_EQ(corner.size(), clean.size());
    EXPECT_TRUE(all_finite(corner));
}

TEST(GeoMarchingCubes, OriginAndSpacingTransformTheSurface) {
    std::vector<double> f(8, 1.0);
    f[0] = -1.0;
    const auto tris = marching_cubes(f, 2, 2, 2, 0.0, {10.0, 20.0, 30.0}, {2.0, 4.0, 8.0});
    ASSERT_EQ(tris.size(), 1u);
    EXPECT_NEAR(tris[0].a.x, 11.0, 1e-12);
    EXPECT_NEAR(tris[0].a.y, 20.0, 1e-12);
    EXPECT_NEAR(tris[0].a.z, 30.0, 1e-12);
    EXPECT_NEAR(tris[0].b.x, 10.0, 1e-12);
    EXPECT_NEAR(tris[0].b.y, 22.0, 1e-12);
    EXPECT_NEAR(tris[0].b.z, 30.0, 1e-12);
    EXPECT_NEAR(tris[0].c.x, 10.0, 1e-12);
    EXPECT_NEAR(tris[0].c.y, 20.0, 1e-12);
    EXPECT_NEAR(tris[0].c.z, 34.0, 1e-12);
    EXPECT_NEAR(mesh_surface_area(tris), std::sqrt(21.0), 1e-12);   // 4.58257569495584
}

TEST(GeoMarchingCubes, DegenerateSpacingDoesNotCrash) {
    std::vector<double> f(8, 1.0);
    f[0] = -1.0;
    // Negative spacing mirrors the geometry; the area is unchanged.
    const auto mirrored = marching_cubes(f, 2, 2, 2, 0.0, {0.0,0.0,0.0}, {-1.0,-1.0,-1.0});
    ASSERT_EQ(mirrored.size(), 1u);
    EXPECT_NEAR(mirrored[0].a.x, -0.5, 1e-15);
    EXPECT_NEAR(mesh_surface_area(mirrored), std::sqrt(3.0) / 8.0, 1e-14);
    // Zero spacing along z collapses the cell into the z = 0 plane.
    const auto flat = marching_cubes(f, 2, 2, 2, 0.0, {0.0,0.0,0.0}, {1.0,1.0,0.0});
    ASSERT_EQ(flat.size(), 1u);
    EXPECT_NEAR(mesh_surface_area(flat), 0.125, 1e-15);
    EXPECT_TRUE(all_finite(flat));
}

TEST(GeoMarchingCubes, TwoDisjointSpheresGiveOneWatertightMeshWithSummedMeasures) {
    // f = min of two spheres of radius 0.5 centred at (+-1.5, 0, 0), on a 61^3 grid over
    // [-2.5, 2.5]^3 (h = 1/12).
    const int n = 61;
    const double lo = -2.5, hi = 2.5, h = (hi - lo) / (n - 1);
    std::vector<double> f(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) *
                          static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const double x = lo + i * h, y = lo + j * h, z = lo + k * h;
                const double a = (x - 1.5) * (x - 1.5) + y*y + z*z - 0.25;
                const double b = (x + 1.5) * (x + 1.5) + y*y + z*z - 0.25;
                f[static_cast<std::size_t>(i) + static_cast<std::size_t>(n) *
                  (static_cast<std::size_t>(j) + static_cast<std::size_t>(n) *
                   static_cast<std::size_t>(k))] = std::min(a, b);
            }
    const auto m = marching_cubes_mesh(f, n, n, n, 0.0, {lo, lo, lo}, {h, h, h});
    EXPECT_TRUE(is_watertight(m));
    EXPECT_NEAR(mesh_surface_area(m), 2.0 * 4.0 * M_PI * 0.25, 0.1);          // 6.283185
    EXPECT_NEAR(mesh_volume(m),       2.0 * 4.0 / 3.0 * M_PI * 0.125, 0.05);  // 1.047198
    EXPECT_GT(mesh_volume(m), 0.0);
}

TEST(GeoMarchingCubes, EveryCaseProducesTheEdgesItsCaseIndexImplies) {
    // Exercise all 256 cube cases through the public API and check that the triangles the table
    // produced use exactly the edges that the corner signs say are cut. Each case is fed as a
    // single 2x2x2 cell with corner values -1 (below) or +1 (above), so every cut edge is
    // bisected and its crossing point is a known cube-edge midpoint.
    static const int corner[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                     {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    static const int edge_corners[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},
                                            {6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    for (int c = 0; c < 256; ++c) {
        std::vector<double> f(8, 1.0);
        // Field index i + 2*(j + 2*k); corner b sits at (dx,dy,dz) above.
        for (int b = 0; b < 8; ++b)
            if ((c >> b) & 1)
                f[static_cast<std::size_t>(corner[b][0] +
                                           2 * (corner[b][1] + 2 * corner[b][2]))] = -1.0;
        const auto tris = marching_cubes(f, 2, 2, 2, 0.0);
        // Expected midpoints of the cut edges.
        std::vector<Point3D> expect;
        for (int e = 0; e < 12; ++e) {
            const int a = edge_corners[e][0], b2 = edge_corners[e][1];
            if (((c >> a) & 1) == ((c >> b2) & 1)) continue;
            expect.push_back({0.5 * (corner[a][0] + corner[b2][0]),
                              0.5 * (corner[a][1] + corner[b2][1]),
                              0.5 * (corner[a][2] + corner[b2][2])});
        }
        if (expect.empty()) {
            EXPECT_TRUE(tris.empty()) << "case " << c;
            continue;
        }
        EXPECT_FALSE(tris.empty()) << "case " << c;
        // Every emitted vertex must be one of the expected midpoints...
        for (const auto& t : tris) {
            for (const Point3D& p : {t.a, t.b, t.c}) {
                bool found = false;
                for (const auto& q : expect)
                    if (dist(p, q) < 1e-12) { found = true; break; }
                EXPECT_TRUE(found) << "case " << c << " emitted an unexpected vertex";
            }
        }
        // ...and every expected midpoint must be used at least once.
        for (const auto& q : expect) {
            bool found = false;
            for (const auto& t : tris) {
                for (const Point3D& p : {t.a, t.b, t.c})
                    if (dist(p, q) < 1e-12) { found = true; break; }
                if (found) break;
            }
            EXPECT_TRUE(found) << "case " << c << " left a cut edge unused";
        }
    }
}

TEST(GeoMarchingCubes, EveryCaseIsWatertightAgainstItsComplement) {
    // Two cells stacked along z, with the 12 shared/unshared corner signs enumerated
    // exhaustively, must weld into a mesh whose only unmatched directed edges lie on the outer
    // boundary of the 2x2x3 box. This is the crack-free property, checked over all 4096 sign
    // assignments rather than sampled.
    for (int bits = 0; bits < 4096; ++bits) {
        std::vector<double> f(12);
        for (int n = 0; n < 12; ++n)
            f[static_cast<std::size_t>(n)] = ((bits >> n) & 1) ? -1.0 : 1.0;
        const auto m = marching_cubes_mesh(f, 2, 2, 3, 0.0);
        std::map<std::pair<int,int>, int> d;
        for (const auto& t : m.triangles) {
            d[{t.a, t.b}] += 1;
            d[{t.b, t.c}] += 1;
            d[{t.c, t.a}] += 1;
        }
        for (const auto& kv : d) {
            EXPECT_EQ(kv.second, 1) << "bits " << bits;
            // The reverse may be absent only when the edge lies on the box boundary; the
            // interior plane z = 1 is the one this test is about, and every edge on it is
            // shared by both cells, so it must always be matched.
            const Point3D& p = m.vertices[static_cast<std::size_t>(kv.first.first)];
            const Point3D& q = m.vertices[static_cast<std::size_t>(kv.first.second)];
            if (std::abs(p.z - 1.0) < 1e-15 && std::abs(q.z - 1.0) < 1e-15) {
                auto it = d.find({kv.first.second, kv.first.first});
                ASSERT_NE(it, d.end()) << "crack at bits " << bits;
                EXPECT_EQ(it->second, 1) << "bits " << bits;
            }
        }
    }
}

// ---- Mesh measurements ----

TEST(GeoMeshMeasures, UnitCubeShellAndTetrahedron) {
    const std::vector<Point3D> v = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                    {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    TriMesh3D cube;
    cube.vertices = v;
    cube.triangles = {{0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},
                      {1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7}};   // outward
    EXPECT_TRUE(is_watertight(cube));
    EXPECT_NEAR(mesh_surface_area(cube), 6.0, 1e-14);
    EXPECT_NEAR(mesh_volume(cube), 1.0, 1e-14);

    TriMesh3D tetra;
    tetra.vertices = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    tetra.triangles = {{0,2,1},{0,1,3},{0,3,2},{1,2,3}};                  // outward
    EXPECT_NEAR(mesh_surface_area(tetra), 1.5 + std::sqrt(3.0) / 2.0, 1e-14);  // 2.36602540378
    EXPECT_NEAR(mesh_volume(tetra), 1.0 / 6.0, 1e-14);

    // Reversing the winding negates the volume and leaves the area alone.
    TriMesh3D flipped = cube;
    for (auto& t : flipped.triangles) std::swap(t.b, t.c);
    EXPECT_NEAR(mesh_surface_area(flipped), 6.0, 1e-14);
    EXPECT_NEAR(mesh_volume(flipped), -1.0, 1e-14);

    // Translating a closed mesh leaves the enclosed volume alone (the origin cancels).
    TriMesh3D moved = cube;
    for (auto& p : moved.vertices) { p.x += 1000.0; p.y -= 250.0; p.z += 7.0; }
    EXPECT_NEAR(mesh_volume(moved), 1.0, 1e-9);
}

TEST(GeoMeshMeasures, SoupOverloadsMatchTheIndexedOnes) {
    TriMesh3D tetra;
    tetra.vertices = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    tetra.triangles = {{0,2,1},{0,1,3},{0,3,2},{1,2,3}};
    std::vector<Triangle3D> soup;
    for (const auto& t : tetra.triangles)
        soup.push_back({tetra.vertices[static_cast<std::size_t>(t.a)],
                        tetra.vertices[static_cast<std::size_t>(t.b)],
                        tetra.vertices[static_cast<std::size_t>(t.c)]});
    EXPECT_DOUBLE_EQ(mesh_surface_area(soup), mesh_surface_area(tetra));
    EXPECT_DOUBLE_EQ(mesh_volume(soup), mesh_volume(tetra));

    std::vector<Triangle3D> reversed;
    for (const auto& t : soup) reversed.push_back({t.a, t.c, t.b});
    EXPECT_DOUBLE_EQ(mesh_volume(reversed), -mesh_volume(soup));
    EXPECT_DOUBLE_EQ(mesh_surface_area(reversed), mesh_surface_area(soup));
}

TEST(GeoMeshMeasures, EmptyAndOutOfRangeAreQuiet) {
    EXPECT_DOUBLE_EQ(mesh_surface_area(std::vector<Triangle3D>{}), 0.0);
    EXPECT_DOUBLE_EQ(mesh_volume(std::vector<Triangle3D>{}), 0.0);
    TriMesh3D empty;
    EXPECT_DOUBLE_EQ(mesh_surface_area(empty), 0.0);
    EXPECT_DOUBLE_EQ(mesh_volume(empty), 0.0);

    TriMesh3D bad;
    bad.vertices = {{0,0,0}};
    bad.triangles = {{0, 5, 9}, {-1, 0, 0}};
    EXPECT_DOUBLE_EQ(mesh_surface_area(bad), 0.0);
    EXPECT_DOUBLE_EQ(mesh_volume(bad), 0.0);

    // A mix: the valid triangle is measured, the out-of-range one is skipped.
    TriMesh3D mixed;
    mixed.vertices = {{0,0,0},{1,0,0},{0,1,0}};
    mixed.triangles = {{0,1,2}, {0,1,7}};
    EXPECT_NEAR(mesh_surface_area(mixed), 0.5, 1e-15);
}

// ---- Marching squares ----

TEST(GeoMarchingSquares, CircleFieldPerimeterApproachesTwoPiR) {
    {   // 41x41 grid over [-1.5, 1.5]^2, h = 0.075, r = 1
        const auto f = grid2(41, -1.5, 1.5, circle_f);
        const auto segs = marching_squares(f, 41, 41, 0.0, {-1.5, -1.5}, {0.075, 0.075});
        EXPECT_EQ(segs.size(), 108u);
        EXPECT_NEAR(contour_length(segs), 6.278805, 1e-4);              // measured
        EXPECT_LT(std::abs(contour_length(segs) / (2.0 * M_PI) - 1.0), 0.002);
        // Positive => the disc is on the left of every segment (CCW contour).
        EXPECT_NEAR(contour_signed_area(segs), 3.135772, 1e-4);         // measured
        EXPECT_LT(std::abs(contour_signed_area(segs) / M_PI - 1.0), 0.005);
    }
    {   // 101x101 grid over [-1.5, 1.5]^2, h = 0.03: error shrinks with h
        const auto f = grid2(101, -1.5, 1.5, circle_f);
        const auto segs = marching_squares(f, 101, 101, 0.0, {-1.5, -1.5}, {0.03, 0.03});
        EXPECT_EQ(segs.size(), 268u);
        EXPECT_NEAR(contour_length(segs), 6.282488, 1e-4);              // measured
        EXPECT_LT(std::abs(contour_length(segs) / (2.0 * M_PI) - 1.0), 0.0005);
        EXPECT_NEAR(contour_signed_area(segs), 3.140655, 1e-4);         // measured
        EXPECT_LT(std::abs(contour_signed_area(segs) / M_PI - 1.0), 0.001);
    }
}

TEST(GeoMarchingSquares, CircleContourIsAClosedLoop) {
    const auto f = grid2(41, -1.5, 1.5, circle_f);
    const auto segs = marching_squares(f, 41, 41, 0.0, {-1.5, -1.5}, {0.075, 0.075});
    // Every endpoint is the tail of exactly one segment and the head of exactly one: the
    // segments chain into closed loops, with no cracks between neighbouring cells. Endpoints
    // are compared bitwise, which is only sound because both cells sharing a grid edge
    // interpolate from the same canonical endpoint order.
    std::map<std::pair<double,double>, int> out_deg, in_deg;
    for (const auto& g : segs) {
        out_deg[{g.a.x, g.a.y}] += 1;
        in_deg[{g.b.x, g.b.y}] += 1;
    }
    ASSERT_FALSE(out_deg.empty());
    EXPECT_EQ(out_deg.size(), in_deg.size());
    for (const auto& kv : out_deg) {
        EXPECT_EQ(kv.second, 1);
        auto it = in_deg.find(kv.first);
        ASSERT_NE(it, in_deg.end());
        EXPECT_EQ(it->second, 1);
    }
}

TEST(GeoMarchingSquares, NegatingTheFieldReversesTheContour) {
    // {-f < -iso} is the exact complement of {f < iso} when no sample sits on the level, so the
    // contour is the same curve traversed backwards and the two shoelace sums cancel exactly.
    const int n = 25;
    const double h = 3.0 / (n - 1);
    const auto f = grid2(n, -1.5, 1.5, circle_f);
    const auto g = grid2(n, -1.5, 1.5, neg_circle_f);
    const auto a = marching_squares(f, n, n, 0.0, {-1.5, -1.5}, {h, h});
    const auto b = marching_squares(g, n, n, 0.0, {-1.5, -1.5}, {h, h});
    ASSERT_FALSE(a.empty());
    ASSERT_FALSE(b.empty());
    EXPECT_GT(contour_signed_area(a), 0.0);
    EXPECT_LT(contour_signed_area(b), 0.0);
    EXPECT_NEAR(contour_signed_area(a) + contour_signed_area(b), 0.0, 1e-12);
    EXPECT_NEAR(contour_length(a), contour_length(b), 1e-12);
}

TEST(GeoMarchingSquares, SingleCellCasesAndSegmentDirection) {
    // Row-major 2x2 field: field[0]=c0(0,0), field[1]=c1(1,0), field[2]=c3(0,1), field[3]=c2(1,1).
    {   // Case 1: only corner 0 below the level -> one segment e0 -> e3.
        const std::vector<double> f = {-1.0, 1.0, 1.0, 1.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 1u);
        EXPECT_NEAR(s[0].a.x, 0.5, 1e-15);
        EXPECT_NEAR(s[0].a.y, 0.0, 1e-15);
        EXPECT_NEAR(s[0].b.x, 0.0, 1e-15);
        EXPECT_NEAR(s[0].b.y, 0.5, 1e-15);
        // Interior on the left: rotating (b - a) by +90 degrees must point at corner 0.
        const Vec2D d = vec2(s[0].a, s[0].b);
        const Vec2D left{-d.y, d.x};
        EXPECT_LT(left.x, 0.0);
        EXPECT_LT(left.y, 0.0);
        EXPECT_NEAR(contour_length(s), std::sqrt(0.5), 1e-15);
    }
    {   // Case 3: corners 0 and 1 below -> the bottom half is solid; one segment e1 -> e3.
        const std::vector<double> f = {-1.0, -1.0, 1.0, 1.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 1u);
        EXPECT_NEAR(s[0].a.x, 1.0, 1e-15);
        EXPECT_NEAR(s[0].a.y, 0.5, 1e-15);
        EXPECT_NEAR(s[0].b.x, 0.0, 1e-15);
        EXPECT_NEAR(s[0].b.y, 0.5, 1e-15);
    }
    {   // Complementary cases are the same segment reversed: case 14 complements case 1.
        const std::vector<double> f = {1.0, -1.0, -1.0, -1.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 1u);
        EXPECT_NEAR(s[0].a.x, 0.0, 1e-15);
        EXPECT_NEAR(s[0].a.y, 0.5, 1e-15);
        EXPECT_NEAR(s[0].b.x, 0.5, 1e-15);
        EXPECT_NEAR(s[0].b.y, 0.0, 1e-15);
    }
    {   // All four corners on the same side of the level: nothing at all.
        EXPECT_TRUE(marching_squares(std::vector<double>(4, -1.0), 2, 2, 0.0).empty());
        EXPECT_TRUE(marching_squares(std::vector<double>(4,  1.0), 2, 2, 0.0).empty());
    }
}

TEST(GeoMarchingSquares, AllTwelveUnambiguousCasesMatchTheReferenceTable) {
    // Corner order in the flat field is {c0, c1, c3, c2}; below-corner sets are the case index.
    // Reference segments (unit cell, midpoint crossings), from the marching-squares case table.
    struct Ref { int mask; double ax, ay, bx, by; };
    static const Ref ref[] = {
        { 1, 0.5, 0.0, 0.0, 0.5}, { 2, 1.0, 0.5, 0.5, 0.0},
        { 3, 1.0, 0.5, 0.0, 0.5}, { 4, 0.5, 1.0, 1.0, 0.5},
        { 6, 0.5, 1.0, 0.5, 0.0}, { 7, 0.5, 1.0, 0.0, 0.5},
        { 8, 0.0, 0.5, 0.5, 1.0}, { 9, 0.5, 0.0, 0.5, 1.0},
        {11, 1.0, 0.5, 0.5, 1.0}, {12, 0.0, 0.5, 1.0, 0.5},
        {13, 0.5, 0.0, 1.0, 0.5}, {14, 0.0, 0.5, 0.5, 0.0}
    };
    static const int slot[4] = {0, 1, 3, 2};   // corner c -> flat field index
    for (const auto& r : ref) {
        std::vector<double> f(4, 1.0);
        for (int c = 0; c < 4; ++c)
            if ((r.mask >> c) & 1) f[static_cast<std::size_t>(slot[c])] = -1.0;
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 1u) << "case " << r.mask;
        EXPECT_NEAR(s[0].a.x, r.ax, 1e-15) << "case " << r.mask;
        EXPECT_NEAR(s[0].a.y, r.ay, 1e-15) << "case " << r.mask;
        EXPECT_NEAR(s[0].b.x, r.bx, 1e-15) << "case " << r.mask;
        EXPECT_NEAR(s[0].b.y, r.by, 1e-15) << "case " << r.mask;
    }
}

TEST(GeoMarchingSquares, AsymptoticDeciderResolvesTheDiagonalCases) {
    // Case 5 (corners 0 and 2 below), row-major {c0, c1, c3, c2}.
    // f0=-2, f1=1, f2=-2, f3=1 -> saddle = (f0*f2 - f1*f3)/(f0-f1+f2-f3) = (4-1)/(-6) = -0.5 < 0
    // -> the cell centre is solid, so 0 and 2 are JOINED: segments e0->e1 and e2->e3.
    {
        const std::vector<double> f = {-2.0, 1.0, 1.0, -2.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 2u);
        EXPECT_NEAR(s[0].a.x, 2.0/3.0, 1e-12); EXPECT_NEAR(s[0].a.y, 0.0,     1e-12);
        EXPECT_NEAR(s[0].b.x, 1.0,     1e-12); EXPECT_NEAR(s[0].b.y, 1.0/3.0, 1e-12);
        EXPECT_NEAR(s[1].a.x, 1.0/3.0, 1e-12); EXPECT_NEAR(s[1].a.y, 1.0,     1e-12);
        EXPECT_NEAR(s[1].b.x, 0.0,     1e-12); EXPECT_NEAR(s[1].b.y, 2.0/3.0, 1e-12);
    }
    // Same case, saddle on the other side: f0=-1, f1=2, f2=-1, f3=2 -> (1-4)/(-6) = +0.5 >= 0
    // -> centre is outside, so 0 and 2 are SEPARATED: segments e0->e3 and e2->e1.
    {
        const std::vector<double> f = {-1.0, 2.0, 2.0, -1.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 2u);
        EXPECT_NEAR(s[0].a.x, 1.0/3.0, 1e-12); EXPECT_NEAR(s[0].a.y, 0.0,     1e-12);
        EXPECT_NEAR(s[0].b.x, 0.0,     1e-12); EXPECT_NEAR(s[0].b.y, 1.0/3.0, 1e-12);
        EXPECT_NEAR(s[1].a.x, 2.0/3.0, 1e-12); EXPECT_NEAR(s[1].a.y, 1.0,     1e-12);
        EXPECT_NEAR(s[1].b.x, 1.0,     1e-12); EXPECT_NEAR(s[1].b.y, 2.0/3.0, 1e-12);
    }
    // Case 10 (corners 1 and 3 below), joined: f0=1, f1=-2, f2=1, f3=-2 -> saddle = -0.5 < 0.
    {
        const std::vector<double> f = {1.0, -2.0, -2.0, 1.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 2u);
        EXPECT_NEAR(s[0].a.x, 0.0,     1e-12); EXPECT_NEAR(s[0].a.y, 1.0/3.0, 1e-12);
        EXPECT_NEAR(s[0].b.x, 1.0/3.0, 1e-12); EXPECT_NEAR(s[0].b.y, 0.0,     1e-12);
        EXPECT_NEAR(s[1].a.x, 1.0,     1e-12); EXPECT_NEAR(s[1].a.y, 2.0/3.0, 1e-12);
        EXPECT_NEAR(s[1].b.x, 2.0/3.0, 1e-12); EXPECT_NEAR(s[1].b.y, 1.0,     1e-12);
    }
    // Case 10, separated: f0=2, f1=-1, f2=2, f3=-1 -> saddle = +0.5.
    {
        const std::vector<double> f = {2.0, -1.0, -1.0, 2.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 2u);
        EXPECT_NEAR(s[0].a.x, 1.0,     1e-12); EXPECT_NEAR(s[0].a.y, 1.0/3.0, 1e-12);
        EXPECT_NEAR(s[0].b.x, 2.0/3.0, 1e-12); EXPECT_NEAR(s[0].b.y, 0.0,     1e-12);
        EXPECT_NEAR(s[1].a.x, 0.0,     1e-12); EXPECT_NEAR(s[1].a.y, 2.0/3.0, 1e-12);
        EXPECT_NEAR(s[1].b.x, 1.0/3.0, 1e-12); EXPECT_NEAR(s[1].b.y, 1.0,     1e-12);
    }
    // Perfectly symmetric corners make the saddle denominator zero; the fallback is the mean of
    // the four corners, which here is 0 and therefore not below the level -> SEPARATED.
    {
        const std::vector<double> f = {-1.0, 1.0, 1.0, -1.0};
        const auto s = marching_squares(f, 2, 2, 0.0);
        ASSERT_EQ(s.size(), 2u);
        EXPECT_NEAR(s[0].a.x, 0.5, 1e-15); EXPECT_NEAR(s[0].a.y, 0.0, 1e-15);
        EXPECT_NEAR(s[0].b.x, 0.0, 1e-15); EXPECT_NEAR(s[0].b.y, 0.5, 1e-15);
    }
}

TEST(GeoMarchingSquares, LinearFieldAndDegenerateInputs) {
    // f = y - 0.5 on a 3x3 grid over the unit square: the contour is the exact segment y = 0.5,
    // total length exactly 1.
    const auto f = grid2(3, 0.0, 1.0, hline_f);
    const auto segs = marching_squares(f, 3, 3, 0.0, {0.0, 0.0}, {0.5, 0.5});
    EXPECT_EQ(segs.size(), 2u);
    EXPECT_NEAR(contour_length(segs), 1.0, 1e-14);
    for (const auto& g : segs) {
        EXPECT_NEAR(g.a.y, 0.5, 1e-15);
        EXPECT_NEAR(g.b.y, 0.5, 1e-15);
        EXPECT_LT(g.b.x, g.a.x);   // solid (y < 0.5) on the left => travelling in -x
    }

    EXPECT_TRUE(marching_squares({}, 0, 0, 0.0).empty());
    EXPECT_TRUE(marching_squares(std::vector<double>(2, 1.0), 1, 2, 0.0).empty());
    EXPECT_TRUE(marching_squares(std::vector<double>(2, 1.0), 2, 1, 0.0).empty());
    EXPECT_TRUE(marching_squares(std::vector<double>(3, 1.0), 2, 2, 0.0).empty());  // size mismatch
    const auto c = grid2(9, -1.5, 1.5, circle_f);
    EXPECT_TRUE(marching_squares(c, 9, 9,  100.0).empty());
    EXPECT_TRUE(marching_squares(c, 9, 9, -100.0).empty());
    EXPECT_TRUE(marching_squares(c, 9, 9,   -1.0).empty());  // iso == min, strict `<`
    std::vector<double> nanf(4, 1.0);
    nanf[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(marching_squares(nanf, 2, 2, 0.0).empty());
    std::vector<double> inff(4, -1.0);
    inff[3] = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(marching_squares(inff, 2, 2, 0.0).empty());
}

TEST(GeoMarchingSquares, OriginAndSpacingTransformTheContour) {
    const std::vector<double> f = {-1.0, 1.0, 1.0, 1.0};
    const auto s = marching_squares(f, 2, 2, 0.0, {10.0, 20.0}, {4.0, 6.0});
    ASSERT_EQ(s.size(), 1u);
    EXPECT_NEAR(s[0].a.x, 12.0, 1e-12);
    EXPECT_NEAR(s[0].a.y, 20.0, 1e-12);
    EXPECT_NEAR(s[0].b.x, 10.0, 1e-12);
    EXPECT_NEAR(s[0].b.y, 23.0, 1e-12);
    EXPECT_NEAR(contour_length(s), std::sqrt(4.0 + 9.0), 1e-12);
}
