// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: two closed-form geometry routines whose VALUE nothing asserted.
//
// The 24-mutant sample over `src/geo/geo.cpp` at seed 89 leaves two survivors
// outside the marching-cubes table, and both are formulas of the kind a shape
// assertion cannot see:
//
//   circumcenter            one `by - cy` became `by + cy`, moving the centre of
//                           the circumscribed circle. Reached through `voronoi`,
//                           whose vertices ARE the circumcentres -- and nothing
//                           checked where they are.
//   dist_point_segment3     the projection parameter t = (ap.ab)/|ab|^2 had one
//                           `+` in its dot product become a `-`. That changes
//                           where along the segment the closest point sits, and
//                           it is invisible whenever the term it corrupts is
//                           zero -- which it is for any segment along an axis
//                           with the point offset along a different one.
//
// A circumcentre is defined by being equidistant from three points, and a
// point-segment distance by being the minimum over the segment. Both are checked
// here against their definitions rather than against remembered numbers.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/geo/geo.hpp"

using namespace ms::geo;

namespace {

double dist2(const Point2D& a, const Point2D& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace

TEST(GeoFormulaValues, VoronoiVerticesAreEquidistantFromTheirThreePoints) {
    // The defining property: a circumcentre is the one point equally far from all
    // three corners. Checked over several configurations, including ones where
    // the centre falls outside the triangle.
    const std::vector<std::vector<Point2D>> inputs{
        {{0.0, 0.0}, {4.0, 0.0}, {0.0, 3.0}, {4.0, 3.0}},
        {{0.0, 0.0}, {6.0, 1.0}, {2.0, 5.0}, {7.0, 6.0}, {3.0, 2.0}},
        {{-3.0, -2.0}, {5.0, -1.0}, {1.0, 7.0}, {-2.0, 4.0}, {6.0, 5.0}, {0.5, 0.25}},
    };
    for (const auto& pts : inputs) {
        const auto tris = delaunay_2d(pts);
        const auto centers = voronoi(pts);
        ASSERT_EQ(centers.size(), tris.size()) << "one Voronoi vertex per triangle";
        ASSERT_FALSE(tris.empty());

        for (std::size_t i = 0; i < tris.size(); ++i) {
            const Point2D& a = pts[static_cast<std::size_t>(tris[i].a)];
            const Point2D& b = pts[static_cast<std::size_t>(tris[i].b)];
            const Point2D& c = pts[static_cast<std::size_t>(tris[i].c)];
            const Point2D& o = centers[i];

            const double ra = dist2(o, a);
            const double rb = dist2(o, b);
            const double rc = dist2(o, c);
            const double scale = std::max({1.0, ra, rb, rc});
            EXPECT_NEAR(ra, rb, scale * 1e-9)
                << "triangle " << i << ": circumcentre is not equidistant from a and b";
            EXPECT_NEAR(ra, rc, scale * 1e-9)
                << "triangle " << i << ": circumcentre is not equidistant from a and c";
            EXPECT_TRUE(std::isfinite(o.x) && std::isfinite(o.y));
        }
    }

    // A right triangle's circumcentre is the midpoint of its hypotenuse -- the
    // one configuration where the answer can be written down by hand.
    const std::vector<Point2D> right{{0.0, 0.0}, {6.0, 0.0}, {0.0, 8.0}};
    const auto centers = voronoi(right);
    ASSERT_EQ(centers.size(), 1u);
    EXPECT_NEAR(centers[0].x, 3.0, 1e-9);
    EXPECT_NEAR(centers[0].y, 4.0, 1e-9);
}

TEST(GeoFormulaValues, PointSegmentDistanceInThreeDimensions) {
    // Each case is a distance that can be written down, and between them they
    // make every term of the dot product `ap.ab` carry the answer at least once:
    // a segment along x with the point offset in y, along y with the offset in z,
    // along z with the offset in x, and a slanted one where all three contribute.
    struct Case {
        Point3D p;
        Segment3D s;
        double want;
        const char* what;
    };
    const Case cases[] = {
        {{1.0, 3.0, 0.0}, {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}, 3.0, "x-segment, y offset"},
        {{0.0, 1.0, 5.0}, {{0.0, 0.0, 0.0}, {0.0, 4.0, 0.0}}, 5.0, "y-segment, z offset"},
        {{7.0, 0.0, 2.0}, {{0.0, 0.0, 0.0}, {0.0, 0.0, 6.0}}, 7.0, "z-segment, x offset"},
        // Beyond an endpoint: the closest point is the endpoint itself.
        {{5.0, 0.0, 0.0}, {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}, 3.0, "past the far end"},
        {{-4.0, 0.0, 0.0}, {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}, 4.0, "before the near end"},
        // On the segment.
        {{1.0, 0.0, 0.0}, {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}, 0.0, "on the segment"},
        // Degenerate segment: the distance to the point it collapses to.
        {{3.0, 4.0, 0.0}, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, 5.0, "degenerate segment"},
    };
    for (const Case& c : cases) {
        EXPECT_NEAR(dist_point_segment3(c.p, c.s), c.want, 1e-12) << c.what;
    }

    // A slanted segment, checked against a direct minimisation over it: the
    // formula and a sampled minimum have to agree, and the sampling knows
    // nothing about the projection.
    const Segment3D slant{{-1.0, 2.0, 0.5}, {3.0, -1.0, 4.5}};
    for (const Point3D& p : {Point3D{0.0, 0.0, 0.0}, Point3D{2.0, 2.0, 2.0},
                             Point3D{-5.0, 4.0, 1.0}, Point3D{10.0, -3.0, 7.0}}) {
        double best = 1e300;
        for (int i = 0; i <= 200000; ++i) {
            const double t = static_cast<double>(i) / 200000.0;
            const double x = slant.a.x + t * (slant.b.x - slant.a.x) - p.x;
            const double y = slant.a.y + t * (slant.b.y - slant.a.y) - p.y;
            const double z = slant.a.z + t * (slant.b.z - slant.a.z) - p.z;
            best = std::min(best, std::sqrt(x * x + y * y + z * z));
        }
        EXPECT_NEAR(dist_point_segment3(p, slant), best, 1e-6)
            << "slanted segment from (" << p.x << ", " << p.y << ", " << p.z << ")";
    }
}
