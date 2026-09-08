// General (non-convex) polygon booleans.
//
// The pre-existing poly_union / poly_intersect / poly_diff helpers are documented as convex
// MVPs: when the true result is non-convex they return a convex-hull over-approximation.
// poly_boolean is the general clipper, so these tests pin the cases the convex helpers
// cannot express -- concave operands, results that split into several disjoint pieces, and
// results with holes -- against areas derived independently of the implementation.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "ms/geo/geo.hpp"

namespace {

using ms::geo::area;
using ms::geo::BooleanOp;
using ms::geo::Point2D;
using ms::geo::Polygon2D;
using ms::geo::PolygonSet;
using ms::geo::poly_boolean;
using ms::geo::poly_set_area;
using ms::geo::signed_area;

// Riemann sum of the indicator of a set predicate over a box: an area estimate that shares
// no code with poly_boolean. The error is O(h * perimeter).
double sampled_area(double lo, double hi, int n, const std::function<bool(Point2D)>& inside) {
    const double h = (hi - lo) / n;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (inside({lo + (i + 0.5) * h, lo + (j + 0.5) * h})) acc += h * h;
        }
    }
    return acc;
}

// A star-shaped (hence simple) polygon about (cx, cy): sorted angles, random radii.
Polygon2D random_star(std::mt19937& rng, double cx, double cy, int n) {
    std::uniform_real_distribution<double> radius(1.0, 4.0);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * M_PI);
    std::vector<double> a(static_cast<std::size_t>(n));
    for (auto& v : a) v = angle(rng);
    std::sort(a.begin(), a.end());
    Polygon2D p;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i > 0 && a[i] - a[i - 1] < 1e-3) continue;  // avoid coincident vertices
        const double r = radius(rng);
        p.push_back({cx + r * std::cos(a[i]), cy + r * std::sin(a[i])});
    }
    return p;
}

bool is_simple(const Polygon2D& p) {
    const int n = static_cast<int>(p.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (j == i + 1 || (i == 0 && j == n - 1)) continue;
            if (ms::geo::intersect_seg_seg({p[i], p[(i + 1) % n]}, {p[j], p[(j + 1) % n]})) {
                return false;
            }
        }
    }
    return true;
}

const Polygon2D kSquareA{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
const Polygon2D kSquareB{{2, 2}, {6, 2}, {6, 6}, {2, 6}};

// An L: the 6x2 foot plus the 2x4 upright, 20 units of area, concave at (2,2).
const Polygon2D kEll{{0, 0}, {6, 0}, {6, 2}, {2, 2}, {2, 6}, {0, 6}};

}  // namespace

TEST(GeoPolyBooleanGeneral, OverlappingSquaresAreExactNotHulled) {
    // |A| = |B| = 16 and the overlap is the 2x2 square [2,4]^2, so the union is 28. The
    // union is an L-shaped octagon, i.e. non-convex, which is exactly what the convex
    // poly_union cannot represent: it returns the hull, of area 32.
    const auto u = ms::geo::poly_union_general(kSquareA, kSquareB);
    ASSERT_EQ(u.size(), 1u);
    EXPECT_NEAR(poly_set_area(u), 28.0, 1e-12);
    EXPECT_GT(area(ms::geo::poly_union(kSquareA, kSquareB)), 28.0 + 1e-9)
        << "the convex helper is supposed to over-approximate here";

    const auto i = ms::geo::poly_intersect_general(kSquareA, kSquareB);
    ASSERT_EQ(i.size(), 1u);
    EXPECT_NEAR(poly_set_area(i), 4.0, 1e-12);

    const auto d = ms::geo::poly_diff_general(kSquareA, kSquareB);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_NEAR(poly_set_area(d), 12.0, 1e-12);

    // A xor B is two disjoint L-shaped pieces of 12 each: one polygon cannot hold it.
    const auto x = ms::geo::poly_symmetric_diff_general(kSquareA, kSquareB);
    EXPECT_EQ(x.size(), 2u);
    EXPECT_NEAR(poly_set_area(x), 24.0, 1e-12);
}

TEST(GeoPolyBooleanGeneral, DifferenceOfNestedSquaresIsAnAnnulusWithAHole) {
    const Polygon2D big{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    const Polygon2D small{{3, 3}, {7, 3}, {7, 7}, {3, 7}};

    const auto d = ms::geo::poly_diff_general(big, small);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_NEAR(poly_set_area(d), 100.0 - 16.0, 1e-12);

    // Exactly one shell (CCW, positive) and one hole (CW, negative).
    int shells = 0;
    int holes = 0;
    for (const auto& c : d) {
        if (signed_area(c) > 0.0) ++shells; else ++holes;
    }
    EXPECT_EQ(shells, 1);
    EXPECT_EQ(holes, 1);

    // Membership must respect the hole.
    EXPECT_TRUE(ms::geo::point_in_polygon_set({1.0, 1.0}, d));
    EXPECT_FALSE(ms::geo::point_in_polygon_set({5.0, 5.0}, d)) << "the hole is not in the set";
    EXPECT_FALSE(ms::geo::point_in_polygon_set({-1.0, -1.0}, d));

    EXPECT_NEAR(poly_set_area(ms::geo::poly_union_general(big, small)), 100.0, 1e-12);
    EXPECT_NEAR(poly_set_area(ms::geo::poly_intersect_general(big, small)), 16.0, 1e-12);
}

TEST(GeoPolyBooleanGeneral, UnionCanCreateAHole) {
    // A U-shape closed off by a lid traps a 2x3 pocket. |U| = 36 - 8 = 28, |lid| = 8, and
    // they overlap in two 1x1 tabs, so the union is 28 + 8 - 2 = 34 with a hole of 6.
    const Polygon2D u_shape{{0, 0}, {6, 0}, {6, 6}, {4, 6}, {4, 2}, {2, 2}, {2, 6}, {0, 6}};
    const Polygon2D lid{{1, 5}, {5, 5}, {5, 7}, {1, 7}};
    ASSERT_NEAR(area(u_shape), 28.0, 1e-12);

    const auto u = ms::geo::poly_union_general(u_shape, lid);
    ASSERT_EQ(u.size(), 2u);
    EXPECT_NEAR(poly_set_area(u), 34.0, 1e-12);

    double hole = 0.0;
    for (const auto& c : u) {
        if (signed_area(c) < 0.0) hole = -signed_area(c);
    }
    EXPECT_NEAR(hole, 6.0, 1e-12);
    EXPECT_FALSE(ms::geo::point_in_polygon_set({3.0, 3.5}, u)) << "the trapped pocket";
    EXPECT_TRUE(ms::geo::point_in_polygon_set({3.0, 6.5}, u)) << "under the lid";
}

TEST(GeoPolyBooleanGeneral, ConcaveOperandSplitsTheResult) {
    // A bar across the L's foot cuts the difference into two pieces: the 6 x 0.5 sliver
    // below it (area 3) and the remaining 11 above it.
    const Polygon2D bar{{-1, 0.5}, {7, 0.5}, {7, 1.5}, {-1, 1.5}};

    const auto d = ms::geo::poly_diff_general(kEll, bar);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_NEAR(poly_set_area(d), 14.0, 1e-12);

    std::vector<double> areas;
    for (const auto& c : d) areas.push_back(signed_area(c));
    std::sort(areas.begin(), areas.end());
    EXPECT_NEAR(areas[0], 3.0, 1e-12);
    EXPECT_NEAR(areas[1], 11.0, 1e-12);

    EXPECT_NEAR(poly_set_area(ms::geo::poly_intersect_general(kEll, bar)), 6.0, 1e-12);
    EXPECT_NEAR(poly_set_area(ms::geo::poly_union_general(kEll, bar)), 22.0, 1e-12);
    EXPECT_NEAR(poly_set_area(ms::geo::poly_symmetric_diff_general(kEll, bar)), 16.0, 1e-12);
}

TEST(GeoPolyBooleanGeneral, DisjointOperands) {
    const Polygon2D a{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const Polygon2D b{{5, 5}, {6, 5}, {6, 6}, {5, 6}};

    const auto u = ms::geo::poly_union_general(a, b);
    EXPECT_EQ(u.size(), 2u);
    EXPECT_NEAR(poly_set_area(u), 2.0, 1e-12);
    EXPECT_TRUE(ms::geo::poly_intersect_general(a, b).empty());

    const auto d = ms::geo::poly_diff_general(a, b);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_NEAR(poly_set_area(d), 1.0, 1e-12);
}

TEST(GeoPolyBooleanGeneral, IdenticalOperands) {
    EXPECT_NEAR(poly_set_area(ms::geo::poly_union_general(kSquareA, kSquareA)), 16.0, 1e-12);
    EXPECT_NEAR(poly_set_area(ms::geo::poly_intersect_general(kSquareA, kSquareA)), 16.0, 1e-12);
    EXPECT_TRUE(ms::geo::poly_diff_general(kSquareA, kSquareA).empty());
    EXPECT_TRUE(ms::geo::poly_symmetric_diff_general(kSquareA, kSquareA).empty());
}

TEST(GeoPolyBooleanGeneral, OperandsSharingAWholeEdge) {
    // Two unit-of-area squares meeting along x = 2. The union is one 4x2 rectangle and the
    // intersection is the shared edge, which has no area and must not come back as a sliver.
    const Polygon2D t1{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
    const Polygon2D t2{{2, 0}, {4, 0}, {4, 2}, {2, 2}};

    const auto u = ms::geo::poly_union_general(t1, t2);
    ASSERT_EQ(u.size(), 1u);
    EXPECT_NEAR(poly_set_area(u), 8.0, 1e-12);
    EXPECT_TRUE(ms::geo::poly_intersect_general(t1, t2).empty());
    EXPECT_NEAR(poly_set_area(ms::geo::poly_diff_general(t1, t2)), 4.0, 1e-12);
}

TEST(GeoPolyBooleanGeneral, WindingOfTheOperandsDoesNotMatter) {
    Polygon2D cw_a = kSquareA;
    Polygon2D cw_b = kSquareB;
    std::reverse(cw_a.begin(), cw_a.end());
    std::reverse(cw_b.begin(), cw_b.end());
    ASSERT_LT(signed_area(cw_a), 0.0);

    for (const auto& a : {kSquareA, cw_a}) {
        for (const auto& b : {kSquareB, cw_b}) {
            EXPECT_NEAR(poly_set_area(ms::geo::poly_union_general(a, b)), 28.0, 1e-12);
            EXPECT_NEAR(poly_set_area(ms::geo::poly_intersect_general(a, b)), 4.0, 1e-12);
            EXPECT_NEAR(poly_set_area(ms::geo::poly_diff_general(a, b)), 12.0, 1e-12);
        }
    }
}

TEST(GeoPolyBooleanGeneral, DegenerateOperandsFollowSetAlgebra) {
    const Polygon2D empty;
    const Polygon2D segment{{0, 0}, {1, 1}};
    const Polygon2D collinear{{0, 0}, {1, 1}, {2, 2}};  // three vertices but zero area

    for (const auto& degenerate : {empty, segment, collinear}) {
        // A u nothing = A, A n nothing = nothing, A \ nothing = A.
        EXPECT_NEAR(poly_set_area(ms::geo::poly_union_general(kSquareA, degenerate)), 16.0, 1e-12);
        EXPECT_TRUE(ms::geo::poly_intersect_general(kSquareA, degenerate).empty());
        EXPECT_NEAR(poly_set_area(ms::geo::poly_diff_general(kSquareA, degenerate)), 16.0, 1e-12);
        // nothing \ A = nothing, and the degenerate operand never contributes area.
        EXPECT_TRUE(ms::geo::poly_diff_general(degenerate, kSquareA).empty());
        EXPECT_NEAR(poly_set_area(ms::geo::poly_union_general(degenerate, kSquareA)), 16.0, 1e-12);
    }
    EXPECT_TRUE(ms::geo::poly_union_general(empty, empty).empty());
    EXPECT_DOUBLE_EQ(poly_set_area({}), 0.0);
    EXPECT_FALSE(ms::geo::point_in_polygon_set({0.0, 0.0}, {}));
}

TEST(GeoPolyBooleanGeneral, ShellsAreCcwAndHolesAreCw) {
    const Polygon2D big{{0, 0}, {12, 0}, {12, 12}, {0, 12}};
    const Polygon2D ring{{2, 2}, {10, 2}, {10, 10}, {2, 10}};
    const auto d = ms::geo::poly_diff_general(big, ring);
    ASSERT_EQ(d.size(), 2u);
    // poly_set_area is the plain sum of signed areas, so the sign convention has to hold for
    // it to equal the true area.
    EXPECT_NEAR(poly_set_area(d), 144.0 - 64.0, 1e-12);
    EXPECT_GT(poly_set_area(d), 0.0);
}

TEST(GeoPolyBooleanGeneral, InclusionExclusionHoldsOnRandomSimplePolygons) {
    // |A u B| + |A n B| == |A| + |B| and |A xor B| == |A u B| - |A n B| are identities the
    // implementation never uses, so they are a genuine cross-check.
    std::mt19937 rng(20260908u);
    int checked = 0;
    for (int trial = 0; trial < 200 && checked < 60; ++trial) {
        const Polygon2D a = random_star(rng, 0.0, 0.0, 7);
        const Polygon2D b = random_star(rng, 1.5, 1.0, 7);
        if (a.size() < 3 || b.size() < 3) continue;
        if (!is_simple(a) || !is_simple(b)) continue;  // self-intersecting inputs are out of scope
        ++checked;

        const double au = poly_set_area(ms::geo::poly_union_general(a, b));
        const double ai = poly_set_area(ms::geo::poly_intersect_general(a, b));
        const double ad = poly_set_area(ms::geo::poly_diff_general(a, b));
        const double ax = poly_set_area(ms::geo::poly_symmetric_diff_general(a, b));

        EXPECT_NEAR(au + ai, area(a) + area(b), 1e-9) << "trial " << trial;
        EXPECT_NEAR(ad, area(a) - ai, 1e-9) << "trial " << trial;
        EXPECT_NEAR(ax, au - ai, 1e-9) << "trial " << trial;
        EXPECT_GE(ai, -1e-12);
        EXPECT_LE(ai, std::min(area(a), area(b)) + 1e-9);
    }
    EXPECT_GE(checked, 20) << "the generator produced too few usable simple polygons";
}

TEST(GeoPolyBooleanGeneral, AgreesWithBruteForceSamplingOnAConcaveCase) {
    const Polygon2D bar{{-1, 0.5}, {7, 0.5}, {7, 1.5}, {-1, 1.5}};
    const auto in_a = [&](Point2D p) { return ms::geo::point_in_polygon(p, kEll); };
    const auto in_b = [&](Point2D p) { return ms::geo::point_in_polygon(p, bar); };

    struct Case {
        BooleanOp op;
        std::function<bool(Point2D)> pred;
    };
    const Case cases[] = {
        {BooleanOp::Union, [&](Point2D p) { return in_a(p) || in_b(p); }},
        {BooleanOp::Intersection, [&](Point2D p) { return in_a(p) && in_b(p); }},
        {BooleanOp::Difference, [&](Point2D p) { return in_a(p) && !in_b(p); }},
        {BooleanOp::SymmetricDifference, [&](Point2D p) { return in_a(p) != in_b(p); }},
    };
    for (const auto& c : cases) {
        const PolygonSet s = poly_boolean(kEll, bar, c.op);
        // 800 samples across [-2, 8] gives h = 0.0125; the perimeters here are under 40, so
        // the Riemann error is bounded well inside 0.3.
        EXPECT_NEAR(poly_set_area(s), sampled_area(-2.0, 8.0, 800, c.pred), 0.3);
    }
}
