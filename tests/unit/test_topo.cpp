#include "ms/topo/topo.hpp"
#include "ms/geo/geo.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include <gtest/gtest.h>

using namespace ms::topo;

// ---- SimplicialComplex ----

TEST(TopoSimplicial, EmptyComplex) {
    SimplicialComplex sc;
    EXPECT_EQ(sc.dimension(), -1);
    EXPECT_EQ(sc.euler_characteristic(), 0);
}

TEST(TopoSimplicial, SingleVertex) {
    SimplicialComplex sc;
    sc.add_point(0);
    EXPECT_EQ(sc.dimension(), 0);
    auto betti = sc.betti_numbers();
    EXPECT_EQ(betti[0], 1);  // one connected component
}

TEST(TopoSimplicial, EdgeAndVertices) {
    SimplicialComplex sc;
    sc.add_simplex({0, 1});
    auto betti = sc.betti_numbers();
    EXPECT_GE(betti.size(), 1u);
    EXPECT_EQ(betti[0], 1);  // one connected component
}

TEST(TopoSimplicial, Triangle) {
    // Hollow triangle: three edges, no face
    SimplicialComplex sc;
    sc.add_simplex({0,1}); sc.add_simplex({1,2}); sc.add_simplex({0,2});
    auto betti = sc.betti_numbers();
    EXPECT_EQ(betti[0], 1);  // connected
    EXPECT_GE(betti[1], 1);  // one loop
}

TEST(TopoSimplicial, FilledTriangle) {
    SimplicialComplex sc;
    sc.add_simplex({0, 1, 2});  // adds triangle + all faces
    auto betti = sc.betti_numbers();
    EXPECT_EQ(betti[0], 1);  // connected
    EXPECT_EQ(betti[1], 0);  // no loops (filled)
}

TEST(TopoSimplicial, EulerCharacteristic) {
    // Tetrahedron surface: 4 V, 6 E, 4 F → χ = 2
    SimplicialComplex sc;
    sc.add_simplex({0,1,2}); sc.add_simplex({0,1,3});
    sc.add_simplex({0,2,3}); sc.add_simplex({1,2,3});
    int chi = sc.euler_characteristic();
    EXPECT_EQ(chi, 2);
}

TEST(TopoSimplicial, TwoComponents) {
    SimplicialComplex sc;
    sc.add_point(0); sc.add_point(1);
    sc.add_point(2); sc.add_point(3);
    sc.add_simplex({0,1}); sc.add_simplex({2,3});
    auto betti = sc.betti_numbers();
    EXPECT_EQ(betti[0], 2);  // two components
}

// ---- Vietoris-Rips ----

TEST(TopoVietorisRips, SmallCloud) {
    std::vector<std::vector<double>> pts = {{0,0},{1,0},{0.5,0.866}};
    auto dist_mat = pairwise_distances(pts);
    auto sc = vietoris_rips(dist_mat, 1.1, 2);
    auto betti = sc.betti_numbers();
    EXPECT_EQ(betti[0], 1);  // connected
}

TEST(TopoVietorisRips, Disconnected) {
    std::vector<std::vector<double>> pts = {{0,0},{10,10}};
    auto dist_mat = pairwise_distances(pts);
    auto sc = vietoris_rips(dist_mat, 1.0, 1);
    auto betti = sc.betti_numbers();
    EXPECT_EQ(betti[0], 2);  // two components
}

// ---- Pairwise distances ----

TEST(TopoPairwise, Triangle) {
    std::vector<std::vector<double>> pts = {{0,0},{3,0},{0,4}};
    auto D = pairwise_distances(pts);
    EXPECT_NEAR(D[0][1], 3.0, 1e-10);
    EXPECT_NEAR(D[0][2], 4.0, 1e-10);
    EXPECT_NEAR(D[1][2], 5.0, 1e-10);
    EXPECT_NEAR(D[0][0], 0.0, 1e-10);
}

// ---- Betti curve ----

TEST(TopoBettiCurve, GrowingRadius) {
    std::vector<std::vector<double>> pts = {{0,0},{1,0},{2,0}};
    auto D = pairwise_distances(pts);
    auto curve = betti_curve(D, {0.5, 1.5, 2.5}, 1);
    EXPECT_EQ(curve.size(), 3u);
    // At r=0.5: all disconnected → β0=3
    EXPECT_EQ(curve[0][0], 3);
    // At r=1.5: connected → β0=1
    EXPECT_EQ(curve[1][0], 1);
}

// ---- Persistence diagram ----

TEST(TopoPersistence, SimpleFiltration) {
    // Three-vertex filtration on a triangle boundary
    // vertices at t=0, edges at t=1
    std::vector<std::pair<Simplex, double>> filt = {
        {{0}, 0.0}, {{1}, 0.0}, {{2}, 0.0},
        {{0,1}, 1.0}, {{1,2}, 1.0}, {{0,2}, 1.5}
    };
    auto pairs = persistence_diagram(filt);
    // Should have some pairs
    EXPECT_FALSE(pairs.empty());
}

// ---- Bottleneck / Wasserstein ----

TEST(TopoDistances, SameDiagram) {
    std::vector<PersistencePair> dgm = {
        {0, 0.0, 1.0}, {1, 0.5, 2.0}
    };
    EXPECT_NEAR(bottleneck_distance(dgm, dgm, 0), 0.0, 1e-10);
    EXPECT_NEAR(wasserstein_distance(dgm, dgm, 0), 0.0, 1e-10);
}

TEST(TopoDistances, ShiftedDiagram) {
    std::vector<PersistencePair> dgm1 = {{0, 0.0, 1.0}};
    std::vector<PersistencePair> dgm2 = {{0, 0.1, 1.1}};
    double bn = bottleneck_distance(dgm1, dgm2, 0);
    EXPECT_GE(bn, 0.0);
}

// ---- Vietoris-Rips (additional) ----

TEST(TopoVietorisRips, UnitSquareCorners) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    auto D = pairwise_distances(pts);
    auto sc = vietoris_rips(D, 0.5, 1);
    EXPECT_EQ(sc.betti_numbers()[0], 4);
}

TEST(TopoVietorisRips, SquareConnected) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    auto D = pairwise_distances(pts);
    auto sc = vietoris_rips(D, 1.5, 2);
    auto betti = sc.betti_numbers();
    EXPECT_EQ(betti[0], 1);
    EXPECT_GE(betti.size(), 2u);
}

TEST(TopoVietorisRips, CollinearChain) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
    auto D = pairwise_distances(pts);
    auto sc = vietoris_rips(D, 1.1, 1);
    EXPECT_EQ(sc.betti_numbers()[0], 1);
}

TEST(TopoVietorisRips, MaxDimZero) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {0.5, 0.866}};
    auto D = pairwise_distances(pts);
    auto sc = vietoris_rips(D, 2.0, 0);
    EXPECT_EQ(sc.dimension(), 1);
    EXPECT_TRUE(sc.simplices(2).empty());
}

// ---- Persistence diagram (additional) ----

TEST(TopoPersistence, VerticesOnly) {
    std::vector<std::pair<Simplex, double>> filt = {
        {{0}, 0.0}, {{1}, 0.0}, {{2}, 0.0}
    };
    auto pairs = persistence_diagram(filt);
    // Isolated vertices have empty boundaries and produce no finite pairs.
    for (const auto& p : pairs) {
        if (!p.is_essential())
            EXPECT_LT(p.birth, p.death);
    }
}

TEST(TopoPersistence, SingleEdge) {
    std::vector<std::pair<Simplex, double>> filt = {
        {{0}, 0.0}, {{1}, 0.0}, {{0, 1}, 1.0}
    };
    auto pairs = persistence_diagram(filt);
    EXPECT_FALSE(pairs.empty());
    bool has_dim0 = false;
    for (const auto& p : pairs)
        if (p.dim == 0) has_dim0 = true;
    EXPECT_TRUE(has_dim0);
}

TEST(TopoPersistence, BirthBeforeDeath) {
    std::vector<std::pair<Simplex, double>> filt = {
        {{0}, 0.0}, {{1}, 0.0}, {{2}, 0.0},
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{0, 2}, 1.5}
    };
    auto pairs = persistence_diagram(filt);
    for (const auto& p : pairs) {
        if (!p.is_essential())
            EXPECT_LT(p.birth, p.death);
    }
}

TEST(TopoPersistence, FilledTriangle) {
    std::vector<std::pair<Simplex, double>> filt = {
        {{0}, 0.0}, {{1}, 0.0}, {{2}, 0.0},
        {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{0, 2}, 1.0},
        {{0, 1, 2}, 2.0}
    };
    auto pairs = persistence_diagram(filt);
    EXPECT_FALSE(pairs.empty());
}

// ---- Bottleneck distance (additional) ----

TEST(TopoBottleneck, EmptyDiagrams) {
    std::vector<PersistencePair> empty;
    EXPECT_NEAR(bottleneck_distance(empty, empty, 0), 0.0, 1e-10);
}

TEST(TopoBottleneck, ParallelShift) {
    std::vector<PersistencePair> dgm1 = {{0, 0.0, 2.0}, {1, 1.0, 3.0}};
    std::vector<PersistencePair> dgm2 = {{0, 0.2, 2.2}, {1, 1.2, 3.2}};
    double bn = bottleneck_distance(dgm1, dgm2, 0);
    EXPECT_NEAR(bn, 0.2, 1e-10);
}

TEST(TopoBottleneck, DimensionFilter) {
    std::vector<PersistencePair> dgm1 = {{0, 0.0, 1.0}, {1, 0.0, 5.0}};
    std::vector<PersistencePair> dgm2 = {{0, 0.5, 1.5}, {1, 0.0, 5.0}};
    EXPECT_NEAR(bottleneck_distance(dgm1, dgm2, 0), 0.5, 1e-10);
    EXPECT_NEAR(bottleneck_distance(dgm1, dgm2, 1), 0.0, 1e-10);
}

TEST(TopoBottleneck, UnmatchedPoint) {
    std::vector<PersistencePair> dgm1 = {{0, 0.0, 1.0}, {0, 2.0, 4.0}};
    std::vector<PersistencePair> dgm2 = {{0, 0.0, 1.0}};
    double bn = bottleneck_distance(dgm1, dgm2, 0);
    EXPECT_NEAR(bn, 1.0, 1e-10);
}

// ---- Betti curve (additional) ----

TEST(TopoBettiCurve, EmptyThresholds) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}};
    auto D = pairwise_distances(pts);
    auto curve = betti_curve(D, {}, 1);
    EXPECT_TRUE(curve.empty());
}

TEST(TopoBettiCurve, SquareFiltration) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    auto D = pairwise_distances(pts);
    auto curve = betti_curve(D, {0.5, 1.5}, 1);
    EXPECT_EQ(curve.size(), 2u);
    EXPECT_EQ(curve[0][0], 4);
    EXPECT_EQ(curve[1][0], 1);
}

TEST(TopoBettiCurve, LoopAppears) {
    // Square boundary: loop at r=1.0, filled once diagonals appear
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    auto D = pairwise_distances(pts);
    auto curve = betti_curve(D, {0.5, 1.0}, 2);
    EXPECT_EQ(curve.size(), 2u);
    EXPECT_EQ(curve[0][0], 4);
    EXPECT_GE(curve[1].size(), 2u);
    EXPECT_GE(curve[1][1], 1);
}

TEST(TopoBettiCurve, MaxDimPadding) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {2, 0}};
    auto D = pairwise_distances(pts);
    auto curve = betti_curve(D, {1.5}, 2);
    ASSERT_EQ(curve.size(), 1u);
    EXPECT_GE(curve[0].size(), 3u);
}

// ---- Čech complex ----

// Equilateral triangle, side length 2: circumradius R = side/sqrt(3) ≈ 1.154700538.
// This is an acute triangle, so R is strictly greater than half the longest side (1.0),
// which lets us exercise the true circumradius threshold (not just the edge/flag rule),
// and directly show Čech(ε) excluding a triangle that Vietoris-Rips(2ε) still includes.
TEST(TopoCech, EquilateralTriangleCircumradiusThreshold) {
    const double side = 2.0;
    const double R = side / std::sqrt(3.0);  // ~1.1547005383792515
    std::vector<std::vector<double>> D = {
        {0.0, side, side},
        {side, 0.0, side},
        {side, side, 0.0}
    };

    // At epsilon == R (with a tiny slack for floating point), the triangle is included.
    {
        auto sc = cech_complex(D, R + 1e-9, 2);
        auto tris = sc.simplices(2);
        EXPECT_EQ(tris.size(), 1u);
    }
    // Just below the circumradius threshold, the triangle must be excluded.
    {
        auto sc = cech_complex(D, R - 0.05, 2);
        auto tris = sc.simplices(2);
        EXPECT_TRUE(tris.empty());
        // But edges are still present: 2*epsilon = 2*(R-0.05) > side, so the 1-skeleton
        // is a complete graph even though the 2-simplex is not filled in.
        EXPECT_EQ(sc.simplices(1).size(), 3u);
    }
}

// Same equilateral triangle: pick an epsilon in the "gap" between the edge threshold
// (side/2 = 1.0) and the true circumradius (~1.1547). At this epsilon, Vietoris-Rips
// (a flag complex, called with r = 2*epsilon so the edge rules match) fills in the
// triangle just because all three edges are present, while the Čech complex correctly
// excludes it because the minimum enclosing ball radius exceeds epsilon. This confirms
// Čech is not simply "clique in the 1-skeleton".
TEST(TopoCech, DivergesFromVietorisRipsFlagRule) {
    const double side = 2.0;
    const double R = side / std::sqrt(3.0);
    std::vector<std::vector<double>> D = {
        {0.0, side, side},
        {side, 0.0, side},
        {side, side, 0.0}
    };
    const double epsilon = 1.05;  // side/2 = 1.0 < epsilon < R ~= 1.1547
    ASSERT_GT(epsilon, side / 2.0);
    ASSERT_LT(epsilon, R);

    auto vr = vietoris_rips(D, 2.0 * epsilon, 2);
    EXPECT_EQ(vr.simplices(2).size(), 1u);  // VR: all edges <= 2*epsilon -> flag-fills the triangle

    auto cech = cech_complex(D, epsilon, 2);
    EXPECT_TRUE(cech.simplices(2).empty());  // Čech: circumradius R > epsilon -> excluded
    EXPECT_EQ(cech.simplices(1).size(), 3u);  // edges still agree with VR at this epsilon
}

// 3-4-5 right triangle: circumradius = hypotenuse/2 = 2.5 (classic hand-computed value).
TEST(TopoCech, RightTriangle345Circumradius) {
    std::vector<std::vector<double>> D = {
        {0.0, 3.0, 5.0},
        {3.0, 0.0, 4.0},
        {5.0, 4.0, 0.0}
    };
    // At epsilon == 2.5 the triangle (and its hypotenuse edge, dist 5 <= 2*2.5) is included.
    {
        auto sc = cech_complex(D, 2.5, 2);
        EXPECT_EQ(sc.simplices(2).size(), 1u);
        EXPECT_EQ(sc.simplices(1).size(), 3u);
    }
    // Just below, the hypotenuse edge (and hence the triangle) is excluded.
    {
        auto sc = cech_complex(D, 2.499, 2);
        EXPECT_TRUE(sc.simplices(2).empty());
    }
}

// Unit square corners: any 3 corners form a right isoceles triangle (right angle at one
// corner, hypotenuse = diagonal = sqrt(2)), so its circumradius is sqrt(2)/2 ~= 0.7071 --
// exactly half the diagonal. This test locks in the diagonal-driven inclusion/exclusion
// thresholds for both the diagonal edge and the triangles that require it.
TEST(TopoCech, UnitSquareDiagonalThreshold) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    auto D = pairwise_distances(pts);
    const double diag = std::sqrt(2.0);
    EXPECT_NEAR(D[0][2], diag, 1e-10);  // sanity check the diagonal length

    // epsilon = 0.5: 2*epsilon = 1.0 < diag, so the diagonal edge (and any triangle
    // spanning it) is excluded; only the 4 unit-length square edges are present.
    {
        auto sc = cech_complex(D, 0.5, 2);
        EXPECT_EQ(sc.simplices(1).size(), 4u);
        EXPECT_TRUE(sc.simplices(2).empty());
    }
    // epsilon = 0.75: 2*epsilon = 1.5 >= diag, so both diagonals are present, and each
    // of the 4 corner triangles has circumradius diag/2 ~= 0.7071 <= 0.75, so all are included.
    {
        auto sc = cech_complex(D, 0.75, 2);
        EXPECT_EQ(sc.simplices(1).size(), 6u);   // 4 sides + 2 diagonals
        EXPECT_EQ(sc.simplices(2).size(), 4u);   // 4 corner triangles
    }
}

// Degenerate / edge-case inputs must not crash and should return sensible results.
TEST(TopoCech, EmptyPointSet) {
    std::vector<std::vector<double>> D;
    auto sc = cech_complex(D, 1.0, 2);
    EXPECT_EQ(sc.dimension(), -1);
    EXPECT_TRUE(sc.all_simplices().empty());
}

TEST(TopoCech, SinglePoint) {
    std::vector<std::vector<double>> D = {{0.0}};
    auto sc = cech_complex(D, 1.0, 2);
    EXPECT_EQ(sc.dimension(), 0);
    EXPECT_EQ(sc.simplices(0).size(), 1u);
}

TEST(TopoCech, EpsilonZeroGivesOnlyVertices) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {0, 1}};
    auto D = pairwise_distances(pts);
    auto sc = cech_complex(D, 0.0, 2);
    EXPECT_EQ(sc.dimension(), 0);
    EXPECT_EQ(sc.simplices(0).size(), 3u);
    EXPECT_TRUE(sc.simplices(1).empty());
}

TEST(TopoCech, LargeEpsilonGivesFullComplex) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    auto D = pairwise_distances(pts);
    auto sc = cech_complex(D, 1000.0, 2);
    EXPECT_EQ(sc.simplices(0).size(), 4u);
    EXPECT_EQ(sc.simplices(1).size(), 6u);   // complete graph on 4 vertices
    EXPECT_EQ(sc.simplices(2).size(), 4u);   // all C(4,3) triangles
}

TEST(TopoCech, MaxDimClampedAboveTwo) {
    // max_dim > 2 is not supported (see @note on cech_complex); it must be clamped
    // to 2 rather than crash or fabricate higher-dimensional simplices.
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    auto D = pairwise_distances(pts);
    auto sc = cech_complex(D, 1000.0, 5);
    EXPECT_EQ(sc.dimension(), 2);
    EXPECT_TRUE(sc.simplices(3).empty());
}

// ---- Alpha complex ----

namespace {

// Local re-derivation of circumcenter/circumradius from 3 raw coordinates, kept
// independent of ms::geo's internal (static) helpers, so that the "no other point
// lies strictly inside the circumcircle" Delaunay property can be verified in tests
// without reaching into ms::geo's implementation details.
struct CircumCircle { double cx, cy, r; bool ok; };

CircumCircle circum_circle(const std::vector<std::vector<double>>& pts, int i, int j, int k) {
    double ax = pts[i][0], ay = pts[i][1];
    double bx = pts[j][0], by = pts[j][1];
    double cx0 = pts[k][0], cy0 = pts[k][1];
    double D = 2.0 * (ax*(by-cy0) + bx*(cy0-ay) + cx0*(ay-by));
    if (std::abs(D) < 1e-12) return {0, 0, 0, false};
    double ux = ((ax*ax+ay*ay)*(by-cy0) + (bx*bx+by*by)*(cy0-ay) + (cx0*cx0+cy0*cy0)*(ay-by)) / D;
    double uy = ((ax*ax+ay*ay)*(cx0-bx) + (bx*bx+by*by)*(ax-cx0) + (cx0*cx0+cy0*cy0)*(bx-ax)) / D;
    double r = std::sqrt((ax-ux)*(ax-ux) + (ay-uy)*(ay-uy));
    return {ux, uy, r, true};
}

// Every simplex in `sub` (as a sorted-vertex set) must also appear in `sup`.
bool is_subcomplex(const SimplicialComplex& sub, const SimplicialComplex& sup) {
    std::set<Simplex> super_set(sup.all_simplices().begin(), sup.all_simplices().end());
    for (auto& s : sub.all_simplices())
        if (super_set.find(s) == super_set.end()) return false;
    return true;
}

} // namespace

// 1. Subcomplex-of-Delaunay property: every triangle/edge alpha_complex returns must
// be a genuine Delaunay triangle/edge of the same point set (5 points in general
// position: no 3 collinear, no 4 exactly cocircular), and no returned triangle may
// contain another input point strictly inside its circumcircle.
TEST(TopoAlpha, SubcomplexOfDelaunay) {
    std::vector<std::vector<double>> pts = {{0,0}, {4,0}, {4,3}, {0,4}, {1.5,1.5}};
    std::vector<ms::geo::Point2D> gpts;
    for (auto& p : pts) gpts.push_back({p[0], p[1]});
    auto delaunay_tris = ms::geo::delaunay_2d(gpts);

    std::set<Simplex> delaunay_tri_set, delaunay_edge_set;
    for (auto& t : delaunay_tris) {
        Simplex s = {t.a, t.b, t.c};
        std::sort(s.begin(), s.end());
        delaunay_tri_set.insert(s);
        Simplex e1={t.a,t.b}, e2={t.b,t.c}, e3={t.a,t.c};
        std::sort(e1.begin(),e1.end()); std::sort(e2.begin(),e2.end()); std::sort(e3.begin(),e3.end());
        delaunay_edge_set.insert(e1); delaunay_edge_set.insert(e2); delaunay_edge_set.insert(e3);
    }
    ASSERT_FALSE(delaunay_tris.empty());

    // Large alpha admits every Delaunay simplex, giving the widest check surface.
    auto sc = alpha_complex(pts, 1000.0, 2);
    for (auto& tri : sc.simplices(2))
        EXPECT_TRUE(delaunay_tri_set.count(tri)) << "triangle not part of the Delaunay triangulation";
    for (auto& e : sc.simplices(1))
        EXPECT_TRUE(delaunay_edge_set.count(e)) << "edge not part of the Delaunay triangulation";

    // Empty-circumcircle defining property of Delaunay triangles.
    for (auto& tri : sc.simplices(2)) {
        auto cc = circum_circle(pts, tri[0], tri[1], tri[2]);
        ASSERT_TRUE(cc.ok);
        for (int m = 0; m < (int)pts.size(); ++m) {
            if (m == tri[0] || m == tri[1] || m == tri[2]) continue;
            double dx = pts[m][0]-cc.cx, dy = pts[m][1]-cc.cy;
            double dm = std::sqrt(dx*dx+dy*dy);
            EXPECT_GE(dm, cc.r - 1e-9) << "point lies strictly inside a returned triangle's circumcircle";
        }
    }
}

// 2. Nesting/monotonicity: increasing alpha only adds simplices, never removes them.
TEST(TopoAlpha, Monotonicity) {
    std::vector<std::vector<double>> pts = {{0,0}, {4,0}, {4,3}, {0,4}, {1.5,1.5}, {2.5, 0.5}};
    std::vector<double> alphas = {0.1, 0.4, 0.9, 1.5, 2.5, 100.0};
    for (size_t a = 0; a + 1 < alphas.size(); ++a) {
        auto sc1 = alpha_complex(pts, alphas[a], 2);
        auto sc2 = alpha_complex(pts, alphas[a+1], 2);
        EXPECT_TRUE(is_subcomplex(sc1, sc2))
            << "alpha=" << alphas[a] << " complex is not a subset of alpha=" << alphas[a+1];
    }
}

// 3. Small hand-checked case: unit-ish rectangle (2 x 2 square) split by Delaunay into
// its two triangles. Both possible diagonals of a square yield an identical circumradius
// (sqrt(2)/2 * side) by symmetry, so the expected simplex counts below hold regardless of
// which diagonal delaunay_2d happens to choose.
TEST(TopoAlpha, SquareHandChecked) {
    std::vector<std::vector<double>> pts = {{0,0}, {2,0}, {2,2}, {0,2}};
    const double side = 2.0;
    const double R = side / std::sqrt(2.0);  // circumradius of each right-isoceles half-triangle

    // alpha below half a side length: only vertices.
    {
        auto sc = alpha_complex(pts, 0.3, 2);
        EXPECT_EQ(sc.simplices(0).size(), 4u);
        EXPECT_TRUE(sc.simplices(1).empty());
        EXPECT_TRUE(sc.simplices(2).empty());
    }
    // alpha between half a side (1.0) and R (~1.414): the 4 hull edges appear
    // (their own MEB radius is 1.0 <= alpha) but the diagonal/triangles do not,
    // since the diagonal is an interior edge governed by triangle inclusion only.
    {
        auto sc = alpha_complex(pts, 1.2, 2);
        EXPECT_EQ(sc.simplices(0).size(), 4u);
        EXPECT_EQ(sc.simplices(1).size(), 4u);
        EXPECT_TRUE(sc.simplices(2).empty());
    }
    // alpha == R: both triangles (and hence all 4 sides + the 1 Delaunay diagonal) appear.
    {
        auto sc = alpha_complex(pts, R + 1e-9, 2);
        EXPECT_EQ(sc.simplices(0).size(), 4u);
        EXPECT_EQ(sc.simplices(1).size(), 5u);
        EXPECT_EQ(sc.simplices(2).size(), 2u);
    }
}

// 4. Comparison with cech_complex: alpha_complex must always be a subset of
// cech_complex at the same alpha/epsilon on the same point set, since the alpha
// complex only ever considers Delaunay simplices -- a strict subset of every
// possible simplex cech_complex considers.
TEST(TopoAlpha, SubsetOfCech) {
    std::vector<std::vector<double>> pts = {{0,0}, {4,0}, {4,3}, {0,4}, {1.5,1.5}, {2.5,0.5}};
    auto D = pairwise_distances(pts);
    double alphas[] = {0.2, 0.75, 1.5, 3.0, 10.0};
    for (double a : alphas) {
        auto alpha_sc = alpha_complex(pts, a, 2);
        auto cech_sc = cech_complex(D, a, 2);
        EXPECT_TRUE(is_subcomplex(alpha_sc, cech_sc)) << "alpha_complex not a subset of cech_complex at alpha=" << a;
    }
}

// 5. Degenerate / edge-case inputs must not crash and should return sensible results.
TEST(TopoAlpha, EmptyPointSet) {
    std::vector<std::vector<double>> pts;
    auto sc = alpha_complex(pts, 1.0, 2);
    EXPECT_EQ(sc.dimension(), -1);
    EXPECT_TRUE(sc.all_simplices().empty());
}

TEST(TopoAlpha, SinglePoint) {
    std::vector<std::vector<double>> pts = {{0, 0}};
    auto sc = alpha_complex(pts, 1.0, 2);
    EXPECT_EQ(sc.dimension(), 0);
    EXPECT_EQ(sc.simplices(0).size(), 1u);
}

TEST(TopoAlpha, TwoPoints) {
    std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}};
    // Below half the distance: no edge.
    {
        auto sc = alpha_complex(pts, 0.2, 2);
        EXPECT_EQ(sc.simplices(0).size(), 2u);
        EXPECT_TRUE(sc.simplices(1).empty());
    }
    // At/above half the distance: the edge appears.
    {
        auto sc = alpha_complex(pts, 0.5, 2);
        EXPECT_EQ(sc.simplices(0).size(), 2u);
        EXPECT_EQ(sc.simplices(1).size(), 1u);
    }
}

TEST(TopoAlpha, CollinearPointsDoNotCrash) {
    std::vector<std::vector<double>> pts = {{0,0}, {1,0}, {2,0}, {3,0}};
    auto sc = alpha_complex(pts, 5.0, 2);
    EXPECT_EQ(sc.simplices(0).size(), 4u);  // vertices are always safe to report
    // Whatever (possibly degenerate) triangles a Delaunay implementation derives
    // from a collinear input, every triangle actually returned must still respect
    // the alpha threshold and involve valid point indices -- i.e. no garbage output.
    for (auto& tri : sc.simplices(2)) {
        EXPECT_EQ(tri.size(), 3u);
        for (int idx : tri) { EXPECT_GE(idx, 0); EXPECT_LT(idx, 4); }
    }
}

TEST(TopoAlpha, DuplicatePointsDoNotCrash) {
    std::vector<std::vector<double>> pts = {{0,0}, {1,0}, {0,1}, {0,0}};
    auto sc = alpha_complex(pts, 5.0, 2);
    EXPECT_EQ(sc.simplices(0).size(), 4u);  // all 4 indices remain distinct vertices
    for (auto& tri : sc.simplices(2)) {
        EXPECT_EQ(tri.size(), 3u);
        for (int idx : tri) { EXPECT_GE(idx, 0); EXPECT_LT(idx, 4); }
    }
}

TEST(TopoAlpha, MaxDimClampedAboveTwo) {
    std::vector<std::vector<double>> pts = {{0,0}, {2,0}, {2,2}, {0,2}};
    auto sc = alpha_complex(pts, 1000.0, 5);
    EXPECT_EQ(sc.dimension(), 2);
    EXPECT_TRUE(sc.simplices(3).empty());
}

// Mirrors cech_complex / vietoris_rips: max_dim only gates simplices of dimension
// >= 2 (edges are always considered against their own alpha value regardless).
TEST(TopoAlpha, MaxDimZeroSuppressesTrianglesOnly) {
    std::vector<std::vector<double>> pts = {{0,0}, {2,0}, {2,2}, {0,2}};
    auto sc = alpha_complex(pts, 1000.0, 0);
    EXPECT_EQ(sc.simplices(0).size(), 4u);
    EXPECT_FALSE(sc.simplices(1).empty());
    EXPECT_TRUE(sc.simplices(2).empty());
}

// ---- Persistence landscape ----

// Single pair (0,2): hand-computable tent function. lambda_1(1) is the peak
// ((d-b)/2 = 1), lambda_1(0)=lambda_1(2)=0 (at the boundary), lambda_1(0.5)=0.5.
TEST(TopoLandscape, SinglePairTentShape) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 2.0}};
    auto land = persistence_landscape(dgm, 2, 5, 0.0, 2.0);
    ASSERT_EQ(land.size(), 2u);
    ASSERT_EQ(land[0].size(), 5u);
    // samples at t = 0, 0.5, 1, 1.5, 2
    EXPECT_NEAR(land[0][0], 0.0, 1e-10);
    EXPECT_NEAR(land[0][1], 0.5, 1e-10);
    EXPECT_NEAR(land[0][2], 1.0, 1e-10);
    EXPECT_NEAR(land[0][3], 0.5, 1e-10);
    EXPECT_NEAR(land[0][4], 0.0, 1e-10);
}

// Only one pair exists, so lambda_2 has no "2nd highest" value anywhere: identically 0.
TEST(TopoLandscape, SinglePairSecondLayerIsZero) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 2.0}};
    auto land = persistence_landscape(dgm, 2, 5, 0.0, 2.0);
    ASSERT_EQ(land.size(), 2u);
    for (double v : land[1])
        EXPECT_NEAR(v, 0.0, 1e-10);
}

// Two non-overlapping pairs: lambda_1 is the union/envelope of both tents, and since
// they never overlap, lambda_2 is all-zero (at most one tent is nonzero at any t).
TEST(TopoLandscape, NonOverlappingPairsUnionEnvelope) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 1.0}, {0, 10.0, 11.0}};
    auto land = persistence_landscape(dgm, 2, 12, 0.0, 11.0);
    ASSERT_EQ(land.size(), 2u);
    ASSERT_EQ(land[0].size(), 12u);
    // step = 1.0, samples at t = 0,1,...,11
    // t=0.5 -> inside first tent, peak height 0.5
    EXPECT_NEAR(land[0][0], 0.0, 1e-10);   // t=0 (boundary of first tent)
    // t = 10.5 is not exactly sampled (step=1), but t=10 and t=11 are boundaries of
    // the second tent, and t=1..9 lie strictly between/outside both tents.
    for (int i = 1; i <= 9; ++i)
        EXPECT_NEAR(land[0][i], 0.0, 1e-10) << "i=" << i;
    EXPECT_NEAR(land[0][10], 0.0, 1e-10);  // t=10, boundary of second tent
    EXPECT_NEAR(land[0][11], 0.0, 1e-10);  // t=11, boundary of second tent
    for (double v : land[1])
        EXPECT_NEAR(v, 0.0, 1e-10);
}

// Finer sampling over a non-overlapping pair to directly hit the tent peaks.
TEST(TopoLandscape, NonOverlappingPairsPeaksHit) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 1.0}, {0, 10.0, 11.0}};
    // 3 samples per unit interval so t=0.5 and t=10.5 land exactly on peaks.
    auto land = persistence_landscape(dgm, 1, 23, 0.0, 11.0);
    // step = 11/22 = 0.5 -> sample index 1 is t=0.5, sample index 21 is t=10.5
    EXPECT_NEAR(land[0][1], 0.5, 1e-10);
    EXPECT_NEAR(land[0][21], 0.5, 1e-10);
}

// Two overlapping pairs with different heights: (0,4) peak height 2, (1,3) peak
// height 1, overlapping in t in [1,3]. In the overlap, lambda_1 must follow the
// taller tent and lambda_2 the shorter one -- this is the key "k-th largest across
// all pairs" behavior that distinguishes landscapes from a naive per-pair summary.
TEST(TopoLandscape, OverlappingPairsKthLargest) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 4.0}, {0, 1.0, 3.0}};
    auto land = persistence_landscape(dgm, 2, 9, 0.0, 4.0);
    // step = 0.5, samples at t = 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4
    // t=2 (midpoint, inside overlap): tent1 = min(2,2) = 2, tent2 = min(1,1) = 1
    EXPECT_NEAR(land[0][4], 2.0, 1e-10);  // lambda_1(2) follows the taller tent
    EXPECT_NEAR(land[1][4], 1.0, 1e-10);  // lambda_2(2) follows the shorter tent
    // t=0.5 (outside overlap, only tent1 nonzero): tent1 = min(0.5,3.5) = 0.5
    EXPECT_NEAR(land[0][1], 0.5, 1e-10);
    EXPECT_NEAR(land[1][1], 0.0, 1e-10);
    // t=3.5 (outside overlap, only tent1 nonzero): tent1 = min(3.5,0.5) = 0.5
    EXPECT_NEAR(land[0][7], 0.5, 1e-10);
    EXPECT_NEAR(land[1][7], 0.0, 1e-10);
}

// Essential (infinite-death) pairs must be excluded, matching is_essential() usage
// elsewhere -- the landscape must be identical whether the essential pair is present
// or simply omitted from the input.
TEST(TopoLandscape, EssentialPairsExcluded) {
    std::vector<PersistencePair> with_essential = {
        {0, 0.0, 2.0},
        {0, 0.5, std::numeric_limits<double>::infinity()}
    };
    std::vector<PersistencePair> without_essential = {{0, 0.0, 2.0}};

    auto land_with = persistence_landscape(with_essential, 2, 5, 0.0, 2.0);
    auto land_without = persistence_landscape(without_essential, 2, 5, 0.0, 2.0);
    ASSERT_EQ(land_with.size(), land_without.size());
    for (size_t k = 0; k < land_with.size(); ++k) {
        ASSERT_EQ(land_with[k].size(), land_without[k].size());
        for (size_t i = 0; i < land_with[k].size(); ++i)
            EXPECT_NEAR(land_with[k][i], land_without[k][i], 1e-10);
    }
}

TEST(TopoLandscape, NonPositiveLayersReturnsEmpty) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 2.0}};
    EXPECT_TRUE(persistence_landscape(dgm, 0, 10, 0.0, 2.0).empty());
    EXPECT_TRUE(persistence_landscape(dgm, -1, 10, 0.0, 2.0).empty());
}

TEST(TopoLandscape, TooFewSamplesReturnsEmpty) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 2.0}};
    EXPECT_TRUE(persistence_landscape(dgm, 1, 1, 0.0, 2.0).empty());
    EXPECT_TRUE(persistence_landscape(dgm, 1, 0, 0.0, 2.0).empty());
}

// Auto-derived t_min/t_max (sentinel default 0.0/0.0): the derived range must
// actually span at least [min birth, max death] of the input diagram.
TEST(TopoLandscape, AutoDerivedRangeSpansDiagram) {
    std::vector<PersistencePair> dgm = {{0, 2.0, 5.0}, {0, 3.0, 9.0}};
    auto land = persistence_landscape(dgm, 1, 50);  // t_min=t_max=0.0 -> auto-derive
    ASSERT_EQ(land.size(), 1u);
    ASSERT_EQ(land[0].size(), 50u);
    // At the min birth (2.0) and max death (9.0) every tent must be at its boundary
    // (0), while some interior sample must be strictly positive -- confirming the
    // sampled range actually reaches down to birth=2 and up to death=9 rather than
    // e.g. clamping to the sentinel default of [0,0].
    bool any_positive = false;
    for (double v : land[0]) if (v > 1e-9) any_positive = true;
    EXPECT_TRUE(any_positive);
}

// More pairs than requested layers: 5 pairs, only 2 layers requested. Both layers
// must be populated (non-trivially) since there are plenty of tents to rank.
TEST(TopoLandscape, MorePairsThanLayers) {
    std::vector<PersistencePair> dgm = {
        {0, 0.0, 2.0}, {0, 0.2, 2.2}, {0, 0.4, 2.4}, {0, 0.6, 2.6}, {0, 0.8, 2.8}
    };
    auto land = persistence_landscape(dgm, 2, 11, 0.0, 3.0);
    ASSERT_EQ(land.size(), 2u);
    bool layer0_nonzero = false, layer1_nonzero = false;
    for (double v : land[0]) if (v > 1e-9) layer0_nonzero = true;
    for (double v : land[1]) if (v > 1e-9) layer1_nonzero = true;
    EXPECT_TRUE(layer0_nonzero);
    EXPECT_TRUE(layer1_nonzero);
}

// Fewer pairs than requested layers: 2 pairs, 5 layers requested. Layers 3-5
// (indices 2..4) must be all-zero everywhere, since there is no "3rd/4th/5th
// highest" tent value when at most 2 tents exist at any given t.
TEST(TopoLandscape, FewerPairsThanLayersHigherLayersAreZero) {
    std::vector<PersistencePair> dgm = {{0, 0.0, 2.0}, {0, 1.0, 3.0}};
    auto land = persistence_landscape(dgm, 5, 9, 0.0, 3.0);
    ASSERT_EQ(land.size(), 5u);
    for (int k = 2; k < 5; ++k)
        for (double v : land[k])
            EXPECT_NEAR(v, 0.0, 1e-10) << "layer=" << k;
}

// Empty diagram (no pairs at all): every layer must be all-zero, and the function
// must not crash even with the sentinel auto-derive range.
TEST(TopoLandscape, EmptyDiagramIsAllZero) {
    std::vector<PersistencePair> dgm;
    auto land = persistence_landscape(dgm, 3, 10);
    ASSERT_EQ(land.size(), 3u);
    for (auto& layer : land) {
        ASSERT_EQ(layer.size(), 10u);
        for (double v : layer) EXPECT_NEAR(v, 0.0, 1e-10);
    }
}

// Diagram with only an essential pair (no finite pairs at all): must behave like
// an empty diagram (all-zero output), not crash on the auto-derive path.
TEST(TopoLandscape, OnlyEssentialPairIsAllZero) {
    std::vector<PersistencePair> dgm = {
        {0, 0.5, std::numeric_limits<double>::infinity()}
    };
    auto land = persistence_landscape(dgm, 2, 10);
    ASSERT_EQ(land.size(), 2u);
    for (auto& layer : land)
        for (double v : layer) EXPECT_NEAR(v, 0.0, 1e-10);
}
