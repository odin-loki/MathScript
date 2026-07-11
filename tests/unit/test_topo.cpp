#include "ms/topo/topo.hpp"
#include <cmath>
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
