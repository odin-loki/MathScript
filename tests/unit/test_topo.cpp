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
