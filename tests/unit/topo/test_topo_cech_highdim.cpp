#include "ms/topo/topo.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <set>
#include <vector>

using namespace ms::topo;

// Coverage for the two pieces that lift cech_complex past its old dim<=2 ceiling:
// meb_radius_from_distances (exact minimum enclosing ball from a distance matrix via
// the bordered Cayley-Menger system) and the dim>=3 incremental expansion built on it.

namespace {

// Regular tetrahedron of edge 2*sqrt(2) on alternating cube corners. Every coordinate
// is exactly representable, so pairwise_distances is as clean as std::sqrt allows.
const std::vector<std::vector<double>> kCubeTetra = {
    {1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};

// The same tetrahedron plus its circumcentre (the origin).
const std::vector<std::vector<double>> kTetraPlusCentre = {
    {1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}, {0, 0, 0}};

std::vector<int> counts_of(const SimplicialComplex& sc) { return sc.simplex_counts(); }

} // namespace

// ---- meb_radius_from_distances ----

// A regular tetrahedron's MEB radius is its circumradius, edge * sqrt(3/8).
TEST(TopoMeb, RegularTetrahedronCircumradius) {
    const double edge = 2.0 * std::sqrt(2.0);
    auto D = pairwise_distances(kCubeTetra);
    ASSERT_NEAR(D[0][1], edge, 1e-12);
    const double expected = edge * std::sqrt(3.0 / 8.0);  // == sqrt(3)
    EXPECT_NEAR(meb_radius_from_distances(D, {0, 1, 2, 3}), expected, 1e-9);
    EXPECT_NEAR(expected, std::sqrt(3.0), 1e-12);
}

TEST(TopoMeb, RegularTetrahedronUnitEdge) {
    const double h = std::sqrt(3.0) / 2.0;
    std::vector<std::vector<double>> pts = {
        {0, 0, 0}, {1, 0, 0}, {0.5, h, 0}, {0.5, h / 3.0, std::sqrt(2.0 / 3.0)}};
    auto D = pairwise_distances(pts);
    EXPECT_NEAR(meb_radius_from_distances(D, {0, 1, 2, 3}), std::sqrt(3.0 / 8.0), 1e-9);
}

// Affinely dependent 4-point sets make the Cayley-Menger system singular; the correct
// answer must still come back, recovered from a smaller support set.
TEST(TopoMeb, DegenerateFlatFourPointSets) {
    {   // planar unit square: MEB is the diagonal ball, supported by 2 opposite corners
        std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        auto D = pairwise_distances(pts);
        EXPECT_NEAR(meb_radius_from_distances(D, {0, 1, 2, 3}), std::sqrt(2.0) / 2.0, 1e-12);
    }
    {   // collinear: MEB spans the two extreme points
        std::vector<std::vector<double>> pts = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
        auto D = pairwise_distances(pts);
        EXPECT_NEAR(meb_radius_from_distances(D, {0, 1, 2, 3}), 1.5, 1e-12);
    }
    {   // four coincident points: radius 0, and +0.0 rather than a numerical -0.0
        std::vector<std::vector<double>> pts = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        auto D = pairwise_distances(pts);
        const double r = meb_radius_from_distances(D, {0, 1, 2, 3});
        EXPECT_NEAR(r, 0.0, 1e-15);
        EXPECT_FALSE(std::signbit(r));
    }
}

// The 2- and 3-point closed forms this module already used are reproduced bit for bit,
// including the triangle helper's absolute obtuseness tolerance. This is load-bearing:
// TopoCech.RightTriangle345Circumradius asserts inclusion at exactly epsilon == 2.5.
TEST(TopoMeb, MatchesLegacyTwoAndThreePointRules) {
    std::vector<std::vector<double>> D345 = {{0, 3, 5}, {3, 0, 4}, {5, 4, 0}};
    EXPECT_EQ(meb_radius_from_distances(D345, {0, 1}), 1.5);     // half the distance
    EXPECT_EQ(meb_radius_from_distances(D345, {0, 2}), 2.5);
    EXPECT_EQ(meb_radius_from_distances(D345, {0, 1, 2}), 2.5);  // right: hypotenuse/2

    const double side = 2.0;
    std::vector<std::vector<double>> Deq = {
        {0.0, side, side}, {side, 0.0, side}, {side, side, 0.0}};
    EXPECT_DOUBLE_EQ(meb_radius_from_distances(Deq, {0, 1, 2}), side / std::sqrt(3.0));

    // Obtuse: the MEB spans the longest side alone.
    std::vector<std::vector<double>> Dob = {{0, 1, 1}, {1, 0, 1.9}, {1, 1.9, 0}};
    EXPECT_DOUBLE_EQ(meb_radius_from_distances(Dob, {0, 1, 2}), 1.9 / 2.0);
}

// 4-point sets whose MEB support set is smaller than the whole set.
TEST(TopoMeb, FourPointSupportSizesTwoAndThree) {
    {   // 3-4-5 right triangle plus the midpoint of its hypotenuse: support is the pair
        std::vector<std::vector<double>> pts = {{0, 0}, {4, 0}, {0, 3}, {2, 1.5}};
        auto D = pairwise_distances(pts);
        EXPECT_NEAR(meb_radius_from_distances(D, {0, 1, 2, 3}), 2.5, 1e-12);
    }
    {   // equilateral triangle (side 1) plus a point just above its centroid: support
        // is the triangle, so the radius stays its circumradius 1/sqrt(3)
        const double h = std::sqrt(3.0) / 2.0;
        std::vector<std::vector<double>> pts = {
            {0, 0, 0}, {1, 0, 0}, {0.5, h, 0}, {0.5, h / 3.0, 0.1}};
        auto D = pairwise_distances(pts);
        EXPECT_NEAR(meb_radius_from_distances(D, {0, 1, 2, 3}), 1.0 / std::sqrt(3.0), 1e-9);
    }
}

// Five points: a square pyramid whose MEB is supported by two opposite base corners
// plus the apex, giving exactly 0.75.
TEST(TopoMeb, SquarePyramidFivePoints) {
    std::vector<std::vector<double>> pts = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1.0}};
    auto D = pairwise_distances(pts);
    EXPECT_NEAR(meb_radius_from_distances(D, {0, 1, 2, 3, 4}), 0.75, 1e-9);
}

// Above kMaxMebExactPoints the Badoiu-Clarkson core-set iteration takes over. It must
// still land on the true radius for a symmetric configuration, and must not under-report.
TEST(TopoMeb, LargeSubsetApproximationPath) {
    const int n = 20;
    ASSERT_GT(n, kMaxMebExactPoints);
    std::vector<std::vector<double>> pts;
    std::vector<int> idx;
    for (int i = 0; i < n; ++i) {
        const double a = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n);
        pts.push_back({5.0 * std::cos(a), 5.0 * std::sin(a)});
        idx.push_back(i);
    }
    auto D = pairwise_distances(pts);
    EXPECT_NEAR(meb_radius_from_distances(D, idx), 5.0, 1e-6);
}

// MEB is monotone under supersets -- the property the dim>=3 expansion relies on for
// face-closure, so it is worth asserting directly.
TEST(TopoMeb, MonotoneUnderSupersets) {
    auto D = pairwise_distances(kTetraPlusCentre);
    const double pair01 = meb_radius_from_distances(D, {0, 1});
    const double tri012 = meb_radius_from_distances(D, {0, 1, 2});
    const double tet0123 = meb_radius_from_distances(D, {0, 1, 2, 3});
    const double all5 = meb_radius_from_distances(D, {0, 1, 2, 3, 4});
    EXPECT_LE(pair01, tri012 + 1e-12);
    EXPECT_LE(tri012, tet0123 + 1e-12);
    EXPECT_LE(tet0123, all5 + 1e-12);
    // The centre is inside the tetrahedron's circumball, so adding it changes nothing.
    EXPECT_NEAR(all5, tet0123, 1e-9);
}

// Malformed input must return 0 rather than read out of bounds (this module returns
// plain values, never Result, so there is no error channel to use).
TEST(TopoMeb, DegenerateArguments) {
    std::vector<std::vector<double>> D = {{0, 1}, {1, 0}};
    EXPECT_EQ(meb_radius_from_distances(D, {}), 0.0);
    EXPECT_EQ(meb_radius_from_distances(D, {0}), 0.0);
    EXPECT_EQ(meb_radius_from_distances(D, {0, 7}), 0.0);
    EXPECT_EQ(meb_radius_from_distances(D, {-1, 0}), 0.0);
    std::vector<std::vector<double>> ragged = {{0, 1}, {1}};
    EXPECT_EQ(meb_radius_from_distances(ragged, {0, 1}), 0.0);
    std::vector<std::vector<double>> empty;
    EXPECT_EQ(meb_radius_from_distances(empty, {0}), 0.0);
}

// ---- cech_complex at max_dim >= 3 ----

// Exact simplex counts on 5 points at three stated epsilons.
TEST(TopoCechHighDim, MaxDim3OnFivePointsSimplexCounts) {
    auto D = pairwise_distances(kTetraPlusCentre);
    // Reference MEB radii for this configuration (all verified above/by hand):
    //   tetra edge pair      sqrt(2)             ~ 1.41421356
    //   centre-vertex pair   sqrt(3)/2           ~ 0.86602540
    //   tetra face           2*sqrt(2)/sqrt(3)   ~ 1.63299316
    //   {v_i,v_j,centre}     sqrt(2)             ~ 1.41421356  (obtuse -> longest/2)
    //   {3 vertices,centre}  2*sqrt(2)/sqrt(3)   ~ 1.63299316
    //   {4 vertices}         sqrt(3)             ~ 1.73205081
    {
        // Only the 6 {v_i,v_j,centre} triangles clear 1.50; no tetrahedron can.
        auto sc = cech_complex(D, 1.50, 3);
        EXPECT_EQ(counts_of(sc), (std::vector<int>{5, 10, 6}));
        EXPECT_EQ(sc.dimension(), 2);
    }
    {
        // All 10 triangles clear 1.70, and so do the 4 tetrahedra containing the centre.
        auto sc = cech_complex(D, 1.70, 3);
        EXPECT_EQ(counts_of(sc), (std::vector<int>{5, 10, 10, 4}));
        EXPECT_EQ(sc.euler_characteristic(), 1);
        EXPECT_EQ(sc.betti_numbers(), (std::vector<int>{1, 0, 0, 0}));
    }
    {
        // Past sqrt(3) the outer tetrahedron joins too: the full 3-skeleton of a
        // 4-simplex, i.e. a triangulated S^3.
        auto sc = cech_complex(D, 1.74, 3);
        EXPECT_EQ(counts_of(sc), (std::vector<int>{5, 10, 10, 5}));
        EXPECT_EQ(sc.euler_characteristic(), 0);
        EXPECT_EQ(sc.betti_numbers(), (std::vector<int>{1, 0, 0, 1}));
    }
}

// The dim-3 analogue of TopoCech.DivergesFromVietorisRipsFlagRule: at epsilon = 1.70
// every facet of {v0,v1,v2,v3} is in the complex, yet the tetrahedron itself is not,
// because its own MEB radius sqrt(3) exceeds epsilon. Čech is not a flag complex at any
// dimension, so the expansion must test each candidate rather than infer it.
TEST(TopoCechHighDim, NotAFlagComplexAtDimensionThree) {
    auto D = pairwise_distances(kTetraPlusCentre);
    const double eps = 1.70;
    ASSERT_LT(meb_radius_from_distances(D, {0, 1, 2}), eps);   // every facet qualifies
    ASSERT_GT(meb_radius_from_distances(D, {0, 1, 2, 3}), eps);  // the 3-simplex does not

    auto sc = cech_complex(D, eps, 3);
    std::set<Simplex> all(sc.all_simplices().begin(), sc.all_simplices().end());
    EXPECT_EQ(all.count(Simplex{0, 1, 2}), 1u);
    EXPECT_EQ(all.count(Simplex{0, 1, 3}), 1u);
    EXPECT_EQ(all.count(Simplex{0, 2, 3}), 1u);
    EXPECT_EQ(all.count(Simplex{1, 2, 3}), 1u);
    EXPECT_EQ(all.count(Simplex{0, 1, 2, 3}), 0u);
    // Vietoris-Rips, being a flag complex, does fill it in at the matching threshold.
    auto vr = vietoris_rips(D, 2.0 * eps, 3);
    std::set<Simplex> rips(vr.all_simplices().begin(), vr.all_simplices().end());
    EXPECT_EQ(rips.count(Simplex{0, 1, 2, 3}), 1u);
}

// Raising max_dim must not perturb the dim<=2 skeleton -- neither its contents nor the
// insertion order that simplices(k) reports.
TEST(TopoCechHighDim, LowSkeletonUnchangedByHigherMaxDim) {
    auto D = pairwise_distances(kTetraPlusCentre);
    for (double eps : {0.5, 0.9, 1.42, 1.5, 1.7, 1.74, 5.0}) {
        auto lo = cech_complex(D, eps, 2);
        auto hi = cech_complex(D, eps, 6);
        for (int k = 0; k <= 2; ++k)
            EXPECT_EQ(lo.simplices(k), hi.simplices(k)) << "eps=" << eps << " k=" << k;
    }
}

// Face-closure at dimension 3, and the standard Čech(ε) ⊆ VR(2ε) containment.
TEST(TopoCechHighDim, FaceClosedAndSubcomplexOfRips) {
    auto D = pairwise_distances(kTetraPlusCentre);
    const double eps = 1.70;
    auto sc = cech_complex(D, eps, 3);
    std::set<Simplex> all(sc.all_simplices().begin(), sc.all_simplices().end());
    for (const auto& s : sc.all_simplices()) {
        if (s.size() < 2) continue;
        for (std::size_t drop = 0; drop < s.size(); ++drop) {
            Simplex face;
            for (std::size_t i = 0; i < s.size(); ++i)
                if (i != drop) face.push_back(s[i]);
            EXPECT_TRUE(all.count(face) > 0);
        }
    }
    auto vr = vietoris_rips(D, 2.0 * eps, 3);
    std::set<Simplex> rips(vr.all_simplices().begin(), vr.all_simplices().end());
    for (const auto& s : sc.all_simplices()) EXPECT_TRUE(rips.count(s) > 0);
}

// Requests above kMaxCechDim are clamped, not honoured, and must not hang.
TEST(TopoCechHighDim, MaxDimClampedAtCeiling) {
    std::vector<std::vector<double>> pts;
    for (int i = 0; i < 12; ++i)
        pts.push_back({0.001 * static_cast<double>(i), 0.0, 0.0});
    auto D = pairwise_distances(pts);
    auto sc = cech_complex(D, 100.0, 100000);
    EXPECT_EQ(sc.dimension(), kMaxCechDim);
    EXPECT_TRUE(sc.simplices(kMaxCechDim + 1).empty());
    // Complete through its own top dimension: C(12, k+1) simplices at every dimension.
    EXPECT_EQ(counts_of(sc),
              (std::vector<int>{12, 66, 220, 495, 792, 924, 792, 495, 220}));
}

// Monotone in epsilon: Čech(e1) is a subcomplex of Čech(e2) whenever e1 <= e2.
TEST(TopoCechHighDim, MonotoneInEpsilon) {
    auto D = pairwise_distances(kTetraPlusCentre);
    auto small = cech_complex(D, 1.50, 4);
    auto large = cech_complex(D, 1.74, 4);
    std::set<Simplex> big(large.all_simplices().begin(), large.all_simplices().end());
    for (const auto& s : small.all_simplices()) EXPECT_TRUE(big.count(s) > 0);
}

// Degenerate inputs at max_dim >= 3 behave exactly as they do at max_dim = 2.
TEST(TopoCechHighDim, DegenerateInputsAtHighMaxDim) {
    {
        std::vector<std::vector<double>> D;
        auto sc = cech_complex(D, 1.0, 5);
        EXPECT_EQ(sc.dimension(), -1);
        EXPECT_TRUE(sc.all_simplices().empty());
    }
    {
        std::vector<std::vector<double>> D = {{0.0}};
        auto sc = cech_complex(D, 1.0, 5);
        EXPECT_EQ(sc.dimension(), 0);
    }
    {
        // Four coincident points at epsilon 0: every subset has MEB radius 0, so the
        // full 3-simplex is present.
        std::vector<std::vector<double>> pts = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        auto D = pairwise_distances(pts);
        auto sc = cech_complex(D, 0.0, 3);
        EXPECT_EQ(sc.dimension(), 3);
        EXPECT_EQ(counts_of(sc), (std::vector<int>{4, 6, 4, 1}));
    }
    {
        // A negative max_dim still runs the unconditional vertex and edge passes, and
        // builds nothing above them -- unchanged from the pre-existing behaviour.
        auto D = pairwise_distances(kTetraPlusCentre);
        auto sc = cech_complex(D, 5.0, -1);
        EXPECT_EQ(sc.dimension(), 1);
        EXPECT_EQ(counts_of(sc), (std::vector<int>{5, 10}));
    }
}
