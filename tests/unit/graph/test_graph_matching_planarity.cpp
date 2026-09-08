#include "ms/graph/graph.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <variant>
#include <vector>
#include <gtest/gtest.h>

using namespace ms::graph;

namespace {

using PairVec = std::vector<std::pair<int, int>>;

// Undirected graph from an explicit weighted edge list.
Graph weighted(int n, const std::vector<Edge>& es) {
    Graph G(n, false);
    for (const auto& e : es) G.add_edge(e.from, e.to, e.weight);
    return G;
}

// Complete graph K_n on vertices [offset, offset+n).
void add_clique(Graph& G, int offset, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            G.add_edge(offset + i, offset + j);
}

// Same validity predicate the cardinality-matching tests use: pairs are
// ordered, vertex-disjoint, and really are edges of G.
bool is_valid_matching(const Graph& G, const PairVec& matching) {
    std::set<int> used;
    std::set<std::pair<int, int>> edge_set;
    for (const auto& e : G.edges()) {
        const int a = std::min(e.from, e.to), b = std::max(e.from, e.to);
        if (a != b) edge_set.insert({a, b});
    }
    for (const auto& [u, v] : matching) {
        if (u >= v) return false;
        if (used.count(u) || used.count(v)) return false;
        if (!edge_set.count({u, v})) return false;
        used.insert(u);
        used.insert(v);
    }
    return true;
}

// Complete bipartite K_{a,b} on left [0,a) and right [a,a+b).
Graph make_complete_bipartite(int a, int b) {
    Graph G(a + b, false);
    for (int i = 0; i < a; ++i)
        for (int j = a; j < a + b; ++j)
            G.add_edge(i, j);
    return G;
}

Graph make_k33() { return make_complete_bipartite(3, 3); }

// Petersen graph: outer 5-cycle 0..4, inner pentagram 5..9, spokes i--i+5.
Graph make_petersen() {
    Graph G(10, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 4); G.add_edge(4, 0);
    for (int i = 0; i < 5; ++i) G.add_edge(i, i + 5);
    G.add_edge(5, 7); G.add_edge(7, 9); G.add_edge(9, 6); G.add_edge(6, 8); G.add_edge(8, 5);
    return G;
}

// r x c square grid; vertex (i,j) has id i*c + j.
Graph make_grid(int r, int c) {
    Graph G(r * c, false);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) {
            if (j + 1 < c) G.add_edge(i * c + j, i * c + j + 1);
            if (i + 1 < r) G.add_edge(i * c + j, (i + 1) * c + j);
        }
    return G;
}

// 3-cube Q3: ids 0..7, edges between ids differing in exactly one bit.
Graph make_cube() {
    Graph G(8, false);
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 3; ++b) {
            const int j = i ^ (1 << b);
            if (j > i) G.add_edge(i, j);
        }
    return G;
}

// Octahedron K_{2,2,2} = K6 minus the perfect matching {(0,1),(2,3),(4,5)}.
Graph make_octahedron() {
    Graph G(6, false);
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
            if (i / 2 != j / 2) G.add_edge(i, j);
    return G;
}

// Wagner graph V8 (Moebius ladder M4): C8 plus the four main diagonals.
Graph make_wagner_v8() {
    Graph G(8, false);
    for (int i = 0; i < 8; ++i) G.add_edge(i, (i + 1) % 8);
    for (int i = 0; i < 4; ++i) G.add_edge(i, i + 4);
    return G;
}

// Traces the faces of a rotation system using the documented rule: after the
// half-edge (v,w) comes (w,x) with x the cyclic PREDECESSOR of v in emb[w].
int count_faces(const std::vector<std::vector<int>>& emb) {
    std::set<std::pair<int, int>> seen;
    int faces = 0;
    for (int v = 0; v < static_cast<int>(emb.size()); ++v) {
        for (int w : emb[static_cast<std::size_t>(v)]) {
            if (seen.count({v, w})) continue;
            ++faces;
            int a = v, b = w;
            do {
                seen.insert({a, b});
                const std::vector<int>& L = emb[static_cast<std::size_t>(b)];
                const auto it = std::find(L.begin(), L.end(), a);
                if (it == L.end()) return -1;   // not a rotation system at all
                const std::size_t idx = static_cast<std::size_t>(it - L.begin());
                const int next = L[(idx + L.size() - 1) % L.size()];
                a = b;
                b = next;
            } while (!seen.count({a, b}));
        }
    }
    return faces;
}

// Every neighbour list is mutual, duplicate-free, and only names real edges.
bool embedding_is_symmetric(const Graph& G, const std::vector<std::vector<int>>& emb) {
    std::set<std::pair<int, int>> edge_set;
    for (const auto& e : G.edges()) {
        const int a = std::min(e.from, e.to), b = std::max(e.from, e.to);
        if (a != b) edge_set.insert({a, b});
    }
    for (int v = 0; v < static_cast<int>(emb.size()); ++v) {
        std::set<int> seen;
        for (int w : emb[static_cast<std::size_t>(v)]) {
            if (w == v) return false;
            if (!seen.insert(w).second) return false;
            if (!edge_set.count({std::min(v, w), std::max(v, w)})) return false;
            const std::vector<int>& L = emb[static_cast<std::size_t>(w)];
            if (std::find(L.begin(), L.end(), v) == L.end()) return false;
        }
    }
    return true;
}

// An edge-minimal non-planar graph is, by Kuratowski's theorem, exactly a
// subdivision of K5 (E - V == 5, five degree-4 branch vertices) or of K3,3
// (E - V == 3, six degree-3 branch vertices); every other vertex has degree 2.
bool is_kuratowski_subdivision(const std::vector<Edge>& es) {
    if (es.empty()) return false;
    std::map<int, int> deg;
    for (const auto& e : es) {
        if (e.from >= e.to) return false;
        ++deg[e.from];
        ++deg[e.to];
    }
    const int ev = static_cast<int>(es.size()) - static_cast<int>(deg.size());
    int d2 = 0, d3 = 0, d4 = 0, other = 0;
    for (const auto& [v, d] : deg) {
        static_cast<void>(v);
        if (d == 2) ++d2;
        else if (d == 3) ++d3;
        else if (d == 4) ++d4;
        else ++other;
    }
    if (other != 0) return false;
    if (ev == 5) return d4 == 5 && d3 == 0;
    if (ev == 3) return d3 == 6 && d4 == 0;
    return false;
}

// Regular icosahedron: 12 vertices, 30 edges (exactly the 3V - 6 bound),
// every vertex of degree 5. Poles 0 and 11, two staggered pentagons.
Graph make_icosahedron() {
    static const int kEdges[30][2] = {
        {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 1},
        {1, 6}, {2, 6}, {2, 7}, {3, 7}, {3, 8}, {4, 8}, {4, 9}, {5, 9}, {5, 10}, {1, 10},
        {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 6}, {6, 11}, {7, 11}, {8, 11}, {9, 11}, {10, 11}};
    Graph G(12, false);
    for (const auto& e : kEdges) G.add_edge(e[0], e[1]);
    return G;
}

// Regular dodecahedron: 20 vertices, 30 edges, every vertex of degree 3.
Graph make_dodecahedron() {
    static const int kEdges[30][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0},
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9},
        {5, 10}, {6, 11}, {7, 12}, {8, 13}, {9, 14},
        {10, 6}, {11, 7}, {12, 8}, {13, 9}, {14, 5},
        {10, 15}, {11, 16}, {12, 17}, {13, 18}, {14, 19},
        {15, 16}, {16, 17}, {17, 18}, {18, 19}, {19, 15}};
    Graph G(20, false);
    for (const auto& e : kEdges) G.add_edge(e[0], e[1]);
    return G;
}

Graph rebuild_from_edges(int n, const std::vector<Edge>& es) {
    Graph G(n, false);
    for (const auto& e : es) G.add_edge(e.from, e.to, e.weight);
    return G;
}

} // namespace

// ---- Maximum weight matching (Edmonds primal-dual blossom) ----

TEST(WeightedMatching, EmptyGraph) {
    Graph G(0, false);
    EXPECT_TRUE(max_weight_matching(G).empty());
    EXPECT_NEAR(max_weight_matching_value(G), 0.0, 1e-12);
    EXPECT_TRUE(max_weight_matching(G, true).empty());
    EXPECT_NEAR(max_weight_matching_value(G, true), 0.0, 1e-12);
}

TEST(WeightedMatching, IsolatedVertices) {
    Graph G(3, false);
    EXPECT_TRUE(max_weight_matching(G).empty());
    EXPECT_NEAR(max_weight_matching_value(G), 0.0, 1e-12);
}

TEST(WeightedMatching, SingleEdge) {
    Graph G = weighted(2, {{0, 1, 5.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_NEAR(max_weight_matching_value(G), 5.0, 1e-12);
}

TEST(WeightedMatching, SingleNegativeEdgeSkipped) {
    Graph G = weighted(2, {{0, 1, -3.0}});
    EXPECT_TRUE(max_weight_matching(G).empty());
    EXPECT_NEAR(max_weight_matching_value(G), 0.0, 1e-12);
    // Forcing maximum cardinality takes the loss.
    const PairVec mc = max_weight_matching(G, true);
    ASSERT_EQ(mc.size(), 1u);
    EXPECT_EQ(mc[0], (std::pair<int, int>{0, 1}));
    EXPECT_NEAR(max_weight_matching_value(G, true), -3.0, 1e-12);
}

TEST(WeightedMatching, PathOfThreePicksHeavier) {
    Graph G = weighted(3, {{0, 1, 5.0}, {1, 2, 11.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{1, 2}));
    EXPECT_NEAR(max_weight_matching_value(G), 11.0, 1e-12);
}

TEST(WeightedMatching, PathOfFourWeightBeatsCardinality) {
    Graph G = weighted(4, {{0, 1, 5.0}, {1, 2, 11.0}, {2, 3, 5.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{1, 2}));
    EXPECT_NEAR(max_weight_matching_value(G), 11.0, 1e-12);

    const PairVec mc = max_weight_matching(G, true);
    ASSERT_EQ(mc.size(), 2u);
    EXPECT_EQ(mc[0], (std::pair<int, int>{0, 1}));
    EXPECT_EQ(mc[1], (std::pair<int, int>{2, 3}));
    EXPECT_NEAR(max_weight_matching_value(G, true), 10.0, 1e-12);
    EXPECT_EQ(maximum_matching(G).size(), 2u);
}

TEST(WeightedMatching, TriangleHeaviestEdge) {
    Graph G = weighted(3, {{0, 1, 3.0}, {1, 2, 1.0}, {0, 2, 2.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_NEAR(max_weight_matching_value(G), 3.0, 1e-12);
}

TEST(WeightedMatching, K4DefeatsGreedy) {
    // Greedy heaviest-first would take (0,1)=6 then (2,3)=1 for 7.
    Graph G = weighted(4, {{0, 1, 6.0}, {0, 2, 5.0}, {0, 3, 1.0},
                           {1, 2, 1.0}, {1, 3, 5.0}, {2, 3, 1.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 2u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 2}));
    EXPECT_EQ(m[1], (std::pair<int, int>{1, 3}));
    EXPECT_NEAR(max_weight_matching_value(G), 10.0, 1e-12);
}

TEST(WeightedMatching, FiveCycleWeighted) {
    Graph G = weighted(5, {{0, 1, 1.0}, {1, 2, 2.0}, {2, 3, 3.0},
                           {3, 4, 4.0}, {0, 4, 5.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 2u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 4}));
    EXPECT_EQ(m[1], (std::pair<int, int>{2, 3}));
    EXPECT_NEAR(max_weight_matching_value(G), 8.0, 1e-12);
    // A 5-cycle's maximum cardinality is also 2, so nothing changes.
    EXPECT_EQ(max_weight_matching(G, true), m);
    EXPECT_NEAR(max_weight_matching_value(G, true), 8.0, 1e-12);
}

TEST(WeightedMatching, SBlossomAugment) {
    Graph G = weighted(4, {{0, 1, 8.0}, {0, 2, 9.0}, {1, 2, 10.0}, {2, 3, 7.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 2u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_EQ(m[1], (std::pair<int, int>{2, 3}));
    EXPECT_NEAR(max_weight_matching_value(G), 15.0, 1e-12);
}

TEST(WeightedMatching, SBlossomRelabelledAsT) {
    Graph G = weighted(6, {{0, 1, 9.0}, {0, 2, 8.0}, {1, 2, 10.0},
                           {0, 3, 5.0}, {3, 4, 4.0}, {0, 5, 3.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 3u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 5}));
    EXPECT_EQ(m[1], (std::pair<int, int>{1, 2}));
    EXPECT_EQ(m[2], (std::pair<int, int>{3, 4}));
    EXPECT_NEAR(max_weight_matching_value(G), 17.0, 1e-12);
}

TEST(WeightedMatching, NestedSBlossom) {
    Graph G = weighted(6, {{0, 1, 9.0}, {0, 2, 9.0}, {1, 2, 10.0}, {1, 3, 8.0},
                           {2, 4, 8.0}, {3, 4, 10.0}, {4, 5, 6.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 3u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 2}));
    EXPECT_EQ(m[1], (std::pair<int, int>{1, 3}));
    EXPECT_EQ(m[2], (std::pair<int, int>{4, 5}));
    EXPECT_NEAR(max_weight_matching_value(G), 23.0, 1e-12);
}

TEST(WeightedMatching, BlossomRelabelledAsSInsideNested) {
    Graph G = weighted(8, {{0, 1, 10.0}, {0, 6, 10.0}, {1, 2, 12.0}, {2, 3, 20.0},
                           {2, 4, 20.0}, {3, 4, 25.0}, {4, 5, 10.0}, {5, 6, 10.0},
                           {6, 7, 8.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 4u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_EQ(m[1], (std::pair<int, int>{2, 3}));
    EXPECT_EQ(m[2], (std::pair<int, int>{4, 5}));
    EXPECT_EQ(m[3], (std::pair<int, int>{6, 7}));
    EXPECT_NEAR(max_weight_matching_value(G), 48.0, 1e-12);
}

TEST(WeightedMatching, NestedBlossomExpandRecursively) {
    Graph G = weighted(8, {{0, 1, 8.0}, {0, 2, 8.0}, {1, 2, 10.0}, {1, 3, 12.0},
                           {2, 4, 12.0}, {3, 4, 14.0}, {3, 5, 12.0}, {4, 6, 12.0},
                           {5, 6, 14.0}, {6, 7, 12.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 4u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_EQ(m[1], (std::pair<int, int>{2, 4}));
    EXPECT_EQ(m[2], (std::pair<int, int>{3, 5}));
    EXPECT_EQ(m[3], (std::pair<int, int>{6, 7}));
    EXPECT_NEAR(max_weight_matching_value(G), 44.0, 1e-12);
}

TEST(WeightedMatching, NegativeWeightsMaxCardinality) {
    Graph G = weighted(4, {{0, 1, 2.0}, {0, 2, -2.0}, {1, 2, 1.0},
                           {1, 3, -1.0}, {2, 3, -6.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_NEAR(max_weight_matching_value(G), 2.0, 1e-12);

    const PairVec mc = max_weight_matching(G, true);
    ASSERT_EQ(mc.size(), 2u);
    EXPECT_EQ(mc[0], (std::pair<int, int>{0, 2}));
    EXPECT_EQ(mc[1], (std::pair<int, int>{1, 3}));
    EXPECT_NEAR(max_weight_matching_value(G, true), -3.0, 1e-12);
}

TEST(WeightedMatching, Disconnected) {
    Graph G = weighted(5, {{0, 1, 3.0}, {2, 3, 1.0}, {3, 4, 7.0}});
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 2u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_EQ(m[1], (std::pair<int, int>{3, 4}));
    EXPECT_NEAR(max_weight_matching_value(G), 10.0, 1e-12);
}

TEST(WeightedMatching, SelfLoopIgnored) {
    Graph G(2, false);
    G.add_edge(0, 0, 100.0);
    G.add_edge(0, 1, 3.0);
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_NEAR(max_weight_matching_value(G), 3.0, 1e-12);
}

TEST(WeightedMatching, ParallelEdgesTakeHeaviest) {
    Graph G(2, false);
    G.add_edge(0, 1, 3.0);
    G.add_edge(0, 1, 9.0);
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_NEAR(max_weight_matching_value(G), 9.0, 1e-12);
}

TEST(WeightedMatching, DirectedTreatedAsUndirected) {
    Graph G(3, true);
    G.add_edge(0, 1, 4.0);
    G.add_edge(1, 0, 9.0);   // opposite direction, heavier: collapses to 9
    G.add_edge(1, 2, 1.0);
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0], (std::pair<int, int>{0, 1}));
    EXPECT_NEAR(max_weight_matching_value(G), 9.0, 1e-12);
}

TEST(WeightedMatching, UnitWeightsMatchCardinality) {
    // Triangle 0-1-2 with pendant edges 0-3 and 1-4 (needs a blossom).
    Graph G(5, false);
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(0, 2);
    G.add_edge(0, 3);
    G.add_edge(1, 4);
    const PairVec m = max_weight_matching(G);
    EXPECT_EQ(m.size(), 2u);
    EXPECT_TRUE(is_valid_matching(G, m));
    EXPECT_NEAR(max_weight_matching_value(G), 2.0, 1e-12);
    EXPECT_EQ(m.size(), maximum_matching(G).size());
}

TEST(WeightedMatching, PetersenUnitWeightsPerfect) {
    Graph G = make_petersen();
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 5u);
    EXPECT_TRUE(is_valid_matching(G, m));
    EXPECT_NEAR(max_weight_matching_value(G), 5.0, 1e-12);
    std::vector<int> covered(10, 0);
    for (const auto& [u, v] : m) {
        ++covered[static_cast<std::size_t>(u)];
        ++covered[static_cast<std::size_t>(v)];
    }
    for (int v = 0; v < 10; ++v) EXPECT_EQ(covered[static_cast<std::size_t>(v)], 1);
}

TEST(WeightedMatching, CompleteGraphK6UnitWeights) {
    Graph G(6, false);
    add_clique(G, 0, 6);
    const PairVec m = max_weight_matching(G);
    ASSERT_EQ(m.size(), 3u);
    EXPECT_TRUE(is_valid_matching(G, m));
    EXPECT_NEAR(max_weight_matching_value(G), 3.0, 1e-12);
}

TEST(WeightedMatching, ZeroWeightGraph) {
    Graph G = weighted(4, {{0, 1, 0.0}, {1, 2, 0.0}, {2, 3, 0.0}});
    const PairVec m = max_weight_matching(G);
    EXPECT_NEAR(max_weight_matching_value(G), 0.0, 1e-12);
    EXPECT_TRUE(is_valid_matching(G, m));
    const PairVec mc = max_weight_matching(G, true);
    EXPECT_EQ(mc.size(), 2u);
    EXPECT_TRUE(is_valid_matching(G, mc));
    EXPECT_NEAR(max_weight_matching_value(G, true), 0.0, 1e-12);
}

TEST(WeightedMatching, ValueAgreesWithEdges) {
    Graph G = weighted(8, {{0, 1, 10.0}, {0, 6, 10.0}, {1, 2, 12.0}, {2, 3, 20.0},
                           {2, 4, 20.0}, {3, 4, 25.0}, {4, 5, 10.0}, {5, 6, 10.0},
                           {6, 7, 8.0}});
    const PairVec m = max_weight_matching(G);
    std::map<std::pair<int, int>, double> wt;
    for (const auto& e : G.edges()) {
        const std::pair<int, int> k{std::min(e.from, e.to), std::max(e.from, e.to)};
        if (!wt.count(k) || e.weight > wt[k]) wt[k] = e.weight;
    }
    double total = 0.0;
    for (const auto& p : m) total += wt[p];
    EXPECT_NEAR(max_weight_matching_value(G), total, 1e-12);
    EXPECT_NEAR(max_weight_matching_value(G), 48.0, 1e-12);
}

TEST(WeightedMatching, NeverBeatsBruteForceOnSmallGraphs) {
    // A weighted 6-vertex graph whose optimum needs a blossom expansion.
    Graph G = weighted(6, {{0, 1, 4.0}, {1, 2, 5.0}, {2, 0, 6.0}, {2, 3, 7.0},
                           {3, 4, 3.0}, {4, 5, 9.0}, {5, 3, 2.0}, {0, 5, 1.0}});
    const PairVec m = max_weight_matching(G);
    EXPECT_TRUE(is_valid_matching(G, m));
    // {(0,2), (1, ...)} cannot beat {(1,2) 5} + {(4,5) 9} = 14 vs
    // {(0,1) 4} + {(2,3) 7} + ... ; the optimum is (0,1)=4,(2,3)=7,(4,5)=9 = 20.
    ASSERT_EQ(m.size(), 3u);
    EXPECT_NEAR(max_weight_matching_value(G), 20.0, 1e-12);
}

// ---- Exact planarity (Left-Right criterion) ----

TEST(Planarity, EmptyGraph) {
    Graph G(0, false);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(emb->size(), 0u);
    EXPECT_TRUE(kuratowski_subgraph(G).empty());
}

TEST(Planarity, SingleVertexAndEdgelessArePlanar) {
    Graph G1(1, false);
    EXPECT_TRUE(is_planar(G1));
    const auto e1 = planar_embedding(G1);
    ASSERT_TRUE(e1.has_value());
    ASSERT_EQ(e1->size(), 1u);
    EXPECT_TRUE((*e1)[0].empty());

    Graph G5(5, false);
    EXPECT_TRUE(is_planar(G5));
    const auto e5 = planar_embedding(G5);
    ASSERT_TRUE(e5.has_value());
    ASSERT_EQ(e5->size(), 5u);
    for (const auto& row : *e5) EXPECT_TRUE(row.empty());
}

TEST(Planarity, SingleEdgeIsPlanar) {
    Graph G(2, false);
    G.add_edge(0, 1);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    ASSERT_EQ(emb->size(), 2u);
    EXPECT_EQ((*emb)[0], std::vector<int>{1});
    EXPECT_EQ((*emb)[1], std::vector<int>{0});
    EXPECT_EQ(count_faces(*emb), 1);
}

TEST(Planarity, TreeIsPlanar) {
    Graph G(4, false);
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    // V - E + F == 2 for one component with edges: 4 - 3 + 1.
    EXPECT_EQ(count_faces(*emb), 1);
    EXPECT_TRUE(embedding_is_symmetric(G, *emb));
}

TEST(Planarity, K4IsPlanar) {
    Graph G(4, false);
    add_clique(G, 0, 4);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 4);          // 4 - 6 + 4 == 2
    EXPECT_TRUE(embedding_is_symmetric(G, *emb));
    for (const auto& row : *emb) EXPECT_EQ(row.size(), 3u);
}

TEST(Planarity, K5IsNotPlanar) {
    Graph G(5, false);
    add_clique(G, 0, 5);
    EXPECT_FALSE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_FALSE(emb.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(emb.error()));
}

TEST(Planarity, K5MinusEdgeIsPlanar) {
    Graph G(5, false);
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (!(i == 3 && j == 4)) G.add_edge(i, j);
    EXPECT_EQ(G.n_edges(), 9);                // exactly the 3V - 6 bound
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 6);          // 5 - 9 + 6 == 2
}

TEST(Planarity, K33IsNotPlanarButHeuristicSaysYes) {
    Graph G = make_k33();
    // The headline regression: the Euler screen accepts K3,3 (9 <= 12) ...
    EXPECT_TRUE(is_planar_k5_k33_check(G));
    // ... and the exact test correctly rejects it.
    EXPECT_FALSE(is_planar(G));
}

TEST(Planarity, K33MinusEdgeIsPlanar) {
    Graph G(6, false);
    for (int a = 0; a < 3; ++a)
        for (int b = 3; b < 6; ++b)
            if (!(a == 2 && b == 5)) G.add_edge(a, b);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 4);          // 6 - 8 + 4 == 2
}

TEST(Planarity, K5SubdivisionIsNotPlanar) {
    // K5 on 0..4 with the edge (3,4) subdivided by the new vertex 5.
    Graph G(6, false);
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (!(i == 3 && j == 4)) G.add_edge(i, j);
    G.add_edge(3, 5);
    G.add_edge(4, 5);
    EXPECT_TRUE(is_planar_k5_k33_check(G));   // 11 <= 12
    EXPECT_FALSE(is_planar(G));
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    EXPECT_EQ(ks.size(), 11u);                // every edge is essential
    EXPECT_TRUE(is_kuratowski_subdivision(ks));
}

TEST(Planarity, K33SubdivisionIsNotPlanar) {
    // K3,3 with the edge (0,3) subdivided by the new vertex 6.
    Graph G(7, false);
    for (int a = 0; a < 3; ++a)
        for (int b = 3; b < 6; ++b)
            if (!(a == 0 && b == 3)) G.add_edge(a, b);
    G.add_edge(0, 6);
    G.add_edge(3, 6);
    EXPECT_TRUE(is_planar_k5_k33_check(G));   // 10 <= 15
    EXPECT_FALSE(is_planar(G));
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    ASSERT_EQ(ks.size(), 10u);
    std::map<int, int> deg;
    std::set<int> vs;
    for (const auto& e : ks) { ++deg[e.from]; ++deg[e.to]; vs.insert(e.from); vs.insert(e.to); }
    EXPECT_EQ(static_cast<int>(ks.size()) - static_cast<int>(vs.size()), 3);
    int d3 = 0;
    for (const auto& [v, d] : deg) { static_cast<void>(v); if (d == 3) ++d3; }
    EXPECT_EQ(d3, 6);
    EXPECT_TRUE(is_kuratowski_subdivision(ks));
}

TEST(Planarity, PetersenIsNotPlanar) {
    Graph G = make_petersen();
    EXPECT_TRUE(is_planar_k5_k33_check(G));   // 15 <= 24
    EXPECT_FALSE(is_planar(G));
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    EXPECT_FALSE(ks.empty());
    EXPECT_TRUE(is_kuratowski_subdivision(ks));
    std::set<std::pair<int, int>> present;
    for (const auto& e : G.edges())
        present.insert({std::min(e.from, e.to), std::max(e.from, e.to)});
    for (const auto& e : ks) EXPECT_TRUE(present.count({e.from, e.to}) == 1u);
}

TEST(Planarity, WagnerV8IsNotPlanar) {
    Graph G = make_wagner_v8();
    EXPECT_TRUE(is_planar_k5_k33_check(G));   // 12 <= 18
    EXPECT_FALSE(is_planar(G));
    EXPECT_TRUE(is_kuratowski_subdivision(kuratowski_subgraph(G)));
}

TEST(Planarity, K34IsNotPlanar) {
    Graph G = make_complete_bipartite(3, 4);
    EXPECT_TRUE(is_planar_k5_k33_check(G));   // 12 <= 15
    EXPECT_FALSE(is_planar(G));
}

TEST(Planarity, K23IsPlanar) {
    Graph G = make_complete_bipartite(2, 3);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 3);          // 5 - 6 + 3 == 2
}

TEST(Planarity, GridIsPlanar) {
    Graph G = make_grid(3, 3);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 5);          // 9 - 12 + 5 == 2
    EXPECT_TRUE(embedding_is_symmetric(G, *emb));
}

TEST(Planarity, CubeIsPlanar) {
    Graph G = make_cube();
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 6);          // 8 - 12 + 6 == 2
    for (const auto& row : *emb) EXPECT_EQ(row.size(), 3u);
}

TEST(Planarity, OctahedronIsPlanar) {
    Graph G = make_octahedron();
    EXPECT_EQ(G.n_edges(), 12);               // exactly the 3V - 6 bound
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 8);          // 6 - 12 + 8 == 2
    for (const auto& row : *emb) EXPECT_EQ(row.size(), 4u);
}

TEST(Planarity, DisconnectedAllPlanar) {
    Graph G(6, false);
    add_clique(G, 0, 3);
    add_clique(G, 3, 3);
    EXPECT_TRUE(is_planar(G));
    const auto emb = planar_embedding(G);
    ASSERT_TRUE(emb.has_value());
    // A rotation system traces faces per component: 2 components with edges,
    // 6 vertices, 6 edges -> 2*2 - 6 + 6 == 4 (a plane drawing would show 3,
    // because it merges the two outer faces; the rotation system cannot).
    EXPECT_EQ(count_faces(*emb), 4);
    EXPECT_TRUE(embedding_is_symmetric(G, *emb));
}

TEST(Planarity, DisconnectedWithNonPlanarComponent) {
    Graph G(8, false);
    add_clique(G, 0, 5);   // K5
    add_clique(G, 5, 3);   // triangle, planar
    EXPECT_FALSE(is_planar(G));
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    ASSERT_EQ(ks.size(), 10u);
    for (const auto& e : ks) {
        EXPECT_LT(e.from, 5);
        EXPECT_LT(e.to, 5);
    }
    EXPECT_TRUE(is_kuratowski_subdivision(ks));
}

TEST(Planarity, SelfLoopsIgnored) {
    Graph K5loop(5, false);
    add_clique(K5loop, 0, 5);
    K5loop.add_edge(0, 0);
    EXPECT_FALSE(is_planar(K5loop));
    const std::vector<Edge> ks = kuratowski_subgraph(K5loop);
    EXPECT_EQ(ks.size(), 10u);                // the loop is not part of it
    for (const auto& e : ks) EXPECT_NE(e.from, e.to);

    Graph tri(3, false);
    add_clique(tri, 0, 3);
    tri.add_edge(1, 1);
    EXPECT_TRUE(is_planar(tri));
    const auto emb = planar_embedding(tri);
    ASSERT_TRUE(emb.has_value());
    for (int v = 0; v < 3; ++v) {
        const std::vector<int>& row = (*emb)[static_cast<std::size_t>(v)];
        EXPECT_EQ(std::find(row.begin(), row.end(), v), row.end());
        EXPECT_EQ(row.size(), 2u);
    }
    EXPECT_EQ(count_faces(*emb), 2);
}

TEST(Planarity, ParallelEdgesIgnored) {
    Graph two(2, false);
    two.add_edge(0, 1);
    two.add_edge(0, 1);
    EXPECT_TRUE(is_planar(two));
    const auto e2 = planar_embedding(two);
    ASSERT_TRUE(e2.has_value());
    EXPECT_EQ((*e2)[0].size(), 1u);
    EXPECT_EQ((*e2)[1].size(), 1u);

    Graph tri(3, false);
    add_clique(tri, 0, 3);
    add_clique(tri, 0, 3);   // every edge added a second time
    EXPECT_TRUE(is_planar(tri));
    const auto emb = planar_embedding(tri);
    ASSERT_TRUE(emb.has_value());
    for (const auto& row : *emb) {
        EXPECT_EQ(row.size(), 2u);
        std::set<int> uniq(row.begin(), row.end());
        EXPECT_EQ(uniq.size(), row.size());
    }
    EXPECT_EQ(count_faces(*emb), 2);
}

TEST(Planarity, DirectedTreatedAsUndirected) {
    Graph G(6, true);
    for (int a = 0; a < 3; ++a)
        for (int b = 3; b < 6; ++b)
            G.add_edge(a, b);   // one-way only
    EXPECT_FALSE(is_planar(G));
    EXPECT_TRUE(is_kuratowski_subdivision(kuratowski_subgraph(G)));
}

TEST(Planarity, KuratowskiOnK5IsExactlyK5) {
    Graph G(5, false);
    add_clique(G, 0, 5);
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    ASSERT_EQ(ks.size(), 10u);
    std::set<std::pair<int, int>> got;
    for (const auto& e : ks) got.insert({e.from, e.to});
    std::set<std::pair<int, int>> want;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j) want.insert({i, j});
    EXPECT_EQ(got, want);
    EXPECT_TRUE(is_kuratowski_subdivision(ks));
}

TEST(Planarity, KuratowskiOnK33IsExactlyK33) {
    Graph G = make_k33();
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    ASSERT_EQ(ks.size(), 9u);
    std::set<std::pair<int, int>> got;
    std::map<int, int> deg;
    for (const auto& e : ks) { got.insert({e.from, e.to}); ++deg[e.from]; ++deg[e.to]; }
    std::set<std::pair<int, int>> want;
    for (int a = 0; a < 3; ++a)
        for (int b = 3; b < 6; ++b) want.insert({a, b});
    EXPECT_EQ(got, want);
    EXPECT_EQ(static_cast<int>(ks.size()) - static_cast<int>(deg.size()), 3);
    for (const auto& [v, d] : deg) { static_cast<void>(v); EXPECT_EQ(d, 3); }
    EXPECT_TRUE(is_kuratowski_subdivision(ks));
}

TEST(Planarity, KuratowskiEmptyOnPlanar) {
    Graph grid = make_grid(3, 3);
    EXPECT_TRUE(kuratowski_subgraph(grid).empty());

    Graph k4(4, false);
    add_clique(k4, 0, 4);
    EXPECT_TRUE(kuratowski_subgraph(k4).empty());

    Graph tree(4, false);
    tree.add_edge(0, 1);
    tree.add_edge(1, 2);
    tree.add_edge(2, 3);
    EXPECT_TRUE(kuratowski_subgraph(tree).empty());

    Graph empty(0, false);
    EXPECT_TRUE(kuratowski_subgraph(empty).empty());
}

TEST(Planarity, KuratowskiCertificateIsMinimal) {
    Graph G = make_petersen();
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    ASSERT_FALSE(ks.empty());
    Graph H = rebuild_from_edges(G.n_vertices(), ks);
    EXPECT_FALSE(is_planar(H));
    // Edge-minimal: dropping any single edge makes it planar.
    for (std::size_t skip = 0; skip < ks.size(); ++skip) {
        std::vector<Edge> reduced;
        for (std::size_t i = 0; i < ks.size(); ++i)
            if (i != skip) reduced.push_back(ks[i]);
        EXPECT_TRUE(is_planar(rebuild_from_edges(G.n_vertices(), reduced)));
    }
}

TEST(Planarity, EmbeddingIsSymmetric) {
    std::vector<Graph> graphs;
    {
        Graph k4(4, false);
        add_clique(k4, 0, 4);
        graphs.push_back(k4);
    }
    graphs.push_back(make_cube());
    graphs.push_back(make_grid(3, 3));
    graphs.push_back(make_icosahedron());
    graphs.push_back(make_dodecahedron());
    for (const Graph& G : graphs) {
        const auto emb = planar_embedding(G);
        ASSERT_TRUE(emb.has_value());
        EXPECT_TRUE(embedding_is_symmetric(G, *emb));
        std::set<std::pair<int, int>> simple;
        for (const auto& e : G.edges())
            if (e.from != e.to)
                simple.insert({std::min(e.from, e.to), std::max(e.from, e.to)});
        std::size_t darts = 0;
        for (const auto& row : *emb) darts += row.size();
        EXPECT_EQ(darts, 2u * simple.size());
    }
}

TEST(Planarity, HeuristicDisagreementsDocumented) {
    // The documented small-graph wart: the Euler screen needs V >= 3.
    Graph edge2(2, false);
    edge2.add_edge(0, 1);
    EXPECT_FALSE(is_planar_k5_k33_check(edge2));
    EXPECT_TRUE(is_planar(edge2));

    Graph empty(0, false);
    EXPECT_FALSE(is_planar_k5_k33_check(empty));
    EXPECT_TRUE(is_planar(empty));

    Graph one(1, false);
    EXPECT_FALSE(is_planar_k5_k33_check(one));
    EXPECT_TRUE(is_planar(one));
}

TEST(Planarity, PlatonicSolidsArePlanar) {
    Graph ico = make_icosahedron();
    EXPECT_EQ(ico.n_edges(), 30);             // exactly the 3V - 6 bound
    EXPECT_TRUE(is_planar(ico));
    const auto ie = planar_embedding(ico);
    ASSERT_TRUE(ie.has_value());
    EXPECT_EQ(count_faces(*ie), 20);          // 12 - 30 + 20 == 2
    for (const auto& row : *ie) EXPECT_EQ(row.size(), 5u);
    EXPECT_TRUE(embedding_is_symmetric(ico, *ie));
    EXPECT_TRUE(kuratowski_subgraph(ico).empty());

    Graph dod = make_dodecahedron();
    EXPECT_TRUE(is_planar(dod));
    const auto de = planar_embedding(dod);
    ASSERT_TRUE(de.has_value());
    EXPECT_EQ(count_faces(*de), 12);          // 20 - 30 + 12 == 2
    for (const auto& row : *de) EXPECT_EQ(row.size(), 3u);
    EXPECT_TRUE(embedding_is_symmetric(dod, *de));
}

TEST(Planarity, PetersenSkewnessIsTwo) {
    // Two edges have to go before the Petersen graph becomes planar; one is
    // not enough, which is exactly what a heuristic screen cannot see.
    Graph one = make_petersen();
    one.remove_edge(0, 5);
    EXPECT_TRUE(is_planar_k5_k33_check(one));
    EXPECT_FALSE(is_planar(one));
    EXPECT_TRUE(is_kuratowski_subdivision(kuratowski_subgraph(one)));

    Graph two = make_petersen();
    two.remove_edge(0, 5);
    two.remove_edge(1, 6);
    EXPECT_TRUE(is_planar(two));
    const auto emb = planar_embedding(two);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 5);          // 10 - 13 + 5 == 2
    EXPECT_TRUE(embedding_is_symmetric(two, *emb));
}

TEST(Planarity, PetersenMinusAVertexIsStillNonPlanar) {
    // The Petersen graph is not apex: deleting a vertex (here vertex 0, by
    // deleting its three incident edges) leaves a K3,3 subdivision behind.
    Graph G = make_petersen();
    G.remove_edge(0, 1);
    G.remove_edge(4, 0);
    G.remove_edge(0, 5);
    EXPECT_FALSE(is_planar(G));
    const std::vector<Edge> ks = kuratowski_subgraph(G);
    EXPECT_TRUE(is_kuratowski_subdivision(ks));
    for (const auto& e : ks) {                // vertex 0 is isolated now
        EXPECT_NE(e.from, 0);
        EXPECT_NE(e.to, 0);
    }
}

TEST(Planarity, WheelIsPlanar) {
    Graph W(7, false);
    for (int i = 1; i < 7; ++i) {
        W.add_edge(0, i);
        W.add_edge(i, i % 6 + 1);
    }
    EXPECT_TRUE(is_planar(W));
    const auto emb = planar_embedding(W);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(count_faces(*emb), 7);          // 7 - 12 + 7 == 2
    EXPECT_EQ((*emb)[0].size(), 6u);
}
