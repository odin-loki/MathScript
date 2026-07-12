#include "ms/graph/graph.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <gtest/gtest.h>

using namespace ms::graph;

namespace {

// Local Jacobi helper for verifying spectral properties in tests.
std::vector<double> test_symmetric_eigenvalues(std::vector<std::vector<double>> A) {
    int n = static_cast<int>(A.size());
    if (n == 0) return {};
    const double tol = 1e-10;
    for (int iter = 0; iter < 50 * n * n; ++iter) {
        int p = 0, q = 1;
        double max_off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (std::abs(A[i][j]) > max_off) {
                    max_off = std::abs(A[i][j]);
                    p = i; q = j;
                }
        if (max_off < tol) break;
        double phi = 0.5 * std::atan2(2.0 * A[p][q], A[q][q] - A[p][p]);
        double c = std::cos(phi), s = std::sin(phi);
        for (int k = 0; k < n; ++k) {
            double akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[p][k] = A[k][p];
            A[k][q] = s * akp + c * akq;
            A[q][k] = A[k][q];
        }
        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = A[q][p] = 0.0;
    }
    std::vector<double> evals(n);
    for (int i = 0; i < n; ++i) evals[i] = A[i][i];
    std::sort(evals.begin(), evals.end());
    return evals;
}

Graph make_barbell() {
    // Two triangles joined by a single bridge edge (2-3); articulation points 2 and 3.
    Graph G(6, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);
    G.add_edge(2, 3);
    G.add_edge(3, 4); G.add_edge(4, 5); G.add_edge(5, 3);
    return G;
}

Graph make_star(int n) {
    Graph G(n, false);
    for (int i = 1; i < n; ++i) G.add_edge(0, i);
    return G;
}

// Complete graph K_n on vertices [offset, offset+n).
void add_clique(Graph& G, int offset, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            G.add_edge(offset + i, offset + j);
}

// n disjoint triangles: vertices [0,3), [3,6), ...
Graph make_disjoint_triangles(int count) {
    Graph G(count * 3, false);
    for (int t = 0; t < count; ++t) add_clique(G, t * 3, 3);
    return G;
}

// Two K4 cliques with no connecting edge.
Graph make_two_disjoint_k4() {
    Graph G(8, false);
    add_clique(G, 0, 4);
    add_clique(G, 4, 4);
    return G;
}

// Two K5 cliques joined by a single bridge edge (4-5).
Graph make_two_k5_bridge() {
    Graph G(10, false);
    add_clique(G, 0, 5);
    add_clique(G, 5, 5);
    G.add_edge(4, 5);
    return G;
}

// Partition helper: every vertex in its own singleton community.
std::vector<std::vector<int>> singleton_partition(int n) {
    std::vector<std::vector<int>> parts;
    for (int i = 0; i < n; ++i) parts.push_back({i});
    return parts;
}

// Partition helper: every vertex in one community.
std::vector<std::vector<int>> one_partition(int n) {
    std::vector<int> all(n);
    for (int i = 0; i < n; ++i) all[i] = i;
    return {all};
}

// Sorts each community's members and sorts communities by their first
// (smallest) member, so partitions can be compared for set-equality
// regardless of the order louvain() happens to emit them in.
std::vector<std::vector<int>> canonicalise(std::vector<std::vector<int>> parts) {
    for (auto& p : parts) std::sort(p.begin(), p.end());
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) {
        return a.front() < b.front();
    });
    return parts;
}

// Structural completeness check for an Eulerian path/circuit result: every
// consecutive pair in `path` must be a real edge of G, and the walk must
// consume every physical edge of G exactly once (checked via a multiset of
// normalised (min,max) endpoint pairs, so multi-edges are each counted
// separately). Used broadly across every positive eulerian_path() test case
// below, rather than spot-checking a handful of properties.
bool eulerian_path_uses_every_edge_exactly_once(const Graph& G, const std::vector<int>& path) {
    if (path.size() < 2) return false;
    if (path.size() - 1 != static_cast<size_t>(G.n_edges())) return false;
    std::multiset<std::pair<int,int>> remaining;
    for (auto& e : G.edges()) {
        int a = e.from, b = e.to;
        if (a > b) std::swap(a, b);
        remaining.insert({a, b});
    }
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        int a = path[i], b = path[i + 1];
        if (a > b) std::swap(a, b);
        auto it = remaining.find({a, b});
        if (it == remaining.end()) return false;  // not a real edge, or already consumed
        remaining.erase(it);
    }
    return remaining.empty();
}

} // namespace

// ---- Basic graph ----
TEST(GraphBasic, AddEdges) {
    Graph G(5, false);
    G.add_edge(0, 1, 1.0);
    G.add_edge(1, 2, 2.0);
    G.add_edge(2, 3, 1.5);
    EXPECT_EQ(G.n_vertices(), 5);
    EXPECT_EQ(G.n_edges(), 3);
}

// ---- BFS ----
TEST(GraphTraversal, BFS) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    auto order = bfs(G, 0);
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 0);
}

// ---- DFS ----
TEST(GraphTraversal, DFS) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(0, 2); G.add_edge(2, 3);
    auto order = dfs(G, 0);
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 0);
}

// ---- Topological sort ----
TEST(GraphTraversal, TopoSort) {
    Graph G(4, true);
    G.add_edge(0, 1); G.add_edge(0, 2); G.add_edge(1, 3); G.add_edge(2, 3);
    auto order = topological_sort(G);
    ASSERT_TRUE(order.has_value());
    EXPECT_EQ(order.value()[0], 0);
    EXPECT_EQ(order.value().back(), 3);
}

TEST(GraphTraversal, TopoSortCycleDetect) {
    Graph G(3, true);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);
    auto order = topological_sort(G);
    EXPECT_FALSE(order.has_value());
}

// ---- Dijkstra ----
TEST(GraphShortestPath, Dijkstra) {
    Graph G(5, true);
    G.add_edge(0, 1, 4.0); G.add_edge(0, 2, 1.0);
    G.add_edge(2, 1, 2.0); G.add_edge(1, 3, 1.0);
    G.add_edge(2, 3, 5.0); G.add_edge(3, 4, 3.0);
    auto [dist, parent] = dijkstra(G, 0);
    EXPECT_NEAR(dist[4], 7.0, 1e-10);  // 0→2→1→3→4 = 1+2+1+3 = 7
}

// ---- Bellman-Ford ----
TEST(GraphShortestPath, BellmanFord) {
    Graph G(3, true);
    G.add_edge(0, 1, 1.0); G.add_edge(1, 2, 2.0);
    auto result = bellman_ford(G, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value().first[2], 3.0, 1e-10);
}

// ---- Floyd-Warshall ----
TEST(GraphShortestPath, FloydWarshall) {
    Graph G(3, true);
    G.add_edge(0, 1, 1.0); G.add_edge(1, 2, 1.0); G.add_edge(0, 2, 5.0);
    auto d = floyd_warshall(G);
    EXPECT_NEAR(d[0][2], 2.0, 1e-10);
}

// ---- Connectivity ----
TEST(GraphConnectivity, IsConnected) {
    Graph G(3, false);
    G.add_edge(0, 1); G.add_edge(1, 2);
    EXPECT_TRUE(is_connected(G));
    Graph G2(3, false);
    G2.add_edge(0, 1);
    EXPECT_FALSE(is_connected(G2));
}

TEST(GraphConnectivity, IsDAG) {
    Graph G(3, true);
    G.add_edge(0, 1); G.add_edge(1, 2);
    EXPECT_TRUE(is_dag(G));
    Graph G2(3, true);
    G2.add_edge(0, 1); G2.add_edge(1, 2); G2.add_edge(2, 0);
    EXPECT_FALSE(is_dag(G2));
}

TEST(GraphConnectivity, IsBipartite) {
    // Path graph is bipartite
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    EXPECT_TRUE(is_bipartite(G));
    // Odd cycle
    Graph G2(3, false);
    G2.add_edge(0, 1); G2.add_edge(1, 2); G2.add_edge(2, 0);
    EXPECT_FALSE(is_bipartite(G2));
}

// ---- SCC ----
TEST(GraphConnectivity, SCC) {
    Graph G(4, true);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0); G.add_edge(3, 2);
    auto sccs = strongly_connected_components(G);
    EXPECT_EQ(sccs.size(), 2u);
}

// ---- MST ----
TEST(GraphMST, Kruskal) {
    Graph G(4, false);
    G.add_edge(0, 1, 1.0); G.add_edge(1, 2, 2.0);
    G.add_edge(2, 3, 3.0); G.add_edge(0, 3, 10.0);
    auto mst = mst_kruskal(G);
    EXPECT_EQ(mst.size(), 3u);
    double total_w = 0.0;
    for (auto& e : mst) total_w += e.weight;
    EXPECT_NEAR(total_w, 6.0, 1e-10);
}

TEST(GraphMST, Prim) {
    Graph G(4, false);
    G.add_edge(0, 1, 1.0); G.add_edge(1, 2, 2.0);
    G.add_edge(2, 3, 3.0); G.add_edge(0, 3, 10.0);
    auto mst = mst_prim(G, 0);
    EXPECT_EQ(mst.size(), 3u);
}

// ---- PageRank ----
TEST(GraphCentrality, PageRank) {
    Graph G(3, true);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);
    auto pr = pagerank(G, 0.85, 100);
    EXPECT_EQ(pr.size(), 3u);
    double sum = 0.0;
    for (auto r : pr) sum += r;
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

// ---- Diameter ----
TEST(GraphProperties, Diameter) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    EXPECT_EQ(diameter(G), 3);
}

// ---- Graph coloring ----
TEST(GraphColoring, Greedy) {
    // 2-colorable cycle
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    auto colors = greedy_colour(G);
    // Adjacent vertices must have different colors
    for (int v = 0; v < 4; ++v)
        for (auto& [u, w] : G.neighbors(v))
            EXPECT_NE(colors[v], colors[u]);
}

// ---- Euler circuit ----
TEST(GraphMisc, EulerCircuit) {
    // 4-cycle: all vertices degree 2 → Euler circuit exists
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    auto circuit = euler_circuit(G);
    ASSERT_FALSE(circuit.empty());
    EXPECT_EQ(circuit.front(), circuit.back());
    // Each undirected edge is stored twice in adjacency lists, so circuit
    // visits all edge-incidences: 4 undirected × 2 = 8 directed edges + 1
    EXPECT_GT(circuit.size(), 1u);
}

// ---- Is tree ----
TEST(GraphProperties, IsTree) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    EXPECT_TRUE(is_tree(G));
    G.add_edge(0, 3);
    EXPECT_FALSE(is_tree(G));
}

// ---- A* ----
TEST(GraphShortestPath, AStar) {
    Graph G(4, false);
    G.add_edge(0, 1, 1.0); G.add_edge(1, 2, 1.0); G.add_edge(2, 3, 1.0);
    std::vector<double> heuristic = {3.0, 2.0, 1.0, 0.0};
    auto path = astar(G, 0, 3, heuristic);
    ASSERT_TRUE(path.has_value());
    EXPECT_EQ(path.value().front(), 0);
    EXPECT_EQ(path.value().back(), 3);
}

TEST(GraphShortestPath, AStarPrefersCheaperRoute) {
    Graph G(4, true);
    G.add_edge(0, 1, 1.0); G.add_edge(1, 3, 1.0);
    G.add_edge(0, 2, 10.0); G.add_edge(2, 3, 1.0);
    std::vector<double> heuristic = {2.0, 1.0, 1.0, 0.0};
    auto path = astar(G, 0, 3, heuristic);
    ASSERT_TRUE(path.has_value());
    ASSERT_EQ(path.value().size(), 3u);
    EXPECT_EQ(path.value()[1], 1);
}

// ---- Bellman-Ford (negative weights) ----
TEST(GraphShortestPath, BellmanFordNegativeWeight) {
    Graph G(3, true);
    G.add_edge(0, 1, 1.0);
    G.add_edge(1, 2, -2.0);
    G.add_edge(0, 2, 5.0);
    auto result = bellman_ford(G, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value().first[2], -1.0, 1e-10);
}

// ---- Betweenness centrality ----
TEST(GraphCentrality, Betweenness) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    auto bc = betweenness_centrality(G);
    EXPECT_EQ(bc.size(), 4u);
    EXPECT_GT(bc[1], bc[0]);
    EXPECT_GT(bc[2], bc[3]);
    EXPECT_GT(bc[1], 0.0);
}

// ---- Max flow ----
TEST(GraphFlow, MaxFlow) {
    Graph G(4, true);
    G.add_edge(0, 1, 3.0);
    G.add_edge(0, 2, 2.0);
    G.add_edge(1, 3, 2.0);
    G.add_edge(2, 3, 3.0);
    auto flow = max_flow(G, 0, 3);
    ASSERT_TRUE(flow.has_value());
    EXPECT_NEAR(flow.value(), 4.0, 1e-10);
}

// ---- Articulation points & bridges ----
TEST(GraphConnectivity, ArticulationPointsBarbell) {
    auto G = make_barbell();
    auto aps = articulation_points(G);
    ASSERT_EQ(aps.size(), 2u);
    EXPECT_EQ(aps[0], 2);
    EXPECT_EQ(aps[1], 3);
}

TEST(GraphConnectivity, BridgesBarbell) {
    auto G = make_barbell();
    auto br = bridges(G);
    ASSERT_EQ(br.size(), 1u);
    EXPECT_EQ(br[0].from, 2);
    EXPECT_EQ(br[0].to, 3);
}

TEST(GraphConnectivity, ArticulationPointsCycleNone) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    EXPECT_TRUE(articulation_points(G).empty());
}

TEST(GraphConnectivity, BridgesCycleNone) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    EXPECT_TRUE(bridges(G).empty());
}

TEST(GraphConnectivity, ArticulationPointsPath) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    auto aps = articulation_points(G);
    ASSERT_EQ(aps.size(), 2u);
    EXPECT_EQ(aps[0], 1);
    EXPECT_EQ(aps[1], 2);
}

TEST(GraphConnectivity, BridgesPath) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    auto br = bridges(G);
    EXPECT_EQ(br.size(), 3u);
}

// ---- Biconnected components ----

namespace {

// Order-independent comparison of two edge lists (as unordered (from,to) sets).
bool same_edge_set(std::vector<Edge> a, std::vector<Edge> b) {
    auto norm = [](std::vector<Edge>& v) {
        for (auto& e : v) if (e.from > e.to) std::swap(e.from, e.to);
        std::sort(v.begin(), v.end(), [](const Edge& x, const Edge& y) {
            return x.from < y.from || (x.from == y.from && x.to < y.to);
        });
    };
    norm(a); norm(b);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].from != b[i].from || a[i].to != b[i].to) return false;
    return true;
}

} // namespace

TEST(GraphConnectivity, BiconnectedComponentsBarbellIsTwoTrianglesPlusBridge) {
    auto G = make_barbell();
    auto bcc = biconnected_components(G);
    ASSERT_EQ(bcc.size(), 3u);

    int triangle_blocks = 0, bridge_blocks = 0;
    for (auto& comp : bcc) {
        if (comp.size() == 3u) {
            ++triangle_blocks;
            bool is_012 = same_edge_set(comp, {{0,1,1.0},{1,2,1.0},{2,0,1.0}});
            bool is_345 = same_edge_set(comp, {{3,4,1.0},{4,5,1.0},{5,3,1.0}});
            EXPECT_TRUE(is_012 || is_345);
        } else if (comp.size() == 1u) {
            ++bridge_blocks;
            EXPECT_EQ(comp[0].from, 2);
            EXPECT_EQ(comp[0].to, 3);
        }
    }
    EXPECT_EQ(triangle_blocks, 2);
    EXPECT_EQ(bridge_blocks, 1);
}

TEST(GraphConnectivity, BiconnectedComponentsCycleIsSingleBlock) {
    // 5-cycle: no articulation points, so the whole cycle is one block.
    Graph G(5, false);
    for (int i = 0; i < 5; ++i) G.add_edge(i, (i + 1) % 5);
    auto bcc = biconnected_components(G);
    ASSERT_EQ(bcc.size(), 1u);
    EXPECT_EQ(bcc[0].size(), 5u);
}

TEST(GraphConnectivity, BiconnectedComponentsTreePathEveryEdgeIsOwnBlock) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    auto bcc = biconnected_components(G);
    ASSERT_EQ(bcc.size(), 3u);
    for (auto& comp : bcc) EXPECT_EQ(comp.size(), 1u);
}

TEST(GraphConnectivity, BiconnectedComponentsStarEveryEdgeIsOwnBlock) {
    auto G = make_star(6);
    auto bcc = biconnected_components(G);
    ASSERT_EQ(bcc.size(), 5u);  // star on 6 vertices has 5 edges, all bridges
    for (auto& comp : bcc) EXPECT_EQ(comp.size(), 1u);
}

TEST(GraphConnectivity, BiconnectedComponentsBridgesConsistency) {
    // Every bridge must appear as a singleton block, and vice versa.
    for (const Graph& G : {make_barbell(), make_two_k5_bridge(), make_two_disjoint_k4()}) {
        auto br = bridges(G);
        auto bcc = biconnected_components(G);

        std::vector<std::pair<int,int>> singleton_blocks;
        for (auto& comp : bcc)
            if (comp.size() == 1u) {
                int a = comp[0].from, b = comp[0].to;
                if (a > b) std::swap(a, b);
                singleton_blocks.push_back({a, b});
            }
        std::vector<std::pair<int,int>> bridge_edges;
        for (auto& e : br) {
            int a = e.from, b = e.to;
            if (a > b) std::swap(a, b);
            bridge_edges.push_back({a, b});
        }
        std::sort(singleton_blocks.begin(), singleton_blocks.end());
        std::sort(bridge_edges.begin(), bridge_edges.end());
        EXPECT_EQ(singleton_blocks, bridge_edges);
    }
}

TEST(GraphConnectivity, BiconnectedComponentsPartitionEveryEdgeExactlyOnce) {
    for (const Graph& G : {make_barbell(), make_two_k5_bridge(), make_two_disjoint_k4(),
                           make_disjoint_triangles(3), make_star(7)}) {
        auto all_edges = G.edges();
        auto bcc = biconnected_components(G);

        std::map<std::pair<int,int>, int> count;
        for (auto& e : all_edges) {
            int a = e.from, b = e.to;
            if (a > b) std::swap(a, b);
            ++count[{a, b}];
        }
        std::map<std::pair<int,int>, int> covered;
        for (auto& comp : bcc)
            for (auto& e : comp) {
                int a = e.from, b = e.to;
                if (a > b) std::swap(a, b);
                ++covered[{a, b}];
            }
        EXPECT_EQ(covered, count);  // every edge covered exactly as many times as it exists
    }
}

TEST(GraphConnectivity, BiconnectedComponentsEmptyGraph) {
    Graph G(0, false);
    EXPECT_TRUE(biconnected_components(G).empty());
}

TEST(GraphConnectivity, BiconnectedComponentsSingleVertex) {
    Graph G(1, false);
    EXPECT_TRUE(biconnected_components(G).empty());
}

TEST(GraphConnectivity, BiconnectedComponentsSingleEdge) {
    Graph G(2, false);
    G.add_edge(0, 1);
    auto bcc = biconnected_components(G);
    ASSERT_EQ(bcc.size(), 1u);
    ASSERT_EQ(bcc[0].size(), 1u);
    EXPECT_EQ(bcc[0][0].from, 0);
    EXPECT_EQ(bcc[0][0].to, 1);
}

TEST(GraphConnectivity, BiconnectedComponentsDisconnectedGraph) {
    // Two disjoint triangles: expect two 3-edge blocks, no cross-contamination.
    auto G = make_disjoint_triangles(2);
    auto bcc = biconnected_components(G);
    ASSERT_EQ(bcc.size(), 2u);
    EXPECT_EQ(bcc[0].size(), 3u);
    EXPECT_EQ(bcc[1].size(), 3u);
    EXPECT_TRUE(same_edge_set(bcc[0], {{0,1,1.0},{1,2,1.0},{2,0,1.0}}));
    EXPECT_TRUE(same_edge_set(bcc[1], {{3,4,1.0},{4,5,1.0},{5,3,1.0}}));
}

TEST(GraphConnectivity, BiconnectedComponentsDisconnectedWithIsolatedVertex) {
    // Isolated vertex contributes zero blocks; other component unaffected.
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);  // triangle 0-1-2
    // vertex 3 isolated
    auto bcc = biconnected_components(G);
    ASSERT_EQ(bcc.size(), 1u);
    EXPECT_EQ(bcc[0].size(), 3u);
}

// ---- Eigenvector & Katz centrality ----
TEST(GraphCentrality, EigenvectorStarHubHighest) {
    auto G = make_star(5);
    auto ec = eigenvector_centrality(G);
    ASSERT_EQ(ec.size(), 5u);
    for (int i = 1; i < 5; ++i)
        EXPECT_GT(ec[0], ec[i]);
}

TEST(GraphCentrality, EigenvectorCycleSymmetric) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    auto ec = eigenvector_centrality(G);
    for (int i = 1; i < 4; ++i)
        EXPECT_NEAR(ec[0], ec[i], 1e-6);
}

TEST(GraphCentrality, KatzStarHubHighest) {
    auto G = make_star(5);
    auto kc = katz_centrality(G, 0.1, 1.0);
    ASSERT_EQ(kc.size(), 5u);
    for (int i = 1; i < 5; ++i)
        EXPECT_GT(kc[0], kc[i]);
}

TEST(GraphCentrality, KatzCycleSymmetric) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    auto kc = katz_centrality(G, 0.1, 1.0);
    for (int i = 1; i < 4; ++i)
        EXPECT_NEAR(kc[0], kc[i], 1e-6);
}

// ---- Laplacian matrices ----
TEST(GraphSpectral, LaplacianTriangle) {
    Graph G(3, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);
    auto L = laplacian(G);
    EXPECT_NEAR(L[0][0], 2.0, 1e-10);
    EXPECT_NEAR(L[0][1], -1.0, 1e-10);
    EXPECT_NEAR(L[1][1], 2.0, 1e-10);
    EXPECT_NEAR(L[2][2], 2.0, 1e-10);
    EXPECT_NEAR(L[0][2], -1.0, 1e-10);
}

TEST(GraphSpectral, LaplacianPathThree) {
    Graph G(3, false);
    G.add_edge(0, 1); G.add_edge(1, 2);
    auto L = laplacian(G);
    EXPECT_NEAR(L[0][0], 1.0, 1e-10);
    EXPECT_NEAR(L[1][1], 2.0, 1e-10);
    EXPECT_NEAR(L[2][2], 1.0, 1e-10);
    EXPECT_NEAR(L[0][1], -1.0, 1e-10);
    EXPECT_NEAR(L[1][2], -1.0, 1e-10);
    EXPECT_NEAR(L[0][2], 0.0, 1e-10);
}

TEST(GraphSpectral, LaplacianRowSumsZero) {
    auto G = make_barbell();
    auto L = laplacian(G);
    for (const auto& row : L) {
        double sum = 0.0;
        for (double v : row) sum += v;
        EXPECT_NEAR(sum, 0.0, 1e-10);
    }
}

TEST(GraphSpectral, NormalisedLaplacianPathThree) {
    Graph G(3, false);
    G.add_edge(0, 1); G.add_edge(1, 2);
    auto Ln = normalised_laplacian(G);
    EXPECT_NEAR(Ln[0][0], 1.0, 1e-10);
    EXPECT_NEAR(Ln[1][1], 1.0, 1e-10);
    EXPECT_NEAR(Ln[2][2], 1.0, 1e-10);
    EXPECT_NEAR(Ln[0][1], -1.0 / std::sqrt(1.0 * 2.0), 1e-10);
    EXPECT_NEAR(Ln[1][2], -1.0 / std::sqrt(2.0 * 1.0), 1e-10);
}

TEST(GraphSpectral, NormalisedLaplacianEigenvaluesInZeroTwo) {
    auto G = make_barbell();
    auto evals = test_symmetric_eigenvalues(normalised_laplacian(G));
    for (double lam : evals) {
        EXPECT_GE(lam, -1e-9);
        EXPECT_LE(lam, 2.0 + 1e-9);
    }
}

// ---- Algebraic connectivity ----
TEST(GraphSpectral, AlgebraicConnectivityDisconnected) {
    Graph G(4, false);
    G.add_edge(0, 1);
    G.add_edge(2, 3);
    EXPECT_NEAR(algebraic_connectivity(G), 0.0, 1e-8);
}

TEST(GraphSpectral, AlgebraicConnectivityConnectedPositive) {
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    EXPECT_GT(algebraic_connectivity(G), 0.0);
}

TEST(GraphSpectral, AlgebraicConnectivityCompleteGraph) {
    Graph G(4, false);
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            G.add_edge(i, j);
    EXPECT_NEAR(algebraic_connectivity(G), 4.0, 1e-6);
}

// ---- Min cut ----
TEST(GraphFlow, MinCutMatchesMaxFlow) {
    Graph G(4, true);
    G.add_edge(0, 1, 3.0);
    G.add_edge(0, 2, 2.0);
    G.add_edge(1, 3, 2.0);
    G.add_edge(2, 3, 3.0);
    auto flow = max_flow(G, 0, 3);
    auto cut = min_cut(G, 0, 3);
    ASSERT_TRUE(flow.has_value());
    ASSERT_TRUE(cut.has_value());
    EXPECT_NEAR(cut.value().value, flow.value(), 1e-10);
}

TEST(GraphFlow, MinCutBottleneckEdge) {
    // Two 3-cliques joined by a low-capacity bottleneck edge.
    Graph G(6, true);
    G.add_edge(0, 1, 10.0); G.add_edge(1, 2, 10.0); G.add_edge(2, 0, 10.0);
    G.add_edge(2, 3, 1.0);
    G.add_edge(3, 4, 10.0); G.add_edge(4, 5, 10.0); G.add_edge(5, 3, 10.0);
    auto cut = min_cut(G, 0, 5);
    ASSERT_TRUE(cut.has_value());
    EXPECT_NEAR(cut.value().value, 1.0, 1e-10);
    ASSERT_EQ(cut.value().cut_edges.size(), 1u);
    EXPECT_EQ(cut.value().cut_edges[0].from, 2);
    EXPECT_EQ(cut.value().cut_edges[0].to, 3);
}

TEST(GraphFlow, MinCutMatchesMaxFlowParallelPaths) {
    Graph G(3, true);
    G.add_edge(0, 1, 5.0);
    G.add_edge(0, 1, 3.0);
    G.add_edge(1, 2, 8.0);
    auto flow = max_flow(G, 0, 2);
    auto cut = min_cut(G, 0, 2);
    ASSERT_TRUE(flow.has_value());
    ASSERT_TRUE(cut.has_value());
    EXPECT_NEAR(cut.value().value, flow.value(), 1e-10);
}

// ---- Isomorphism ----

TEST(GraphIsomorphism, ReflexiveCycle) {
    Graph G = from_edge_list(5, {{0,1,1.0},{1,2,1.0},{2,3,1.0},{3,4,1.0},{4,0,1.0}});
    EXPECT_TRUE(is_isomorphic(G, G));
}

TEST(GraphIsomorphism, PermutedTriangleIsIsomorphic) {
    Graph G1 = from_edge_list(3, {{0,1,1.0},{1,2,1.0},{2,0,1.0}});
    // Relabel via permutation {2,0,1}: vertex i of G1 becomes vertex perm[i] of G2.
    std::vector<int> perm = {2, 0, 1};
    Graph G2 = from_edge_list(3, {{perm[0],perm[1],1.0},{perm[1],perm[2],1.0},{perm[2],perm[0],1.0}});
    EXPECT_TRUE(is_isomorphic(G1, G2));
}

TEST(GraphIsomorphism, PathVsStarNotIsomorphic) {
    // Path 0-1-2-3: degree sequence [1,2,2,1]; Star centered at 0: [3,1,1,1].
    Graph path = from_edge_list(4, {{0,1,1.0},{1,2,1.0},{2,3,1.0}});
    Graph star = from_edge_list(4, {{0,1,1.0},{0,2,1.0},{0,3,1.0}});
    EXPECT_FALSE(is_isomorphic(path, star));
}

TEST(GraphIsomorphism, DifferentVertexCountNotIsomorphic) {
    Graph G1 = from_edge_list(3, {{0,1,1.0},{1,2,1.0}});
    Graph G2 = from_edge_list(4, {{0,1,1.0},{1,2,1.0},{2,3,1.0}});
    EXPECT_FALSE(is_isomorphic(G1, G2));
}

TEST(GraphIsomorphism, K33VsPrismSameDegreeSequenceNotIsomorphic) {
    // K3,3: bipartite {0,1,2} vs {3,4,5}, every cross pair connected; triangle-free.
    Graph k33 = from_edge_list(6, {
        {0,3,1.0},{0,4,1.0},{0,5,1.0},
        {1,3,1.0},{1,4,1.0},{1,5,1.0},
        {2,3,1.0},{2,4,1.0},{2,5,1.0}});
    // Prism (K3 x K2): two triangles {0,1,2} and {3,4,5} joined by a perfect matching.
    Graph prism = from_edge_list(6, {
        {0,1,1.0},{1,2,1.0},{2,0,1.0},
        {3,4,1.0},{4,5,1.0},{5,3,1.0},
        {0,3,1.0},{1,4,1.0},{2,5,1.0}});
    // Both are 3-regular on 6 vertices with 9 edges (same degree sequence),
    // but K3,3 is triangle-free while the prism has triangles, so they are
    // not isomorphic; this exercises the backtracking search itself, not
    // just the cheap degree-sequence pre-check.
    EXPECT_EQ(k33.n_vertices(), prism.n_vertices());
    EXPECT_EQ(k33.n_edges(), prism.n_edges());
    EXPECT_FALSE(is_isomorphic(k33, prism));
}

TEST(GraphIsomorphism, PermutedCompleteGraphIsIsomorphic) {
    Graph K4a(4, false);
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) K4a.add_edge(i, j);
    std::vector<int> perm = {3, 1, 0, 2};
    Graph K4b(4, false);
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) K4b.add_edge(perm[i], perm[j]);
    EXPECT_TRUE(is_isomorphic(K4a, K4b));
}

TEST(GraphIsomorphism, ExtraEdgeNotIsomorphic) {
    Graph G1 = from_edge_list(4, {{0,1,1.0},{1,2,1.0},{2,3,1.0}});
    Graph G2 = from_edge_list(4, {{0,1,1.0},{1,2,1.0},{2,3,1.0},{3,0,1.0}});
    EXPECT_FALSE(is_isomorphic(G1, G2));
}

TEST(GraphIsomorphism, DirectedCyclePermutedIsIsomorphic) {
    Graph G1 = from_edge_list(3, {{0,1,1.0},{1,2,1.0},{2,0,1.0}}, /*directed=*/true);
    std::vector<int> perm = {2, 0, 1};
    Graph G2 = from_edge_list(3,
        {{perm[0],perm[1],1.0},{perm[1],perm[2],1.0},{perm[2],perm[0],1.0}}, /*directed=*/true);
    EXPECT_TRUE(is_isomorphic(G1, G2));
}

TEST(GraphIsomorphism, DirectedVsUndirectedNotIsomorphic) {
    Graph directedTri = from_edge_list(3, {{0,1,1.0},{1,2,1.0},{2,0,1.0}}, /*directed=*/true);
    Graph undirectedTri = from_edge_list(3, {{0,1,1.0},{1,2,1.0},{2,0,1.0}}, /*directed=*/false);
    EXPECT_FALSE(is_isomorphic(directedTri, undirectedTri));
}

TEST(GraphIsomorphism, DirectedTransitiveTournamentReversalIsIsomorphic) {
    // 0->1, 0->2, 1->2 is a transitive tournament (total order 0 > 1 > 2).
    Graph G1 = from_edge_list(3, {{0,1,1.0},{0,2,1.0},{1,2,1.0}}, /*directed=*/true);
    // Reversing every edge gives the reverse order 2 > 1 > 0, which is
    // structurally the same tournament under the relabeling 0<->2.
    Graph reversed = from_edge_list(3, {{1,0,1.0},{2,0,1.0},{2,1,1.0}}, /*directed=*/true);
    EXPECT_TRUE(is_isomorphic(G1, reversed));
}

TEST(GraphIsomorphism, DirectedNonIsomorphicDifferentStructure) {
    // Directed 3-cycle 0->1->2->0: every vertex has in-degree 1, out-degree 1.
    Graph cycle = from_edge_list(3, {{0,1,1.0},{1,2,1.0},{2,0,1.0}}, /*directed=*/true);
    // Directed "star-ish" graph with same edge/vertex count but different
    // per-vertex (in,out) degrees: 0->1, 0->2, 1->0.
    Graph other = from_edge_list(3, {{0,1,1.0},{0,2,1.0},{1,0,1.0}}, /*directed=*/true);
    EXPECT_FALSE(is_isomorphic(cycle, other));
}

TEST(GraphIsomorphism, EmptyGraphsAreIsomorphic) {
    Graph G1(0, false);
    Graph G2(0, false);
    EXPECT_TRUE(is_isomorphic(G1, G2));
}

TEST(GraphIsomorphism, SingleVertexNoEdgesIsomorphic) {
    Graph G1(1, false);
    Graph G2(1, false);
    EXPECT_TRUE(is_isomorphic(G1, G2));
}

TEST(GraphIsomorphism, DisconnectedGraphsPermutedIsomorphic) {
    // Two disjoint edges {0-1, 2-3} vs a permuted relabeling {0-2, 1-3}.
    Graph G1 = from_edge_list(4, {{0,1,1.0},{2,3,1.0}});
    Graph G2 = from_edge_list(4, {{0,2,1.0},{1,3,1.0}});
    EXPECT_TRUE(is_isomorphic(G1, G2));
}

TEST(GraphIsomorphism, DisconnectedVsConnectedSameCountsNotIsomorphic) {
    // Both have 4 vertices and 3 edges, but the path is connected with
    // degree sequence [1,1,2,2] while this graph is disconnected (a triangle
    // plus an isolated vertex) with degree sequence [0,2,2,2].
    Graph path = from_edge_list(4, {{0,1,1.0},{1,2,1.0},{2,3,1.0}});
    Graph triangleAndIsolated = from_edge_list(4, {{0,1,1.0},{1,2,1.0},{2,0,1.0}});
    EXPECT_FALSE(is_isomorphic(path, triangleAndIsolated));
}

// ---- Modularity ----

TEST(GraphCommunity, ModularitySingleCommunityIsZero) {
    auto G = make_barbell();  // two triangles joined by a bridge
    EXPECT_NEAR(modularity(G, one_partition(6)), 0.0, 1e-10);
}

TEST(GraphCommunity, ModularityCorrectPartitionBeatsTrivialPartitions) {
    auto G = make_barbell();
    double q_correct = modularity(G, {{0, 1, 2}, {3, 4, 5}});
    double q_singleton = modularity(G, singleton_partition(6));
    double q_one = modularity(G, one_partition(6));
    EXPECT_GT(q_correct, q_singleton);
    EXPECT_GT(q_correct, q_one);
    // Hand-computed: m=7, correct partition Q = 2*(3/7 - (7/14)^2) = 6/7 - 0.5 = 0.3571...
    EXPECT_NEAR(q_correct, 5.0 / 14.0, 1e-9);
}

TEST(GraphCommunity, ModularitySingletonsIsNegativeForBarbell) {
    auto G = make_barbell();
    EXPECT_LT(modularity(G, singleton_partition(6)), 0.0);
}

TEST(GraphCommunity, ModularityNoEdgesIsZero) {
    Graph G(4, false);
    EXPECT_NEAR(modularity(G, singleton_partition(4)), 0.0, 1e-12);
}

TEST(GraphCommunity, ModularityEmptyGraph) {
    Graph G(0, false);
    EXPECT_NEAR(modularity(G, {}), 0.0, 1e-12);
}

// ---- Louvain ----

TEST(GraphCommunity, LouvainTwoDisjointCliquesRecoversCliques) {
    auto G = make_two_disjoint_k4();
    auto comms = canonicalise(louvain(G));
    ASSERT_EQ(comms.size(), 2u);
    std::vector<std::vector<int>> expected = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    EXPECT_EQ(comms, canonicalise(expected));
}

TEST(GraphCommunity, LouvainTwoK5CliquesWithBridgeRecoversClusters) {
    auto G = make_two_k5_bridge();
    auto comms = canonicalise(louvain(G));
    ASSERT_EQ(comms.size(), 2u);
    std::vector<std::vector<int>> expected = {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}};
    EXPECT_EQ(comms, canonicalise(expected));
}

TEST(GraphCommunity, LouvainSingleCliqueIsOneCommunity) {
    Graph G(6, false);
    add_clique(G, 0, 6);
    auto comms = louvain(G);
    ASSERT_EQ(comms.size(), 1u);
    EXPECT_EQ(comms[0].size(), 6u);
}

TEST(GraphCommunity, LouvainDisjointTrianglesDoNotSpanComponents) {
    auto G = make_disjoint_triangles(3);
    auto comms = canonicalise(louvain(G));
    // Each triangle must be fully contained in exactly one community, and no
    // community may straddle two different (disconnected) triangles.
    std::vector<int> comm_of(9, -1);
    for (int c = 0; c < static_cast<int>(comms.size()); ++c)
        for (int v : comms[c]) comm_of[v] = c;
    for (int t = 0; t < 3; ++t) {
        int c0 = comm_of[t * 3];
        for (int i = 1; i < 3; ++i) EXPECT_EQ(comm_of[t * 3 + i], c0);
    }
    for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 3; ++b)
            EXPECT_NE(comm_of[a * 3], comm_of[b * 3]);
}

TEST(GraphCommunity, LouvainBarbellRecoversTwoTriangles) {
    auto G = make_barbell();
    auto comms = canonicalise(louvain(G));
    ASSERT_EQ(comms.size(), 2u);
    std::vector<std::vector<int>> expected = {{0, 1, 2}, {3, 4, 5}};
    EXPECT_EQ(comms, canonicalise(expected));
}

TEST(GraphCommunity, LouvainBeatsTrivialPartitionsOnVariousGraphs) {
    auto check = [](const Graph& G) {
        int n = G.n_vertices();
        auto comms = louvain(G);
        double q_louvain = modularity(G, comms);
        double q_singleton = modularity(G, singleton_partition(n));
        double q_one = modularity(G, one_partition(n));
        EXPECT_GE(q_louvain, q_singleton - 1e-9);
        EXPECT_GE(q_louvain, q_one - 1e-9);
    };
    check(make_barbell());
    check(make_two_disjoint_k4());
    check(make_two_k5_bridge());
    check(make_disjoint_triangles(3));
    Graph star = make_star(6);
    check(star);
}

TEST(GraphCommunity, LouvainPartitionCoversAllVertices) {
    auto G = make_two_k5_bridge();
    auto comms = louvain(G);
    std::vector<bool> seen(G.n_vertices(), false);
    for (auto& c : comms)
        for (int v : c) {
            EXPECT_FALSE(seen[v]);  // no vertex appears twice
            seen[v] = true;
        }
    for (bool s : seen) EXPECT_TRUE(s);  // every vertex appears
}

TEST(GraphCommunity, LouvainEmptyGraph) {
    Graph G(0, false);
    EXPECT_TRUE(louvain(G).empty());
}

TEST(GraphCommunity, LouvainSingleVertex) {
    Graph G(1, false);
    auto comms = louvain(G);
    ASSERT_EQ(comms.size(), 1u);
    EXPECT_EQ(comms[0], std::vector<int>{0});
}

TEST(GraphCommunity, LouvainNoEdgesGivesSingletons) {
    // Convention: a graph with no edges places every vertex in its own
    // singleton community (there is no incentive to merge any pair).
    Graph G(5, false);
    auto comms = canonicalise(louvain(G));
    EXPECT_EQ(comms, canonicalise(singleton_partition(5)));
}

TEST(GraphCommunity, LouvainFullyDisconnectedMultiVertex) {
    Graph G(4, false);  // no edges at all among 4 vertices
    auto comms = canonicalise(louvain(G));
    ASSERT_EQ(comms.size(), 4u);
    for (auto& c : comms) EXPECT_EQ(c.size(), 1u);
}

TEST(GraphCommunity, LouvainDeterministicAcrossRuns) {
    auto G = make_two_k5_bridge();
    auto comms1 = canonicalise(louvain(G));
    auto comms2 = canonicalise(louvain(G));
    EXPECT_EQ(comms1, comms2);
}

TEST(GraphCommunity, LouvainDirectedGraphIsSymmetrisedAndRecoversCliques) {
    // Same two-K4 structure, but every edge stored as a directed pair.
    Graph G(8, true);
    for (int t = 0; t < 2; ++t)
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j) {
                G.add_edge(t * 4 + i, t * 4 + j);
                G.add_edge(t * 4 + j, t * 4 + i);
            }
    auto comms = canonicalise(louvain(G));
    ASSERT_EQ(comms.size(), 2u);
    std::vector<std::vector<int>> expected = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    EXPECT_EQ(comms, canonicalise(expected));
}

TEST(GraphCommunity, LouvainWeightedEdgesFavourHeavyNeighbour) {
    // Two triangles {0,1,2} and {3,4,5}, plus a pendant vertex 6 attached to
    // both by a light edge (6-0, weight 1) and a much heavier edge (6-3,
    // weight 5). Modularity should favour attaching 6 to vertex 3's triangle
    // over vertex 0's, purely because of the edge-weight difference -- verify
    // this independently via modularity() (so the expectation doesn't rely on
    // hand-derived arithmetic), then check louvain() agrees.
    Graph G(7, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);
    G.add_edge(3, 4); G.add_edge(4, 5); G.add_edge(5, 3);
    G.add_edge(6, 0, 1.0);
    G.add_edge(6, 3, 5.0);
    std::vector<std::vector<int>> joins_light_side = {{0, 1, 2, 6}, {3, 4, 5}};
    std::vector<std::vector<int>> joins_heavy_side = {{0, 1, 2}, {3, 4, 5, 6}};
    EXPECT_GT(modularity(G, joins_heavy_side), modularity(G, joins_light_side));

    auto comms = canonicalise(louvain(G));
    int comm_of_6 = -1, comm_of_3 = -1;
    for (int c = 0; c < static_cast<int>(comms.size()); ++c)
        for (int v : comms[c]) {
            if (v == 6) comm_of_6 = c;
            if (v == 3) comm_of_3 = c;
        }
    EXPECT_EQ(comm_of_6, comm_of_3);
}

// ---- Eulerian path/circuit (Hierholzer's) ----

TEST(GraphEulerian, SquareCycleIsEulerianCircuit) {
    // 4-cycle: every vertex has degree 2 (even) and it's connected.
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    auto res = eulerian_path(G);
    EXPECT_TRUE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    ASSERT_TRUE(eulerian_path_uses_every_edge_exactly_once(G, res.path));
    EXPECT_EQ(res.path.front(), res.path.back());
}

TEST(GraphEulerian, FigureEightTwoTrianglesIsEulerianCircuit) {
    // Two triangles sharing vertex 0: {0,1,2} and {0,3,4}. Vertex 0 has
    // degree 4, all others degree 2 -- all even, single connected graph.
    Graph G(5, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);
    G.add_edge(0, 3); G.add_edge(3, 4); G.add_edge(4, 0);
    auto res = eulerian_path(G);
    EXPECT_TRUE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    ASSERT_TRUE(eulerian_path_uses_every_edge_exactly_once(G, res.path));
    EXPECT_EQ(res.path.front(), res.path.back());
}

TEST(GraphEulerian, SimplePathGraphHasEulerianPathNotCircuit) {
    // 0-1-2-3: endpoints 0 and 3 have odd degree 1, interior vertices even.
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3);
    auto res = eulerian_path(G);
    EXPECT_FALSE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    ASSERT_TRUE(eulerian_path_uses_every_edge_exactly_once(G, res.path));
    std::set<int> ends = {res.path.front(), res.path.back()};
    EXPECT_EQ(ends, std::set<int>({0, 3}));
}

TEST(GraphEulerian, SingleEdgeHasEulerianPath) {
    Graph G(2, false);
    G.add_edge(0, 1);
    auto res = eulerian_path(G);
    EXPECT_FALSE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    ASSERT_TRUE(eulerian_path_uses_every_edge_exactly_once(G, res.path));
    std::set<int> ends = {res.path.front(), res.path.back()};
    EXPECT_EQ(ends, std::set<int>({0, 1}));
}

TEST(GraphEulerian, BarbellHasEulerianPathBetweenTheTwoOddVertices) {
    // Two triangles joined by bridge 2-3 (see make_barbell): vertices 2 and
    // 3 are the only odd-degree vertices (degree 3 each), everything else
    // even -- so an Eulerian path (not circuit) must start/end at {2,3}.
    auto G = make_barbell();
    auto res = eulerian_path(G);
    EXPECT_FALSE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    ASSERT_TRUE(eulerian_path_uses_every_edge_exactly_once(G, res.path));
    std::set<int> ends = {res.path.front(), res.path.back()};
    EXPECT_EQ(ends, std::set<int>({2, 3}));
}

TEST(GraphEulerian, StarGraphWithThreeLeavesHasNeitherPathNorCircuit) {
    // Star on 4 vertices: center (degree 3) plus 3 leaves (degree 1 each)
    // -- 4 odd-degree vertices, far more than the 0-or-2 that's allowed.
    auto G = make_star(4);
    auto res = eulerian_path(G);
    EXPECT_FALSE(res.has_circuit);
    EXPECT_FALSE(res.has_path);
    EXPECT_TRUE(res.path.empty());
}

TEST(GraphEulerian, KonigsbergBridgesHasNeitherPathNorCircuit) {
    // The original 1736 Konigsberg bridges problem: 4 landmasses, 7
    // bridges, all 4 vertices have odd degree (3, 3, 5, 3) -- Euler's
    // famous proof that no such walk exists.
    Graph G(4, false);
    G.add_edge(0, 2); G.add_edge(0, 2);  // two bridges between north bank and island
    G.add_edge(1, 2); G.add_edge(1, 2);  // two bridges between south bank and island
    G.add_edge(0, 3);                    // north bank <-> east
    G.add_edge(1, 3);                    // south bank <-> east
    G.add_edge(2, 3);                    // island <-> east
    ASSERT_EQ(G.n_edges(), 7);
    auto res = eulerian_path(G);
    EXPECT_FALSE(res.has_circuit);
    EXPECT_FALSE(res.has_path);
    EXPECT_TRUE(res.path.empty());
}

TEST(GraphEulerian, DisconnectedTwoCyclesBothLocallyEvenButNoOverallCircuit) {
    // Two disjoint 4-cycles: every vertex has even degree (globally 0 odd
    // vertices), which would satisfy the naive degree-only circuit test --
    // but the graph is disconnected, so no single walk can cover both.
    Graph G(8, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    G.add_edge(4, 5); G.add_edge(5, 6); G.add_edge(6, 7); G.add_edge(7, 4);
    auto res = eulerian_path(G);
    EXPECT_FALSE(res.has_circuit);
    EXPECT_FALSE(res.has_path);
    EXPECT_TRUE(res.path.empty());
}

TEST(GraphEulerian, DisconnectedCircuitComponentPlusPathComponentIsNeitherOverall) {
    // Component A = 4-cycle on {0,1,2,3} (0 odd vertices, its own circuit).
    // Component B = path 4-5-6 (exactly 2 odd vertices, its own path).
    // Combined globally: exactly 2 odd-degree vertices (4 and 6), which
    // passes the degree-only test -- but the two components are disjoint,
    // so the connectivity requirement must still reject both flags.
    Graph G(7, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0);
    G.add_edge(4, 5); G.add_edge(5, 6);
    auto res = eulerian_path(G);
    EXPECT_FALSE(res.has_circuit);
    EXPECT_FALSE(res.has_path);
    EXPECT_TRUE(res.path.empty());
}

TEST(GraphEulerian, IsolatedVertexIsIgnoredAndCircuitStillFound) {
    // Triangle {0,1,2} plus an isolated vertex 3 with no edges at all.
    // Connectivity is judged only over vertices with nonzero degree, so
    // the isolated vertex must not block the circuit.
    Graph G(4, false);
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 0);
    auto res = eulerian_path(G);
    EXPECT_TRUE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    ASSERT_TRUE(eulerian_path_uses_every_edge_exactly_once(G, res.path));
    EXPECT_EQ(res.path.front(), res.path.back());
}

TEST(GraphEulerian, ZeroEdgeGraphConventionIsVacuousCircuitWithEmptyPath) {
    // Documented convention: a graph with no edges vacuously satisfies
    // "every vertex has even degree" and "connected ignoring isolated
    // vertices" (there's nothing to connect), so both flags are true, but
    // there's nothing to traverse so `path` stays empty.
    Graph G(5, false);
    auto res = eulerian_path(G);
    EXPECT_TRUE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    EXPECT_TRUE(res.path.empty());
}

TEST(GraphEulerian, ZeroVertexGraphConventionIsVacuousCircuitWithEmptyPath) {
    Graph G(0, false);
    auto res = eulerian_path(G);
    EXPECT_TRUE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
    EXPECT_TRUE(res.path.empty());
}

TEST(GraphEulerian, HasPathIsTrueWheneverHasCircuitIsTrueAcrossManyGraphs) {
    // has_path documents "Eulerian path (possibly a circuit) exists", so
    // it must be true any time has_circuit is true.
    for (const Graph& G : {make_barbell(), make_disjoint_triangles(1)}) {
        auto res = eulerian_path(G);
        if (res.has_circuit) EXPECT_TRUE(res.has_path);
    }
    Graph square(4, false);
    square.add_edge(0, 1); square.add_edge(1, 2); square.add_edge(2, 3); square.add_edge(3, 0);
    auto res = eulerian_path(square);
    EXPECT_TRUE(res.has_circuit);
    EXPECT_TRUE(res.has_path);
}

TEST(GraphEulerian, AllPositiveCaseGraphsProduceStructurallyValidPaths) {
    // Broad regression sweep: every graph here is expected to have either
    // an Eulerian circuit or path, and in every case the returned `path`
    // must pass the full structural-completeness check (real edges,
    // exactly matching the edge count) -- not just a spot-check.
    Graph square(4, false);
    square.add_edge(0, 1); square.add_edge(1, 2); square.add_edge(2, 3); square.add_edge(3, 0);

    Graph figure_eight(5, false);
    figure_eight.add_edge(0, 1); figure_eight.add_edge(1, 2); figure_eight.add_edge(2, 0);
    figure_eight.add_edge(0, 3); figure_eight.add_edge(3, 4); figure_eight.add_edge(4, 0);

    Graph simple_path(4, false);
    simple_path.add_edge(0, 1); simple_path.add_edge(1, 2); simple_path.add_edge(2, 3);

    for (const Graph& G : {square, figure_eight, simple_path, make_barbell(),
                           make_disjoint_triangles(1)}) {
        auto res = eulerian_path(G);
        ASSERT_TRUE(res.has_circuit || res.has_path);
        EXPECT_TRUE(eulerian_path_uses_every_edge_exactly_once(G, res.path));
    }
}
