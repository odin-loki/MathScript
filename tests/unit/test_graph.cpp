#include "ms/graph/graph.hpp"
#include <algorithm>
#include <cmath>
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
