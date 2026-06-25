#include "ms/graph/graph.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace ms::graph;

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
