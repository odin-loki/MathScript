#pragma once
#include "ms/error/error_types.hpp"
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ms {
namespace graph {

// --- Graph data structure ---
struct Edge { int from, to; double weight; };

class Graph {
public:
    explicit Graph(int n_vertices, bool directed = false);
    void add_edge(int u, int v, double weight = 1.0);
    void remove_edge(int u, int v);
    int  n_vertices() const { return static_cast<int>(adj_.size()); }
    int  n_edges()    const { return n_edges_; }
    bool is_directed() const { return directed_; }
    const std::vector<std::pair<int,double>>& neighbors(int v) const { return adj_[v]; }
    std::vector<Edge> edges() const;

private:
    std::vector<std::vector<std::pair<int,double>>> adj_;
    bool directed_;
    int  n_edges_ = 0;
};

// Construction helpers
Graph from_edge_list(int n, const std::vector<Edge>& edges, bool directed = false);

// --- Traversal ---
std::vector<int> bfs(const Graph& G, int source);
std::vector<int> dfs(const Graph& G, int source);
Result<std::vector<int>> topological_sort(const Graph& G);   // DAGs only

// --- Shortest paths ---
constexpr double INF = std::numeric_limits<double>::infinity();

// Dijkstra: returns {dist, parent}
std::pair<std::vector<double>, std::vector<int>>
    dijkstra(const Graph& G, int source);

// Bellman-Ford: returns {dist, parent}, error if negative cycle
Result<std::pair<std::vector<double>, std::vector<int>>>
    bellman_ford(const Graph& G, int source);

// Floyd-Warshall: returns n×n distance matrix
std::vector<std::vector<double>> floyd_warshall(const Graph& G);

// A*: returns path from source to target using heuristic h
Result<std::vector<int>> astar(const Graph& G, int source, int target,
                                const std::vector<double>& heuristic);

// --- Connectivity ---
bool                  is_connected(const Graph& G);
bool                  is_strongly_connected(const Graph& G);
std::vector<std::vector<int>> connected_components(const Graph& G);
std::vector<std::vector<int>> strongly_connected_components(const Graph& G);  // Tarjan
std::vector<int>      articulation_points(const Graph& G);   // undirected DFS low-link
std::vector<Edge>     bridges(const Graph& G);               // undirected DFS low-link

// Biconnected components (blocks) of an undirected graph, via Tarjan's DFS
// low-link algorithm with an explicit edge stack. A biconnected component is
// a maximal subgraph containing no articulation point of itself -- i.e. it
// stays connected after removing any single one of its vertices. Every edge
// of G belongs to exactly one block, so the returned components partition
// G's edge set (edges() elsewhere returns each edge once for undirected
// graphs, matching this convention). Blocks may share vertices (an
// articulation point sits at the junction of two or more blocks) but never
// share edges. A bridge edge (see bridges()) always forms its own singleton
// (1-edge, 2-vertex) block; conversely every singleton block returned here
// corresponds to a bridge. Directed edges are treated as undirected, as with
// articulation_points/bridges.
// @param G undirected (or directed-treated-as-undirected) input graph
// @return list of blocks, each a list of edges (u,v) belonging to that block;
//         empty for a graph with no edges (including n == 0)
// @note O(V+E) time and space, same DFS pass family as articulation_points/bridges
std::vector<std::vector<Edge>> biconnected_components(const Graph& G);

bool                  is_bipartite(const Graph& G);
bool                  is_dag(const Graph& G);

// --- Spanning trees ---
// Returns MST edges (Kruskal)
std::vector<Edge> mst_kruskal(const Graph& G);
// Returns MST edges (Prim)
std::vector<Edge> mst_prim(const Graph& G, int start = 0);

// --- Graph properties ---
int    diameter(const Graph& G);   // longest shortest path
int    radius(const Graph& G);
bool   is_tree(const Graph& G);
bool   is_planar_k5_k33_check(const Graph& G);  // heuristic

// Backtracking vertex-bijection search (VF2-style, simplified): true iff a
// bijection between G1 and G2 vertices exists that preserves adjacency
// (and edge direction, for directed graphs). Exponential worst case; intended
// for the small graphs (roughly up to 12-15 vertices) this library's other
// exact graph algorithms already target, not large-scale isomorphism testing.
bool   is_isomorphic(const Graph& G1, const Graph& G2);

// --- Centrality ---
std::vector<double> pagerank(const Graph& G, double alpha = 0.85, int max_iter = 100);
std::vector<double> betweenness_centrality(const Graph& G);
std::vector<double> closeness_centrality(const Graph& G);
std::vector<double> degree_centrality(const Graph& G);
// Power iteration on adjacency (dominant eigenvector); L2-normalised scores
std::vector<double> eigenvector_centrality(const Graph& G, int max_iter = 100, double tol = 1e-9);
// x = alpha * A^T * x + beta; requires alpha < 1/spectral_radius(A) for convergence
std::vector<double> katz_centrality(const Graph& G, double alpha = 0.1, double beta = 1.0,
                                    int max_iter = 100, double tol = 1e-9);

// --- Spectral ---
std::vector<std::vector<double>> laplacian(const Graph& G);              // L = D - A
std::vector<std::vector<double>> normalised_laplacian(const Graph& G);  // I - D^{-1/2} A D^{-1/2}
// Second-smallest Laplacian eigenvalue (0 for disconnected graphs)
double algebraic_connectivity(const Graph& G);

// --- Max flow (Dinic's algorithm) ---
Result<double> max_flow(const Graph& G, int source, int sink);

struct MinCutResult { double value; std::vector<Edge> cut_edges; };
Result<MinCutResult> min_cut(const Graph& G, int source, int sink);

// --- Bipartite matching (Hopcroft-Karp) ---
// G must be bipartite with left vertices [0, left_size) and right [left_size, n)
Result<int> bipartite_match(const Graph& G, int left_size);

// --- Coloring (greedy) ---
std::vector<int> greedy_colour(const Graph& G);
int chromatic_number_approx(const Graph& G);  // greedy upper bound

// --- Community detection ---
// Louvain community detection (greedy modularity optimization, Blondel et al. 2008).
// Treats the graph as undirected and weighted (directed edges are symmetrised;
// unweighted edges carry weight 1, matching Graph's own default). Returns a
// partition of vertex indices into communities (groups), analogous to
// connected_components' return shape. Deterministic given the same input
// (fixed vertex-visitation order each local-moving pass -- no RNG). A graph
// with no edges yields every vertex in its own singleton community.
std::vector<std::vector<int>> louvain(const Graph& G);

// Modularity Q of a given partition of G's vertices (Newman-Girvan weighted
// modularity). `communities` must partition [0, G.n_vertices()); vertices not
// covered by any group are treated as contributing zero. Useful standalone
// for scoring/comparing partitions, and to verify louvain()'s output.
double modularity(const Graph& G, const std::vector<std::vector<int>>& communities);

// --- Miscellaneous ---
std::vector<double> adjacency_spectrum(const Graph& G);  // eigenvalues of adj matrix
std::vector<int>    euler_circuit(const Graph& G);        // Hierholzer
Result<std::vector<int>> hamiltonian_path(const Graph& G);  // backtracking

} // namespace graph
} // namespace ms
