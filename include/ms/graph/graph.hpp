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

// Minimum spanning arborescence rooted at `root` (directed graphs only).
// Chu-Liu/Edmonds: greedily pick each non-root vertex's cheapest incoming
// edge; if that set contains a cycle, contract the cycle, recurse, expand.
// Requires every non-root vertex to have at least one incoming edge.
struct ArborescenceResult { double total_weight; std::vector<Edge> edges; };
ArborescenceResult min_arborescence(const Graph& G, int root);

// --- Graph properties ---
int    diameter(const Graph& G);   // longest shortest path
int    radius(const Graph& G);
bool   is_tree(const Graph& G);
bool   is_planar_k5_k33_check(const Graph& G);  // heuristic

// K-core decomposition of an undirected graph via the standard Batagelj-
// Zaversnik bucket-queue degree-peeling algorithm (O(V+E)). The core number
// of a vertex is the largest k such that it belongs to the k-core (the
// maximal induced subgraph where every vertex has degree >= k within that
// subgraph). Directed edges are symmetrised (as with louvain/biconnected
// components); self-loops and multi-edges are handled via adjacency-list
// degree (each stored half-edge entry contributes to degree). Isolated
// vertices receive core number 0.
// @param G undirected (or directed-treated-as-undirected) input graph
// @return core number per vertex, indexed by vertex id (size n_vertices())
// @note O(V+E) time and space; identical convention to degree_centrality's
//       use of neighbors(v).size() for undirected degree
std::vector<int> k_core_decomposition(const Graph& G);

// Maximal k-core subgraph: the induced subgraph on all vertices whose core
// number is >= k (from k_core_decomposition). Vertices are compactly
// renumbered to [0, m) in ascending original-id order; edges are copied
// once per undirected pair. An empty graph is returned when no vertex
// survives the threshold (including k < 0 or k larger than the degeneracy).
// @param G input graph (same undirected convention as k_core_decomposition)
// @param k minimum core number for membership
// @return undirected subgraph on surviving vertices
Graph k_core_subgraph(const Graph& G, int k);

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

// Eulerian path/circuit existence check and construction, via Hierholzer's
// algorithm. An Eulerian CIRCUIT is a closed walk using every edge of G
// exactly once and returning to its start vertex; an Eulerian PATH is an
// open walk using every edge exactly once (start and end vertices differ).
//
// @note This module represents graphs with a single Graph type whose
//       is_directed() flag toggles whether add_edge mirrors an edge into
//       both endpoints' adjacency lists or just one (see Graph::add_edge);
//       there is no separate directed-graph representation. Nearly all of
//       ms::graph's structural-analysis algorithms (mst_kruskal, mst_prim,
//       articulation_points, bridges, biconnected_components, ...) are
//       specified for undirected graphs, with directed input either
//       rejected or treated in an unspecified/best-effort way. This
//       function follows that same convention: it is specified for
//       UNDIRECTED graphs only (using degree = neighbors(v).size(), which
//       double-counts each undirected edge across its two endpoints as
//       classic graph theory expects). Passing a directed Graph gives
//       unspecified results (degree would only reflect out-edges), exactly
//       as with biconnected_components/articulation_points/bridges. A full
//       directed-Eulerian-circuit implementation (in/out-degree balance
//       plus single-SCC-over-nonzero-degree-vertices connectivity) is
//       intentionally out of scope here to stay consistent with the rest
//       of the module.
//
// Existence criteria (classic graph theory, undirected case):
//   - Eulerian circuit: every vertex has even degree, AND the graph is
//     connected when isolated (zero-degree) vertices are ignored.
//   - Eulerian path that is not a circuit: EXACTLY 2 vertices have odd
//     degree (all others even), AND the graph is connected when isolated
//     vertices are ignored. Any odd-degree-vertex count other than 0 or 2
//     makes an Eulerian path/circuit impossible.
//
// @note Degenerate zero-edge convention: a graph with no edges (including
//       n == 0 vertices) vacuously satisfies "every vertex has even degree"
//       (0 is even) and is vacuously "connected ignoring isolated vertices"
//       (there are none to connect), so this function reports
//       has_circuit = has_path = true for it -- but leaves `path` empty
//       since there is nothing to traverse. This is a deliberate, documented
//       choice among the genuinely ambiguous conventions in the literature;
//       callers that want "no edges" treated as failure should special-case
//       G.n_edges() == 0 themselves.
struct EulerianResult {
    bool has_circuit = false;   // Eulerian circuit exists
    bool has_path = false;      // Eulerian path (possibly a circuit) exists
    std::vector<int> path;      // vertex sequence of the actual Eulerian path/circuit found
                                 // (empty if neither exists, or if the graph has no edges)
};
// @param G the graph, assumed undirected per this function's documented scope (see above).
// @return has_circuit/has_path flags plus a constructed vertex sequence (via Hierholzer's
//         algorithm) if either exists; `path` is empty when neither exists.
// @note O(V+E) time and space: one connectivity BFS plus one Hierholzer pass, each linear.
EulerianResult eulerian_path(const Graph& G);

// Traveling Salesman Problem heuristic: finds a low-cost (not necessarily optimal -- TSP is
// NP-hard, this is a practical heuristic) Hamiltonian cycle visiting every vertex exactly once
// and returning to the start, via the classic two-phase approach:
//   1. CONSTRUCTION: nearest-neighbor greedy tour -- starting from a given vertex, repeatedly
//      move to the nearest unvisited vertex, until all vertices are visited, then return to
//      the start.
//   2. IMPROVEMENT: 2-opt local search -- repeatedly look for a pair of edges (i,i+1) and
//      (j,j+1) in the current tour such that REVERSING the tour segment between them (i.e.
//      replacing edges (tour[i],tour[i+1]) and (tour[j],tour[j+1]) with (tour[i],tour[j]) and
//      (tour[i+1],tour[j+1])) would REDUCE the total tour length; apply the best such
//      improvement found each pass, repeat until no improving swap exists (or a max iteration
//      cap is hit, to guarantee termination on adversarial/large inputs).
//
// Distances: this operates on a COMPLETE graph given by a full pairwise distance matrix (NOT
// this module's sparse Graph/Edge representation, since TSP conceptually needs a distance
// between every pair of vertices, not just explicit edges) -- @param dist an n x n symmetric
// distance matrix (dist[i][j] = distance from vertex i to vertex j, dist[i][i] should be 0).
// This is a deliberate scope choice to keep the API simple and directly usable (callers with a
// sparse graph can first compute all-pairs shortest paths via this module's existing
// Floyd-Warshall or similar to get a dense distance matrix, then pass that in here).
struct TSPResult {
    std::vector<int> tour;        // vertex visit order (length n, tour[0] is the start,
                                   // implicitly returns to tour[0] at the end -- the cycle is
                                   // NOT explicitly repeated at the end of this vector)
    double total_distance = 0.0;  // total cycle length including the return edge
};
// @param dist n x n symmetric distance matrix. n < 2 is a degenerate/trivial case: n == 0
//        returns an empty tour with distance 0, n == 1 returns tour {0} with distance 0.
// @param start_vertex which vertex to start the nearest-neighbor construction from (default 0).
// @param max_2opt_iterations safety cap on 2-opt improvement passes to guarantee termination
//        (a reasonable default, e.g. 1000, should be more than enough for the moderate-size
//        test inputs this will actually see, since 2-opt on small instances converges
//        quickly); 0 disables the improvement phase entirely, returning the plain
//        nearest-neighbor construction unchanged.
// @return the improved tour and its total cycle distance.
// @note O(n^2) construction plus O(iterations * n^2) improvement; intended for the small/
//       moderate instance sizes this library's other exact/heuristic graph algorithms target.
TSPResult tsp_heuristic(const std::vector<std::vector<double>>& dist,
                        int start_vertex = 0, int max_2opt_iterations = 1000);

} // namespace graph
} // namespace ms
