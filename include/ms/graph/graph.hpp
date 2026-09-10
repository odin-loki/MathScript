// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
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

// Transitive closure (Floyd-Warshall-style boolean reachability): returns an
// n×n matrix where result[i][j] is true iff vertex j is reachable from vertex
// i via a directed path of length >= 0. Self-reachability is included:
// result[i][i] is always true (the zero-length path convention), even when G
// has no self-loop at i; input self-loops are redundant with this diagonal.
// Follows G's directed/undirected semantics (undirected edges appear in both
// endpoints' adjacency lists, so closure matches component reachability).
// @param G input graph
// @return n×n boolean reachability matrix; empty for n == 0
// @note O(V^3) time and O(V^2) space, same triple loop structure as floyd_warshall
std::vector<std::vector<bool>> transitive_closure(const Graph& G);

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
// Per-vertex eccentricity: max finite shortest-path distance from v to any
// vertex reachable from v (Floyd–Warshall on G). On a connected graph,
// ecc[v] is the usual graph eccentricity and max(ecc) == diameter(G).
// If v cannot reach every other vertex (disconnected G), ecc[v] is -1
// (unreachable/infinite eccentricity sentinel; floyd_warshall uses INF).
std::vector<int> eccentricity(const Graph& G);
int    diameter(const Graph& G);   // longest shortest path
int    radius(const Graph& G);
bool   is_tree(const Graph& G);
// Necessary-but-not-sufficient planarity SCREEN: the Euler-formula bound
// E <= 3V - 6 that every simple planar graph with V >= 3 satisfies. Kept
// bit-for-bit unchanged for backwards compatibility -- it is what the REPL's
// graph_is_planar binding has always called, and what the existing
// GraphProperties.PlanarHeuristic* tests assert.
//
// It is a HEURISTIC, and it is wrong in both directions:
//   - False POSITIVES: it accepts K3,3 (9 <= 12), the Petersen graph
//     (15 <= 24), the Wagner graph V8 (12 <= 18) and every sparse
//     subdivision of a Kuratowski graph -- all of which are non-planar.
//   - False NEGATIVES: the bound is only valid for V >= 3, so it reports
//     false for n == 0 and n == 1, and for any 2-vertex graph carrying an
//     edge (1 <= 0 is false), all of which are trivially planar.
//   - It counts G.n_edges(), i.e. add_edge calls, so self-loops and parallel
//     edges inflate E even though neither can affect planarity.
//
// USE is_planar(G) FOR THE EXACT ANSWER; this function remains only as the
// cheap O(1) screen it always was.
bool   is_planar_k5_k33_check(const Graph& G);  // heuristic

// Exact planarity test: true iff G can be drawn in the plane with no two
// edges crossing. Implements the Left-Right planarity criterion (de
// Fraysseix / Ossona de Mendez / Rosenstiehl, in Brandes' formulation): one
// DFS orients the graph and computes lowpoints and per-edge nesting depths, a
// second DFS over the nesting-depth-ordered adjacency lists maintains a stack
// of conflict pairs of return-edge intervals, and G is planar exactly when no
// two return edges are forced onto the same side of a DFS tree path.
//
// Conventions:
//   - The test runs on the UNDERLYING SIMPLE UNDIRECTED graph: directed edges
//     are symmetrised (same convention as louvain / k_core_decomposition),
//     self-loops are dropped, parallel edges collapse to one. Nothing is lost
//     -- a loop or a parallel twin can always be drawn alongside its partner
//     -- so G is planar iff its simple underlying graph is.
//   - Neighbour ids outside [0, n_vertices()) are skipped defensively.
//   - Disconnected graphs are handled directly (one DFS root per component,
//     lowest-numbered vertex first); G is planar iff every component is.
//   - n == 0, n == 1 and edgeless graphs are planar (vacuously). NOTE this
//     differs from is_planar_k5_k33_check, which reports false for them.
// @param G input graph
// @return true iff G is planar
// @note O(V + E): building the simple edge list dominates, after which the
//       E > 3V - 6 (V > 2) Euler rejection makes the three DFS passes O(V).
bool is_planar(const Graph& G);

// Combinatorial planar embedding (rotation system) of G, produced by the same
// Left-Right run as is_planar. Returns, for each vertex v, the neighbours of v
// in clockwise cyclic order around v. The list is a CYCLE: which neighbour
// appears first is an implementation detail, but the cyclic order and the
// orientation are consistent across every vertex, which is what makes the
// rotation system an embedding.
//
// Face tracing: starting from a directed half-edge (v,w), the next half-edge
// of the same face is (w,x) where x is the CYCLIC PREDECESSOR of v in
// result[w] (i.e. v's counter-clockwise neighbour around w). Following that
// rule from every one of the 2E directed half-edges partitions them into
// faces. A rotation system carries no information tying one component to
// another, so faces are traced PER COMPONENT: each connected component that
// has at least one edge satisfies Euler's formula V_i - E_i + F_i == 2 on its
// own, and the total face count is therefore
//     F == 2 * C_e - V_e + E
// where C_e counts the components carrying at least one edge and V_e counts
// the vertices those components contain. (Two disjoint triangles give F == 4,
// not the 3 a plane DRAWING would show -- a drawing merges the components'
// outer faces, a rotation system cannot know they were merged. Isolated
// vertices have no half-edges and so contribute no face at all.)
//
// Conventions: same simple-underlying-graph normalisation as is_planar, so
// self-loops and parallel edges never appear in the returned lists, directed
// input is symmetrised, and an isolated vertex gets an empty list. The
// returned vector always has exactly G.n_vertices() entries (empty for
// n == 0).
// @param G input graph
// @return per-vertex clockwise neighbour cycle, or a DomainError when G is
//         not planar (there is no embedding to return)
// @note O(V + E), the same single Left-Right run as is_planar plus the
//       embedding pass.
Result<std::vector<std::vector<int>>> planar_embedding(const Graph& G);

// Kuratowski subgraph: when G is NOT planar, the edges of a subdivision of K5
// or of K3,3 contained in G -- the concrete certificate of non-planarity that
// Kuratowski's theorem promises. Empty when G IS planar.
//
// Found by edge-minimisation: starting from G's simple underlying edge set (in
// ascending (u,v) order), each edge in turn is deleted and the rest re-tested;
// an edge whose deletion makes the graph planar is essential and put back, an
// edge whose deletion leaves it non-planar is left out. What survives is an
// edge-minimal non-planar graph, which by Kuratowski's theorem is exactly a
// subdivision of K5 or K3,3: E - V == 5 with five degree-4 branch vertices
// (K5 case), or E - V == 3 with six degree-3 branch vertices (K3,3 case), all
// other vertices having degree 2 (V counting only vertices the returned edges
// actually touch).
//
// Conventions: same simple-underlying-graph normalisation as is_planar.
// Returned edges have from < to, carry the same (parallel-collapsed) weight
// canonical simple-edge extraction assigns, and are sorted ascending by
// (from, to) -- the shape and ordering bridges() already uses.
// @param G input graph
// @return edges of a K5 or K3,3 subdivision; empty iff is_planar(G)
// @note O(E * (V + E)): one planarity test per candidate edge.
std::vector<Edge> kuratowski_subgraph(const Graph& G);

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

// Maximum cardinality matching in a general undirected graph (Edmonds blossom).
// Directed edges are treated as undirected (same convention as bridges /
// biconnected_components). Self-loops are ignored. Returns matching edges as
// (u,v) pairs with u < v. Empty when n == 0 or the graph has no matchable edges.
// @param G input graph
// @return list of matched undirected edges (cardinality-maximal)
// @note O(V^3) time via BFS blossom shrinking; intended for moderate n
std::vector<std::pair<int, int>> maximum_matching(const Graph& G);

// Maximum-WEIGHT matching in a general (non-bipartite) undirected graph, via
// Edmonds' primal-dual blossom algorithm in Galil's O(V^3) formulation (a
// stage per augmentation, an S/T-labelled alternating forest, blossom
// shrinking, dual adjustment by the least of four candidate deltas, and
// blossom expansion). Where maximum_matching maximises the NUMBER of matched
// edges, this maximises the SUM of their weights.
//
// Conventions, matching maximum_matching's where they overlap:
//   - Directed edges are treated as undirected (same convention as bridges /
//     biconnected_components / maximum_matching).
//   - Self-loops are ignored: a loop can never join two distinct vertices.
//   - Parallel edges between the same pair collapse to the single HEAVIEST
//     copy -- an optimal matching would never prefer a lighter parallel twin.
//   - Neighbour ids outside [0, n_vertices()) are skipped defensively.
//   - Isolated vertices simply stay unmatched.
//   - Matched edges are returned as (u,v) pairs with u < v, sorted ascending
//     by u then by v -- the same shape and ordering maximum_matching returns.
//     Empty for n == 0, for an edgeless graph, and whenever no edge improves
//     the total.
//
// Weight handling:
//   - Negative-weight edges are never matched when maxcardinality is false
//     (leaving both endpoints free is strictly better), so an all-negative
//     graph yields an EMPTY matching, not a maximal one.
//   - Zero-weight edges neither help nor hurt; whether one appears is fixed
//     by the search order and never changes the reported total.
//   - With every weight 1.0 (Graph::add_edge's default) the total weight IS
//     the cardinality, so the result has the same SIZE as maximum_matching(G)
//     -- though possibly a different edge set, since ties are broken by
//     search order rather than by any documented rule.
//
// @param G input graph
// @param maxcardinality when true, search only among matchings of MAXIMUM
//        CARDINALITY and return the heaviest of those: the size equals
//        maximum_matching(G).size() exactly, even when reaching it forces
//        negative-weight edges in and lowers the total. When false (the
//        default) cardinality is free and only total weight is maximised.
// @return matched undirected edges as (u,v) pairs, u < v, ascending
// @note O(V^3) time; O(V + E) space, with an O(V^2) worst case for the
//       per-blossom least-slack edge lists. Intended for moderate n, exactly
//       like maximum_matching. Exact for integer-valued weights (every dual
//       stays integral in the doubled formulation); for general real weights
//       a relative tolerance of 1e-9 * max(1, max|w|) decides edge slack.
std::vector<std::pair<int, int>> max_weight_matching(const Graph& G,
                                                     bool maxcardinality = false);

// Total weight of max_weight_matching(G, maxcardinality): the sum of the
// parallel-collapsed weights of the edges that function returns. 0.0 for an
// empty matching, including n == 0 and edgeless graphs.
// @param G input graph
// @param maxcardinality see max_weight_matching
// @return sum of matched edge weights (may be negative when maxcardinality
//         is true and the graph's maximum-cardinality matchings all cost)
// @note Runs the same O(V^3) search; it is not a cheaper query.
double max_weight_matching_value(const Graph& G, bool maxcardinality = false);

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
