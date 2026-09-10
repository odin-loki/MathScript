// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/graph/graph.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace ms {
namespace graph {

namespace {

// Build binary adjacency matrix (matches adjacency_spectrum convention).
std::vector<std::vector<double>> adjacency_matrix(const Graph& G) {
    int n = G.n_vertices();
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (int u = 0; u < n; ++u)
        for (auto& [v, w] : G.neighbors(u)) A[u][v] = 1.0;
    return A;
}

// Jacobi eigenvalue decomposition for small symmetric matrices.
std::vector<double> symmetric_eigenvalues(std::vector<std::vector<double>> A) {
    int n = static_cast<int>(A.size());
    if (n == 0) return {};
    if (n == 1) return {A[0][0]};
    const double tol = 1e-12;
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
        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        double c = std::cos(phi), s = std::sin(phi);
        for (int k = 0; k < n; ++k) {
            double akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[p][k] = A[k][p];
            A[k][q] = s * akp + c * akq;
            A[q][k] = A[k][q];
        }
        double new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        double new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][p] = new_pp;
        A[q][q] = new_qq;
        A[p][q] = A[q][p] = 0.0;
    }
    std::vector<double> evals(n);
    for (int i = 0; i < n; ++i) evals[i] = A[i][i];
    std::sort(evals.begin(), evals.end());
    return evals;
}

struct DinicResult {
    double flow = 0.0;
    std::vector<std::vector<std::pair<int, double>>> residual;
};

DinicResult dinic_max_flow_residual(const Graph& G, int source, int sink) {
    int n = G.n_vertices();
    DinicResult result;
    auto& res = result.residual;
    res.assign(n, {});
    std::vector<std::vector<int>> rev_idx(n);
    for (int u = 0; u < n; ++u) {
        for (auto& [v, w] : G.neighbors(u)) {
            int uidx = static_cast<int>(res[u].size());
            int vidx = static_cast<int>(res[v].size());
            res[u].push_back({v, w});
            res[v].push_back({u, 0.0});
            rev_idx[u].push_back(vidx);
            rev_idx[v].push_back(uidx);
        }
    }
    while (true) {
        std::vector<int> level(n, -1);
        level[source] = 0;
        std::queue<int> q;
        q.push(source);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (auto& [u, cap] : res[v])
                if (cap > 1e-12 && level[u] < 0) { level[u] = level[v] + 1; q.push(u); }
        }
        if (level[sink] < 0) break;
        std::vector<int> iter(n, 0);
        std::function<double(int, double)> dfs_flow = [&](int v, double pushed) -> double {
            if (v == sink) return pushed;
            for (int& i = iter[v]; i < static_cast<int>(res[v].size()); ++i) {
                auto& [u, cap] = res[v][i];
                if (cap < 1e-12 || level[u] != level[v] + 1) continue;
                double d = dfs_flow(u, std::min(pushed, cap));
                if (d > 0) { cap -= d; res[u][rev_idx[v][i]].second += d; return d; }
            }
            return 0.0;
        };
        while (true) {
            double pushed = dfs_flow(source, INF);
            if (pushed < 1e-12) break;
            result.flow += pushed;
        }
    }
    return result;
}

void dfs_articulation_bridges(const Graph& G, int u, int parent, int& timer,
                              std::vector<int>& disc, std::vector<int>& low,
                              std::vector<bool>& is_ap,
                              std::vector<Edge>& bridges) {
    disc[u] = low[u] = ++timer;
    int children = 0;
    for (auto& [v, w] : G.neighbors(u)) {
        if (v == parent) continue;
        if (disc[v] == -1) {
            ++children;
            dfs_articulation_bridges(G, v, u, timer, disc, low, is_ap, bridges);
            low[u] = std::min(low[u], low[v]);
            if (parent != -1 && low[v] >= disc[u]) is_ap[u] = true;
            if (low[v] > disc[u]) {
                int a = std::min(u, v), b = std::max(u, v);
                bridges.push_back({a, b, w});
            }
        } else {
            low[u] = std::min(low[u], disc[v]);
        }
    }
    if (parent == -1 && children >= 2) is_ap[u] = true;
}

// Same low-link DFS as dfs_articulation_bridges, augmented with an explicit
// edge stack: every edge is pushed exactly once, when first traversed from
// its lower-disc endpoint (tree edges when the child is discovered, back
// edges when a descendant looks up at a not-yet-finished ancestor). When a
// child v returns with low[v] >= disc[u] -- the same condition that flags u
// as an articulation point (or the DFS root closing off a subtree) -- every
// edge back to and including (u, v) is popped off the stack: that batch is
// exactly one biconnected component.
void dfs_biconnected(const Graph& G, int u, int parent, int& timer,
                     std::vector<int>& disc, std::vector<int>& low,
                     std::vector<Edge>& edge_stack,
                     std::vector<std::vector<Edge>>& components) {
    disc[u] = low[u] = ++timer;
    for (auto& [v, w] : G.neighbors(u)) {
        if (v == parent) continue;
        if (disc[v] == -1) {
            edge_stack.push_back({u, v, w});
            dfs_biconnected(G, v, u, timer, disc, low, edge_stack, components);
            low[u] = std::min(low[u], low[v]);
            if (low[v] >= disc[u]) {
                std::vector<Edge> comp;
                while (true) {
                    Edge e = edge_stack.back();
                    edge_stack.pop_back();
                    comp.push_back(e);
                    if (e.from == u && e.to == v) break;
                }
                components.push_back(std::move(comp));
            }
        } else if (disc[v] < disc[u]) {
            // Back edge, encountered from the descendant looking up at an
            // ancestor still on the call stack; push once here only (the
            // ancestor will later see the same edge with disc[v] > disc[u]
            // and must not push it again).
            edge_stack.push_back({u, v, w});
            low[u] = std::min(low[u], disc[v]);
        }
    }
}

// Symmetrise a directed graph into an undirected one (each directed edge
// becomes one undirected edge); undirected inputs are just copied as-is.
Graph to_undirected(const Graph& G) {
    if (!G.is_directed()) return G;
    Graph U(G.n_vertices(), false);
    for (int u = 0; u < G.n_vertices(); ++u)
        for (auto& [v, w] : G.neighbors(u)) U.add_edge(u, v, w);
    return U;
}


// Underlying SIMPLE UNDIRECTED edge list of G, the common normalisation used
// by max_weight_matching and the planarity family. Directed graphs are
// symmetrised (a directed edge becomes one undirected edge, as with
// to_undirected); self-loops are dropped (they can neither be matched nor
// affect planarity); parallel copies of the same unordered pair collapse to
// the single HEAVIEST copy; neighbour ids outside [0, n) are skipped
// defensively (the same guard maximum_matching already applies). Every
// returned Edge has from < to, and the list is sorted ascending by
// (from, to), so the downstream edge numbering -- and therefore every
// tie-break in the algorithms built on it -- is a deterministic function of
// G alone.
// O(E log E) time, O(E) space.
std::vector<Edge> canonical_simple_edges(const Graph& G) {
    const int n = G.n_vertices();
    std::map<std::pair<int, int>, double> best;
    for (int u = 0; u < n; ++u) {
        for (const auto& [v, w] : G.neighbors(u)) {
            if (v < 0 || v >= n || v == u) continue;
            const std::pair<int, int> key{std::min(u, v), std::max(u, v)};
            const auto it = best.find(key);
            if (it == best.end()) best.emplace(key, w);
            else if (w > it->second) it->second = w;
        }
    }
    std::vector<Edge> out;
    out.reserve(best.size());
    for (const auto& [key, w] : best) out.push_back(Edge{key.first, key.second, w});
    return out;
}
} // namespace

// ---- Graph ----

Graph::Graph(int n, bool directed)
    : adj_(n), directed_(directed) {}

void Graph::add_edge(int u, int v, double weight) {
    adj_[u].push_back({v, weight});
    if (!directed_) adj_[v].push_back({u, weight});
    ++n_edges_;
}

void Graph::remove_edge(int u, int v) {
    auto& au = adj_[u];
    au.erase(std::remove_if(au.begin(), au.end(),
        [v](const std::pair<int,double>& p){ return p.first == v; }), au.end());
    if (!directed_) {
        auto& av = adj_[v];
        av.erase(std::remove_if(av.begin(), av.end(),
            [u](const std::pair<int,double>& p){ return p.first == u; }), av.end());
    }
    --n_edges_;
}

std::vector<Edge> Graph::edges() const {
    std::vector<Edge> result;
    for (int u = 0; u < (int)adj_.size(); ++u)
        for (auto& [v, w] : adj_[u])
            if (directed_ || u <= v)
                result.push_back({u, v, w});
    return result;
}

Graph from_edge_list(int n, const std::vector<Edge>& edges, bool directed) {
    Graph G(n, directed);
    for (auto& e : edges) G.add_edge(e.from, e.to, e.weight);
    return G;
}

// ---- Traversal ----

namespace {

// Index-based queue: one reserve, no pop/shift overhead from std::queue.
inline void bfs_visit_neighbors(const Graph& G, int v,
                                std::vector<std::uint8_t>& visited,
                                std::vector<int>& q) {
    const auto& nbrs = G.neighbors(v);
    for (size_t i = 0, end = nbrs.size(); i < end; ++i) {
        const int u = nbrs[i].first;
        if (!visited[static_cast<size_t>(u)]) {
            visited[static_cast<size_t>(u)] = 1;
            q.push_back(u);
        }
    }
}

// Sparse graphs: skip weight reads when scanning adjacency lists.
inline void bfs_visit_neighbors_sparse(const Graph& G, int v,
                                       std::vector<std::uint8_t>& visited,
                                       std::vector<int>& q) {
    for (const auto& nbr : G.neighbors(v)) {
        const int u = nbr.first;
        if (!visited[static_cast<size_t>(u)]) {
            visited[static_cast<size_t>(u)] = 1;
            q.push_back(u);
        }
    }
}

std::vector<int> bfs_impl(const Graph& G, int source, bool dense) {
    const int n = G.n_vertices();
    std::vector<std::uint8_t> visited(static_cast<size_t>(n), 0);
    std::vector<int> order;
    std::vector<int> q;
    order.reserve(static_cast<size_t>(n));
    q.reserve(static_cast<size_t>(n));

    q.push_back(source);
    visited[static_cast<size_t>(source)] = 1;

    size_t head = 0;
    while (head < q.size()) {
        const int v = q[head++];
        order.push_back(v);
        if (dense)
            bfs_visit_neighbors(G, v, visited, q);
        else
            bfs_visit_neighbors_sparse(G, v, visited, q);
    }
    return order;
}

} // namespace

std::vector<int> bfs(const Graph& G, int source) {
    const int n = G.n_vertices();
    if (n == 0) return {};
    const int m = G.n_edges();
    const bool dense =
        static_cast<long long>(m) * 4LL >= static_cast<long long>(n) * static_cast<long long>(n);
    return bfs_impl(G, source, dense);
}

std::vector<int> dfs(const Graph& G, int source) {
    int n = G.n_vertices();
    std::vector<bool> visited(n, false);
    std::vector<int> order;
    std::stack<int> stk;
    stk.push(source);
    while (!stk.empty()) {
        int v = stk.top(); stk.pop();
        if (visited[v]) continue;
        visited[v] = true;
        order.push_back(v);
        for (auto& [u, w] : G.neighbors(v))
            if (!visited[u]) stk.push(u);
    }
    return order;
}

Result<std::vector<int>> topological_sort(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> indegree(n, 0);
    for (int u = 0; u < n; ++u)
        for (auto& [v, w] : G.neighbors(u)) ++indegree[v];
    std::queue<int> q;
    for (int v = 0; v < n; ++v) if (indegree[v] == 0) q.push(v);
    std::vector<int> order;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        order.push_back(v);
        for (auto& [u, w] : G.neighbors(v))
            if (--indegree[u] == 0) q.push(u);
    }
    if ((int)order.size() != n)
        return std::unexpected(Error{DomainError{"topological_sort", "graph has cycle"}});
    return order;
}

// ---- Shortest paths ----

namespace {

using DijkstraEntry = std::pair<double, int>;

inline void dijkstra_relax_neighbors(const Graph& G, int u, double du,
                                     std::vector<double>& dist,
                                     std::vector<int>& parent,
                                     std::priority_queue<DijkstraEntry,
                                                         std::vector<DijkstraEntry>,
                                                         std::greater<DijkstraEntry>>& pq) {
    const auto& nbrs = G.neighbors(u);
    for (size_t i = 0, end = nbrs.size(); i < end; ++i) {
        const int v = nbrs[i].first;
        const double nd = du + nbrs[i].second;
        if (nd < dist[static_cast<size_t>(v)]) {
            dist[static_cast<size_t>(v)] = nd;
            parent[static_cast<size_t>(v)] = u;
            pq.push({nd, v});
        }
    }
}

inline void dijkstra_relax_neighbors_sparse(const Graph& G, int u, double du,
                                            std::vector<double>& dist,
                                            std::vector<int>& parent,
                                            std::priority_queue<DijkstraEntry,
                                                                std::vector<DijkstraEntry>,
                                                                std::greater<DijkstraEntry>>& pq) {
    for (const auto& nbr : G.neighbors(u)) {
        const int v = nbr.first;
        const double nd = du + nbr.second;
        if (nd < dist[static_cast<size_t>(v)]) {
            dist[static_cast<size_t>(v)] = nd;
            parent[static_cast<size_t>(v)] = u;
            pq.push({nd, v});
        }
    }
}

std::pair<std::vector<double>, std::vector<int>>
dijkstra_impl(const Graph& G, int source, bool dense) {
    const int n = G.n_vertices();
    std::vector<double> dist(static_cast<size_t>(n), INF);
    std::vector<int> parent(static_cast<size_t>(n), -1);
    dist[static_cast<size_t>(source)] = 0.0;

    std::vector<DijkstraEntry> heap;
    heap.reserve(static_cast<size_t>(n));
    std::priority_queue<DijkstraEntry, std::vector<DijkstraEntry>, std::greater<DijkstraEntry>>
        pq(std::greater<DijkstraEntry>(), std::move(heap));
    pq.push({0.0, source});

    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[static_cast<size_t>(u)]) continue;
        if (dense)
            dijkstra_relax_neighbors(G, u, d, dist, parent, pq);
        else
            dijkstra_relax_neighbors_sparse(G, u, d, dist, parent, pq);
    }
    return {dist, parent};
}

} // namespace

std::pair<std::vector<double>, std::vector<int>>
dijkstra(const Graph& G, int source) {
    const int n = G.n_vertices();
    if (n == 0) return {{}, {}};
    const int m = G.n_edges();
    const bool dense =
        static_cast<long long>(m) * 4LL >= static_cast<long long>(n) * static_cast<long long>(n);
    return dijkstra_impl(G, source, dense);
}

Result<std::pair<std::vector<double>, std::vector<int>>>
bellman_ford(const Graph& G, int source) {
    int n = G.n_vertices();
    std::vector<double> dist(n, INF);
    std::vector<int> parent(n, -1);
    dist[source] = 0.0;
    auto edges = G.edges();
    for (int i = 0; i < n - 1; ++i)
        for (auto& e : edges)
            if (dist[e.from] < INF && dist[e.from] + e.weight < dist[e.to]) {
                dist[e.to]    = dist[e.from] + e.weight;
                parent[e.to]  = e.from;
            }
    // Check negative cycles
    for (auto& e : edges)
        if (dist[e.from] < INF && dist[e.from] + e.weight < dist[e.to])
            return std::unexpected(Error{DomainError{"bellman_ford", "negative cycle"}});
    return {{dist, parent}};
}

std::vector<std::vector<double>> floyd_warshall(const Graph& G) {
    const int n = G.n_vertices();
    const size_t ns = static_cast<size_t>(n);
    std::vector<double> flat(ns * ns, INF);
    for (int i = 0; i < n; ++i)
        flat[static_cast<size_t>(i) * ns + static_cast<size_t>(i)] = 0.0;
    for (int u = 0; u < n; ++u) {
        const size_t row = static_cast<size_t>(u) * ns;
        for (auto& [v, w] : G.neighbors(u)) {
            const size_t col = static_cast<size_t>(v);
            flat[row + col] = std::min(flat[row + col], w);
        }
    }
    for (int k = 0; k < n; ++k) {
        const size_t k_row = static_cast<size_t>(k) * ns;
        const double* row_k = flat.data() + k_row;
        for (int i = 0; i < n; ++i) {
            const size_t i_row = static_cast<size_t>(i) * ns;
            const double dik = flat[i_row + static_cast<size_t>(k)];
            if (dik >= INF) continue;
            double* row_i = flat.data() + i_row;
            for (int j = 0; j < n; ++j) {
                const double dkj = row_k[static_cast<size_t>(j)];
                if (dkj >= INF) continue;
                const double alt = dik + dkj;
                if (alt < row_i[static_cast<size_t>(j)]) row_i[static_cast<size_t>(j)] = alt;
            }
        }
    }
    std::vector<std::vector<double>> d(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            d[i][j] = flat[static_cast<size_t>(i) * ns + static_cast<size_t>(j)];
    return d;
}

std::vector<std::vector<bool>> transitive_closure(const Graph& G) {
    int n = G.n_vertices();
    std::vector<std::vector<bool>> reach(n, std::vector<bool>(n, false));
    for (int i = 0; i < n; ++i) {
        reach[i][i] = true;
        for (auto& [v, w] : G.neighbors(i)) reach[i][v] = true;
    }
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
    return reach;
}

Result<std::vector<int>> astar(const Graph& G, int source, int target,
                                const std::vector<double>& h) {
    int n = G.n_vertices();
    std::vector<double> g(n, INF), f(n, INF);
    std::vector<int> parent(n, -1);
    g[source] = 0.0; f[source] = h[source];
    using P = std::pair<double,int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    pq.push({f[source], source});
    std::vector<bool> closed(n, false);
    while (!pq.empty()) {
        auto [fv, v] = pq.top(); pq.pop();
        if (v == target) {
            std::vector<int> path;
            for (int u = target; u != -1; u = parent[u]) path.push_back(u);
            std::reverse(path.begin(), path.end());
            return path;
        }
        if (closed[v]) continue;
        closed[v] = true;
        for (auto& [u, w] : G.neighbors(v)) {
            double ng = g[v] + w;
            if (ng < g[u]) {
                g[u] = ng; f[u] = ng + h[u]; parent[u] = v;
                pq.push({f[u], u});
            }
        }
    }
    return std::unexpected(Error{DomainError{"astar", "no path"}});
}

// ---- Connectivity ----

bool is_connected(const Graph& G) {
    if (G.n_vertices() == 0) return true;
    return (int)bfs(G, 0).size() == G.n_vertices();
}

bool is_strongly_connected(const Graph& G) {
    auto comps = strongly_connected_components(G);
    return comps.size() == 1;
}

std::vector<std::vector<int>> connected_components(const Graph& G) {
    int n = G.n_vertices();
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> comps;
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            auto order = bfs(G, i);
            for (int v : order) visited[v] = true;
            comps.push_back(order);
        }
    }
    return comps;
}

// Tarjan's SCC
std::vector<std::vector<int>> strongly_connected_components(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> idx(n, -1), low(n, -1), comp(n, -1);
    std::vector<bool> on_stack(n, false);
    std::stack<int> stk;
    int counter = 0;
    std::vector<std::vector<int>> sccs;

    std::function<void(int)> tarjan = [&](int v) {
        idx[v] = low[v] = counter++;
        stk.push(v); on_stack[v] = true;
        for (auto& [u, w] : G.neighbors(v)) {
            if (idx[u] == -1) { tarjan(u); low[v] = std::min(low[v], low[u]); }
            else if (on_stack[u]) low[v] = std::min(low[v], idx[u]);
        }
        if (low[v] == idx[v]) {
            std::vector<int> scc;
            while (true) {
                int u = stk.top(); stk.pop();
                on_stack[u] = false;
                scc.push_back(u);
                if (u == v) break;
            }
            sccs.push_back(scc);
        }
    };
    for (int i = 0; i < n; ++i) if (idx[i] == -1) tarjan(i);
    return sccs;
}

std::vector<int> articulation_points(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> disc(n, -1), low(n, -1);
    std::vector<bool> is_ap(n, false);
    std::vector<Edge> unused;
    int timer = 0;
    for (int i = 0; i < n; ++i)
        if (disc[i] == -1)
            dfs_articulation_bridges(G, i, -1, timer, disc, low, is_ap, unused);
    std::vector<int> aps;
    for (int i = 0; i < n; ++i)
        if (is_ap[i]) aps.push_back(i);
    std::sort(aps.begin(), aps.end());
    return aps;
}

std::vector<Edge> bridges(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> disc(n, -1), low(n, -1);
    std::vector<bool> unused_ap(n, false);
    std::vector<Edge> result;
    int timer = 0;
    for (int i = 0; i < n; ++i)
        if (disc[i] == -1)
            dfs_articulation_bridges(G, i, -1, timer, disc, low, unused_ap, result);
    std::sort(result.begin(), result.end(), [](const Edge& a, const Edge& b) {
        return a.from < b.from || (a.from == b.from && a.to < b.to);
    });
    return result;
}

std::vector<std::vector<Edge>> biconnected_components(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> disc(n, -1), low(n, -1);
    std::vector<Edge> edge_stack;
    std::vector<std::vector<Edge>> components;
    int timer = 0;
    for (int i = 0; i < n; ++i)
        if (disc[i] == -1)
            dfs_biconnected(G, i, -1, timer, disc, low, edge_stack, components);

    auto edge_less = [](const Edge& a, const Edge& b) {
        return a.from < b.from || (a.from == b.from && a.to < b.to);
    };
    for (auto& comp : components) {
        for (auto& e : comp)
            if (e.from > e.to) std::swap(e.from, e.to);
        std::sort(comp.begin(), comp.end(), edge_less);
    }
    std::sort(components.begin(), components.end(), [&](const auto& a, const auto& b) {
        return edge_less(a.front(), b.front());
    });
    return components;
}

bool is_bipartite(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> color(n, -1);
    for (int start = 0; start < n; ++start) {
        if (color[start] != -1) continue;
        std::queue<int> q;
        q.push(start); color[start] = 0;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (auto& [u, w] : G.neighbors(v)) {
                if (color[u] == -1) { color[u] = 1 - color[v]; q.push(u); }
                else if (color[u] == color[v]) return false;
            }
        }
    }
    return true;
}

bool is_dag(const Graph& G) {
    if (!G.is_directed()) return false;
    return topological_sort(G).has_value();
}

// ---- MST ----

struct UnionFind {
    std::vector<int> parent, rank_;
    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }
};

std::vector<Edge> mst_kruskal(const Graph& G) {
    auto edges = G.edges();
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b){
        return a.weight < b.weight; });
    UnionFind uf(G.n_vertices());
    std::vector<Edge> mst;
    for (auto& e : edges)
        if (uf.unite(e.from, e.to)) mst.push_back(e);
    return mst;
}

std::vector<Edge> mst_prim(const Graph& G, int start) {
    int n = G.n_vertices();
    std::vector<double> key(n, INF);
    std::vector<int> parent(n, -1);
    std::vector<bool> in_mst(n, false);
    key[start] = 0.0;
    using P = std::pair<double,int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    pq.push({0.0, start});
    std::vector<Edge> mst;
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (in_mst[u]) continue;
        in_mst[u] = true;
        if (parent[u] != -1) mst.push_back({parent[u], u, d});
        for (auto& [v, w] : G.neighbors(u))
            if (!in_mst[v] && w < key[v]) { key[v] = w; parent[v] = u; pq.push({w, v}); }
    }
    return mst;
}

namespace {

struct ContractedEdge {
    int from, to;
    double weight;
    int orig_from, orig_to;
    double orig_weight;
};

ArborescenceResult edmonds_arborescence(const Graph& G, int root) {
    int n = G.n_vertices();
    if (n == 0) return {0.0, {}};
    if (n == 1) return {0.0, {}};

    std::vector<int> pred(n, -1);
    std::vector<double> pred_w(n, INF);
    for (int u = 0; u < n; ++u)
        for (auto& [v, w] : G.neighbors(u))
            if (v != root && w < pred_w[v]) { pred_w[v] = w; pred[v] = u; }

    for (int v = 0; v < n; ++v)
        if (v != root && pred[v] == -1) return {INF, {}};

    std::vector<bool> in_cycle(n, false);
    std::vector<int> stamp(n, -1);
    int cur_stamp = 0;
    for (int start = 0; start < n; ++start) {
        if (start == root) continue;
        if (stamp[start] != -1) continue;
        std::vector<int> path;
        int v = start;
        while (v != root && pred[v] != -1) {
            if (stamp[v] == cur_stamp) {
                auto it = std::find(path.begin(), path.end(), v);
                for (auto jt = it; jt != path.end(); ++jt) in_cycle[*jt] = true;
                in_cycle[v] = true;
                break;
            }
            if (stamp[v] != -1) break;
            stamp[v] = cur_stamp;
            path.push_back(v);
            v = pred[v];
        }
        ++cur_stamp;
    }

    bool has_cycle = false;
    for (bool b : in_cycle) if (b) { has_cycle = true; break; }

    if (!has_cycle) {
        ArborescenceResult result{0.0, {}};
        result.edges.reserve(static_cast<size_t>(n - 1));
        for (int v = 0; v < n; ++v) {
            if (v == root) continue;
            result.edges.push_back({pred[v], v, pred_w[v]});
            result.total_weight += pred_w[v];
        }
        return result;
    }

    std::vector<int> new_id(n, -1);
    int next_id = 0;
    int super = -1;
    for (int v = 0; v < n; ++v) {
        if (in_cycle[v]) {
            if (super == -1) super = next_id++;
            new_id[v] = super;
        } else {
            new_id[v] = next_id++;
        }
    }
    int new_n = next_id;
    int new_root = new_id[root];

    Graph H(new_n, true);
    std::vector<ContractedEdge> mapping;
    for (int u = 0; u < n; ++u) {
        for (auto& [v, w] : G.neighbors(u)) {
            int nu = new_id[u], nv = new_id[v];
            if (nu == nv) continue;
            double nw = w;
            if (in_cycle[v] && !in_cycle[u]) nw = w - pred_w[v];
            H.add_edge(nu, nv, nw);
            mapping.push_back({nu, nv, nw, u, v, w});
        }
    }

    auto sub = edmonds_arborescence(H, new_root);
    if (sub.edges.size() != static_cast<size_t>(new_n - 1)) return {INF, {}};

    auto lookup = [&](int nu, int nv, double nw) -> ContractedEdge {
        for (const auto& e : mapping)
            if (e.from == nu && e.to == nv && std::abs(e.weight - nw) < 1e-9)
                return e;
        for (const auto& e : mapping)
            if (e.from == nu && e.to == nv) return e;
        return {nu, nv, nw, -1, -1, nw};
    };

    int entry = -1;
    ArborescenceResult result{0.0, {}};
    result.edges.reserve(static_cast<size_t>(n - 1));
    for (const auto& e : sub.edges) {
        auto mapped = lookup(e.from, e.to, e.weight);
        if (mapped.orig_from < 0) return {INF, {}};
        result.edges.push_back({mapped.orig_from, mapped.orig_to, mapped.orig_weight});
        result.total_weight += mapped.orig_weight;
        if (e.to == super) entry = mapped.orig_to;
    }

    for (int v = 0; v < n; ++v) {
        if (!in_cycle[v] || v == entry) continue;
        result.edges.push_back({pred[v], v, pred_w[v]});
        result.total_weight += pred_w[v];
    }
    return result;
}

} // namespace

ArborescenceResult min_arborescence(const Graph& G, int root) {
    if (root < 0 || root >= G.n_vertices()) return {INF, {}};
    return edmonds_arborescence(G, root);
}

// ---- Properties ----

std::vector<int> eccentricity(const Graph& G) {
    auto d = floyd_warshall(G);
    int n = G.n_vertices();
    std::vector<int> ecc(n);
    for (int i = 0; i < n; ++i) {
        bool all_reachable = true;
        double max_d = 0.0;
        for (int j = 0; j < n; ++j) {
            if (d[i][j] >= INF) {
                if (j != i) all_reachable = false;
            } else {
                max_d = std::max(max_d, d[i][j]);
            }
        }
        ecc[i] = all_reachable ? static_cast<int>(max_d) : -1;
    }
    return ecc;
}

int diameter(const Graph& G) {
    auto d = floyd_warshall(G);
    int n = G.n_vertices();
    double diam = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (d[i][j] < INF) diam = std::max(diam, d[i][j]);
    return static_cast<int>(diam);
}

int radius(const Graph& G) {
    auto d = floyd_warshall(G);
    int n = G.n_vertices();
    double rad = INF;
    for (int i = 0; i < n; ++i) {
        double ecc = 0;
        for (int j = 0; j < n; ++j)
            if (d[i][j] < INF) ecc = std::max(ecc, d[i][j]);
        if (ecc > 0) rad = std::min(rad, ecc);
    }
    return rad < INF ? static_cast<int>(rad) : 0;
}

bool is_tree(const Graph& G) {
    int n = G.n_vertices();
    return is_connected(G) && G.n_edges() == n - 1;
}

bool is_planar_k5_k33_check(const Graph& G) {
    // Simple Euler formula check: planar => E <= 3V - 6
    int V = G.n_vertices(), E = G.n_edges();
    return E <= 3 * V - 6;
}

// ---- Exact planarity: the Left-Right criterion ----

namespace {

// Explicit signed -> unsigned index conversion, so every container subscript
// in the Left-Right and weighted-matching code below states the crossing.
inline std::size_t uz(int i) { return static_cast<std::size_t>(i); }

// One directed half-edge of the canonical simple edge list: `nbr` is the far
// endpoint, `edge` the undirected edge index, `half` the half-edge id
// (2*edge seen from the lower endpoint, 2*edge+1 from the higher one). A
// named struct rather than a 3-element array so no extra header is needed.
struct AdjEntry { int nbr = -1; int edge = -1; int half = -1; };

// A maximal run of return (back) edges that must all be drawn on the same
// side of the current DFS tree path; `low`/`high` are oriented edge ids with
// -1 meaning "none", so an interval is empty exactly when both are -1.
struct LRInterval {
    int low = -1;
    int high = -1;
    bool empty() const { return low == -1 && high == -1; }
};

// The two sides (left/right) of one constraint on a DFS tree path.
struct LRConflictPair {
    LRInterval left{};
    LRInterval right{};
};

// Left-Right planarity test (de Fraysseix / Ossona de Mendez / Rosenstiehl,
// in Brandes' iterative formulation), the engine behind is_planar and
// planar_embedding. Three passes over the canonical simple edge list:
//   1. dfs_orientation  -- orient every edge away from the DFS tree and
//      compute height, lowpt, lowpt2 and per-edge nesting depth;
//   2. dfs_testing      -- walk the nesting-depth-ordered adjacency lists
//      maintaining a stack of conflict pairs; the two `return false` sites in
//      add_constraints are the only places non-planarity is detected;
//   3. dfs_embedding    -- resolve every relative side to an absolute +-1,
//      re-sort by signed nesting depth and splice half-edges into per-vertex
//      circular lists, yielding the rotation system.
// Everything is std::vector/std::unordered_map state; no owning raw pointers,
// no floating point, and no recursion (all three passes use explicit stacks).
struct LRPlanarity {
    int n = 0;
    int m = 0;
    std::vector<Edge> E;
    std::vector<std::vector<AdjEntry>> adj;

    // Pass 1 state.
    std::vector<int> height;        // n; -1 = not yet visited
    std::vector<int> parent_edge;   // n; oriented edge id of the tree edge into v
    std::vector<int> src;           // m
    std::vector<int> dst;           // m
    std::vector<char> oriented;     // m
    std::vector<int> lowpt;         // m
    std::vector<int> lowpt2;        // m
    std::vector<int> nesting_depth; // m
    std::vector<std::vector<int>> out_edges;     // n; orientation order
    std::vector<std::vector<int>> ordered_adjs;  // n; nesting-depth order
    std::vector<int> roots;

    // Pass 2 state.
    std::vector<int> ref_;          // m; reference edge for side resolution
    std::vector<int> side_;         // m; relative, then absolute, side (+1/-1)
    std::vector<int> lowpt_edge;    // m
    std::vector<int> stack_bottom;  // m; index of S's top when the edge was entered
    std::vector<LRConflictPair> S;

    // Pass 3 state: per-vertex circular half-edge lists.
    std::vector<std::unordered_map<int, int>> emb_cw;
    std::vector<std::unordered_map<int, int>> emb_ccw;
    std::vector<int> emb_first;
    std::vector<int> left_ref;
    std::vector<int> right_ref;
    std::vector<int> chain_;

    // Shared per-pass scratch (each vertex belongs to exactly one DFS tree,
    // so one array per pass serves every root).
    std::vector<std::size_t> ind;
    std::vector<char> skip_init;

    // --- Pass 1: orientation -------------------------------------------
    void dfs_orientation(int root) {
        std::vector<int> dfs_stack;
        dfs_stack.push_back(root);
        while (!dfs_stack.empty()) {
            const int v = dfs_stack.back();
            dfs_stack.pop_back();
            const int e = parent_edge[uz(v)];
            while (ind[uz(v)] < adj[uz(v)].size()) {
                const AdjEntry a = adj[uz(v)][ind[uz(v)]];
                bool descended = false;
                if (!skip_init[uz(a.half)]) {
                    if (oriented[uz(a.edge)]) { ++ind[uz(v)]; continue; }
                    oriented[uz(a.edge)] = 1;
                    src[uz(a.edge)] = v;
                    dst[uz(a.edge)] = a.nbr;
                    out_edges[uz(v)].push_back(a.edge);
                    lowpt[uz(a.edge)] = height[uz(v)];
                    lowpt2[uz(a.edge)] = height[uz(v)];
                    if (height[uz(a.nbr)] == -1) {          // tree edge
                        parent_edge[uz(a.nbr)] = a.edge;
                        height[uz(a.nbr)] = height[uz(v)] + 1;
                        dfs_stack.push_back(v);             // revisit v after w
                        dfs_stack.push_back(a.nbr);
                        skip_init[uz(a.half)] = 1;
                        descended = true;
                    } else {                                // back edge
                        lowpt[uz(a.edge)] = height[uz(a.nbr)];
                    }
                }
                if (descended) break;                       // deliberately no ++ind
                const int k = a.edge;
                nesting_depth[uz(k)] = 2 * lowpt[uz(k)];
                if (lowpt2[uz(k)] < height[uz(v)]) nesting_depth[uz(k)] += 1;
                if (e != -1) {
                    if (lowpt[uz(k)] < lowpt[uz(e)]) {
                        lowpt2[uz(e)] = std::min(lowpt[uz(e)], lowpt2[uz(k)]);
                        lowpt[uz(e)] = lowpt[uz(k)];
                    } else if (lowpt[uz(k)] > lowpt[uz(e)]) {
                        lowpt2[uz(e)] = std::min(lowpt2[uz(e)], lowpt[uz(k)]);
                    } else {
                        lowpt2[uz(e)] = std::min(lowpt2[uz(e)], lowpt2[uz(k)]);
                    }
                }
                ++ind[uz(v)];
            }
        }
    }

    // --- Pass 2: testing -----------------------------------------------
    void set_ref(int a, int b) { if (a >= 0) ref_[uz(a)] = b; }

    bool conflicting(const LRInterval& I, int b) const {
        return !I.empty() && I.high >= 0 && lowpt[uz(I.high)] > lowpt[uz(b)];
    }

    int lowest(const LRConflictPair& P) const {
        const bool has_l = P.left.low >= 0;
        const bool has_r = P.right.low >= 0;
        if (!has_l && !has_r) return std::numeric_limits<int>::max();
        if (!has_l) return lowpt[uz(P.right.low)];
        if (!has_r) return lowpt[uz(P.left.low)];
        return std::min(lowpt[uz(P.left.low)], lowpt[uz(P.right.low)]);
    }

    bool add_constraints(int ei, int e) {
        // e is the tree edge into the current vertex; a root would have
        // height 0 and every lowpt is >= 0, so this guard is unreachable.
        if (e < 0) return true;
        LRConflictPair P;
        // Merge the return edges of e_i into P.right.
        while (!S.empty()) {
            LRConflictPair Q = S.back();
            S.pop_back();
            if (!Q.left.empty()) std::swap(Q.left, Q.right);
            if (!Q.left.empty()) return false;              // NOT PLANAR
            if (Q.right.low >= 0 && lowpt[uz(Q.right.low)] > lowpt[uz(e)]) {
                if (P.right.empty()) P.right = Q.right;     // topmost interval
                else set_ref(P.right.low, Q.right.high);
                P.right.low = Q.right.low;
            } else {
                set_ref(Q.right.low, lowpt_edge[uz(e)]);    // align
            }
            if (static_cast<int>(S.size()) - 1 == stack_bottom[uz(ei)]) break;
        }
        // Merge the conflicting return edges of e_1 .. e_{i-1} into P.left.
        while (!S.empty() &&
               (conflicting(S.back().left, ei) || conflicting(S.back().right, ei))) {
            LRConflictPair Q = S.back();
            S.pop_back();
            if (conflicting(Q.right, ei)) std::swap(Q.left, Q.right);
            if (conflicting(Q.right, ei)) return false;     // NOT PLANAR
            set_ref(P.right.low, Q.right.high);
            if (Q.right.low != -1) P.right.low = Q.right.low;
            if (P.left.empty()) P.left = Q.left;            // topmost interval
            else set_ref(P.left.low, Q.left.high);
            P.left.low = Q.left.low;
        }
        if (!(P.left.empty() && P.right.empty())) S.push_back(P);
        return true;
    }

    void remove_back_edges(int e) {
        const int u = src[uz(e)];
        // Drop the conflict pairs whose return edges all end at u.
        while (!S.empty() && lowest(S.back()) == height[uz(u)]) {
            const LRConflictPair P = S.back();
            S.pop_back();
            if (P.left.low != -1) side_[uz(P.left.low)] = -1;
        }
        if (!S.empty()) {                                   // trim one more pair
            LRConflictPair P = S.back();
            S.pop_back();
            while (P.left.high != -1 && dst[uz(P.left.high)] == u)
                P.left.high = ref_[uz(P.left.high)];
            if (P.left.high == -1 && P.left.low != -1) {
                set_ref(P.left.low, P.right.low);
                side_[uz(P.left.low)] = -1;
                P.left.low = -1;
            }
            while (P.right.high != -1 && dst[uz(P.right.high)] == u)
                P.right.high = ref_[uz(P.right.high)];
            if (P.right.high == -1 && P.right.low != -1) {
                set_ref(P.right.low, P.left.low);
                side_[uz(P.right.low)] = -1;
                P.right.low = -1;
            }
            S.push_back(P);
        }
        // The side of e is the side of a highest return edge still open.
        if (lowpt[uz(e)] < height[uz(u)]) {
            const int hl = S.empty() ? -1 : S.back().left.high;
            const int hr = S.empty() ? -1 : S.back().right.high;
            if (hl != -1 && (hr == -1 || lowpt[uz(hl)] > lowpt[uz(hr)])) ref_[uz(e)] = hl;
            else ref_[uz(e)] = hr;
        }
    }

    bool dfs_testing(int root) {
        std::vector<int> dfs_stack;
        dfs_stack.push_back(root);
        while (!dfs_stack.empty()) {
            const int v = dfs_stack.back();
            dfs_stack.pop_back();
            const int e = parent_edge[uz(v)];
            bool skip_final = false;
            while (ind[uz(v)] < ordered_adjs[uz(v)].size()) {
                const int ei = ordered_adjs[uz(v)][ind[uz(v)]];
                const int w = dst[uz(ei)];
                if (!skip_init[uz(ei)]) {
                    stack_bottom[uz(ei)] = static_cast<int>(S.size()) - 1;
                    if (ei == parent_edge[uz(w)]) {         // tree edge
                        dfs_stack.push_back(v);
                        dfs_stack.push_back(w);
                        skip_init[uz(ei)] = 1;
                        skip_final = true;
                        break;                              // deliberately no ++ind
                    }
                    lowpt_edge[uz(ei)] = ei;                // back edge
                    LRConflictPair fresh;
                    fresh.right = LRInterval{ei, ei};
                    S.push_back(fresh);
                }
                if (lowpt[uz(ei)] < height[uz(v)]) {        // integrate return edges
                    if (ind[uz(v)] == 0) {                  // e_i is v's first edge
                        if (e >= 0) lowpt_edge[uz(e)] = lowpt_edge[uz(ei)];
                    } else if (!add_constraints(ei, e)) {
                        return false;
                    }
                }
                ++ind[uz(v)];
            }
            if (!skip_final && e != -1) remove_back_edges(e);
        }
        return true;
    }

    // --- Pass 3: sign resolution and embedding --------------------------
    // Iterative equivalent of the reference's recursive sign(): follow the
    // ref_ chain to its end, clearing links on the way (so the walk also
    // memoises and cannot loop), then multiply the sides back down the chain.
    int resolve_sign(int e) {
        chain_.clear();
        int f = e;
        while (ref_[uz(f)] != -1) {
            chain_.push_back(f);
            const int next = ref_[uz(f)];
            ref_[uz(f)] = -1;
            f = next;
        }
        while (!chain_.empty()) {
            const int g = chain_.back();
            chain_.pop_back();
            side_[uz(g)] *= side_[uz(f)];
            f = g;
        }
        return side_[uz(e)];
    }

    void half_edge_cw(int v, int w, int r) {                // insert w cw of r
        std::unordered_map<int, int>& cw = emb_cw[uz(v)];
        std::unordered_map<int, int>& ccw = emb_ccw[uz(v)];
        if (r == -1) {
            cw[w] = w;
            ccw[w] = w;
            emb_first[uz(v)] = w;
            return;
        }
        const int cw_r = cw[r];
        cw[r] = w;
        cw[w] = cw_r;
        ccw[cw_r] = w;
        ccw[w] = r;
    }

    void half_edge_ccw(int v, int w, int r) {               // insert w ccw of r
        if (r == -1) { half_edge_cw(v, w, -1); return; }
        const int ccw_r = emb_ccw[uz(v)][r];
        half_edge_cw(v, w, ccw_r);
        if (r == emb_first[uz(v)]) emb_first[uz(v)] = w;
    }

    void half_edge_first(int v, int w) { half_edge_ccw(v, w, emb_first[uz(v)]); }

    void dfs_embedding(int root) {
        std::vector<int> dfs_stack;
        dfs_stack.push_back(root);
        while (!dfs_stack.empty()) {
            const int v = dfs_stack.back();
            dfs_stack.pop_back();
            while (ind[uz(v)] < ordered_adjs[uz(v)].size()) {
                const int ei = ordered_adjs[uz(v)][ind[uz(v)]];
                const int w = dst[uz(ei)];
                ++ind[uz(v)];                               // note: before the break
                if (ei == parent_edge[uz(w)]) {             // tree edge
                    half_edge_first(w, v);
                    left_ref[uz(v)] = w;
                    right_ref[uz(v)] = w;
                    dfs_stack.push_back(v);
                    dfs_stack.push_back(w);
                    break;
                }
                if (side_[uz(ei)] == 1) {                   // back edge
                    half_edge_cw(w, v, right_ref[uz(w)]);
                } else {
                    half_edge_ccw(w, v, left_ref[uz(w)]);
                    left_ref[uz(w)] = v;
                }
            }
        }
    }

    void sort_by_nesting_depth() {
        for (int v = 0; v < n; ++v) {
            ordered_adjs[uz(v)] = out_edges[uz(v)];
            // stable: ties keep orientation order, which is what fixes the
            // embedding's tie-breaking; an unstable sort would silently
            // change it.
            std::stable_sort(ordered_adjs[uz(v)].begin(), ordered_adjs[uz(v)].end(),
                             [this](int a, int b) {
                                 return nesting_depth[uz(a)] < nesting_depth[uz(b)];
                             });
        }
    }

    // Runs passes 1 and 2 (and pass 3 when want_embedding); returns planarity.
    bool run(const Graph& G, bool want_embedding) {
        n = G.n_vertices();
        E = canonical_simple_edges(G);
        m = static_cast<int>(E.size());
        // Euler's bound, exact for simple graphs with at least 3 vertices.
        if (n > 2 && m > 3 * n - 6) return false;

        adj.clear();
        adj.resize(uz(n));
        for (int k = 0; k < m; ++k) {
            const Edge& ed = E[uz(k)];
            adj[uz(ed.from)].push_back(AdjEntry{ed.to, k, 2 * k});
            adj[uz(ed.to)].push_back(AdjEntry{ed.from, k, 2 * k + 1});
        }

        height.assign(uz(n), -1);
        parent_edge.assign(uz(n), -1);
        src.assign(uz(m), -1);
        dst.assign(uz(m), -1);
        oriented.assign(uz(m), 0);
        lowpt.assign(uz(m), 0);
        lowpt2.assign(uz(m), 0);
        nesting_depth.assign(uz(m), 0);
        out_edges.clear();
        out_edges.resize(uz(n));
        ordered_adjs.clear();
        ordered_adjs.resize(uz(n));
        roots.clear();
        ind.assign(uz(n), 0);
        skip_init.assign(uz(2 * m), 0);
        for (int v = 0; v < n; ++v) {
            if (height[uz(v)] == -1) {
                height[uz(v)] = 0;
                roots.push_back(v);
                dfs_orientation(v);
            }
        }

        sort_by_nesting_depth();

        ref_.assign(uz(m), -1);
        side_.assign(uz(m), 1);
        lowpt_edge.assign(uz(m), -1);
        stack_bottom.assign(uz(m), -1);
        S.clear();
        ind.assign(uz(n), 0);
        skip_init.assign(uz(m), 0);
        for (std::size_t i = 0; i < roots.size(); ++i)
            if (!dfs_testing(roots[i])) return false;

        if (!want_embedding) return true;

        for (int k = 0; k < m; ++k)
            nesting_depth[uz(k)] = resolve_sign(k) * nesting_depth[uz(k)];
        sort_by_nesting_depth();

        emb_cw.clear();
        emb_cw.resize(uz(n));
        emb_ccw.clear();
        emb_ccw.resize(uz(n));
        emb_first.assign(uz(n), -1);
        left_ref.assign(uz(n), -1);
        right_ref.assign(uz(n), -1);
        for (int v = 0; v < n; ++v) {
            int prev = -1;
            for (std::size_t i = 0; i < ordered_adjs[uz(v)].size(); ++i) {
                const int w = dst[uz(ordered_adjs[uz(v)][i])];
                half_edge_cw(v, w, prev);
                prev = w;
            }
        }
        ind.assign(uz(n), 0);
        for (std::size_t i = 0; i < roots.size(); ++i) dfs_embedding(roots[i]);
        return true;
    }

    // Walks each vertex's circular list once, starting from emb_first.
    std::vector<std::vector<int>> embedding() {
        std::vector<std::vector<int>> out(uz(n));
        for (int v = 0; v < n; ++v) {
            const int first = emb_first[uz(v)];
            if (first == -1) continue;
            const std::size_t cap = emb_cw[uz(v)].size();
            int cur = first;
            for (std::size_t step = 0; step < cap; ++step) {
                out[uz(v)].push_back(cur);
                cur = emb_cw[uz(v)][cur];
                if (cur == first) break;
            }
        }
        return out;
    }
};

} // namespace

bool is_planar(const Graph& G) {
    LRPlanarity lr;
    return lr.run(G, false);
}

Result<std::vector<std::vector<int>>> planar_embedding(const Graph& G) {
    LRPlanarity lr;
    if (!lr.run(G, true)) {
        return std::unexpected(
            Error{DomainError{"planar_embedding", "graph is not planar"}});
    }
    return lr.embedding();
}

std::vector<Edge> kuratowski_subgraph(const Graph& G) {
    const std::vector<Edge> E = canonical_simple_edges(G);
    const int n = G.n_vertices();
    std::vector<char> present(E.size(), 1);
    auto rebuild = [&]() {
        Graph H(n, false);
        for (std::size_t i = 0; i < E.size(); ++i)
            if (present[i]) H.add_edge(E[i].from, E[i].to, E[i].weight);
        return H;
    };
    if (is_planar(rebuild())) return {};                 // planar: no certificate
    // Greedy edge minimisation. Every edge put back was, at that moment,
    // required for non-planarity; since the surviving set only shrinks after
    // that and subgraphs of planar graphs are planar, the final set is
    // edge-minimal non-planar -- i.e. exactly a K5 or K3,3 subdivision.
    for (std::size_t i = 0; i < E.size(); ++i) {
        present[i] = 0;
        if (is_planar(rebuild())) present[i] = 1;        // essential: put it back
    }
    std::vector<Edge> out;
    for (std::size_t i = 0; i < E.size(); ++i)
        if (present[i]) out.push_back(E[i]);
    return out;
}

// ---- K-core decomposition (Batagelj-Zaversnik degree peeling) ----

std::vector<int> k_core_decomposition(const Graph& G) {
    Graph U = to_undirected(G);
    int n = U.n_vertices();
    std::vector<int> core(n, 0);
    if (n == 0) return core;

    std::vector<int> deg(n);
    int max_deg = 0;
    for (int v = 0; v < n; ++v) {
        deg[v] = static_cast<int>(U.neighbors(v).size());
        max_deg = std::max(max_deg, deg[v]);
    }

    std::vector<int> vert(n), pos(n);
    std::vector<int> bin(max_deg + 2, 0);
    for (int v = 0; v < n; ++v) ++bin[deg[v]];
    int start = 0;
    for (int d = 0; d <= max_deg; ++d) {
        int num = bin[d];
        bin[d] = start;
        start += num;
    }
    for (int v = 0; v < n; ++v) {
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        ++bin[deg[v]];
    }
    for (int d = max_deg; d >= 1; --d) bin[d] = bin[d - 1];
    bin[0] = 0;

    for (int k = 0; k <= max_deg; ++k) {
        for (int i = bin[k]; i < ((k < max_deg) ? bin[k + 1] : n); ++i) {
            int v = vert[i];
            core[v] = k;
            for (auto& [u, w] : U.neighbors(v)) {
                if (deg[u] > k) {
                    int du = deg[u];
                    int pu = pos[u];
                    int pw = vert[bin[du]];
                    if (u != pw) {
                        pos[u] = bin[du];
                        pos[pw] = pu;
                        vert[pu] = pw;
                        vert[bin[du]] = u;
                    }
                    ++bin[du];
                    --deg[u];
                }
            }
        }
    }
    return core;
}

Graph k_core_subgraph(const Graph& G, int k) {
    auto core = k_core_decomposition(G);
    int n = G.n_vertices();
    std::vector<int> new_id(n, -1);
    int m = 0;
    for (int v = 0; v < n; ++v)
        if (core[v] >= k) new_id[v] = m++;

    Graph H(m, false);
    if (m == 0) return H;

    Graph U = to_undirected(G);
    for (int v = 0; v < n; ++v) {
        if (new_id[v] < 0) continue;
        for (auto& [u, w] : U.neighbors(v))
            if (new_id[u] >= 0 && new_id[v] < new_id[u])
                H.add_edge(new_id[v], new_id[u], w);
    }
    return H;
}

// ---- Isomorphism ----

namespace {

bool has_edge(const Graph& G, int u, int v) {
    for (auto& [nbr, w] : G.neighbors(u))
        if (nbr == v) return true;
    return false;
}

// {in-degree, out-degree} per vertex; add_edge mirrors both directions for
// undirected graphs, so in == out == neighbors().size() there, and this
// reduces to the plain degree.
std::vector<std::pair<int,int>> in_out_degrees(const Graph& G) {
    int n = G.n_vertices();
    std::vector<std::pair<int,int>> deg(n, {0, 0});
    for (int u = 0; u < n; ++u) {
        deg[u].second = static_cast<int>(G.neighbors(u).size());
        for (auto& [v, w] : G.neighbors(u)) ++deg[v].first;
    }
    return deg;
}

bool isomorphic_backtrack(const Graph& G1, const Graph& G2,
                           const std::vector<std::pair<int,int>>& deg1,
                           const std::vector<std::pair<int,int>>& deg2,
                           std::vector<int>& mapping, std::vector<bool>& used, int u) {
    int n = G1.n_vertices();
    if (u == n) return true;
    for (int v = 0; v < n; ++v) {
        if (used[v] || deg1[u] != deg2[v]) continue;
        bool consistent = true;
        for (int up = 0; up < u && consistent; ++up) {
            int vp = mapping[up];
            if (has_edge(G1, u, up) != has_edge(G2, v, vp)) consistent = false;
            if (G1.is_directed() && has_edge(G1, up, u) != has_edge(G2, vp, v)) consistent = false;
        }
        if (!consistent) continue;
        mapping[u] = v; used[v] = true;
        if (isomorphic_backtrack(G1, G2, deg1, deg2, mapping, used, u + 1)) return true;
        used[v] = false;
    }
    return false;
}

} // namespace

bool is_isomorphic(const Graph& G1, const Graph& G2) {
    if (G1.n_vertices() != G2.n_vertices() ||
        G1.n_edges() != G2.n_edges() ||
        G1.is_directed() != G2.is_directed())
        return false;
    int n = G1.n_vertices();
    if (n == 0) return true;

    auto deg1 = in_out_degrees(G1);
    auto deg2 = in_out_degrees(G2);
    auto sorted1 = deg1, sorted2 = deg2;
    std::sort(sorted1.begin(), sorted1.end());
    std::sort(sorted2.begin(), sorted2.end());
    if (sorted1 != sorted2) return false;

    std::vector<int> mapping(n, -1);
    std::vector<bool> used(n, false);
    return isomorphic_backtrack(G1, G2, deg1, deg2, mapping, used, 0);
}

// ---- Centrality ----

std::vector<double> pagerank(const Graph& G, double alpha, int max_iter) {
    int n = G.n_vertices();
    std::vector<double> rank(n, 1.0 / n);
    std::vector<int> outdeg(n, 0);
    for (int u = 0; u < n; ++u) outdeg[u] = static_cast<int>(G.neighbors(u).size());
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> new_rank(n, (1.0 - alpha) / n);
        for (int u = 0; u < n; ++u)
            if (outdeg[u] > 0)
                for (auto& [v, w] : G.neighbors(u))
                    new_rank[v] += alpha * rank[u] / outdeg[u];
        double diff = 0.0;
        for (int i = 0; i < n; ++i) diff += std::abs(new_rank[i] - rank[i]);
        rank = new_rank;
        if (diff < 1e-10) break;
    }
    return rank;
}

std::vector<double> betweenness_centrality(const Graph& G) {
    int n = G.n_vertices();
    std::vector<double> bc(n, 0.0);
    for (int s = 0; s < n; ++s) {
        std::vector<double> sigma(n, 0.0), delta(n, 0.0);
        std::vector<double> dist(n, -1);
        std::vector<std::vector<int>> pred(n);
        std::queue<int> q;
        std::stack<int> stk;
        sigma[s] = 1.0; dist[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            stk.push(v);
            for (auto& [u, w] : G.neighbors(v)) {
                if (dist[u] < 0) { dist[u] = dist[v] + 1; q.push(u); }
                if (dist[u] == dist[v] + 1) { sigma[u] += sigma[v]; pred[u].push_back(v); }
            }
        }
        while (!stk.empty()) {
            int w = stk.top(); stk.pop();
            for (int v : pred[w]) delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            if (w != s) bc[w] += delta[w];
        }
    }
    double scale = 1.0 / ((n - 1) * (n - 2));
    for (auto& b : bc) b *= scale;
    return bc;
}

std::vector<double> closeness_centrality(const Graph& G) {
    int n = G.n_vertices();
    std::vector<double> cc(n, 0.0);
    auto d = floyd_warshall(G);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0; int count = 0;
        for (int j = 0; j < n; ++j)
            if (i != j && d[i][j] < INF) { sum += d[i][j]; ++count; }
        if (sum > 0.0) cc[i] = static_cast<double>(count) / sum;
    }
    return cc;
}

std::vector<double> degree_centrality(const Graph& G) {
    int n = G.n_vertices();
    std::vector<double> dc(n);
    double norm = (n > 1) ? 1.0 / (n - 1) : 1.0;
    for (int v = 0; v < n; ++v)
        dc[v] = G.neighbors(v).size() * norm;
    return dc;
}

std::vector<double> eigenvector_centrality(const Graph& G, int max_iter, double tol) {
    int n = G.n_vertices();
    if (n == 0) return {};
    std::vector<double> x(n);
    for (int v = 0; v < n; ++v) x[v] = 1.0 + static_cast<double>(G.neighbors(v).size());
    double init_norm = 0.0;
    for (double v : x) init_norm += v * v;
    init_norm = std::sqrt(init_norm);
    for (auto& v : x) v /= init_norm;
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> y(n, 0.0);
        for (int u = 0; u < n; ++u)
            for (auto& [v, w] : G.neighbors(u))
                y[v] += x[u];
        double norm = 0.0;
        for (double v : y) norm += v * v;
        norm = std::sqrt(norm);
        if (norm < 1e-15) return y;
        double diff = 0.0;
        for (int i = 0; i < n; ++i) {
            y[i] /= norm;
            diff += std::abs(y[i] - x[i]);
        }
        x = y;
        if (diff < tol) break;
    }
    return x;
}

std::vector<double> katz_centrality(const Graph& G, double alpha, double beta,
                                    int max_iter, double tol) {
    int n = G.n_vertices();
    if (n == 0) return {};
    std::vector<double> x(n, beta);
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> y(n, beta);
        for (int u = 0; u < n; ++u)
            for (auto& [v, w] : G.neighbors(u))
                y[v] += alpha * x[u];
        double diff = 0.0;
        for (int i = 0; i < n; ++i) diff += std::abs(y[i] - x[i]);
        x = y;
        if (diff < tol) break;
    }
    return x;
}

std::vector<std::vector<double>> laplacian(const Graph& G) {
    int n = G.n_vertices();
    auto A = adjacency_matrix(G);
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        double deg = 0.0;
        for (int j = 0; j < n; ++j) deg += A[i][j];
        L[i][i] = deg;
        for (int j = 0; j < n; ++j)
            if (i != j) L[i][j] = -A[i][j];
    }
    return L;
}

std::vector<std::vector<double>> normalised_laplacian(const Graph& G) {
    int n = G.n_vertices();
    auto A = adjacency_matrix(G);
    std::vector<double> deg(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) deg[i] += A[i][j];
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        if (deg[i] < 1e-15) continue;  // isolated vertex: row/col remain zero
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                L[i][j] = 1.0;
            } else if (deg[j] >= 1e-15 && A[i][j] != 0.0) {
                L[i][j] = -1.0 / std::sqrt(deg[i] * deg[j]);
            }
        }
    }
    return L;
}

double algebraic_connectivity(const Graph& G) {
    int n = G.n_vertices();
    if (n <= 1) return 0.0;
    auto evals = symmetric_eigenvalues(laplacian(G));
    if (evals.size() < 2) return 0.0;
    return evals[1];  // second-smallest; smallest is always 0
}

// ---- Max flow (Dinic's) ----

Result<double> max_flow(const Graph& G, int source, int sink) {
    return dinic_max_flow_residual(G, source, sink).flow;
}

Result<MinCutResult> min_cut(const Graph& G, int source, int sink) {
    auto dinic = dinic_max_flow_residual(G, source, sink);
    int n = G.n_vertices();
    std::vector<bool> reachable(n, false);
    std::queue<int> q;
    q.push(source);
    reachable[source] = true;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (auto& [u, cap] : dinic.residual[v])
            if (cap > 1e-12 && !reachable[u]) {
                reachable[u] = true;
                q.push(u);
            }
    }
    MinCutResult result;
    result.value = dinic.flow;
    for (auto& e : G.edges()) {
        if (reachable[e.from] && !reachable[e.to])
            result.cut_edges.push_back(e);
    }
    return result;
}

// ---- Bipartite matching (Hopcroft-Karp) ----

Result<int> bipartite_match(const Graph& G, int left_size) {
    if (!is_bipartite(G))
        return std::unexpected(Error{DomainError{"bipartite_match", "not bipartite"}});
    int n = G.n_vertices();
    std::vector<int> match_l(left_size, -1), match_r(n - left_size, -1);
    int total = 0;
    // Augmenting path
    std::function<bool(int, std::vector<bool>&)> dfs_aug = [&](int u, std::vector<bool>& visited) -> bool {
        for (auto& [v, w] : G.neighbors(u)) {
            int rv = v - left_size;
            if (rv < 0 || visited[rv]) continue;
            visited[rv] = true;
            if (match_r[rv] == -1 || dfs_aug(match_r[rv], visited)) {
                match_l[u] = rv; match_r[rv] = u; return true;
            }
        }
        return false;
    };
    for (int u = 0; u < left_size; ++u) {
        std::vector<bool> visited(n - left_size, false);
        if (dfs_aug(u, visited)) ++total;
    }
    return total;
}

// ---- General maximum cardinality matching (Edmonds blossom) ----

std::vector<std::pair<int, int>> maximum_matching(const Graph& G) {
    const int n = G.n_vertices();
    if (n == 0) return {};

    std::vector<std::vector<int>> adj(static_cast<size_t>(n));
    for (int u = 0; u < n; ++u) {
        for (const auto& [v, w] : G.neighbors(u)) {
            if (v < 0 || v >= n || v == u) continue;
            adj[static_cast<size_t>(u)].push_back(v);
        }
    }
    for (auto& row : adj) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    std::vector<int> mate(static_cast<size_t>(n), -1);
    std::vector<int> parent(static_cast<size_t>(n), -1);
    std::vector<int> base(static_cast<size_t>(n));
    std::vector<int> q(static_cast<size_t>(n));
    std::vector<char> used(static_cast<size_t>(n), 0);
    std::vector<char> blossom(static_cast<size_t>(n), 0);

    auto lca = [&](int a, int b) {
        std::vector<char> marked(static_cast<size_t>(n), 0);
        while (true) {
            a = base[static_cast<size_t>(a)];
            marked[static_cast<size_t>(a)] = 1;
            if (mate[static_cast<size_t>(a)] == -1) break;
            a = parent[static_cast<size_t>(mate[static_cast<size_t>(a)])];
        }
        while (true) {
            b = base[static_cast<size_t>(b)];
            if (marked[static_cast<size_t>(b)]) return b;
            b = parent[static_cast<size_t>(mate[static_cast<size_t>(b)])];
        }
    };

    auto mark_path = [&](int v, int b, int children) {
        while (base[static_cast<size_t>(v)] != b) {
            blossom[static_cast<size_t>(base[static_cast<size_t>(v)])] = 1;
            blossom[static_cast<size_t>(base[static_cast<size_t>(mate[static_cast<size_t>(v)])])] = 1;
            parent[static_cast<size_t>(v)] = children;
            children = mate[static_cast<size_t>(v)];
            v = parent[static_cast<size_t>(mate[static_cast<size_t>(v)])];
        }
    };

    auto find_path = [&](int root) -> int {
        used.assign(static_cast<size_t>(n), 0);
        parent.assign(static_cast<size_t>(n), -1);
        for (int i = 0; i < n; ++i) base[static_cast<size_t>(i)] = i;
        used[static_cast<size_t>(root)] = 1;
        int qh = 0, qt = 0;
        q[static_cast<size_t>(qt++)] = root;
        while (qh < qt) {
            const int v = q[static_cast<size_t>(qh++)];
            for (int to : adj[static_cast<size_t>(v)]) {
                if (base[static_cast<size_t>(v)] == base[static_cast<size_t>(to)] ||
                    mate[static_cast<size_t>(v)] == to)
                    continue;
                if (to == root ||
                    (mate[static_cast<size_t>(to)] != -1 &&
                     parent[static_cast<size_t>(mate[static_cast<size_t>(to)])] != -1)) {
                    const int cur = lca(v, to);
                    blossom.assign(static_cast<size_t>(n), 0);
                    mark_path(v, cur, to);
                    mark_path(to, cur, v);
                    for (int i = 0; i < n; ++i) {
                        if (blossom[static_cast<size_t>(base[static_cast<size_t>(i)])]) {
                            base[static_cast<size_t>(i)] = cur;
                            if (!used[static_cast<size_t>(i)]) {
                                used[static_cast<size_t>(i)] = 1;
                                q[static_cast<size_t>(qt++)] = i;
                            }
                        }
                    }
                } else if (parent[static_cast<size_t>(to)] == -1) {
                    parent[static_cast<size_t>(to)] = v;
                    if (mate[static_cast<size_t>(to)] == -1) return to;
                    to = mate[static_cast<size_t>(to)];
                    used[static_cast<size_t>(to)] = 1;
                    q[static_cast<size_t>(qt++)] = to;
                }
            }
        }
        return -1;
    };

    for (int i = 0; i < n; ++i) {
        if (mate[static_cast<size_t>(i)] == -1) {
            int v = find_path(i);
            while (v != -1) {
                const int pv = parent[static_cast<size_t>(v)];
                const int ppv = mate[static_cast<size_t>(pv)];
                mate[static_cast<size_t>(v)] = pv;
                mate[static_cast<size_t>(pv)] = v;
                v = ppv;
            }
        }
    }

    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < n; ++i) {
        if (mate[static_cast<size_t>(i)] > i)
            edges.emplace_back(i, mate[static_cast<size_t>(i)]);
    }
    return edges;
}

// ---- General maximum WEIGHT matching (Edmonds primal-dual blossom) ----

namespace {

struct WeightedMatchingResult {
    std::vector<std::pair<int, int>> edges;
    double weight = 0.0;
};

// Relative tolerance base: an edge counts as tight, and a blossom dual as
// zero, within kBlossomRelTol * max(1, max|weight|). Integer-valued weights
// keep every dual exactly integral in the doubled formulation below, so the
// tolerance only matters for genuinely real-valued inputs.
constexpr double kBlossomRelTol = 1e-9;

// Edmonds' primal-dual maximum-weight matching in Galil's O(V^3) formulation.
//
// The doubled dual convention is what keeps integer weights exact: dualvar[v]
// holds 2*u_v for a vertex and z_b for a blossom, so an edge's slack is
//     slack(k) = dualvar[u] + dualvar[v] - 2*w(k)
// and every dual adjustment moves by an integral amount.
//
// Endpoint numbering: undirected edge k owns the two endpoint ids 2k and
// 2k+1, with endpoint[2k] = eu[k] and endpoint[2k+1] = ev[k]. Hence p ^ 1 is
// always the other end of the same edge and p / 2 is the edge index, which is
// what makes the alternating-tree bookkeeping below a handful of xors.
//
// Blossom ids in [0, nvertex) are trivial blossoms (single vertices); ids in
// [nvertex, 2*nvertex) are shrunk blossoms handed out from unusedblossoms.
struct BlossomSolver {
    int nvertex = 0;
    int nedge = 0;
    bool maxcardinality = false;
    double tol = 0.0;

    std::vector<int> eu;                     // nedge; eu[k] < ev[k]
    std::vector<int> ev;                     // nedge
    std::vector<double> ew;                  // nedge
    std::vector<int> endpoint;               // 2*nedge
    std::vector<std::vector<int>> neighbend; // nvertex; remote endpoint ids

    std::vector<int> mate;                   // nvertex; endpoint id, -1 = free
    std::vector<int> label;                  // 2*nvertex; 0 free, 1 S, 2 T, 5 mark
    std::vector<int> labelend;               // 2*nvertex
    std::vector<int> inblossom;              // nvertex; top-level blossom of v
    std::vector<int> blossomparent;          // 2*nvertex
    std::vector<int> blossombase;            // 2*nvertex; -1 = unused slot
    std::vector<int> bestedge;               // 2*nvertex; least-slack edge out
    std::vector<std::vector<int>> blossomchilds;    // 2*nvertex; cyclic order
    std::vector<std::vector<int>> blossomendps;     // 2*nvertex
    std::vector<std::vector<int>> blossombestedges; // 2*nvertex
    // Distinguishes "no list computed yet" from "computed and empty", which
    // the reference expresses as None vs []; both are meaningful states.
    std::vector<char> bestedges_valid;       // 2*nvertex
    std::vector<int> unusedblossoms;
    std::vector<double> dualvar;             // 2*nvertex; doubled duals
    std::vector<char> allowedge;             // nedge; slack is (nearly) zero
    std::vector<int> queue_;                 // S-vertices pending scan (LIFO)

    double slack(int k) const {
        return dualvar[uz(eu[uz(k)])] + dualvar[uz(ev[uz(k)])] - 2.0 * ew[uz(k)];
    }

    static int wrap(int j, int len) { return ((j % len) + len) % len; }

    static int index_of(const std::vector<int>& v, int x) {
        for (std::size_t i = 0; i < v.size(); ++i)
            if (v[i] == x) return static_cast<int>(i);
        return -1;
    }

    // Every trivial (single-vertex) blossom inside b, in cyclic child order.
    // Recursion depth is the blossom nesting depth, at most nvertex / 2.
    void collect_leaves(int b, std::vector<int>& out) const {
        if (b < nvertex) { out.push_back(b); return; }
        for (std::size_t i = 0; i < blossomchilds[uz(b)].size(); ++i) {
            const int t = blossomchilds[uz(b)][i];
            if (t < nvertex) out.push_back(t);
            else collect_leaves(t, out);
        }
    }

    // Label w's top-level blossom S (t == 1) or T (t == 2), arriving through
    // endpoint p. A T label immediately forces the matched partner to S,
    // which is why this recurses exactly one level.
    void assign_label(int w, int t, int p) {
        const int b = inblossom[uz(w)];
        label[uz(w)] = t;
        label[uz(b)] = t;
        labelend[uz(w)] = p;
        labelend[uz(b)] = p;
        bestedge[uz(w)] = -1;
        bestedge[uz(b)] = -1;
        if (t == 1) {
            std::vector<int> leaves;
            collect_leaves(b, leaves);
            for (std::size_t i = 0; i < leaves.size(); ++i) queue_.push_back(leaves[i]);
        } else if (t == 2) {
            const int mb = mate[uz(blossombase[uz(b)])];
            if (mb >= 0) assign_label(endpoint[uz(mb)], 1, mb ^ 1);
        }
    }

    // Walk the two alternating paths up towards their roots one step at a
    // time, alternating sides and leaving breadcrumbs (label 5 has bit 2 set).
    // Returns the base of the blossom the two paths close, or -1 when they
    // reach different roots -- which means an augmenting path was found.
    int scan_blossom(int v, int w) {
        std::vector<int> path;
        int base = -1;
        while (v != -1 || w != -1) {
            int b = inblossom[uz(v)];
            if ((label[uz(b)] & 4) != 0) { base = blossombase[uz(b)]; break; }
            path.push_back(b);
            label[uz(b)] = 5;
            if (labelend[uz(b)] == -1) {
                v = -1;
            } else {
                v = endpoint[uz(labelend[uz(b)])];
                b = inblossom[uz(v)];               // a T-blossom
                v = endpoint[uz(labelend[uz(b)])];
            }
            if (w != -1) std::swap(v, w);
        }
        for (std::size_t i = 0; i < path.size(); ++i) label[uz(path[i])] = 1;
        return base;
    }

    // Shrink the odd cycle through edge k into a new blossom based at `base`.
    void add_blossom(int base, int k) {
        int v = eu[uz(k)];
        int w = ev[uz(k)];
        const int bb = inblossom[uz(base)];
        int bv = inblossom[uz(v)];
        int bw = inblossom[uz(w)];
        const int b = unusedblossoms.back();
        unusedblossoms.pop_back();
        blossombase[uz(b)] = base;
        blossomparent[uz(b)] = -1;
        blossomparent[uz(bb)] = b;
        std::vector<int> path;
        std::vector<int> endps;
        while (bv != bb) {                          // trace v back to the base
            blossomparent[uz(bv)] = b;
            path.push_back(bv);
            endps.push_back(labelend[uz(bv)]);
            v = endpoint[uz(labelend[uz(bv)])];
            bv = inblossom[uz(v)];
        }
        path.push_back(bb);
        std::reverse(path.begin(), path.end());
        std::reverse(endps.begin(), endps.end());
        endps.push_back(2 * k);
        while (bw != bb) {                          // trace w back to the base
            blossomparent[uz(bw)] = b;
            path.push_back(bw);
            endps.push_back(labelend[uz(bw)] ^ 1);
            w = endpoint[uz(labelend[uz(bw)])];
            bw = inblossom[uz(w)];
        }
        label[uz(b)] = 1;
        labelend[uz(b)] = labelend[uz(bb)];
        dualvar[uz(b)] = 0.0;
        blossomchilds[uz(b)] = path;
        blossomendps[uz(b)] = endps;
        {
            std::vector<int> leaves;
            collect_leaves(b, leaves);
            for (std::size_t i = 0; i < leaves.size(); ++i) {
                const int x = leaves[i];
                // A T-vertex swallowed by an S-blossom becomes an S-vertex.
                if (label[uz(inblossom[uz(x)])] == 2) queue_.push_back(x);
                inblossom[uz(x)] = b;
            }
        }
        // Least-slack edge from b to each neighbouring S-blossom. Iterating
        // bestedgeto in ascending blossom id keeps the collected list, and so
        // every later tie-break, a deterministic function of the input.
        std::vector<int> bestedgeto(uz(2 * nvertex), -1);
        for (std::size_t pi = 0; pi < path.size(); ++pi) {
            const int sub = path[pi];
            std::vector<std::vector<int>> nblists;
            if (!bestedges_valid[uz(sub)]) {
                std::vector<int> leaves;
                collect_leaves(sub, leaves);
                for (std::size_t li = 0; li < leaves.size(); ++li) {
                    const std::vector<int>& nb = neighbend[uz(leaves[li])];
                    std::vector<int> ks;
                    ks.reserve(nb.size());
                    for (std::size_t ni = 0; ni < nb.size(); ++ni) ks.push_back(nb[ni] / 2);
                    nblists.push_back(ks);
                }
            } else {
                nblists.push_back(blossombestedges[uz(sub)]);
            }
            for (std::size_t li = 0; li < nblists.size(); ++li) {
                const std::vector<int>& nblist = nblists[li];
                for (std::size_t ki = 0; ki < nblist.size(); ++ki) {
                    const int kk = nblist[ki];
                    int i = eu[uz(kk)];
                    int j = ev[uz(kk)];
                    if (inblossom[uz(j)] == b) std::swap(i, j);
                    const int bj = inblossom[uz(j)];
                    if (bj != b && label[uz(bj)] == 1 &&
                        (bestedgeto[uz(bj)] == -1 || slack(kk) < slack(bestedgeto[uz(bj)]))) {
                        bestedgeto[uz(bj)] = kk;
                    }
                }
            }
            blossombestedges[uz(sub)].clear();
            bestedges_valid[uz(sub)] = 0;
            bestedge[uz(sub)] = -1;
        }
        blossombestedges[uz(b)].clear();
        for (int t = 0; t < 2 * nvertex; ++t)
            if (bestedgeto[uz(t)] != -1) blossombestedges[uz(b)].push_back(bestedgeto[uz(t)]);
        bestedges_valid[uz(b)] = 1;
        bestedge[uz(b)] = -1;
        for (std::size_t i = 0; i < blossombestedges[uz(b)].size(); ++i) {
            const int kk = blossombestedges[uz(b)][i];
            if (bestedge[uz(b)] == -1 || slack(kk) < slack(bestedge[uz(b)])) bestedge[uz(b)] = kk;
        }
    }

    // Undo a blossom, relabelling its children along the alternating path
    // that entered it. endstage == true means the stage is over and only
    // zero-dual blossoms are being expanded, so no relabelling is needed.
    void expand_blossom(int b, bool endstage) {
        for (std::size_t i = 0; i < blossomchilds[uz(b)].size(); ++i) {
            const int s = blossomchilds[uz(b)][i];
            blossomparent[uz(s)] = -1;
            if (s < nvertex) {
                inblossom[uz(s)] = s;
            } else if (endstage && std::abs(dualvar[uz(s)]) <= tol) {
                expand_blossom(s, endstage);
            } else {
                std::vector<int> leaves;
                collect_leaves(s, leaves);
                for (std::size_t li = 0; li < leaves.size(); ++li)
                    inblossom[uz(leaves[li])] = s;
            }
        }
        if (!endstage && label[uz(b)] == 2) {
            const int len = static_cast<int>(blossomchilds[uz(b)].size());
            const int entrychild = inblossom[uz(endpoint[uz(labelend[uz(b)] ^ 1)])];
            int j = index_of(blossomchilds[uz(b)], entrychild);
            int jstep = 0;
            int endptrick = 0;
            if ((j & 1) != 0) { j -= len; jstep = 1; endptrick = 0; }
            else { jstep = -1; endptrick = 1; }
            int p = labelend[uz(b)];
            while (j != 0) {                        // relabel along the path
                label[uz(endpoint[uz(p ^ 1)])] = 0;
                const int q1 = blossomendps[uz(b)][uz(wrap(j - endptrick, len))];
                label[uz(endpoint[uz(q1 ^ endptrick ^ 1)])] = 0;
                assign_label(endpoint[uz(p ^ 1)], 2, p);
                allowedge[uz(q1 / 2)] = 1;
                j += jstep;
                const int q2 = blossomendps[uz(b)][uz(wrap(j - endptrick, len))];
                p = q2 ^ endptrick;
                allowedge[uz(p / 2)] = 1;
                j += jstep;
            }
            int bv = blossomchilds[uz(b)][uz(wrap(j, len))];
            label[uz(endpoint[uz(p ^ 1)])] = 2;
            label[uz(bv)] = 2;
            labelend[uz(endpoint[uz(p ^ 1)])] = p;
            labelend[uz(bv)] = p;
            bestedge[uz(bv)] = -1;
            j += jstep;
            while (blossomchilds[uz(b)][uz(wrap(j, len))] != entrychild) {
                bv = blossomchilds[uz(b)][uz(wrap(j, len))];
                if (label[uz(bv)] == 1) { j += jstep; continue; }
                std::vector<int> leaves;
                collect_leaves(bv, leaves);
                int v = -1;
                for (std::size_t li = 0; li < leaves.size(); ++li) {
                    v = leaves[li];
                    if (label[uz(v)] != 0) break;
                }
                if (v != -1 && label[uz(v)] != 0) {
                    const int mb = mate[uz(blossombase[uz(bv)])];
                    label[uz(v)] = 0;
                    if (mb >= 0) label[uz(endpoint[uz(mb)])] = 0;
                    assign_label(v, 2, labelend[uz(v)]);
                }
                j += jstep;
            }
        }
        label[uz(b)] = -1;
        labelend[uz(b)] = -1;
        blossomchilds[uz(b)].clear();
        blossomendps[uz(b)].clear();
        blossombase[uz(b)] = -1;
        blossombestedges[uz(b)].clear();
        bestedges_valid[uz(b)] = 0;
        bestedge[uz(b)] = -1;
        unusedblossoms.push_back(b);
    }

    // Swap the matched/unmatched edges around blossom b so that its base
    // becomes v, recursing into every sub-blossom the path passes through.
    void augment_blossom(int b, int v) {
        int t = v;
        while (blossomparent[uz(t)] != b) t = blossomparent[uz(t)];
        if (t >= nvertex) augment_blossom(t, v);
        const int len = static_cast<int>(blossomchilds[uz(b)].size());
        const int i = index_of(blossomchilds[uz(b)], t);
        int j = i;
        int jstep = 0;
        int endptrick = 0;
        if ((i & 1) != 0) { j -= len; jstep = 1; endptrick = 0; }
        else { jstep = -1; endptrick = 1; }
        while (j != 0) {
            j += jstep;
            int tt = blossomchilds[uz(b)][uz(wrap(j, len))];
            const int p = blossomendps[uz(b)][uz(wrap(j - endptrick, len))] ^ endptrick;
            if (tt >= nvertex) augment_blossom(tt, endpoint[uz(p)]);
            j += jstep;
            tt = blossomchilds[uz(b)][uz(wrap(j, len))];
            if (tt >= nvertex) augment_blossom(tt, endpoint[uz(p ^ 1)]);
            mate[uz(endpoint[uz(p)])] = p ^ 1;
            mate[uz(endpoint[uz(p ^ 1)])] = p;
        }
        std::rotate(blossomchilds[uz(b)].begin(), blossomchilds[uz(b)].begin() + i,
                    blossomchilds[uz(b)].end());
        std::rotate(blossomendps[uz(b)].begin(), blossomendps[uz(b)].begin() + i,
                    blossomendps[uz(b)].end());
        blossombase[uz(b)] = blossombase[uz(blossomchilds[uz(b)][0])];
    }

    // Flip the alternating path found through edge k, growing the matching
    // by one edge at each of the path's two ends.
    void augment_matching(int k) {
        for (int side = 0; side < 2; ++side) {
            int s = (side == 0) ? eu[uz(k)] : ev[uz(k)];
            int p = (side == 0) ? (2 * k + 1) : (2 * k);
            while (true) {
                const int bs = inblossom[uz(s)];
                if (bs >= nvertex) augment_blossom(bs, s);
                mate[uz(s)] = p;
                if (labelend[uz(bs)] == -1) break;
                const int t = endpoint[uz(labelend[uz(bs)])];
                const int bt = inblossom[uz(t)];
                s = endpoint[uz(labelend[uz(bt)])];
                const int j = endpoint[uz(labelend[uz(bt)] ^ 1)];
                if (bt >= nvertex) augment_blossom(bt, j);
                mate[uz(j)] = labelend[uz(bt)];
                p = labelend[uz(bt)] ^ 1;
            }
        }
    }

    double min_vertex_dual() const {
        double d = dualvar[0];
        for (int v = 1; v < nvertex; ++v) d = std::min(d, dualvar[uz(v)]);
        return d;
    }

    WeightedMatchingResult solve(const Graph& G, bool maxcard) {
        WeightedMatchingResult result;
        maxcardinality = maxcard;
        nvertex = G.n_vertices();
        if (nvertex == 0) return result;
        const std::vector<Edge> E = canonical_simple_edges(G);
        nedge = static_cast<int>(E.size());
        if (nedge == 0) return result;

        eu.assign(uz(nedge), 0);
        ev.assign(uz(nedge), 0);
        ew.assign(uz(nedge), 0.0);
        double maxweight = 0.0;   // clamped at 0 so duals start dual-feasible
        double maxabsweight = 0.0;
        for (int k = 0; k < nedge; ++k) {
            eu[uz(k)] = E[uz(k)].from;
            ev[uz(k)] = E[uz(k)].to;
            ew[uz(k)] = E[uz(k)].weight;
            maxweight = std::max(maxweight, ew[uz(k)]);
            maxabsweight = std::max(maxabsweight, std::abs(ew[uz(k)]));
        }
        tol = kBlossomRelTol * std::max(1.0, maxabsweight);

        endpoint.assign(uz(2 * nedge), 0);
        neighbend.clear();
        neighbend.resize(uz(nvertex));
        for (int k = 0; k < nedge; ++k) {
            endpoint[uz(2 * k)] = eu[uz(k)];
            endpoint[uz(2 * k + 1)] = ev[uz(k)];
            neighbend[uz(eu[uz(k)])].push_back(2 * k + 1);
            neighbend[uz(ev[uz(k)])].push_back(2 * k);
        }

        mate.assign(uz(nvertex), -1);
        label.assign(uz(2 * nvertex), 0);
        labelend.assign(uz(2 * nvertex), -1);
        inblossom.assign(uz(nvertex), 0);
        for (int v = 0; v < nvertex; ++v) inblossom[uz(v)] = v;
        blossomparent.assign(uz(2 * nvertex), -1);
        blossombase.assign(uz(2 * nvertex), -1);
        for (int v = 0; v < nvertex; ++v) blossombase[uz(v)] = v;
        bestedge.assign(uz(2 * nvertex), -1);
        blossomchilds.clear();
        blossomchilds.resize(uz(2 * nvertex));
        blossomendps.clear();
        blossomendps.resize(uz(2 * nvertex));
        blossombestedges.clear();
        blossombestedges.resize(uz(2 * nvertex));
        bestedges_valid.assign(uz(2 * nvertex), 0);
        unusedblossoms.clear();
        for (int b = nvertex; b < 2 * nvertex; ++b) unusedblossoms.push_back(b);
        dualvar.assign(uz(2 * nvertex), 0.0);
        for (int v = 0; v < nvertex; ++v) dualvar[uz(v)] = maxweight;
        allowedge.assign(uz(nedge), 0);
        queue_.clear();

        // At most one augmentation per stage, and each augmentation matches
        // two more vertices, so nvertex stages always suffice.
        for (int stage = 0; stage < nvertex; ++stage) {
            std::fill(label.begin(), label.end(), 0);
            std::fill(bestedge.begin(), bestedge.end(), -1);
            for (int b = nvertex; b < 2 * nvertex; ++b) {
                blossombestedges[uz(b)].clear();
                bestedges_valid[uz(b)] = 0;
            }
            std::fill(allowedge.begin(), allowedge.end(), 0);
            queue_.clear();
            for (int v = 0; v < nvertex; ++v)
                if (mate[uz(v)] == -1 && label[uz(inblossom[uz(v)])] == 0)
                    assign_label(v, 1, -1);

            bool augmented = false;
            // Each substage either makes a new edge allowable or expands a
            // blossom, so this cap is unreachable in exact arithmetic; it
            // exists only so rounding can never spin here forever.
            const int substage_cap = 4 * (nvertex + nedge) + 16;
            int substage_guard = 0;
            while (true) {
                if (++substage_guard > substage_cap) break;

                // ---- grow the alternating forest ----
                while (!queue_.empty() && !augmented) {
                    const int v = queue_.back();
                    queue_.pop_back();
                    const std::vector<int>& nb = neighbend[uz(v)];
                    for (std::size_t pi = 0; pi < nb.size(); ++pi) {
                        const int p = nb[pi];
                        const int k = p / 2;
                        const int w = endpoint[uz(p)];
                        if (inblossom[uz(v)] == inblossom[uz(w)]) continue;
                        double kslack = 0.0;
                        if (!allowedge[uz(k)]) {
                            kslack = slack(k);
                            if (kslack <= tol) allowedge[uz(k)] = 1;
                        }
                        if (allowedge[uz(k)]) {
                            if (label[uz(inblossom[uz(w)])] == 0) {
                                assign_label(w, 2, p ^ 1);
                            } else if (label[uz(inblossom[uz(w)])] == 1) {
                                const int base = scan_blossom(v, w);
                                if (base >= 0) {
                                    add_blossom(base, k);
                                } else {
                                    augment_matching(k);
                                    augmented = true;
                                    break;
                                }
                            } else if (label[uz(w)] == 0) {
                                label[uz(w)] = 2;
                                labelend[uz(w)] = p ^ 1;
                            }
                        } else if (label[uz(inblossom[uz(w)])] == 1) {
                            const int bv = inblossom[uz(v)];
                            if (bestedge[uz(bv)] == -1 || kslack < slack(bestedge[uz(bv)]))
                                bestedge[uz(bv)] = k;
                        } else if (label[uz(w)] == 0) {
                            if (bestedge[uz(w)] == -1 || kslack < slack(bestedge[uz(w)]))
                                bestedge[uz(w)] = k;
                        }
                    }
                }
                if (augmented) break;

                // ---- dual adjustment: take the least of the four deltas ----
                int deltatype = -1;
                double delta = 0.0;
                int deltaedge = -1;
                int deltablossom = -1;

                if (!maxcardinality) {                       // delta1
                    deltatype = 1;
                    delta = min_vertex_dual();
                }
                for (int v = 0; v < nvertex; ++v) {          // delta2
                    if (label[uz(inblossom[uz(v)])] == 0 && bestedge[uz(v)] != -1) {
                        const double d = slack(bestedge[uz(v)]);
                        if (deltatype == -1 || d < delta) {
                            delta = d;
                            deltatype = 2;
                            deltaedge = bestedge[uz(v)];
                        }
                    }
                }
                for (int b = 0; b < 2 * nvertex; ++b) {      // delta3
                    if (blossomparent[uz(b)] == -1 && label[uz(b)] == 1 &&
                        bestedge[uz(b)] != -1) {
                        const double d = 0.5 * slack(bestedge[uz(b)]);
                        if (deltatype == -1 || d < delta) {
                            delta = d;
                            deltatype = 3;
                            deltaedge = bestedge[uz(b)];
                        }
                    }
                }
                for (int b = nvertex; b < 2 * nvertex; ++b) {  // delta4
                    if (blossombase[uz(b)] >= 0 && blossomparent[uz(b)] == -1 &&
                        label[uz(b)] == 2 && (deltatype == -1 || dualvar[uz(b)] < delta)) {
                        delta = dualvar[uz(b)];
                        deltatype = 4;
                        deltablossom = b;
                    }
                }
                if (deltatype == -1) {                       // maxcardinality optimum
                    deltatype = 1;
                    delta = std::max(0.0, min_vertex_dual());
                }
                if (delta < 0.0) delta = 0.0;                // rounding clamp

                for (int v = 0; v < nvertex; ++v) {
                    const int lv = label[uz(inblossom[uz(v)])];
                    if (lv == 1) dualvar[uz(v)] -= delta;
                    else if (lv == 2) dualvar[uz(v)] += delta;
                }
                for (int b = nvertex; b < 2 * nvertex; ++b) {
                    if (blossombase[uz(b)] >= 0 && blossomparent[uz(b)] == -1) {
                        if (label[uz(b)] == 1) dualvar[uz(b)] += delta;
                        else if (label[uz(b)] == 2) dualvar[uz(b)] -= delta;
                    }
                }

                if (deltatype == 1) break;                   // optimum reached
                if (deltatype == 2) {
                    allowedge[uz(deltaedge)] = 1;
                    int i = eu[uz(deltaedge)];
                    if (label[uz(inblossom[uz(i)])] == 0) i = ev[uz(deltaedge)];
                    queue_.push_back(i);
                } else if (deltatype == 3) {
                    allowedge[uz(deltaedge)] = 1;
                    queue_.push_back(eu[uz(deltaedge)]);
                } else {
                    expand_blossom(deltablossom, false);
                }
            }

            if (!augmented) break;                           // no augmenting path left
            for (int b = nvertex; b < 2 * nvertex; ++b) {    // end of stage
                if (blossomparent[uz(b)] == -1 && blossombase[uz(b)] >= 0 &&
                    label[uz(b)] == 1 && std::abs(dualvar[uz(b)]) <= tol) {
                    expand_blossom(b, true);
                }
            }
        }

        for (int v = 0; v < nvertex; ++v) {
            const int mv = mate[uz(v)];
            if (mv >= 0 && endpoint[uz(mv)] > v) {
                result.edges.emplace_back(v, endpoint[uz(mv)]);
                result.weight += ew[uz(mv / 2)];
            }
        }
        return result;
    }
};

} // namespace

std::vector<std::pair<int, int>> max_weight_matching(const Graph& G, bool maxcardinality) {
    BlossomSolver solver;
    return solver.solve(G, maxcardinality).edges;
}

double max_weight_matching_value(const Graph& G, bool maxcardinality) {
    BlossomSolver solver;
    return solver.solve(G, maxcardinality).weight;
}

// ---- Coloring ----

std::vector<int> greedy_colour(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> color(n, -1);
    for (int v = 0; v < n; ++v) {
        std::unordered_set<int> used;
        for (auto& [u, w] : G.neighbors(v))
            if (color[u] >= 0) used.insert(color[u]);
        int c = 0;
        while (used.count(c)) ++c;
        color[v] = c;
    }
    return color;
}

int chromatic_number_approx(const Graph& G) {
    auto colors = greedy_colour(G);
    return *std::max_element(colors.begin(), colors.end()) + 1;
}

// ---- Community detection (Louvain) ----

namespace {

// One phase-1 "local moving" run to convergence: repeatedly scan all vertices
// (fixed order, no RNG) and greedily move each to whichever neighbouring
// community (or its own) yields the best modularity gain, until a full pass
// produces no moves. Returns a community label per vertex (labels are a
// subset of [0, n), not necessarily consecutive -- see aggregate() below).
//
// Uses the incremental gain formula (Blondel et al. 2008) rather than
// recomputing full modularity per candidate:
//   gain(i, C) = k_i,in(C)/m - Sigma_tot(C) * k_i / (2*m^2)
// where k_i,in(C) excludes i's own self-loop (a self-loop never connects i to
// a *different* community) and Sigma_tot(C) is C's total weighted degree with
// i already removed.
std::vector<int> local_moving(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> community(n);
    std::iota(community.begin(), community.end(), 0);
    if (n == 0) return community;

    std::vector<double> k(n, 0.0);
    double total = 0.0;
    for (int v = 0; v < n; ++v)
        for (auto& [u, w] : G.neighbors(v)) { k[v] += w; total += w; }
    double m = total / 2.0;
    if (m <= 1e-15) return community;  // no edges: leave every vertex singleton

    std::vector<double> sigma_tot = k;  // each vertex starts as its own community

    const int max_passes = 100;
    for (int pass = 0; pass < max_passes; ++pass) {
        bool moved = false;
        for (int i = 0; i < n; ++i) {
            int ci = community[i];
            sigma_tot[ci] -= k[i];

            std::map<int, double> weight_to_comm;
            weight_to_comm[ci];  // always a candidate, even with zero neighbor weight
            for (auto& [v, w] : G.neighbors(i)) {
                if (v == i) continue;  // exclude self-loops from k_i,in
                weight_to_comm[community[v]] += w;
            }

            int best_comm = ci;
            double best_gain = weight_to_comm[ci] / m - sigma_tot[ci] * k[i] / (2.0 * m * m);
            for (auto& [c, w_in] : weight_to_comm) {
                if (c == ci) continue;
                double gain = w_in / m - sigma_tot[c] * k[i] / (2.0 * m * m);
                if (gain > best_gain + 1e-12) { best_gain = gain; best_comm = c; }
            }
            sigma_tot[best_comm] += k[i];
            if (best_comm != ci) { community[i] = best_comm; moved = true; }
        }
        if (!moved) break;
    }
    return community;
}

struct Aggregation { Graph meta; std::vector<int> mapping; };

// Phase-2 aggregation: collapse each community from local_moving() into a
// single super-vertex. Inter-community edges sum into a single weighted edge
// between the corresponding super-vertices; intra-community edges (including
// any pre-existing self-loops) sum into a self-loop on the super-vertex.
//
// Every neighbour-list entry (u, v, w) is a "half" of some nominal edge of
// weight w: a regular edge (u != v) contributes one such entry at each
// endpoint, and a self-loop (u == v) contributes two such entries at its one
// vertex (this is exactly how Graph::add_edge stores both cases). So summing
// w/2 over all entries recovers each nominal edge's true weight exactly once,
// whether it lands on a self-loop or a cross-community pair. Feeding the
// resulting self-loop weight back through add_edge(c, c, w) re-creates the
// same doubled-entry convention one level up, so the invariant is preserved
// across repeated aggregation.
Aggregation aggregate(const Graph& G, const std::vector<int>& community) {
    int n = G.n_vertices();
    std::unordered_map<int, int> new_id;
    std::vector<int> mapping(n);
    for (int v = 0; v < n; ++v) {
        auto [it, inserted] = new_id.try_emplace(community[v], static_cast<int>(new_id.size()));
        mapping[v] = it->second;
    }
    int k = static_cast<int>(new_id.size());

    std::vector<double> self_w(k, 0.0);
    std::map<std::pair<int,int>, double> pair_w;
    for (int u = 0; u < n; ++u) {
        int cu = mapping[u];
        for (auto& [v, w] : G.neighbors(u)) {
            int cv = mapping[v];
            if (cu == cv) self_w[cu] += 0.5 * w;
            else pair_w[{std::min(cu, cv), std::max(cu, cv)}] += 0.5 * w;
        }
    }

    Graph meta(k, false);
    for (int c = 0; c < k; ++c)
        if (self_w[c] > 1e-15) meta.add_edge(c, c, self_w[c]);
    for (auto& [key, w] : pair_w)
        if (w > 1e-15) meta.add_edge(key.first, key.second, w);

    return {std::move(meta), std::move(mapping)};
}

} // namespace

std::vector<std::vector<int>> louvain(const Graph& G) {
    int n0 = G.n_vertices();
    if (n0 == 0) return {};

    Graph current = to_undirected(G);
    std::vector<std::vector<int>> level_mappings;

    const int max_levels = 50;
    for (int level = 0; level < max_levels; ++level) {
        int n = current.n_vertices();
        auto community = local_moving(current);
        auto agg = aggregate(current, community);
        level_mappings.push_back(agg.mapping);
        if (agg.meta.n_vertices() >= n) break;  // no further coarsening -- local optimum
        current = std::move(agg.meta);
    }

    std::vector<int> final_comm(n0);
    int k_final = 0;
    for (int v = 0; v < n0; ++v) {
        int c = v;
        for (auto& mapping : level_mappings) c = mapping[c];
        final_comm[v] = c;
        k_final = std::max(k_final, c + 1);
    }
    std::vector<std::vector<int>> result(k_final);
    for (int v = 0; v < n0; ++v) result[final_comm[v]].push_back(v);
    result.erase(std::remove_if(result.begin(), result.end(),
        [](const std::vector<int>& c){ return c.empty(); }), result.end());
    return result;
}

double modularity(const Graph& G, const std::vector<std::vector<int>>& communities) {
    int n = G.n_vertices();
    if (n == 0) return 0.0;

    std::vector<int> comm_of(n, -1);
    for (int c = 0; c < static_cast<int>(communities.size()); ++c)
        for (int v : communities[c])
            if (v >= 0 && v < n) comm_of[v] = c;

    std::vector<double> k(n, 0.0);
    double total = 0.0;
    for (int v = 0; v < n; ++v)
        for (auto& [u, w] : G.neighbors(v)) { k[v] += w; total += w; }
    double m = total / 2.0;
    if (m <= 1e-15) return 0.0;

    int nc = static_cast<int>(communities.size());
    std::vector<double> internal(nc, 0.0), degree_sum(nc, 0.0);
    for (int v = 0; v < n; ++v) {
        int cv = comm_of[v];
        if (cv < 0) continue;
        degree_sum[cv] += k[v];
        for (auto& [u, w] : G.neighbors(v))
            if (comm_of[u] == cv) internal[cv] += w;
    }

    double Q = 0.0;
    for (int c = 0; c < nc; ++c) {
        double frac = degree_sum[c] / (2.0 * m);
        Q += (0.5 * internal[c]) / m - frac * frac;
    }
    return Q;
}

// ---- Adjacency spectrum (power iteration for largest eigenvalue) ----

std::vector<double> adjacency_spectrum(const Graph& G) {
    int n = G.n_vertices();
    // Build adjacency matrix
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (int u = 0; u < n; ++u)
        for (auto& [v, w] : G.neighbors(u)) A[u][v] = 1.0;
    // Power iteration for spectral radius
    std::vector<double> x(n, 1.0 / std::sqrt(n));
    double lam = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) y[i] += A[i][j] * x[j];
        lam = 0.0;
        for (double v : y) lam += v * v;
        lam = std::sqrt(lam);
        if (lam < 1e-15) break;
        for (auto& v : y) v /= lam;
        x = y;
    }
    return {lam};
}

// ---- Euler circuit (Hierholzer's) ----

std::vector<int> euler_circuit(const Graph& G) {
    int n = G.n_vertices();
    // Check if Euler circuit exists
    for (int v = 0; v < n; ++v)
        if (G.neighbors(v).size() % 2 != 0) return {};
    // Hierholzer's algorithm
    std::vector<std::vector<std::pair<int,double>>> adj(n);
    for (int u = 0; u < n; ++u) adj[u] = G.neighbors(u);
    std::vector<size_t> idx(n, 0);
    std::stack<int> stk;
    std::vector<int> circuit;
    stk.push(0);
    while (!stk.empty()) {
        int v = stk.top();
        if (idx[v] < adj[v].size()) {
            stk.push(adj[v][idx[v]++].first);
        } else {
            circuit.push_back(v);
            stk.pop();
        }
    }
    std::reverse(circuit.begin(), circuit.end());
    return circuit;
}

// ---- Eulerian path/circuit (Hierholzer's), undirected graphs ----

EulerianResult eulerian_path(const Graph& G) {
    EulerianResult result;
    int n = G.n_vertices();

    // Degenerate zero-edge case (see header @note): vacuously a circuit,
    // but there is nothing to traverse so `path` stays empty.
    if (n == 0 || G.n_edges() == 0) {
        result.has_circuit = true;
        result.has_path = true;
        return result;
    }

    std::vector<int> degree(n, 0);
    for (int v = 0; v < n; ++v) degree[v] = static_cast<int>(G.neighbors(v).size());

    int any_nonzero = -1;
    for (int v = 0; v < n; ++v)
        if (degree[v] > 0) { any_nonzero = v; break; }

    // Connectivity, ignoring isolated (zero-degree) vertices: every vertex
    // with an incident edge must be reachable from any one of them.
    std::vector<bool> visited(n, false);
    for (int v : bfs(G, any_nonzero)) visited[v] = true;
    for (int v = 0; v < n; ++v)
        if (degree[v] > 0 && !visited[v]) return result;  // disconnected -> both false

    int odd_count = 0, first_odd = -1;
    for (int v = 0; v < n; ++v)
        if (degree[v] % 2 != 0) {
            ++odd_count;
            if (first_odd == -1) first_odd = v;
        }

    if (odd_count != 0 && odd_count != 2) return result;  // neither circuit nor path

    result.has_circuit = (odd_count == 0);
    result.has_path = true;
    int start = (odd_count == 2) ? first_odd : any_nonzero;

    // Edge-indexed adjacency: each physical (undirected) edge gets one
    // shared index into `used`, referenced from both endpoints' lists, so
    // traversing it from either side marks the same slot used.
    auto edge_list = G.edges();
    int m = static_cast<int>(edge_list.size());
    std::vector<std::vector<std::pair<int,int>>> adj(n);  // (neighbor, edge index)
    for (int i = 0; i < m; ++i) {
        const Edge& e = edge_list[i];
        adj[e.from].push_back({e.to, i});
        if (e.to != e.from) adj[e.to].push_back({e.from, i});
    }
    std::vector<bool> used(m, false);
    std::vector<size_t> ptr(n, 0);

    // Iterative Hierholzer: descend on unused edges, backtrack (emitting
    // to `circuit`) once the top of the stack has none left, then reverse.
    std::vector<int> stack_path{start};
    std::vector<int> circuit;
    while (!stack_path.empty()) {
        int v = stack_path.back();
        while (ptr[v] < adj[v].size() && used[adj[v][ptr[v]].second]) ++ptr[v];
        if (ptr[v] < adj[v].size()) {
            auto [nv, eidx] = adj[v][ptr[v]];
            used[eidx] = true;
            stack_path.push_back(nv);
        } else {
            circuit.push_back(v);
            stack_path.pop_back();
        }
    }
    std::reverse(circuit.begin(), circuit.end());
    result.path = std::move(circuit);
    return result;
}

// ---- Hamiltonian path (backtracking) ----

Result<std::vector<int>> hamiltonian_path(const Graph& G) {
    int n = G.n_vertices();
    std::vector<int> path;
    std::vector<bool> visited(n, false);
    path.push_back(0); visited[0] = true;

    std::function<bool()> backtrack = [&]() -> bool {
        if ((int)path.size() == n) return true;
        int v = path.back();
        for (auto& [u, w] : G.neighbors(v)) {
            if (!visited[u]) {
                visited[u] = true;
                path.push_back(u);
                if (backtrack()) return true;
                path.pop_back();
                visited[u] = false;
            }
        }
        return false;
    };
    if (backtrack()) return path;
    return std::unexpected(Error{DomainError{"hamiltonian_path", "no Hamiltonian path"}});
}

// ---- Traveling Salesman Problem heuristic (nearest-neighbor + 2-opt) ----

namespace {

// Sums the cycle length of `tour` under `dist`, including the wraparound
// edge from the last vertex back to tour[0]. Shared by both the
// nearest-neighbor construction and the final TSPResult bookkeeping so the
// two phases can never disagree on how a tour's length is computed.
double tour_cycle_length(const std::vector<int>& tour,
                          const std::vector<std::vector<double>>& dist) {
    int n = static_cast<int>(tour.size());
    if (n < 2) return 0.0;
    double total = 0.0;
    for (int i = 0; i + 1 < n; ++i) total += dist[tour[i]][tour[i + 1]];
    total += dist[tour[n - 1]][tour[0]];
    return total;
}

// Nearest-neighbor greedy construction: from `start`, repeatedly hop to the
// closest unvisited vertex until every vertex has been appended.
std::vector<int> nearest_neighbor_tour(const std::vector<std::vector<double>>& dist, int start) {
    int n = static_cast<int>(dist.size());
    std::vector<int> tour;
    tour.reserve(n);
    std::vector<bool> visited(n, false);
    int current = start;
    tour.push_back(current);
    visited[current] = true;
    for (int step = 1; step < n; ++step) {
        int best = -1;
        double best_dist = INF;
        for (int v = 0; v < n; ++v) {
            if (!visited[v] && dist[current][v] < best_dist) {
                best_dist = dist[current][v];
                best = v;
            }
        }
        tour.push_back(best);
        visited[best] = true;
        current = best;
    }
    return tour;
}

} // namespace

TSPResult tsp_heuristic(const std::vector<std::vector<double>>& dist,
                         int start_vertex, int max_2opt_iterations) {
    TSPResult result;
    int n = static_cast<int>(dist.size());
    if (n == 0) return result;
    if (n == 1) { result.tour = {0}; return result; }

    std::vector<int> tour = nearest_neighbor_tour(dist, start_vertex);

    // 2-opt local search: each pass scans every pair of tour edges (i,i+1)
    // and (j,j+1) (cyclically, so the edge out of the last vertex wraps
    // back to tour[0]) and finds the single most-improving segment
    // reversal; apply it and start a fresh pass, converging once no
    // improving reversal remains anywhere in a full pass.
    for (int iter = 0; iter < max_2opt_iterations; ++iter) {
        double best_delta = 0.0;
        int best_i = -1, best_j = -1;
        for (int i = 0; i < n; ++i) {
            int ip1 = (i + 1) % n;
            for (int j = i + 1; j < n; ++j) {
                int jp1 = (j + 1) % n;
                if (ip1 == j || jp1 == i) continue;  // adjacent edges share a vertex, no-op
                double delta = dist[tour[i]][tour[j]] + dist[tour[ip1]][tour[jp1]]
                             - dist[tour[i]][tour[ip1]] - dist[tour[j]][tour[jp1]];
                if (delta < best_delta) {
                    best_delta = delta;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (best_i < 0) break;  // no improving reversal found in this pass -> converged
        std::reverse(tour.begin() + best_i + 1, tour.begin() + best_j + 1);
    }

    result.tour = tour;
    result.total_distance = tour_cycle_length(tour, dist);
    return result;
}

} // namespace graph
} // namespace ms
