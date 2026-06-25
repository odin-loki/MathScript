#include "ms/graph/graph.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <stack>
#include <unordered_set>

namespace ms {
namespace graph {

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

std::vector<int> bfs(const Graph& G, int source) {
    int n = G.n_vertices();
    std::vector<bool> visited(n, false);
    std::vector<int> order;
    std::queue<int> q;
    q.push(source);
    visited[source] = true;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        order.push_back(v);
        for (auto& [u, w] : G.neighbors(v))
            if (!visited[u]) { visited[u] = true; q.push(u); }
    }
    return order;
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

std::pair<std::vector<double>, std::vector<int>>
dijkstra(const Graph& G, int source) {
    int n = G.n_vertices();
    std::vector<double> dist(n, INF);
    std::vector<int> parent(n, -1);
    dist[source] = 0.0;
    using P = std::pair<double,int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    pq.push({0.0, source});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& [v, w] : G.neighbors(u)) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    return {dist, parent};
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
    int n = G.n_vertices();
    std::vector<std::vector<double>> d(n, std::vector<double>(n, INF));
    for (int i = 0; i < n; ++i) d[i][i] = 0.0;
    for (int u = 0; u < n; ++u)
        for (auto& [v, w] : G.neighbors(u)) d[u][v] = std::min(d[u][v], w);
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (d[i][k] < INF && d[k][j] < INF)
                    d[i][j] = std::min(d[i][j], d[i][k] + d[k][j]);
    return d;
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

// ---- Properties ----

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

// ---- Max flow (Dinic's) ----

Result<double> max_flow(const Graph& G, int source, int sink) {
    int n = G.n_vertices();
    // Build residual graph
    std::vector<std::vector<std::pair<int,double>>> res(n);
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
    double total_flow = 0.0;
    while (true) {
        // BFS to find level graph
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
            for (int& i = iter[v]; i < (int)res[v].size(); ++i) {
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
            total_flow += pushed;
        }
    }
    return total_flow;
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

} // namespace graph
} // namespace ms
