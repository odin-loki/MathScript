// MathScript Benchmark: graph traversal (BFS, Dijkstra, Floyd-Warshall)

#include <benchmark/benchmark.h>
#include "ms/graph/graph.hpp"

using ms::graph::Graph;
using ms::graph::bfs;
using ms::graph::dijkstra;
using ms::graph::floyd_warshall;

namespace {

Graph make_path_graph(int n) {
    Graph G(n, false);
    for (int i = 0; i + 1 < n; ++i)
        G.add_edge(i, i + 1);
    return G;
}

Graph make_grid_graph(int side) {
    const int n = side * side;
    Graph G(n, false);
    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side; ++c) {
            const int v = r * side + c;
            if (c + 1 < side) G.add_edge(v, v + 1);
            if (r + 1 < side) G.add_edge(v, v + side);
        }
    }
    return G;
}

Graph make_dense_graph(int n) {
    Graph G(n, false);
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            G.add_edge(u, v);
    return G;
}

} // namespace

static void BM_Bfs(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Graph G = make_path_graph(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(bfs(G, 0));
}
BENCHMARK(BM_Bfs)->Arg(1000)->Arg(10000)->Arg(50000);

static void BM_Bfs_Grid(benchmark::State& state) {
    const int side = static_cast<int>(state.range(0));
    const Graph G = make_grid_graph(side);
    for (auto _ : state)
        benchmark::DoNotOptimize(bfs(G, 0));
}
BENCHMARK(BM_Bfs_Grid)->Arg(32)->Arg(100)->Arg(224);

static void BM_Bfs_Dense(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Graph G = make_dense_graph(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(bfs(G, 0));
}
BENCHMARK(BM_Bfs_Dense)->Arg(64)->Arg(128)->Arg(192);

static void BM_Dijkstra(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Graph G = make_path_graph(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(dijkstra(G, 0));
}
BENCHMARK(BM_Dijkstra)->Arg(1000)->Arg(10000)->Arg(50000);

static void BM_Dijkstra_Grid(benchmark::State& state) {
    const int side = static_cast<int>(state.range(0));
    const Graph G = make_grid_graph(side);
    for (auto _ : state)
        benchmark::DoNotOptimize(dijkstra(G, 0));
}
BENCHMARK(BM_Dijkstra_Grid)->Arg(32)->Arg(100)->Arg(224);

static void BM_Dijkstra_Dense(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Graph G = make_dense_graph(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(dijkstra(G, 0));
}
BENCHMARK(BM_Dijkstra_Dense)->Arg(64)->Arg(128)->Arg(192);

static void BM_FloydWarshall(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Graph G = make_dense_graph(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(floyd_warshall(G));
}
BENCHMARK(BM_FloydWarshall)->Arg(64)->Arg(128)->Arg(192);

static void BM_FloydWarshall_Grid(benchmark::State& state) {
    const int side = static_cast<int>(state.range(0));
    const Graph G = make_grid_graph(side);
    for (auto _ : state)
        benchmark::DoNotOptimize(floyd_warshall(G));
}
BENCHMARK(BM_FloydWarshall_Grid)->Arg(32)->Arg(100)->Arg(224);
