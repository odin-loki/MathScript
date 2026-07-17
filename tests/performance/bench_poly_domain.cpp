// MathScript Benchmark: Polynomial, Domain, and DiffGeo Operations
// Benchmarks for poly_eval, poly_mul, poly_deriv, factorial, nchoosek, gcd, Graph, geodesic

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>
#include <numeric>
#include "ms/poly/poly.hpp"
#include "ms/domain/domain.hpp"
#include "ms/diffgeo/diffgeo.hpp"

using namespace ms;

// Helper: create a degree-N polynomial with coefficients 1..N+1
static std::vector<double> make_poly(int n) {
    std::vector<double> p(n + 1);
    for (int i = 0; i <= n; ++i) p[i] = static_cast<double>(i + 1);
    return p;
}

// ---------------------------------------------------------------------------
// Polynomial evaluation
// ---------------------------------------------------------------------------

static void BM_PolyEval_Degree10(benchmark::State& state) {
    auto p = make_poly(10);
    double x = 1.5;
    for (auto _ : state) {
        auto r = poly_eval(p, x);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyEval_Degree10);

static void BM_PolyEval_Degree50(benchmark::State& state) {
    auto p = make_poly(50);
    double x = 0.9;
    for (auto _ : state) {
        auto r = poly_eval(p, x);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyEval_Degree50);

static void BM_PolyEval_Degree100(benchmark::State& state) {
    auto p = make_poly(100);
    double x = 1.0;
    for (auto _ : state) {
        auto r = poly_eval(p, x);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyEval_Degree100);

static void BM_PolyEvalBatch(benchmark::State& state) {
    auto p = make_poly(50);
    // 4096 xs ~8s smoke; 1024 ~2s.
    std::vector<double> xs(1024);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(xs.size() - 1);
    }
    for (auto _ : state) {
        auto r = poly_eval_at(p, xs);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(xs.size()));
}
BENCHMARK(BM_PolyEvalBatch);

static void BM_PolyEvalAt_Degree100_Large(benchmark::State& state) {
    auto p = make_poly(100);
    // 65536 xs ~40s smoke; 4096 ~2.5s.
    std::vector<double> xs(4096);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(xs.size() - 1);
    }
    for (auto _ : state) {
        auto r = poly_eval_at(p, xs);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(xs.size()));
}
BENCHMARK(BM_PolyEvalAt_Degree100_Large);

static void BM_PolyEvalAt_ScalarSweep(benchmark::State& state) {
    auto p = make_poly(50);
    const int count = 8192;
    for (auto _ : state) {
        double sum = 0.0;
        for (int i = 0; i < count; ++i) {
            const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(count - 1);
            sum += poly_eval(p, x)[0];
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_PolyEvalAt_ScalarSweep);

// ---------------------------------------------------------------------------
// Polynomial addition
// ---------------------------------------------------------------------------

static void BM_PolyAdd_Degree20(benchmark::State& state) {
    auto a = make_poly(20);
    auto b = make_poly(20);
    for (auto _ : state) {
        auto r = poly_add(a, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyAdd_Degree20);

static void BM_PolyAdd_Degree200(benchmark::State& state) {
    auto a = make_poly(200);
    auto b = make_poly(200);
    for (auto _ : state) {
        auto r = poly_add(a, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyAdd_Degree200);

// ---------------------------------------------------------------------------
// Polynomial multiplication
// ---------------------------------------------------------------------------

static void BM_PolyMul_Degree10(benchmark::State& state) {
    auto a = make_poly(10);
    auto b = make_poly(10);
    for (auto _ : state) {
        auto r = poly_mul(a, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyMul_Degree10);

static void BM_PolyMul_Degree50(benchmark::State& state) {
    auto a = make_poly(50);
    auto b = make_poly(50);
    for (auto _ : state) {
        auto r = poly_mul(a, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyMul_Degree50);

// ---------------------------------------------------------------------------
// Polynomial derivative
// ---------------------------------------------------------------------------

static void BM_PolyDeriv_Degree100(benchmark::State& state) {
    auto p = make_poly(100);
    for (auto _ : state) {
        auto r = poly_deriv(p);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_PolyDeriv_Degree100);

// ---------------------------------------------------------------------------
// Factorial
// ---------------------------------------------------------------------------

static void BM_Factorial_20(benchmark::State& state) {
    for (auto _ : state) {
        auto r = factorial(20);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Factorial_20);

static void BM_Factorial_Sweep(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t n = 0; n <= 20; ++n) {
            auto r = factorial(n);
            benchmark::DoNotOptimize(r);
        }
    }
}
BENCHMARK(BM_Factorial_Sweep);

// ---------------------------------------------------------------------------
// nChooseK
// ---------------------------------------------------------------------------

static void BM_NChooseK_100_50(benchmark::State& state) {
    for (auto _ : state) {
        auto r = nchoosek(30, 15);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_NChooseK_100_50);

// ---------------------------------------------------------------------------
// GCD
// ---------------------------------------------------------------------------

static void BM_GCD_LargeNumbers(benchmark::State& state) {
    for (auto _ : state) {
        auto r = gcd(123456789u, 987654321u);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_GCD_LargeNumbers);

static void BM_GCD_Sweep(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t i = 1; i <= 100; ++i) {
            auto r = gcd(i * 7, i * 11);
            benchmark::DoNotOptimize(r);
        }
    }
}
BENCHMARK(BM_GCD_Sweep);

// ---------------------------------------------------------------------------
// Graph operations
// ---------------------------------------------------------------------------

static void BM_GraphEdges_Dense_50Nodes(benchmark::State& state) {
    const size_t N = 50;
    Graph g;
    g.n = N;
    g.adj.resize(N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = i + 1; j < N; ++j)
            g.adj[i].push_back(j);

    for (auto _ : state) {
        auto r = graph_num_edges(g);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_GraphEdges_Dense_50Nodes);

static void BM_GraphEdges_Sparse_100Nodes(benchmark::State& state) {
    const size_t N = 100;
    Graph g;
    g.n = N;
    g.adj.resize(N);
    for (size_t i = 0; i + 1 < N; ++i)
        g.adj[i].push_back(i + 1);  // Chain graph

    for (auto _ : state) {
        auto r = graph_num_edges(g);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_GraphEdges_Sparse_100Nodes);

// ---------------------------------------------------------------------------
// DiffGeo geodesic (Wave 228 certification)
// ---------------------------------------------------------------------------

namespace {

ms::diffgeo::MetricFn euclidean_2d_metric() {
    return [](const ms::diffgeo::Coords& x) -> std::vector<std::vector<double>> {
        (void)x;
        return {{1, 0}, {0, 1}};
    };
}

ms::diffgeo::MetricFn sphere_metric_2d() {
    return [](const ms::diffgeo::Coords& x) -> std::vector<std::vector<double>> {
        const double s2 = std::sin(x[0]) * std::sin(x[0]);
        return {{1, 0}, {0, s2}};
    };
}

} // namespace

static void BM_Geodesic_Euclidean(benchmark::State& state) {
    const int n_steps = static_cast<int>(state.range(0));
    const ms::diffgeo::Coords x0 = {0.0, 0.0};
    const ms::diffgeo::Coords v0 = {1.0, 0.5};
    for (auto _ : state) {
        auto traj = ms::diffgeo::geodesic(euclidean_2d_metric(), x0, v0, 1.0, n_steps);
        benchmark::DoNotOptimize(traj.back().x[0]);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_steps);
}
BENCHMARK(BM_Geodesic_Euclidean)->Arg(50)->Arg(100);

static void BM_Christoffel_Sphere(benchmark::State& state) {
    const ms::diffgeo::Coords x = {0.7853981633974483, 0.5};  // pi/4, 0.5
    for (auto _ : state) {
        auto Chr = ms::diffgeo::christoffel(sphere_metric_2d(), x);
        benchmark::DoNotOptimize(Chr[0][1][1]);
    }
}
BENCHMARK(BM_Christoffel_Sphere);

BENCHMARK_MAIN();
