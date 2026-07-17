// MathScript FEM benchmark suite

#include <benchmark/benchmark.h>
#include <cmath>

#include "ms/fem/fem.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using namespace ms::fem;

static void BM_FemSolve1D(benchmark::State& state) {
    const std::size_t n_elements = static_cast<std::size_t>(state.range(0));
    const Mesh1D mesh = mesh1d(0.0, 1.0, n_elements);
    ColMatrix<double> K = assemble_stiffness_1d(mesh);
    ColMatrix<double> f = assemble_load_1d(mesh, [](double x) {
        return M_PI * M_PI * std::sin(M_PI * x);
    });
    apply_dirichlet(K, f, {0, mesh.nodes.size() - 1}, {0.0, 0.0});

    for (auto _ : state) {
        auto u = solve_fem(K, f);
        benchmark::DoNotOptimize(u);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(mesh.nodes.size()));
}
BENCHMARK(BM_FemSolve1D)->Arg(100)->Arg(500)->Arg(1000)->Arg(5000);

BENCHMARK_MAIN();
