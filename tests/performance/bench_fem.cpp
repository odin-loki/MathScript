// MathScript 1D FEM benchmark suite

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/fem/fem.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using namespace ms::fem;

namespace {

constexpr std::size_t kAssembleElements = 1000;

Mesh1D make_mesh(std::size_t n_elements) {
    return mesh1d(0.0, 1.0, n_elements);
}

} // namespace

static void BM_FemAssemble1D(benchmark::State& state) {
    const auto mesh = make_mesh(kAssembleElements);
    for (auto _ : state) {
        auto K = assemble_stiffness_1d(mesh);
        benchmark::DoNotOptimize(K.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kAssembleElements));
}
BENCHMARK(BM_FemAssemble1D);

static void BM_FemSolve1D(benchmark::State& state) {
    const std::size_t n_elements = static_cast<std::size_t>(state.range(0));
    const Mesh1D mesh = make_mesh(n_elements);
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
