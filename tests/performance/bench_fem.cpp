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
constexpr std::size_t kSolveElements = 100;

Mesh1D make_mesh(std::size_t n_elements) {
    return mesh1d(0.0, 1.0, n_elements);
}

} // namespace

// ---------------------------------------------------------------------------
// Assemble global stiffness matrix on a 1D P1 mesh
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Full 1D Poisson FEM solve (-u'' = f, Dirichlet BCs)
// ---------------------------------------------------------------------------

static void BM_FemSolve1D(benchmark::State& state) {
    const auto mesh = make_mesh(kSolveElements);
    for (auto _ : state) {
        ColMatrix<double> K = assemble_stiffness_1d(mesh);
        ColMatrix<double> f = assemble_load_1d(mesh, [](double x) {
            return M_PI * M_PI * std::sin(M_PI * x);
        });
        apply_dirichlet(K, f, {0, mesh.nodes.size() - 1}, {0.0, 0.0});
        auto u = solve_fem(K, f);
        benchmark::DoNotOptimize(u.has_value());
        if (u.has_value()) {
            benchmark::DoNotOptimize(u->data());
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kSolveElements));
}
BENCHMARK(BM_FemSolve1D);

BENCHMARK_MAIN();
