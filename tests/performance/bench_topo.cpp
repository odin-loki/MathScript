// MathScript Benchmark: persistence diagram distances (bottleneck, Wasserstein)

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/topo/topo.hpp"

using ms::topo::PersistencePair;
using ms::topo::bottleneck_distance;
using ms::topo::wasserstein_distance;

namespace {

std::vector<PersistencePair> make_synthetic_diagram(int n, double offset) {
    std::vector<PersistencePair> dgm;
    dgm.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double birth = static_cast<double>(i) * 0.01 + offset;
        const double death = birth + 0.5 + 0.01 * static_cast<double>(i % 7);
        dgm.push_back({0, birth, death});
    }
    return dgm;
}

} // namespace

static void BM_Bottleneck(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto dgm1 = make_synthetic_diagram(n, 0.0);
    const auto dgm2 = make_synthetic_diagram(n, 0.05);
    for (auto _ : state)
        benchmark::DoNotOptimize(bottleneck_distance(dgm1, dgm2, 0));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_Bottleneck)->Arg(50)->Arg(200)->Arg(500);

static void BM_Wasserstein(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto dgm1 = make_synthetic_diagram(n, 0.0);
    const auto dgm2 = make_synthetic_diagram(n, 0.05);
    for (auto _ : state)
        benchmark::DoNotOptimize(wasserstein_distance(dgm1, dgm2, 0, 2));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_Wasserstein)->Arg(50)->Arg(200)->Arg(500);
