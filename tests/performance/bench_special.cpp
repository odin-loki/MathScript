// MathScript special function benchmarks

#include <benchmark/benchmark.h>
#include <cmath>
#include "ms/special/special.hpp"

using namespace ms;

static void BM_erf(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    for (auto _ : state) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += ms::erf(static_cast<double>(i) / n);
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_gamma(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    for (auto _ : state) {
        double sum = 0.0;
        for (int i = 1; i <= n; ++i) {
            sum += ms::gamma_func(static_cast<double>(i) / n + 0.5);
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_bessel_j0(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    for (auto _ : state) {
        double sum = 0.0;
        for (int i = 1; i <= n; ++i) {
            sum += ms::bessel_j0(static_cast<double>(i) * 0.1);
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_erf)->Arg(1000)->Arg(10000);
BENCHMARK(BM_gamma)->Arg(1000)->Arg(10000);
BENCHMARK(BM_bessel_j0)->Arg(100)->Arg(1000);
