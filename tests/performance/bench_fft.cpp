// MathScript FFT Benchmark

#include <benchmark/benchmark.h>

#include <vector>

#include "ms/fft/fft.hpp"

using namespace ms;

static void BM_fft(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) * 0.001;
    }

    for (auto _ : state) {
        auto result = fft(x);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}

BENCHMARK(BM_fft)->Arg(256)->Arg(1024);
