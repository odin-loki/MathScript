// MathScript FFT Benchmark

#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

#include "ms/fft/fft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

static void BM_ifft(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) * 0.001;
    }
    const auto spec = fft(x).value();

    for (auto _ : state) {
        auto result = ifft(spec);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}

BENCHMARK(BM_fft)->Arg(256)->Arg(1024)->Arg(4096)->Arg(8192);
BENCHMARK(BM_ifft)->Arg(256)->Arg(1024)->Arg(8192);

static void BM_rfft(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * 17.0 * static_cast<double>(i) / static_cast<double>(n));
    }

    for (auto _ : state) {
        auto result = rfft(x);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}

BENCHMARK(BM_rfft)->Arg(4096);

static void BM_fft2_rect(benchmark::State& state) {
    const auto rows = static_cast<size_t>(state.range(0));
    const auto cols = static_cast<size_t>(state.range(1));
    std::vector<std::complex<double>> data(rows * cols);
    for (size_t i = 0; i < data.size(); ++i) {
        const double t = static_cast<double>(i);
        data[i] = {std::sin(0.13 * t), 0.2 * std::cos(0.07 * t)};
    }

    for (auto _ : state) {
        auto result = fft2(data, rows, cols);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows * cols));
}

BENCHMARK(BM_fft2_rect)->Args({8, 16})->Args({32, 64});
