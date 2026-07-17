// MathScript stats/signal benchmark suite

#include <benchmark/benchmark.h>
#include <numeric>
#include <vector>

#include "ms/signal/signal.hpp"
#include "ms/stats/stats.hpp"

using namespace ms;

static std::vector<double> make_data(int n) {
    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), 0.0);
    for (int i = 0; i < n; ++i) v[i] = 0.1 * v[i] + 1.0;
    return v;
}

static void BM_mean_stddev(benchmark::State& state) {
    const auto data = make_data(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        const double m = mean(data);
        const double s = stddev(data);
        benchmark::DoNotOptimize(m);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void BM_percentile(benchmark::State& state) {
    const auto data = make_data(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        const double p25 = percentile(data, 25.0);
        const double p75 = percentile(data, 75.0);
        benchmark::DoNotOptimize(p25);
        benchmark::DoNotOptimize(p75);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void BM_linear_regression(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto x = make_data(n);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = 2.0 * x[i] + 1.0;
    for (auto _ : state) {
        const auto r = linear_regression(x, y);
        benchmark::DoNotOptimize(r.slope);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_moving_average(benchmark::State& state) {
    const auto data = make_data(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        const auto result = moving_average(data, 5);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static void BM_convolve(benchmark::State& state) {
    const auto data = make_data(static_cast<int>(state.range(0)));
    const std::vector<double> kernel{0.25, 0.5, 0.25};
    for (auto _ : state) {
        const auto result = convolve(data, kernel);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

static std::vector<double> make_kde_grid(size_t count) {
    std::vector<double> grid(count);
    if (count == 0) {
        return grid;
    }
    if (count == 1) {
        grid[0] = 0.0;
        return grid;
    }
    const double step = 20.0 / static_cast<double>(count - 1);
    for (size_t i = 0; i < count; ++i) {
        grid[i] = -10.0 + static_cast<double>(i) * step;
    }
    return grid;
}

static void BM_KDE(benchmark::State& state) {
    const auto samples = make_data(1000);
    const auto grid = make_kde_grid(256);
    for (auto _ : state) {
        const auto density = kde(samples, grid, 0.4, "gaussian");
        benchmark::DoNotOptimize(density);
    }
    state.SetItemsProcessed(state.iterations() * 1000 * 256);
}

BENCHMARK(BM_mean_stddev)->Arg(1000)->Arg(10000)->Arg(100000);
BENCHMARK(BM_percentile)->Arg(1000)->Arg(10000);
BENCHMARK(BM_linear_regression)->Arg(500)->Arg(5000);
BENCHMARK(BM_moving_average)->Arg(1000)->Arg(10000);
BENCHMARK(BM_convolve)->Arg(1000)->Arg(10000);
BENCHMARK(BM_KDE);
