// MathScript Signal Filter Design / Apply Benchmark Suite

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/signal/signal.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

namespace {

constexpr double kFs = 8000.0;
constexpr double kCutoff = 1000.0;
constexpr int kChebyOrder = 4;
constexpr size_t kApplySamples = 8192;
constexpr size_t kWelchSamples = 65536;

std::vector<double> make_sinusoid(size_t n, double fs, double f0) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * f0 * static_cast<double>(i) / fs);
    }
    return x;
}

// scipy.signal.butter(4, 0.25, output='sos') at fs=8000, cutoff=1000 Hz (Wn=0.25).
std::vector<std::array<double, 6>> order4_lowpass_sos() {
    return {
        {0.01020948, 0.02041896, 0.01020948, 1.0, -0.85539793, 0.20971536},
        {1.0, 2.0, 1.0, 1.0, -1.11302985, 0.57406192},
    };
}

} // namespace

// ---------------------------------------------------------------------------
// cheby1 design (order 4)
// ---------------------------------------------------------------------------

static void BM_Cheby1Design(benchmark::State& state) {
    for (auto _ : state) {
        auto coeffs = cheby1(kChebyOrder, 1.0, kCutoff, kFs, FilterType::Lowpass);
        benchmark::DoNotOptimize(coeffs.b.data());
        benchmark::DoNotOptimize(coeffs.a.data());
    }
}
BENCHMARK(BM_Cheby1Design);

// ---------------------------------------------------------------------------
// cheby2 design (order 4)
// ---------------------------------------------------------------------------

static void BM_Cheby2Design(benchmark::State& state) {
    for (auto _ : state) {
        auto coeffs = cheby2(kChebyOrder, 40.0, kCutoff, kFs, FilterType::Lowpass);
        benchmark::DoNotOptimize(coeffs.b.data());
        benchmark::DoNotOptimize(coeffs.a.data());
    }
}
BENCHMARK(BM_Cheby2Design);

// ---------------------------------------------------------------------------
// sosfilt on 8192 samples (order 4 lowpass)
// ---------------------------------------------------------------------------

static void BM_Sosfilt8192(benchmark::State& state) {
    const auto sos = order4_lowpass_sos();
    const auto x = make_sinusoid(kApplySamples, kFs, 500.0);
    for (auto _ : state) {
        auto y = sosfilt(sos, x);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kApplySamples));
}
BENCHMARK(BM_Sosfilt8192);

// ---------------------------------------------------------------------------
// filtfilt on 8192 samples
// ---------------------------------------------------------------------------

static void BM_Filtfilt8192(benchmark::State& state) {
    const auto coeffs = cheby1(kChebyOrder, 1.0, kCutoff, kFs, FilterType::Lowpass);
    const auto x = make_sinusoid(kApplySamples, kFs, 500.0);
    for (auto _ : state) {
        auto y = filtfilt(coeffs.b, coeffs.a, x);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kApplySamples));
}
BENCHMARK(BM_Filtfilt8192);

// ---------------------------------------------------------------------------
// welch_psd on 65536 samples
// ---------------------------------------------------------------------------

static void BM_WelchPsd65536(benchmark::State& state) {
    constexpr double welch_fs = 1000.0;
    constexpr size_t segment_len = 256;
    const auto x = make_sinusoid(kWelchSamples, welch_fs, 50.0);
    for (auto _ : state) {
        auto result = welch_psd(x, welch_fs, segment_len, 0.5);
        benchmark::DoNotOptimize(result.has_value());
        if (result.has_value()) {
            benchmark::DoNotOptimize(result->power.data());
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kWelchSamples));
}
BENCHMARK(BM_WelchPsd65536);

// ---------------------------------------------------------------------------
// welch_psd buffer reuse on 65536 samples (same workload, dedicated tag)
// ---------------------------------------------------------------------------

static void BM_WelchReuse(benchmark::State& state) {
    constexpr double welch_fs = 1000.0;
    constexpr size_t segment_len = 256;
    const auto x = make_sinusoid(kWelchSamples, welch_fs, 50.0);
    for (auto _ : state) {
        auto result = welch_psd(x, welch_fs, segment_len, 0.5);
        benchmark::DoNotOptimize(result.has_value());
        if (result.has_value()) {
            benchmark::DoNotOptimize(result->power.data());
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kWelchSamples));
}
BENCHMARK(BM_WelchReuse);

BENCHMARK_MAIN();
