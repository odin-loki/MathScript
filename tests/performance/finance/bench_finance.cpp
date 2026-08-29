// MathScript Benchmark: finance (Black-Scholes, MC European) + info (entropy)
// Wave 228 certification coverage for Wave 227 finance/info optimizations.

#include <benchmark/benchmark.h>
#include <vector>

#include "ms/finance/finance.hpp"
#include "ms/info/info.hpp"

using ms::finance::bs_call;
using ms::finance::bs_put;
using ms::finance::mc_european_call;
using ms::finance::heston_call;
using ms::finance::heston_put;
using ms::finance::sabr_call;
using ms::finance::sabr_put;
using ms::info::entropy;

namespace {

constexpr double kSpot = 100.0;
constexpr double kStrike = 100.0;
constexpr double kTime = 1.0;
constexpr double kRate = 0.05;
constexpr double kVol = 0.2;

// SABR ATM params (matches test_finance kSabrATM).
constexpr double kSabrAlpha = 0.20;
constexpr double kSabrBeta = 0.5;
constexpr double kSabrRho = -0.30;
constexpr double kSabrNu = 0.40;

// Heston ATM params (matches test_finance kHestonATM).
constexpr double kHestonV0 = 0.04;
constexpr double kHestonKappa = 2.0;
constexpr double kHestonTheta = 0.04;
constexpr double kHestonSigmaV = 0.3;
constexpr double kHestonRho = -0.7;

// Smoke-safe MC path count (unit tests use 500k; REPL uses 20k).
constexpr int kMcPaths = 10000;

std::vector<double> uniform_pmf(int n) {
    const double p_i = 1.0 / static_cast<double>(n);
    return std::vector<double>(static_cast<std::size_t>(n), p_i);
}

} // namespace

static void BM_BlackScholes(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(bs_call(kSpot, kStrike, kTime, kRate, kVol));
        benchmark::DoNotOptimize(bs_put(kSpot, kStrike, kTime, kRate, kVol));
        benchmark::DoNotOptimize(bs_call(kSpot, kStrike * 1.1, kTime, kRate, kVol));
        benchmark::DoNotOptimize(bs_put(kSpot, kStrike * 0.9, kTime, kRate, kVol));
    }
}
BENCHMARK(BM_BlackScholes);

static void BM_MCEuropean(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            mc_european_call(kSpot, kStrike, kTime, kRate, kVol, kMcPaths, 42u));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kMcPaths));
}
BENCHMARK(BM_MCEuropean);

static void BM_SabrCall(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(sabr_call(kSpot, kStrike, kTime, kRate, kSabrAlpha,
                                             kSabrBeta, kSabrRho, kSabrNu));
    }
}
BENCHMARK(BM_SabrCall);

static void BM_SabrPut(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(sabr_put(kSpot, kStrike, kTime, kRate, kSabrAlpha,
                                            kSabrBeta, kSabrRho, kSabrNu));
    }
}
BENCHMARK(BM_SabrPut);

static void BM_HestonCall(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(heston_call(kSpot, kStrike, kTime, kRate, kHestonV0,
                                             kHestonKappa, kHestonTheta, kHestonSigmaV,
                                             kHestonRho));
    }
}
BENCHMARK(BM_HestonCall);

static void BM_HestonPut(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(heston_put(kSpot, kStrike, kTime, kRate, kHestonV0,
                                            kHestonKappa, kHestonTheta, kHestonSigmaV,
                                            kHestonRho));
    }
}
BENCHMARK(BM_HestonPut);

static void BM_Entropy(benchmark::State& state) {
    const auto p = uniform_pmf(1024);
    for (auto _ : state) {
        benchmark::DoNotOptimize(entropy(p, 2.0));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(p.size()));
}
BENCHMARK(BM_Entropy);

BENCHMARK_MAIN();
