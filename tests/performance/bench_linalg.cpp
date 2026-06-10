// MathScript linear algebra benchmarks

#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include <benchmark/benchmark.h>

using namespace ms;
using DMatrix = ColMatrix<double>;

static DMatrix make_spd(size_t n, unsigned seed) {
    const auto M = rand<double>(n, n, seed);
    const auto Mt = transpose(M);
    const auto MtM = matmul(M, Mt).value();
    const auto I = eye<double>(n);
    return MtM + static_cast<double>(n) * I;
}

static void BM_lu(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    const auto A = rand<double>(n, n, 42);

    for (auto _ : state) {
        auto result = lu(A);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(n) * static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

static void BM_solve(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    const auto A = rand<double>(n, n, 44);
    const auto b = rand<double>(n, 1, 45);

    for (auto _ : state) {
        auto result = solve(A, b);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

static void BM_svd(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    const auto A = rand<double>(n, n, 46);

    for (auto _ : state) {
        auto result = svd(A);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(n) * static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

static void BM_chol(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    const auto A = make_spd(n, 47);

    for (auto _ : state) {
        auto result = chol(A);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(n) * static_cast<int64_t>(n) * static_cast<int64_t>(n) / 3);
}

BENCHMARK(BM_lu)->Arg(64)->Arg(128);
BENCHMARK(BM_solve)->Arg(64);
BENCHMARK(BM_svd)->Arg(32);
BENCHMARK(BM_chol)->Arg(64);
