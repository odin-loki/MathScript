#include <benchmark/benchmark.h>
#include <cmath>
#include "ms/tensorops/tensorops.hpp"

using namespace ms::tensorops;

static Tensor make_random_matrix(int rows, int cols, double seed) {
    Tensor T({rows, cols}, 0.0);
    for (long i = 0; i < T.numel(); ++i)
        T.data[static_cast<size_t>(i)] =
            std::sin(static_cast<double>(i) * 0.17 + seed) * 0.5 + 0.25;
    return T;
}

static Tensor make_random_tensor(const std::vector<int>& shape, double seed) {
    Tensor T(shape, 0.0);
    for (long i = 0; i < T.numel(); ++i)
        T.data[static_cast<size_t>(i)] =
            std::sin(static_cast<double>(i) * 0.13 + seed) * 0.5 + 0.25;
    return T;
}

static void BM_TensorContract(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Tensor A = make_random_matrix(n, n, 1.0);
    const Tensor B = make_random_matrix(n, n, 2.0);

    for (auto _ : state) {
        auto C = contract(A, B, {{1, 0}});
        benchmark::DoNotOptimize(C.data.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) *
                            static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

static void BM_TensorContract_TTStep(benchmark::State& state) {
    const int r = static_cast<int>(state.range(0));
    const int n = static_cast<int>(state.range(1));
    Tensor left = make_random_tensor({1, n, r}, 3.0);
    Tensor core = make_random_tensor({r, n, r}, 4.0);

    for (auto _ : state) {
        auto out = contract(left, core, {{2, 0}});
        benchmark::DoNotOptimize(out.data.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(r) *
                            static_cast<int64_t>(n) * static_cast<int64_t>(r));
}

BENCHMARK(BM_TensorContract)->Arg(8)->Arg(16)->Arg(32)->Arg(64);
BENCHMARK(BM_TensorContract_TTStep)->Args({4, 8})->Args({8, 16})->Args({16, 32});

BENCHMARK_MAIN();
