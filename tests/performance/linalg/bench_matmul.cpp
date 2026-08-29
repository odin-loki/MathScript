// MathScript Matrix Multiply Benchmark (CPU only)

#include <benchmark/benchmark.h>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/runtime/dispatch.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

static void BM_matmul(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    const auto A = rand<double>(n, n, 42);
    const auto B = rand<double>(n, n, 43);

    for (auto _ : state) {
        auto result = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(n) * static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

BENCHMARK(BM_matmul)->Arg(64)->Arg(256)->Arg(512);
