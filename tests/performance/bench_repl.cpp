// MathScript REPL interpreter benchmarks

#include "ms/interp/repl_engine.hpp"
#include <benchmark/benchmark.h>

using namespace ms::interp;

static void BM_repl_scalar_expr(benchmark::State& state) {
    Interpreter interp;
    constexpr int kIterations = 1000;

    for (auto _ : state) {
        for (int i = 0; i < kIterations; ++i) {
            auto result = interp.execute("x = sin(1.0) + cos(2.0) * sqrt(3.0)");
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations() * kIterations);
}

static void BM_repl_matrix_assign(benchmark::State& state) {
    Interpreter interp;
    constexpr int kIterations = 100;

    for (auto _ : state) {
        for (int i = 0; i < kIterations; ++i) {
            auto assign = interp.execute("M = [3, 1; 1, 2]");
            benchmark::DoNotOptimize(assign);
            auto det_result = interp.execute("d = det(M)");
            benchmark::DoNotOptimize(det_result);
        }
    }

    state.SetItemsProcessed(state.iterations() * kIterations * 2);
}

BENCHMARK(BM_repl_scalar_expr);
BENCHMARK(BM_repl_matrix_assign);
