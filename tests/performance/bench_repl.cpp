// MathScript REPL interpreter benchmarks

#include "ms/interp/repl_engine.hpp"
#include <benchmark/benchmark.h>

using namespace ms::interp;

static void BM_repl_simple_arith(benchmark::State& state) {
    Interpreter interp;
    constexpr int kIterations = 10000;

    for (auto _ : state) {
        for (int i = 0; i < kIterations; ++i) {
            auto result = interp.execute("r = 1+2*3");
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations() * kIterations);
}

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

static void BM_repl_zeros_eye(benchmark::State& state) {
    Interpreter interp;
    constexpr int kIterations = 100;

    for (auto _ : state) {
        for (int i = 0; i < kIterations; ++i) {
            auto r1 = interp.execute("Z = zeros(4)");
            benchmark::DoNotOptimize(r1);
            auto r2 = interp.execute("I = eye(3)");
            benchmark::DoNotOptimize(r2);
        }
    }

    state.SetItemsProcessed(state.iterations() * kIterations * 2);
}

static void BM_repl_matmul_pipeline(benchmark::State& state) {
    Interpreter interp;
    interp.execute("A = [2,1;1,3]");
    interp.execute("B = [1,0;0,1]");
    constexpr int kIterations = 100;

    for (auto _ : state) {
        for (int i = 0; i < kIterations; ++i) {
            auto r = interp.execute("C = matmul(A, B)");
            benchmark::DoNotOptimize(r);
        }
    }

    state.SetItemsProcessed(state.iterations() * kIterations);
}

BENCHMARK(BM_repl_simple_arith);
BENCHMARK(BM_repl_scalar_expr);
BENCHMARK(BM_repl_matrix_assign);
BENCHMARK(BM_repl_zeros_eye);
BENCHMARK(BM_repl_matmul_pipeline);
