// MathScript SIMD benchmark suite

#include <benchmark/benchmark.h>
#include <numeric>
#include <vector>

#include "ms/simd/simd.hpp"

using namespace ms::simd;

static std::vector<double> make_vec(int n, double start = 1.0) {
    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

// ---------------------------------------------------------------------------
// simd::add
// ---------------------------------------------------------------------------

static void BM_simd_add(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto a = make_vec(n);
    const auto b = make_vec(n, 0.5);
    std::vector<double> out(n);
    for (auto _ : state) {
        add(a, b, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_add)->Arg(8)->Arg(64)->Arg(512)->Arg(4096);

// ---------------------------------------------------------------------------
// simd::mul
// ---------------------------------------------------------------------------

static void BM_simd_mul(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto a = make_vec(n);
    const auto b = make_vec(n, 0.5);
    std::vector<double> out(n);
    for (auto _ : state) {
        mul(a, b, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_mul)->Arg(8)->Arg(64)->Arg(512)->Arg(4096);

// ---------------------------------------------------------------------------
// simd::sub
// ---------------------------------------------------------------------------

static void BM_simd_sub(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto a = make_vec(n);
    const auto b = make_vec(n, 0.5);
    std::vector<double> out(n);
    for (auto _ : state) {
        sub(a, b, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
// 1M elements ~4ms/run; 65536 keeps smoke under per-bench budget.
BENCHMARK(BM_simd_sub)->Arg(65536);

// ---------------------------------------------------------------------------
// simd::abs
// ---------------------------------------------------------------------------

static void BM_simd_abs(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) - static_cast<double>(n) / 2.0;
    }
    std::vector<double> out(n);
    for (auto _ : state) {
        abs(x, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_abs)->Arg(65536);

// ---------------------------------------------------------------------------
// simd::scale
// ---------------------------------------------------------------------------

static void BM_simd_scale(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto x = make_vec(n);
    std::vector<double> out(n);
    for (auto _ : state) {
        scale(2.5, x, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_scale)->Arg(8)->Arg(64)->Arg(512)->Arg(4096);

// ---------------------------------------------------------------------------
// simd::axpy
// ---------------------------------------------------------------------------

static void BM_simd_axpy(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto x = make_vec(n);
    std::vector<double> y = make_vec(n, 0.0);
    for (auto _ : state) {
        axpy(0.5, x, y);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_axpy)->Arg(8)->Arg(64)->Arg(512)->Arg(4096);

// ---------------------------------------------------------------------------
// simd::dot
// ---------------------------------------------------------------------------

static void BM_simd_dot(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto a = make_vec(n);
    const auto b = make_vec(n, 0.5);
    for (auto _ : state) {
        const double result = dot(a, b);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_dot)->Arg(8)->Arg(64)->Arg(512)->Arg(4096)->Arg(65536);

// ---------------------------------------------------------------------------
// simd::sum
// ---------------------------------------------------------------------------

static void BM_simd_sum(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto x = make_vec(n);
    for (auto _ : state) {
        const double result = sum(x);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_sum)->Arg(8)->Arg(64)->Arg(512)->Arg(4096)->Arg(65536);

// ---------------------------------------------------------------------------
// simd::sum_squares
// ---------------------------------------------------------------------------

static void BM_simd_sum_squares(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto x = make_vec(n);
    for (auto _ : state) {
        const double result = sum_squares(x);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_sum_squares)->Arg(8)->Arg(64)->Arg(512)->Arg(4096)->Arg(65536);

// ---------------------------------------------------------------------------
// simd::norm_l2
// ---------------------------------------------------------------------------

static void BM_simd_norm_l2(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto x = make_vec(n);
    for (auto _ : state) {
        const double result = norm_l2(x);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_norm_l2)->Arg(65536);

// ---------------------------------------------------------------------------
// simd::exp_map
// ---------------------------------------------------------------------------

static void BM_simd_exp_map(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = 0.001 * static_cast<double>(i);
    std::vector<double> out(n);
    for (auto _ : state) {
        exp_map(x, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_simd_exp_map)->Arg(8)->Arg(64)->Arg(512)->Arg(4096);

// ---------------------------------------------------------------------------
// simd::gemv
// ---------------------------------------------------------------------------

static void BM_simd_gemv(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto A = make_vec(n * n);
    const auto x = make_vec(n);
    for (auto _ : state) {
        const auto y = gemv(A, static_cast<size_t>(n), static_cast<size_t>(n), x);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * n * n);
}
BENCHMARK(BM_simd_gemv)->Arg(8)->Arg(64)->Arg(512);

BENCHMARK_MAIN();
