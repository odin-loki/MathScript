// MathScript Special Functions and Memory Allocator Benchmark Suite

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/special/special.hpp"
#include "ms/memory/aligned_allocator.hpp"
#include "ms/memory/pool_allocator.hpp"

// ---------------------------------------------------------------------------
// Special: erf(x) over a range of x values
// ---------------------------------------------------------------------------

static void BM_Erf(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> xs(n);
    for (int i = 0; i < n; ++i)
        xs[i] = -3.0 + 6.0 * i / (n > 1 ? n - 1 : 1);
    for (auto _ : state) {
        for (int i = 0; i < n; ++i) {
            double r = ms::erf(xs[i]);
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Erf)->Arg(64)->Arg(256)->Arg(1024);

// ---------------------------------------------------------------------------
// Special: bessel_j0(x) for various x
// ---------------------------------------------------------------------------

static void BM_BesselJ0(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> xs(n);
    for (int i = 0; i < n; ++i)
        xs[i] = 0.1 + 20.0 * i / n;
    for (auto _ : state) {
        for (int i = 0; i < n; ++i) {
            double r = ms::bessel_j0(xs[i]);
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_BesselJ0)->Arg(64)->Arg(256)->Arg(1024);

// ---------------------------------------------------------------------------
// Special: legendre_pn(n, m, x) with m=0 for orders n=5 and n=10
// ---------------------------------------------------------------------------

static void BM_LegendreP(benchmark::State& state) {
    const int order  = static_cast<int>(state.range(0));
    const int n_pts  = 256;
    std::vector<double> xs(n_pts);
    for (int i = 0; i < n_pts; ++i)
        xs[i] = -1.0 + 2.0 * i / (n_pts - 1);
    for (auto _ : state) {
        for (int i = 0; i < n_pts; ++i) {
            double r = ms::legendre_pn(order, 0, xs[i]);
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pts);
}
BENCHMARK(BM_LegendreP)->Arg(5)->Arg(10);

// ---------------------------------------------------------------------------
// Special: gamma_func(x) for various x
// ---------------------------------------------------------------------------

static void BM_GammaFunc(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> xs(n);
    for (int i = 0; i < n; ++i)
        xs[i] = 0.5 + 10.0 * i / n;
    for (auto _ : state) {
        for (int i = 0; i < n; ++i) {
            double r = ms::gamma_func(xs[i]);
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_GammaFunc)->Arg(64)->Arg(256)->Arg(1024);

// ---------------------------------------------------------------------------
// Memory: aligned allocate/deallocate of 1 KB with 64-byte alignment
// ---------------------------------------------------------------------------

static void BM_AlignedAlloc(benchmark::State& state) {
    constexpr size_t kSize  = 1024;
    constexpr size_t kAlign = 64;
    for (auto _ : state) {
        void* ptr = ms::memory::aligned_alloc(kAlign, kSize);
        benchmark::DoNotOptimize(ptr);
        ms::memory::aligned_free(ptr);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kSize);
}
BENCHMARK(BM_AlignedAlloc);

// ---------------------------------------------------------------------------
// Memory: pool allocator allocate/deallocate cycle (64-byte block size)
// ---------------------------------------------------------------------------

static void BM_PoolAllocate(benchmark::State& state) {
    ms::memory::PoolAllocator<64> pool;
    for (auto _ : state) {
        void* ptr = pool.allocate();
        benchmark::DoNotOptimize(ptr);
        pool.deallocate(ptr);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_PoolAllocate);

BENCHMARK_MAIN();
