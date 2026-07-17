// MathScript RNG and Dispatch Benchmarks

#include <benchmark/benchmark.h>
#include "ms/core/rng.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// RNG benchmarks
// ---------------------------------------------------------------------------

static void BM_session_uniform(benchmark::State& state) {
    set_session_rng(
        []() -> double {
            static unsigned s = 12345u;
            s ^= (s << 13); s ^= (s >> 17); s ^= (s << 5);
            return static_cast<double>(s) / 4294967296.0;
        },
        []() -> double {
            static unsigned s = 98765u;
            s ^= (s << 13); s ^= (s >> 17); s ^= (s << 5);
            return static_cast<double>(s & 0x7FFFFFFFu) / 2147483648.0 - 1.0;
        });

    for (auto _ : state) {
        const double v = session_uniform();
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations());
    clear_session_rng();
}

static void BM_session_normal(benchmark::State& state) {
    set_session_rng(
        []() -> double {
            static unsigned s = 12345u;
            s ^= (s << 13); s ^= (s >> 17); s ^= (s << 5);
            return static_cast<double>(s) / 4294967296.0;
        },
        []() -> double {
            static unsigned s = 11111u;
            s ^= (s << 13); s ^= (s >> 17); s ^= (s << 5);
            return static_cast<double>(s & 0x7FFFFFFFu) / 2147483648.0 - 1.0;
        });

    for (auto _ : state) {
        const double v = session_normal();
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations());
    clear_session_rng();
}

static void BM_rng_active_check(benchmark::State& state) {
    set_session_rng(
        []() -> double { return 0.5; },
        []() -> double { return 0.0; });

    for (auto _ : state) {
        const bool active = session_rng_active();
        benchmark::DoNotOptimize(active);
    }
    clear_session_rng();
}

// ---------------------------------------------------------------------------
// Dispatch benchmarks
// ---------------------------------------------------------------------------

static void BM_dispatch_decide_cpu(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        const auto d = decide(n, ExecPolicy::CPU);
        benchmark::DoNotOptimize(d.n_threads);
    }
    state.SetItemsProcessed(state.iterations());
}

static void BM_dispatch_decide_auto(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        const auto d = decide(n, ExecPolicy::AUTO);
        benchmark::DoNotOptimize(d.backend);
    }
    state.SetItemsProcessed(state.iterations());
}

static void BM_dispatch_decide_with_topology(benchmark::State& state) {
    const SystemTopology topo = detect_topology();
    const auto n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        const auto d = decide(n, ExecPolicy::AUTO, topo);
        benchmark::DoNotOptimize(d.n_threads);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_session_uniform);
BENCHMARK(BM_session_normal);
BENCHMARK(BM_rng_active_check);
BENCHMARK(BM_dispatch_decide_cpu)->Arg(1024)->Arg(65536);
BENCHMARK(BM_dispatch_decide_auto)->Arg(1024)->Arg(65536);
BENCHMARK(BM_dispatch_decide_with_topology)->Arg(1024)->Arg(65536);
