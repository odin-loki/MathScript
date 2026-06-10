// MathScript ODE / PDE Numerical Benchmark Suite

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/ode/ode.hpp"
#include "ms/pde/pde.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// ODE: Euler method - exponential decay
// ---------------------------------------------------------------------------

static void BM_ODE_Euler(benchmark::State& state) {
    const size_t steps = static_cast<size_t>(state.range(0));
    auto f = [](double /*t*/, double y) { return -y; };
    for (auto _ : state) {
        auto result = ode_euler(f, 0.0, 1.0, 5.0, steps);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(steps));
}
BENCHMARK(BM_ODE_Euler)->Arg(100)->Arg(1000)->Arg(10000);

// ---------------------------------------------------------------------------
// ODE: RK4 method - exponential decay
// ---------------------------------------------------------------------------

static void BM_ODE_RK4(benchmark::State& state) {
    const size_t steps = static_cast<size_t>(state.range(0));
    auto f = [](double /*t*/, double y) { return -y; };
    for (auto _ : state) {
        auto result = ode_rk4(f, 0.0, 1.0, 5.0, steps);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(steps));
}
BENCHMARK(BM_ODE_RK4)->Arg(100)->Arg(1000)->Arg(10000);

// ---------------------------------------------------------------------------
// PDE: 1-D heat equation - varying grid size, fixed time steps
// ---------------------------------------------------------------------------

static void BM_PDE_Heat1D(benchmark::State& state) {
    const size_t nx    = static_cast<size_t>(state.range(0));
    const double dx    = 1.0 / static_cast<double>(nx + 1);
    const double alpha = 0.01;
    // CFL stability: alpha * dt / dx^2 <= 0.5  =>  dt <= 0.5 * dx^2 / alpha
    const double dt = 0.4 * dx * dx / alpha;
    std::vector<double> x0(nx, 0.0);
    for (size_t i = nx / 3; i < 2 * nx / 3; ++i) x0[i] = 1.0;
    for (auto _ : state) {
        auto result = pde_heat_1d(x0, alpha, dx, dt, 50);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(nx) * 50);
}
BENCHMARK(BM_PDE_Heat1D)->Arg(20)->Arg(100)->Arg(500)->Arg(2000);

BENCHMARK_MAIN();
