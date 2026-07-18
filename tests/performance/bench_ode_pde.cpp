// MathScript ODE / PDE Numerical Benchmark Suite

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/ode/ode.hpp"
#include "ms/pde/pde.hpp"
#include "ms/cfd/cfd.hpp"

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
// ODE: RK4 vector method - coupled linear system
// ---------------------------------------------------------------------------

static void BM_ODE_RK4Vec(benchmark::State& state) {
    const size_t dim  = static_cast<size_t>(state.range(0));
    const size_t steps = static_cast<size_t>(state.range(1));
    std::vector<double> y0(dim, 1.0);
    auto f = [dim](double /*t*/, const std::vector<double>& y) {
        std::vector<double> dy(dim);
        for (size_t i = 0; i + 1 < dim; ++i) {
            dy[i] = y[i + 1];
        }
        if (dim > 0) {
            dy[dim - 1] = -y[0];
        }
        return dy;
    };
    for (auto _ : state) {
        auto result = ode_rk4_vec(f, 0.0, y0, 5.0, steps);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(steps) *
                            static_cast<int64_t>(dim));
}
BENCHMARK(BM_ODE_RK4Vec)->Args({8, 1000})->Args({32, 1000});

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

// ---------------------------------------------------------------------------
// CFD: 1-D upwind advection time integration
// ---------------------------------------------------------------------------

static void BM_CFD_Advection(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    const double dx = 1.0 / static_cast<double>(n);
    const double dt = 0.4 * dx;
    const double t_end = 1.0;
    std::vector<double> u0(n, 0.0);
    for (size_t i = n / 4; i < 3 * n / 4; ++i) {
        u0[i] = 1.0;
    }
    const std::vector<double> v(n, 1.0);
    for (auto _ : state) {
        auto result = ms::cfd::run_advection(u0, v, t_end, dt, dx);
        benchmark::DoNotOptimize(result);
    }
    const int64_t steps = static_cast<int64_t>(std::ceil(t_end / dt));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) * steps);
}
BENCHMARK(BM_CFD_Advection)->Arg(100)->Arg(500)->Arg(2000);

// ---------------------------------------------------------------------------
// CFD: 2-D upwind advection time integration
// ---------------------------------------------------------------------------

static void BM_CfdAdvection2D(benchmark::State& state) {
    const auto grid = ms::cfd::grid2d(0.0, 1.0, 0.0, 1.0, 64, 64);
    const auto u0 = ms::cfd::square_pulse_2d(grid, 0.5, 0.5, 0.25, 0.25, 1.0);
    const auto vx = ms::cfd::constant_velocity(grid.nx * grid.ny, 1.0);
    const auto vy = ms::cfd::constant_velocity(grid.nx * grid.ny, 0.0);
    const double t_end = 0.4;
    const double dt = 0.4 * grid.dx;
    for (auto _ : state) {
        auto result = ms::cfd::run_advection_2d(
            u0,
            vx,
            vy,
            t_end,
            dt,
            grid.dx,
            grid.dy,
            ms::cfd::BoundaryCondition::Periodic,
            ms::cfd::BoundaryCondition::Periodic);
        benchmark::DoNotOptimize(result);
    }
    const int64_t steps = static_cast<int64_t>(std::ceil(t_end / dt));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(grid.nx) *
                            static_cast<int64_t>(grid.ny) * steps);
}
BENCHMARK(BM_CfdAdvection2D);

// ---------------------------------------------------------------------------
// CFD: 2-D single upwind FVM advection step
// ---------------------------------------------------------------------------

static void BM_CfdUpwindStep2D(benchmark::State& state) {
    const auto grid = ms::cfd::grid2d(0.0, 1.0, 0.0, 1.0, 64, 64);
    const auto u0 = ms::cfd::square_pulse_2d(grid, 0.5, 0.5, 0.25, 0.25, 1.0);
    const auto vx = ms::cfd::constant_velocity(grid.nx * grid.ny, 1.0);
    const auto vy = ms::cfd::constant_velocity(grid.nx * grid.ny, 0.0);
    const double dt = 0.4 * grid.dx;
    for (auto _ : state) {
        auto result = ms::cfd::upwind_fvm_advection_2d(
            u0,
            vx,
            vy,
            dt,
            grid.dx,
            grid.dy,
            ms::cfd::BoundaryCondition::Periodic,
            ms::cfd::BoundaryCondition::Periodic);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(grid.nx) *
                            static_cast<int64_t>(grid.ny));
}
BENCHMARK(BM_CfdUpwindStep2D);

BENCHMARK_MAIN();
