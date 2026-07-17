// MathScript Signal Processing and Linear Algebra Benchmark Suite

#include <benchmark/benchmark.h>
#include <vector>
#include <cmath>
#include <numeric>

#include "ms/signal/signal.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// Signal: convolve
// ---------------------------------------------------------------------------

static void BM_Convolve(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> a(n), b(n / 4 + 1);
    for (int i = 0; i < n; ++i) a[i] = std::sin(2.0 * M_PI * i / n);
    for (int i = 0; i < static_cast<int>(b.size()); ++i) b[i] = 1.0 / b.size();
    for (auto _ : state) {
        auto r = convolve(a, b);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Convolve)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

static void BM_ConvolveFFT(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> a(n), b(n / 4 + 1);
    for (int i = 0; i < n; ++i) a[i] = std::sin(2.0 * M_PI * i / n);
    for (int i = 0; i < static_cast<int>(b.size()); ++i) b[i] = 1.0 / b.size();
    for (auto _ : state) {
        auto r = convolve(a, b);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_ConvolveFFT)->Arg(4096)->Arg(16384);

// ---------------------------------------------------------------------------
// Signal: correlate
// ---------------------------------------------------------------------------

static void BM_Correlate(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = std::sin(2.0 * M_PI * i / n);
        b[i] = std::cos(2.0 * M_PI * i / n);
    }
    for (auto _ : state) {
        auto r = correlate(a, b);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Correlate)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// ---------------------------------------------------------------------------
// Signal: moving_average
// ---------------------------------------------------------------------------

static void BM_MovingAverage(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = static_cast<double>(i);
    const int window = n / 8;
    for (auto _ : state) {
        auto r = moving_average(x, window);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_MovingAverage)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536);

// ---------------------------------------------------------------------------
// Signal: butterworth filter
// ---------------------------------------------------------------------------

static void BM_Butterworth(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = std::sin(2.0 * M_PI * i / n);
    for (auto _ : state) {
        auto r = lowpass(x, 0.3, 1.0);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Butterworth)->Arg(256)->Arg(1024)->Arg(4096);

// ---------------------------------------------------------------------------
// Signal: filter (direct-form II transposed IIR/FIR)
// ---------------------------------------------------------------------------

static void BM_Filter8192(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto coeffs = cheby1(4, 1.0, 1000.0, 8000.0, FilterType::Lowpass);
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * 500.0 * static_cast<double>(i) / 8000.0);
    }
    for (auto _ : state) {
        auto y = filter(coeffs.b, coeffs.a, x);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}
BENCHMARK(BM_Filter8192)->Arg(8192);

// ---------------------------------------------------------------------------
// Linear Algebra: solve Ax=b
// ---------------------------------------------------------------------------

static DMatrix make_spd(int n) {
    DMatrix A(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = n * 2.0;  // dominant diagonal
        for (int j = 0; j < i; ++j) {
            A(i, j) = A(j, i) = 1.0;
        }
    }
    return A;
}

static void BM_Solve(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DMatrix A = make_spd(n);
    DMatrix b(n, 1);
    for (int i = 0; i < n; ++i) b(i, 0) = 1.0;
    for (auto _ : state) {
        auto r = solve(A, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Solve)->Arg(4)->Arg(16)->Arg(32)->Arg(64);

// ---------------------------------------------------------------------------
// Linear Algebra: LU decomposition
// ---------------------------------------------------------------------------

static void BM_LU(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DMatrix A = make_spd(n);
    for (auto _ : state) {
        auto r = lu(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_LU)->Arg(4)->Arg(16)->Arg(32)->Arg(64);

// ---------------------------------------------------------------------------
// Linear Algebra: QR decomposition
// ---------------------------------------------------------------------------

static void BM_QR(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DMatrix A = make_spd(n);
    for (auto _ : state) {
        auto r = qr(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_QR)->Arg(4)->Arg(16)->Arg(32)->Arg(64);

// ---------------------------------------------------------------------------
// Linear Algebra: Cholesky decomposition
// ---------------------------------------------------------------------------

static void BM_Chol(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DMatrix A = make_spd(n);
    for (auto _ : state) {
        auto r = chol(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Chol)->Arg(4)->Arg(16)->Arg(32)->Arg(64);

// ---------------------------------------------------------------------------
// Linear Algebra: det and trace
// ---------------------------------------------------------------------------

static void BM_Det(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DMatrix A = make_spd(n);
    for (auto _ : state) {
        auto r = det(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Det)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

static void BM_Trace(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DMatrix A = make_spd(n);
    for (auto _ : state) {
        const double r = trace(A).value();
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Trace)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK_MAIN();
