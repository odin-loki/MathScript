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
// Signal: conv2
// ---------------------------------------------------------------------------

static Matrix<double> make_conv2_matrix(int rows, int cols) {
    Matrix<double> m(static_cast<size_t>(rows), static_cast<size_t>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m(static_cast<size_t>(i), static_cast<size_t>(j)) =
                std::sin(0.13 * static_cast<double>(i)) *
                std::cos(0.07 * static_cast<double>(j));
        }
    }
    return m;
}

static void BM_Conv2(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int k = static_cast<int>(state.range(1));
    const auto A = make_conv2_matrix(n, n);
    const auto B = make_conv2_matrix(k, k);
    for (auto _ : state) {
        auto r = conv2(A, B);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
}
BENCHMARK(BM_Conv2)->Args({8, 3})->Args({16, 4})->Args({32, 5})->Args({64, 7})->Args({128, 9});

static Matrix<double> make_non_separable_kernel(int k) {
    Matrix<double> B(static_cast<size_t>(k), static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            B(static_cast<size_t>(i), static_cast<size_t>(j)) =
                0.2 * static_cast<double>((i + 1) * (j + 1)) +
                0.05 * static_cast<double>(i == j ? 1 : 0);
        }
    }
    return B;
}

static void BM_Conv2FFT(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int k = static_cast<int>(state.range(1));
    const auto A = make_conv2_matrix(n, n);
    const auto B = make_non_separable_kernel(k);
    for (auto _ : state) {
        auto r = conv2(A, B);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
}
BENCHMARK(BM_Conv2FFT)->Args({64, 7})->Args({128, 9})->Args({256, 11});

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
// Signal: xcorr (partial FFT lag extraction)
// ---------------------------------------------------------------------------

static void BM_Xcorr8192(benchmark::State& state) {
    constexpr int n = 8192;
    constexpr int max_lag = 64;
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[static_cast<size_t>(i)] = std::sin(2.0 * M_PI * i / n);
        b[static_cast<size_t>(i)] = std::cos(2.0 * M_PI * i / n);
    }
    for (auto _ : state) {
        auto r = xcorr(a, b, max_lag);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Xcorr8192);

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
// Signal: median_filter
// ---------------------------------------------------------------------------

static void BM_MedianFilter(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int window = static_cast<int>(state.range(1));
    std::vector<double> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = std::sin(2.0 * M_PI * i / 97.0) +
                                     0.3 * std::cos(2.0 * M_PI * i / 23.0);
    }
    for (auto _ : state) {
        auto r = median_filter(x, window);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_MedianFilter)->Args({4096, 3})->Args({4096, 5})->Args({4096, 7})
    ->Args({65536, 7})->Args({65536, 31})->Args({65536, 101});

// ---------------------------------------------------------------------------
// Signal: savgol
// ---------------------------------------------------------------------------

static void BM_Savgol65536(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int window = static_cast<int>(state.range(1));
    const int polyorder = static_cast<int>(state.range(2));
    std::vector<double> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = std::sin(2.0 * M_PI * i / 97.0) +
                                       0.3 * std::cos(2.0 * M_PI * i / 23.0);
    }
    for (auto _ : state) {
        auto r = savgol(x, window, polyorder);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Savgol65536)->Args({65536, 11, 3})->Args({65536, 21, 3})->Args({65536, 51, 5});

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
// Signal: cross-correlation (lag-limited)
// ---------------------------------------------------------------------------

static void BM_Xcorr(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int max_lag = n / 16;
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = std::sin(2.0 * M_PI * i / n);
        b[i] = std::cos(2.0 * M_PI * i / n);
    }
    for (auto _ : state) {
        auto r = xcorr(a, b, max_lag);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Xcorr)->Arg(4096)->Arg(65536);

// ---------------------------------------------------------------------------
// Signal: Savitzky-Golay smoothing
// ---------------------------------------------------------------------------

static void BM_Savgol(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * i / 97.0) + 0.05 * std::cos(2.0 * M_PI * i / 23.0);
    }
    for (auto _ : state) {
        auto r = savgol(x, 11, 3);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Savgol)->Arg(4096)->Arg(65536);

// ---------------------------------------------------------------------------
// Signal: magnitude-squared coherence (65536 samples)
// ---------------------------------------------------------------------------

static void BM_Coherence65536(benchmark::State& state) {
    constexpr double fs = 1000.0;
    constexpr size_t n = 65536;
    constexpr size_t segment_len = 256;
    std::vector<double> x(n), y(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * 50.0 * static_cast<double>(i) / fs);
        y[i] = 0.9 * x[i] + 0.1 * std::sin(2.0 * M_PI * 120.0 * static_cast<double>(i) / fs);
    }
    for (auto _ : state) {
        auto result = coherence(x, y, fs, segment_len, 0.5);
        benchmark::DoNotOptimize(result.has_value());
        if (result.has_value()) {
            benchmark::DoNotOptimize(result->coherence.data());
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_Coherence65536);

// ---------------------------------------------------------------------------
// Signal: periodogram
// ---------------------------------------------------------------------------

static void BM_Periodogram(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    constexpr double fs = 1000.0;
    std::vector<double> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] =
            std::sin(2.0 * M_PI * 50.0 * static_cast<double>(i) / fs) +
            0.2 * std::sin(2.0 * M_PI * 120.0 * static_cast<double>(i) / fs);
    }
    const auto window = hanning(x.size());
    for (auto _ : state) {
        auto result = periodogram(x, fs, window);
        benchmark::DoNotOptimize(result.has_value());
        if (result.has_value()) {
            benchmark::DoNotOptimize(result->power.data());
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}
BENCHMARK(BM_Periodogram)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536);

// ---------------------------------------------------------------------------
// Signal: STFT spectrogram (65536 samples)
// ---------------------------------------------------------------------------

static void BM_Spectrogram(benchmark::State& state) {
    constexpr double fs = 1000.0;
    constexpr size_t n = 65536;
    constexpr size_t segment_len = 256;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * 50.0 * static_cast<double>(i) / fs) +
               0.2 * std::sin(2.0 * M_PI * 120.0 * static_cast<double>(i) / fs);
    }
    for (auto _ : state) {
        auto result = spectrogram(x, fs, segment_len, 0.5);
        benchmark::DoNotOptimize(result.has_value());
        if (result.has_value()) {
            benchmark::DoNotOptimize(result->magnitude.data());
            benchmark::DoNotOptimize(result->times.data());
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_Spectrogram);

// ---------------------------------------------------------------------------
// Signal: Welch PSD (65536 samples)
// ---------------------------------------------------------------------------

static void BM_Welch65536(benchmark::State& state) {
    constexpr double fs = 1000.0;
    constexpr size_t n = 65536;
    constexpr size_t segment_len = 256;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * 50.0 * static_cast<double>(i) / fs) +
               0.2 * std::sin(2.0 * M_PI * 120.0 * static_cast<double>(i) / fs);
    }
    for (auto _ : state) {
        auto result = welch_psd(x, fs, segment_len, 0.5);
        benchmark::DoNotOptimize(result.has_value());
        if (result.has_value()) {
            benchmark::DoNotOptimize(result->power.data());
            benchmark::DoNotOptimize(result->frequencies.data());
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_Welch65536);

// ---------------------------------------------------------------------------
// Signal: autocorrelation (lag-limited)
// ---------------------------------------------------------------------------

static void BM_Autocorr(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int max_lag = n / 16;
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * i / n) + 0.3 * std::cos(2.0 * M_PI * i / 17.0);
    }
    for (auto _ : state) {
        auto r = autocorr(x, max_lag);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Autocorr)->Arg(4096)->Arg(65536);

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
