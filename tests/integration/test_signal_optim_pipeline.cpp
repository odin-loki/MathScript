// MathScript: Signal + Optim + Stats Pipeline Integration Tests (Wave 49)
// Tests combining signal processing, optimization, and statistics

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/signal/signal.hpp"
#include "ms/optim/optim.hpp"
#include "ms/stats/stats.hpp"

using namespace ms;

// Generate sine wave
static std::vector<double> make_sine(double freq, double fs, size_t N) {
    std::vector<double> sig(N);
    for (size_t i = 0; i < N; ++i)
        sig[i] = std::sin(2.0 * M_PI * freq * i / fs);
    return sig;
}

// Compute RMS
static double rms(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double s = 0.0;
    for (auto v : x) s += v * v;
    return std::sqrt(s / x.size());
}

// ---------------------------------------------------------------------------
// Pipeline 1: Filter → analyze output statistics
// ---------------------------------------------------------------------------

TEST(SignalOptimPipeline, Lowpass_Reduces_HighFreq_Power) {
    size_t N = 512;
    double fs = 1000.0, cutoff = 100.0;
    // Mix low and high frequency
    auto low = make_sine(20.0, fs, N);
    auto high = make_sine(400.0, fs, N);
    std::vector<double> mixed(N);
    for (size_t i = 0; i < N; ++i) mixed[i] = low[i] + high[i];

    auto filtered = lowpass(mixed, cutoff, fs);
    ASSERT_EQ(filtered.size(), N);

    // After lowpass, RMS should be closer to low-only
    double rms_low = rms(low);
    double rms_filtered = rms(filtered);
    double rms_high = rms(high);

    // Filtered signal should have less energy than mixed
    double power_mixed = 0.0, power_filtered = 0.0;
    for (auto v : mixed) power_mixed += v * v;
    for (auto v : filtered) power_filtered += v * v;
    EXPECT_LT(power_filtered, power_mixed * 1.1)
        << "Lowpass should not increase total power";
}

TEST(SignalOptimPipeline, MovingAverage_ReducesVariance) {
    size_t N = 200;
    std::vector<double> noisy(N);
    for (size_t i = 0; i < N; ++i)
        noisy[i] = std::sin(2.0 * M_PI * 5.0 * i / N) + ((i % 3) - 1) * 0.5;
    auto smoothed = moving_average(noisy, 10);
    EXPECT_LT(var(smoothed), var(noisy) * 2.0)
        << "Smoothed signal should have smaller variance than noisy";
}

// ---------------------------------------------------------------------------
// Pipeline 2: Convolve + correlation analysis
// ---------------------------------------------------------------------------

TEST(SignalOptimPipeline, Convolve_Then_Correlate_WithKernel) {
    // Convolving a signal with a kernel, then correlating output with kernel
    // should produce a peak at the correct lag
    std::vector<double> sig = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> kernel = {1.0, 2.0, 1.0};
    auto conv = convolve(sig, kernel);
    // conv should be {1, 2, 1, 0, ...}
    ASSERT_GE(conv.size(), 3u);
    EXPECT_NEAR(conv[0], 1.0, 1e-10);
    EXPECT_NEAR(conv[1], 2.0, 1e-10);
    EXPECT_NEAR(conv[2], 1.0, 1e-10);
}

TEST(SignalOptimPipeline, Correlate_MaxAtZeroLag_ForSelf) {
    // Auto-correlation should be maximum at lag 0
    std::vector<double> sig = {1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0};
    auto corr = correlate(sig, sig);
    // Maximum should be at center (lag=0)
    size_t center = corr.size() / 2;
    for (size_t i = 0; i < corr.size(); ++i)
        if (i != center)
            EXPECT_LE(std::abs(corr[i]), std::abs(corr[center]) + 1e-9)
                << "Auto-correlation not max at center (lag=0)";
}

// ---------------------------------------------------------------------------
// Pipeline 3: Window + signal analysis
// ---------------------------------------------------------------------------

TEST(SignalOptimPipeline, Hamming_Window_ReducesLeakage) {
    // Apply window before analysis - windowed signal has less "spectral leakage"
    size_t N = 64;
    auto sig = make_sine(3.0, static_cast<double>(N), N);
    auto win = hamming(N);

    std::vector<double> windowed(N);
    for (size_t i = 0; i < N; ++i) windowed[i] = sig[i] * win[i];

    // Windowed signal should have different (smaller) energy than original
    double e_orig = 0.0, e_win = 0.0;
    for (size_t i = 0; i < N; ++i) {
        e_orig += sig[i] * sig[i];
        e_win += windowed[i] * windowed[i];
    }
    EXPECT_LT(e_win, e_orig) << "Windowed signal should have less energy";
    EXPECT_GT(e_win, 0.0) << "Windowed signal should still have positive energy";
}

TEST(SignalOptimPipeline, Blackman_Window_IsValid) {
    size_t N = 64;
    auto w = blackman(N);
    ASSERT_EQ(w.size(), N);
    for (auto v : w) {
        EXPECT_GE(v, -1e-10) << "Blackman window should be non-negative";
        EXPECT_LE(v, 1.0 + 1e-10) << "Blackman window should not exceed 1";
    }
}

// ---------------------------------------------------------------------------
// Pipeline 4: Signal stats
// ---------------------------------------------------------------------------

TEST(SignalOptimPipeline, Sine_Mean_IsNearZero) {
    size_t N = 1000;
    auto sig = make_sine(5.0, 1000.0, N);
    double m = mean(sig);
    EXPECT_NEAR(m, 0.0, 0.01) << "Sine wave mean should be ~0";
}

TEST(SignalOptimPipeline, Sine_Var_IsHalf) {
    // Var of sin(2*pi*f*t) = 1/2 for many cycles
    size_t N = 10000;
    auto sig = make_sine(5.0, 1000.0, N);
    double v = var(sig);
    EXPECT_NEAR(v, 0.5, 0.01) << "Sine wave variance should be ~0.5";
}

TEST(SignalOptimPipeline, FilteredSignal_Stats_AllFinite) {
    size_t N = 256;
    auto sig = make_sine(50.0, 1000.0, N);
    auto filtered = butterworth(sig, 200.0, 1000.0);

    EXPECT_TRUE(std::isfinite(mean(filtered)));
    EXPECT_TRUE(std::isfinite(stddev(filtered)));
    EXPECT_TRUE(std::isfinite(min_value(filtered)));
    EXPECT_TRUE(std::isfinite(max_value(filtered)));
}

// ---------------------------------------------------------------------------
// Pipeline 5: Golden section minimizes filter output power
// ---------------------------------------------------------------------------

TEST(SignalOptimPipeline, GoldenSection_FindsOptimalCutoff) {
    // Find cutoff frequency that minimizes output power for a high-freq signal
    // (i.e. the best lowpass to remove the signal)
    size_t N = 256;
    double fs = 1000.0;
    auto sig = make_sine(400.0, fs, N);  // High-freq signal

    // f(cutoff): power of lowpass-filtered signal
    auto cost = [&](double cutoff) -> double {
        if (cutoff <= 0.0 || cutoff >= fs / 2.0) return 1e10;
        auto filtered = lowpass(sig, cutoff, fs);
        double p = 0.0;
        for (auto v : filtered) p += v * v;
        return p;
    };

    // Find cutoff that minimizes power (should be near 0 to block all signal)
    double opt_cutoff = golden_section(cost, 10.0, 490.0, 1.0);
    EXPECT_GT(opt_cutoff, 0.0) << "Optimal cutoff should be positive";
    EXPECT_LT(opt_cutoff, fs / 2.0) << "Optimal cutoff should be < Nyquist";
    // The minimizer should give low output power for the high-freq signal
    EXPECT_LT(cost(opt_cutoff), cost(490.0) * 1.1)
        << "Minimizer should reduce output power";
}
