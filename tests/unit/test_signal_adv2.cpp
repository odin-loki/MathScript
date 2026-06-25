// MathScript: Advanced Signal Processing Tests (Wave 43)
// Tests for butterworth, lowpass, highpass, bandpass, window functions

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/signal/signal.hpp"

using namespace ms;

static double signal_energy(const std::vector<double>& x) {
    double e = 0.0;
    for (double v : x) e += v * v;
    return e;
}

// ---------------------------------------------------------------------------
// butterworth filter
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Butterworth_OutputSize_MatchesInput) {
    std::vector<double> x(64, 1.0);
    auto y = butterworth(x, 0.1, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalAdv2, Butterworth_AllFinite) {
    std::vector<double> x(64);
    for (size_t i = 0; i < 64; ++i)
        x[i] = std::sin(2.0 * M_PI * 0.1 * i);
    auto y = butterworth(x, 0.2, 1.0);
    for (auto v : y) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalAdv2, Butterworth_ZeroInput_ZeroOutput) {
    std::vector<double> x(32, 0.0);
    auto y = butterworth(x, 0.2, 1.0);
    for (auto v : y) EXPECT_NEAR(v, 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// lowpass filter
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Lowpass_OutputSize_MatchesInput) {
    std::vector<double> x(64, 1.0);
    auto y = lowpass(x, 0.3, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalAdv2, Lowpass_AllFinite) {
    std::vector<double> x(64);
    for (size_t i = 0; i < 64; ++i)
        x[i] = std::sin(2.0 * M_PI * 0.05 * i);
    auto y = lowpass(x, 0.1, 1.0);
    for (auto v : y) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalAdv2, Lowpass_ConstantInput_ProducesOutput) {
    // DC signal through a low-pass filter should pass through (or attenuate)
    std::vector<double> x(32, 1.0);
    auto y = lowpass(x, 0.4, 1.0);
    EXPECT_EQ(y.size(), x.size());
    for (auto v : y) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// highpass filter
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Highpass_OutputSize_MatchesInput) {
    std::vector<double> x(64, 1.0);
    auto y = highpass(x, 0.3, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalAdv2, Highpass_AllFinite) {
    std::vector<double> x(64);
    for (size_t i = 0; i < 64; ++i)
        x[i] = std::sin(2.0 * M_PI * 0.4 * i);
    auto y = highpass(x, 0.2, 1.0);
    for (auto v : y) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// bandpass filter
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Bandpass_OutputSize_MatchesInput) {
    std::vector<double> x(64, 0.0);
    auto y = bandpass(x, 0.1, 0.3, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalAdv2, Bandpass_AllFinite) {
    std::vector<double> x(64);
    for (size_t i = 0; i < 64; ++i)
        x[i] = std::sin(2.0 * M_PI * 0.2 * i);
    auto y = bandpass(x, 0.1, 0.3, 1.0);
    for (auto v : y) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// Window functions: hamming
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Hamming_CorrectSize) {
    for (size_t n : {8u, 16u, 32u, 64u}) {
        auto w = hamming(n);
        EXPECT_EQ(w.size(), n);
    }
}

TEST(SignalAdv2, Hamming_EndpointsNearZero) {
    // Hamming window: w(0) = 0.08, w(N-1) = 0.08 (not exactly 0)
    auto w = hamming(64);
    EXPECT_NEAR(w[0], w[w.size() - 1], 1e-9) << "Hamming should be symmetric";
    EXPECT_LT(w[0], 0.2) << "Hamming endpoints should be small";
}

TEST(SignalAdv2, Hamming_PeakAtCenter) {
    size_t n = 65;
    auto w = hamming(n);
    double center_val = w[n / 2];
    EXPECT_NEAR(center_val, 1.0, 0.01) << "Hamming peak should be near 1.0 at center";
}

TEST(SignalAdv2, Hamming_AllPositive) {
    auto w = hamming(32);
    for (auto v : w) EXPECT_GE(v, 0.0);
}

TEST(SignalAdv2, Hamming_Symmetric) {
    size_t n = 33;
    auto w = hamming(n);
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], 1e-9) << "Hamming not symmetric at i=" << i;
}

// ---------------------------------------------------------------------------
// Window functions: hanning
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Hanning_CorrectSize) {
    auto w = hanning(32);
    EXPECT_EQ(w.size(), 32u);
}

TEST(SignalAdv2, Hanning_EndpointsNearZero) {
    auto w = hanning(64);
    EXPECT_NEAR(w[0], 0.0, 0.01) << "Hanning(0) should be near 0";
    EXPECT_NEAR(w[w.size() - 1], 0.0, 0.01) << "Hanning(N-1) should be near 0";
}

TEST(SignalAdv2, Hanning_AllNonnegative) {
    auto w = hanning(32);
    for (auto v : w) EXPECT_GE(v, -1e-10);
}

TEST(SignalAdv2, Hanning_Symmetric) {
    size_t n = 33;
    auto w = hanning(n);
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], 1e-9);
}

// ---------------------------------------------------------------------------
// Window functions: blackman
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Blackman_CorrectSize) {
    auto w = blackman(32);
    EXPECT_EQ(w.size(), 32u);
}

TEST(SignalAdv2, Blackman_EndpointsNearZero) {
    auto w = blackman(64);
    EXPECT_NEAR(w[0], 0.0, 0.01);
    EXPECT_NEAR(w[w.size() - 1], 0.0, 0.01);
}

TEST(SignalAdv2, Blackman_PeakAtCenter) {
    size_t n = 65;
    auto w = blackman(n);
    EXPECT_NEAR(w[n / 2], 1.0, 0.01);
}

TEST(SignalAdv2, Blackman_Symmetric) {
    size_t n = 33;
    auto w = blackman(n);
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], 1e-9);
}

// ---------------------------------------------------------------------------
// Window functions: parzen
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Parzen_CorrectSize) {
    auto w = parzen(32);
    EXPECT_EQ(w.size(), 32u);
}

TEST(SignalAdv2, Parzen_AllNonnegative) {
    auto w = parzen(32);
    for (auto v : w) EXPECT_GE(v, -1e-10);
}

TEST(SignalAdv2, Parzen_PeakAtCenter) {
    size_t n = 65;
    auto w = parzen(n);
    double center = w[n / 2];
    // Values near center should be larger than at edges
    EXPECT_GT(center, w[0]);
}

// ---------------------------------------------------------------------------
// correlate
// ---------------------------------------------------------------------------

TEST(SignalAdv2, Correlate_SelfCorrelation_HasPeakAtZeroLag) {
    std::vector<double> x = {1.0, 2.0, 3.0, 2.0, 1.0};
    auto r = correlate(x, x);
    EXPECT_GE(r.size(), 1u);
    // Self-correlation peak should be at center
    size_t center = r.size() / 2;
    double peak = r[center];
    for (size_t i = 0; i < r.size(); ++i) {
        if (i != center)
            EXPECT_LE(r[i], peak + 1e-9) << "Peak at center should be maximum";
    }
}

TEST(SignalAdv2, Correlate_AllFinite) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {3.0, 2.0, 1.0};
    auto r = correlate(a, b);
    for (auto v : r) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// moving_average
// ---------------------------------------------------------------------------

TEST(SignalAdv2, MovingAverage_ConstantSignal_Unchanged) {
    std::vector<double> x(20, 5.0);
    auto r = moving_average(x, 5);
    EXPECT_GE(r.size(), 1u);
    for (auto v : r) EXPECT_NEAR(v, 5.0, 1e-9);
}

TEST(SignalAdv2, MovingAverage_Window1_IsIdentity) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto r = moving_average(x, 1);
    EXPECT_EQ(r.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(r[i], x[i], 1e-9);
}
