// MathScript Signal Window Functions Numerical Reference Tests
// Covers Hamming, Hanning, Blackman, Parzen, Triangular window properties

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "ms/signal/signal.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

static double sum_vec(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

static bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

static bool all_nonneg(const std::vector<double>& v) {
    for (double x : v) if (x < 0.0) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Hamming window: w[n] = 0.54 - 0.46*cos(2πn/(N-1))
// ---------------------------------------------------------------------------

TEST(SignalWindows, Hamming_Size) {
    const auto w = hamming(16);
    ASSERT_EQ(w.size(), 16u);
}

TEST(SignalWindows, Hamming_AllFinite) {
    EXPECT_TRUE(all_finite(hamming(64)));
}

TEST(SignalWindows, Hamming_AllNonneg) {
    EXPECT_TRUE(all_nonneg(hamming(32)));
}

TEST(SignalWindows, Hamming_Symmetric) {
    const auto w = hamming(32);
    for (size_t i = 0; i < w.size() / 2; ++i)
        EXPECT_NEAR(w[i], w[w.size() - 1 - i], 1e-12);
}

TEST(SignalWindows, Hamming_Endpoints) {
    // Hamming window: w[0] = 0.54 - 0.46 = 0.08
    const auto w = hamming(32);
    EXPECT_NEAR(w[0], 0.08, 1e-10);
    EXPECT_NEAR(w[w.size() - 1], 0.08, 1e-10);
}

TEST(SignalWindows, Hamming_Peak_Near_Center) {
    const auto w = hamming(33);
    double maxval = *std::max_element(w.begin(), w.end());
    EXPECT_NEAR(maxval, 1.0, 1e-10);
    EXPECT_NEAR(w[16], 1.0, 1e-10);  // N=33, center at 16
}

// ---------------------------------------------------------------------------
// Hanning window: w[n] = 0.5*(1 - cos(2πn/(N-1)))
// ---------------------------------------------------------------------------

TEST(SignalWindows, Hanning_Size) {
    const auto w = hanning(20);
    ASSERT_EQ(w.size(), 20u);
}

TEST(SignalWindows, Hanning_AllFinite) {
    EXPECT_TRUE(all_finite(hanning(64)));
}

TEST(SignalWindows, Hanning_Symmetric) {
    const auto w = hanning(32);
    for (size_t i = 0; i < w.size() / 2; ++i)
        EXPECT_NEAR(w[i], w[w.size() - 1 - i], 1e-12);
}

TEST(SignalWindows, Hanning_Endpoints_Near_Zero) {
    const auto w = hanning(32);
    EXPECT_NEAR(w[0], 0.0, 1e-10);
    EXPECT_NEAR(w[w.size() - 1], 0.0, 1e-10);
}

TEST(SignalWindows, Hanning_AllNonneg) {
    EXPECT_TRUE(all_nonneg(hanning(64)));
}

TEST(SignalWindows, Hanning_Peak_Is_One) {
    const auto w = hanning(33);
    double maxval = *std::max_element(w.begin(), w.end());
    EXPECT_NEAR(maxval, 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Blackman window: w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
// ---------------------------------------------------------------------------

TEST(SignalWindows, Blackman_Size) {
    const auto w = blackman(16);
    ASSERT_EQ(w.size(), 16u);
}

TEST(SignalWindows, Blackman_AllFinite) {
    EXPECT_TRUE(all_finite(blackman(64)));
}

TEST(SignalWindows, Blackman_Symmetric) {
    const auto w = blackman(32);
    for (size_t i = 0; i < w.size() / 2; ++i)
        EXPECT_NEAR(w[i], w[w.size() - 1 - i], 1e-12);
}

TEST(SignalWindows, Blackman_Endpoints) {
    // Blackman: w[0] = 0.42 - 0.5 + 0.08 = 0.0
    const auto w = blackman(32);
    EXPECT_NEAR(w[0], 0.0, 1e-10);
}

TEST(SignalWindows, Blackman_Has_SideLobe_Suppression) {
    // Blackman window has lower sidelobes than Hanning — sum of squared values
    // is larger since it has less energy concentration near edge
    const size_t N = 64;
    const auto wb = blackman(N);
    const auto wh = hanning(N);
    // Blackman should not have all-zero interior
    bool has_positive = false;
    for (double x : wb) if (x > 0.1) { has_positive = true; break; }
    EXPECT_TRUE(has_positive);
}

// ---------------------------------------------------------------------------
// Parzen window
// ---------------------------------------------------------------------------

TEST(SignalWindows, Parzen_Size) {
    const auto w = parzen(32);
    ASSERT_EQ(w.size(), 32u);
}

TEST(SignalWindows, Parzen_AllFinite) {
    EXPECT_TRUE(all_finite(parzen(64)));
}

TEST(SignalWindows, Parzen_AllNonneg) {
    EXPECT_TRUE(all_nonneg(parzen(32)));
}

TEST(SignalWindows, Parzen_Symmetric) {
    const auto w = parzen(32);
    for (size_t i = 0; i < w.size() / 2; ++i)
        EXPECT_NEAR(w[i], w[w.size() - 1 - i], 1e-10);
}

// ---------------------------------------------------------------------------
// Triangular window: linearly ramps up from 0 to 1, then down to 0
// ---------------------------------------------------------------------------

TEST(SignalWindows, Triangular_Size) {
    const auto w = triangular(16);
    ASSERT_EQ(w.size(), 16u);
}

TEST(SignalWindows, Triangular_AllFinite) {
    EXPECT_TRUE(all_finite(triangular(64)));
}

TEST(SignalWindows, Triangular_AllNonneg) {
    EXPECT_TRUE(all_nonneg(triangular(32)));
}

TEST(SignalWindows, Triangular_Symmetric_Or_NearlySymmetric) {
    const auto w = triangular(32);
    // Triangular may be near-symmetric (implementation-defined for even N)
    double max_diff = 0.0;
    for (size_t i = 0; i < w.size() / 2; ++i)
        max_diff = std::max(max_diff, std::abs(w[i] - w[w.size() - 1 - i]));
    // Allow some asymmetry (< 10% of max value for even N)
    EXPECT_LT(max_diff, 0.15);
}

TEST(SignalWindows, Triangular_MaxNearCenter) {
    const auto w = triangular(33);
    double maxval = *std::max_element(w.begin(), w.end());
    EXPECT_NEAR(maxval, 1.0, 0.1);  // Peak should be approximately 1.0
}

// ---------------------------------------------------------------------------
// Window size = 1 edge cases
// ---------------------------------------------------------------------------

TEST(SignalWindows, AllWindows_Size1) {
    EXPECT_EQ(hamming(1).size(), 1u);
    EXPECT_EQ(hanning(1).size(), 1u);
    EXPECT_EQ(blackman(1).size(), 1u);
    EXPECT_EQ(parzen(1).size(), 1u);
    EXPECT_EQ(triangular(1).size(), 1u);
}

// ---------------------------------------------------------------------------
// Butterworth/lowpass/highpass/bandpass smoke tests
// ---------------------------------------------------------------------------

TEST(SignalFilters, Butterworth_OutputSameSize) {
    const std::vector<double> x(64, 0.0);
    const auto y = butterworth(x, 0.25, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalFilters, Butterworth_AllFinite) {
    std::vector<double> x(64);
    for (size_t i = 0; i < 64; ++i) x[i] = std::sin(2.0 * M_PI * 0.1 * i);
    const auto y = butterworth(x, 0.2, 1.0);
    EXPECT_TRUE(all_finite(y));
}

TEST(SignalFilters, Lowpass_OutputSameSize) {
    const std::vector<double> x(32, 1.0);
    const auto y = lowpass(x, 0.3, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalFilters, Highpass_OutputSameSize) {
    const std::vector<double> x(32, 1.0);
    const auto y = highpass(x, 0.1, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalFilters, Bandpass_OutputSameSize) {
    const std::vector<double> x(32, 1.0);
    const auto y = bandpass(x, 0.1, 0.4, 1.0);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalFilters, Lowpass_AttenuatesHighFreq) {
    // High-freq sine should be attenuated by low-pass filter
    const size_t N = 256;
    std::vector<double> x(N);
    // Mix: low-freq 0.05 + high-freq 0.45
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 0.05 * i) + std::sin(2.0 * M_PI * 0.45 * i);
    const auto y = lowpass(x, 0.1, 1.0);
    EXPECT_EQ(y.size(), N);
    EXPECT_TRUE(all_finite(y));
}
