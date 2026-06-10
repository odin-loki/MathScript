// MathScript Signal Processing Numerical Reference Tests
// Verifies signal processing functions against known mathematical results

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/signal/signal.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Convolve: known results from math
// ---------------------------------------------------------------------------

TEST(NumericalSignal, Convolve_Delta_Identity) {
    // Convolving with [1] (Dirac delta) = identity
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> delta = {1.0};
    const auto result = convolve(x, delta);
    ASSERT_EQ(result.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(result[i], x[i], 1e-12);
}

TEST(NumericalSignal, Convolve_BoxFilter_Smoothing) {
    // convolve returns full linear convolution: size = len(a) + len(b) - 1
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> box = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    const auto result = convolve(x, box);
    ASSERT_EQ(result.size(), x.size() + box.size() - 1);  // 5 + 3 - 1 = 7
    // Indices 2..4 are fully overlapping (no zero-padding): result[i] = (x[i-2]+x[i-1]+x[i])/3
    EXPECT_NEAR(result[2], (1.0 + 2.0 + 3.0) / 3.0, 1e-12);
    EXPECT_NEAR(result[3], (2.0 + 3.0 + 4.0) / 3.0, 1e-12);
    EXPECT_NEAR(result[4], (3.0 + 4.0 + 5.0) / 3.0, 1e-12);
}

TEST(NumericalSignal, Convolve_Commutativity) {
    // Convolution is commutative: a*b == b*a
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {4.0, 5.0};
    const auto ab = convolve(a, b);
    const auto ba = convolve(b, a);
    ASSERT_EQ(ab.size(), ba.size());
    for (size_t i = 0; i < ab.size(); ++i)
        EXPECT_NEAR(ab[i], ba[i], 1e-12);
}

TEST(NumericalSignal, Convolve_Energy_Preserved_ForDelta) {
    // Total sum of conv with normalized box should preserve sum
    const std::vector<double> x(100, 1.0);
    const std::vector<double> kernel = {1.0};  // identity kernel
    const auto result = convolve(x, kernel);
    const double sum_in  = std::accumulate(x.begin(), x.end(), 0.0);
    const double sum_out = std::accumulate(result.begin(), result.end(), 0.0);
    EXPECT_NEAR(sum_out, sum_in, 1e-10);
}

// ---------------------------------------------------------------------------
// Correlate: autocorrelation peak at zero lag
// ---------------------------------------------------------------------------

TEST(NumericalSignal, Correlate_Autocorrelation_Peak_At_Lag0) {
    // Autocorrelation: x ⊗ x has maximum at center (lag 0)
    const std::vector<double> x = {1.0, 2.0, 3.0, 2.0, 1.0};
    const auto acorr = correlate(x, x);
    ASSERT_FALSE(acorr.empty());

    const double max_val = *std::max_element(acorr.begin(), acorr.end());
    const size_t center = acorr.size() / 2;
    EXPECT_NEAR(acorr[center], max_val, 1e-10);
}

TEST(NumericalSignal, Correlate_Orthogonal_Signals) {
    // Sine and cosine are orthogonal; cross-correlation should be ~0 at all lags
    // for full-period signals
    const size_t N = 32;
    std::vector<double> s(N), c(N);
    for (size_t i = 0; i < N; ++i) {
        s[i] = std::sin(2.0 * M_PI * i / N);
        c[i] = std::cos(2.0 * M_PI * i / N);
    }
    const auto xcorr = correlate(s, c);
    ASSERT_FALSE(xcorr.empty());
    // The cross-correlation should be much smaller than the autocorrelation
    const auto sacorr = correlate(s, s);
    const double peak_acorr = *std::max_element(sacorr.begin(), sacorr.end());
    const double max_xcorr = *std::max_element(xcorr.begin(), xcorr.end());
    // Cross-corr peak should be less than the autocorrelation peak
    EXPECT_LT(max_xcorr, peak_acorr);
}

// ---------------------------------------------------------------------------
// Moving average: known analytical results
// ---------------------------------------------------------------------------

TEST(NumericalSignal, MovingAverage_ConstantInput) {
    // Moving average of a constant = the constant
    const std::vector<double> x(20, 5.0);
    const auto result = moving_average(x, 4);
    ASSERT_EQ(result.size(), x.size());
    for (double v : result)
        EXPECT_NEAR(v, 5.0, 1e-12);
}

TEST(NumericalSignal, MovingAverage_LinearInput) {
    // Causal backward-looking moving average: result[i] = mean(x[i-w+1..i])
    // For x[i]=i and window=3: result[i] = (i-2 + i-1 + i)/3 = i-1  (for i >= 2)
    const size_t N = 10;
    std::vector<double> x(N);
    std::iota(x.begin(), x.end(), 0.0);
    const auto result = moving_average(x, 3);
    ASSERT_EQ(result.size(), N);
    // Fully-windowed elements start at index window-1 = 2
    for (size_t i = 2; i < N; ++i)
        EXPECT_NEAR(result[i], static_cast<double>(i) - 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Filter numerical properties
// ---------------------------------------------------------------------------

TEST(NumericalSignal, Lowpass_PreservesDC) {
    // A low-pass filter passes DC (constant signal through unchanged)
    const size_t N = 100;
    const std::vector<double> dc(N, 2.0);
    const auto result = lowpass(dc, 0.4, 1.0);  // cutoff 0.4, fs=1.0
    ASSERT_EQ(result.size(), N);
    // DC value should be approximately preserved at steady state
    EXPECT_NEAR(result[N-1], 2.0, 0.5);  // lenient: filter may have transient
}

TEST(NumericalSignal, Highpass_Attenuates_DC) {
    // High-pass filter should attenuate DC
    const size_t N = 100;
    const std::vector<double> dc(N, 3.0);
    const auto result = highpass(dc, 0.3, 1.0);
    ASSERT_EQ(result.size(), N);
    // At steady state, the high-pass output for DC should approach 0
    const double max_abs = std::abs(*std::max_element(result.begin() + N/2, result.end()));
    EXPECT_LT(max_abs, 1.5);  // Should be attenuated
}

TEST(NumericalSignal, Filter_OutputLengthMatchesInput) {
    const std::vector<double> x(50, 1.0);
    EXPECT_EQ(lowpass(x, 0.2, 1.0).size(), x.size());
    EXPECT_EQ(highpass(x, 0.2, 1.0).size(), x.size());
    EXPECT_EQ(bandpass(x, 0.1, 0.4, 1.0).size(), x.size());
    EXPECT_EQ(butterworth(x, 0.3, 1.0).size(), x.size());
}

TEST(NumericalSignal, Filter_FiniteOutputValues) {
    const std::vector<double> x(50, 1.0);
    for (double v : lowpass(x, 0.2, 1.0))
        EXPECT_TRUE(std::isfinite(v));
    for (double v : highpass(x, 0.2, 1.0))
        EXPECT_TRUE(std::isfinite(v));
}
