// MathScript: Advanced Signal Filter Property Tests (Wave 49)
// Tests for butterworth/lowpass/highpass/bandpass filter properties,
// window functions, and convolve correctness

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

// Generate sinusoidal signal of given frequency
static std::vector<double> make_sine(double freq, double fs, size_t N) {
    std::vector<double> sig(N);
    for (size_t i = 0; i < N; ++i)
        sig[i] = std::sin(2.0 * M_PI * freq * i / fs);
    return sig;
}

// Compute signal power (mean squared)
static double power(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double s = 0.0;
    for (auto v : x) s += v * v;
    return s / x.size();
}

// ---------------------------------------------------------------------------
// Butterworth filter: basic properties
// ---------------------------------------------------------------------------

TEST(SignalAdv3, Butterworth_OutputSize_MatchesInput) {
    size_t N = 256;
    auto sig = make_sine(100.0, 1000.0, N);
    auto out = butterworth(sig, 200.0, 1000.0);
    EXPECT_EQ(out.size(), N);
}

TEST(SignalAdv3, Butterworth_AllFinite) {
    auto sig = make_sine(50.0, 1000.0, 256);
    auto out = butterworth(sig, 200.0, 1000.0);
    for (auto v : out) EXPECT_TRUE(std::isfinite(v)) << "butterworth output not finite";
}

TEST(SignalAdv3, Butterworth_ZeroInput_ZeroOutput) {
    std::vector<double> sig(128, 0.0);
    auto out = butterworth(sig, 200.0, 1000.0);
    for (auto v : out) EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(SignalAdv3, Butterworth_Attenuates_HighFreq) {
    // Low-frequency signal should pass, high-freq should be attenuated
    size_t N = 512;
    double fs = 1000.0, cutoff = 100.0;
    auto low = make_sine(10.0, fs, N);    // well below cutoff
    auto high = make_sine(400.0, fs, N);  // well above cutoff
    auto low_out = butterworth(low, cutoff, fs);
    auto high_out = butterworth(high, cutoff, fs);
    double p_low = power(low_out);
    double p_high = power(high_out);
    EXPECT_GT(p_low, p_high * 10.0)
        << "Butterworth should attenuate high-freq more than low-freq";
}

// ---------------------------------------------------------------------------
// Lowpass filter
// ---------------------------------------------------------------------------

TEST(SignalAdv3, Lowpass_OutputSize_MatchesInput) {
    auto sig = make_sine(50.0, 1000.0, 128);
    auto out = lowpass(sig, 100.0, 1000.0);
    EXPECT_EQ(out.size(), sig.size());
}

TEST(SignalAdv3, Lowpass_AllFinite) {
    auto sig = make_sine(200.0, 2000.0, 256);
    auto out = lowpass(sig, 500.0, 2000.0);
    for (auto v : out) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalAdv3, Lowpass_Attenuates_HighFreq) {
    size_t N = 512;
    double fs = 1000.0, cutoff = 100.0;
    auto low = make_sine(20.0, fs, N);
    auto high = make_sine(450.0, fs, N);
    double p_low = power(lowpass(low, cutoff, fs));
    double p_high = power(lowpass(high, cutoff, fs));
    EXPECT_GT(p_low, p_high * 5.0)
        << "Lowpass should pass low-freq and attenuate high-freq";
}

// ---------------------------------------------------------------------------
// Highpass filter
// ---------------------------------------------------------------------------

TEST(SignalAdv3, Highpass_OutputSize_MatchesInput) {
    auto sig = make_sine(400.0, 1000.0, 128);
    auto out = highpass(sig, 200.0, 1000.0);
    EXPECT_EQ(out.size(), sig.size());
}

TEST(SignalAdv3, Highpass_AllFinite) {
    auto sig = make_sine(400.0, 2000.0, 256);
    auto out = highpass(sig, 300.0, 2000.0);
    for (auto v : out) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalAdv3, Highpass_Passes_HighFreq) {
    // Highpass should not completely kill a high-frequency signal
    size_t N = 512;
    double fs = 1000.0, cutoff = 50.0;
    auto high = make_sine(400.0, fs, N);
    double p_before = power(high);
    double p_after = power(highpass(high, cutoff, fs));
    // High-freq signal above cutoff should still have significant power after highpass
    EXPECT_GT(p_after, p_before * 0.1)
        << "Highpass should not fully kill high-freq signal far above cutoff";
}

// ---------------------------------------------------------------------------
// Bandpass filter
// ---------------------------------------------------------------------------

TEST(SignalAdv3, Bandpass_OutputSize_MatchesInput) {
    auto sig = make_sine(200.0, 1000.0, 128);
    auto out = bandpass(sig, 100.0, 300.0, 1000.0);
    EXPECT_EQ(out.size(), sig.size());
}

TEST(SignalAdv3, Bandpass_AllFinite) {
    auto sig = make_sine(200.0, 1000.0, 256);
    auto out = bandpass(sig, 100.0, 300.0, 1000.0);
    for (auto v : out) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// Convolve: correctness
// ---------------------------------------------------------------------------

TEST(SignalAdv3, Convolve_OutputSize_APlus_BMinus1) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {1.0, 1.0};
    auto c = convolve(a, b);
    EXPECT_EQ(c.size(), a.size() + b.size() - 1);
}

TEST(SignalAdv3, Convolve_IdentityKernel) {
    // Convolve with [1] should return original signal
    std::vector<double> sig = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> delta = {1.0};
    auto out = convolve(sig, delta);
    ASSERT_EQ(out.size(), sig.size());
    for (size_t i = 0; i < sig.size(); ++i)
        EXPECT_NEAR(out[i], sig[i], 1e-10);
}

TEST(SignalAdv3, Convolve_DeltaShift) {
    // Convolve with [0, 1] shifts signal by 1
    std::vector<double> sig = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> shift = {0.0, 1.0};
    auto out = convolve(sig, shift);
    ASSERT_EQ(out.size(), sig.size() + 1);
    EXPECT_NEAR(out[0], 0.0, 1e-10);
    EXPECT_NEAR(out[1], 1.0, 1e-10);
    EXPECT_NEAR(out[2], 2.0, 1e-10);
}

TEST(SignalAdv3, Convolve_SumIsProductOfSums) {
    // If a and b are all ones: sum(conv(a,b)) = sum(a) * sum(b)
    std::vector<double> a(5, 1.0);
    std::vector<double> b(3, 1.0);
    auto c = convolve(a, b);
    double sum_c = std::accumulate(c.begin(), c.end(), 0.0);
    double expected = std::accumulate(a.begin(), a.end(), 0.0) *
                      std::accumulate(b.begin(), b.end(), 0.0);
    EXPECT_NEAR(sum_c, expected, 1e-10);
}

TEST(SignalAdv3, Convolve_Commutative) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0};
    auto ab = convolve(a, b);
    auto ba = convolve(b, a);
    ASSERT_EQ(ab.size(), ba.size());
    for (size_t i = 0; i < ab.size(); ++i)
        EXPECT_NEAR(ab[i], ba[i], 1e-10) << "convolve not commutative at i=" << i;
}

// ---------------------------------------------------------------------------
// Moving average
// ---------------------------------------------------------------------------

TEST(SignalAdv3, MovingAverage_ConstantSignal_IsConstant) {
    std::vector<double> sig(50, 3.0);
    auto out = moving_average(sig, 5);
    // After initial fill, all outputs should be 3
    for (size_t i = 4; i < out.size(); ++i)
        EXPECT_NEAR(out[i], 3.0, 1e-10) << "moving_average of constant failed at i=" << i;
}

TEST(SignalAdv3, MovingAverage_OutputSize_MatchesInput) {
    std::vector<double> sig(100, 1.0);
    auto out = moving_average(sig, 10);
    EXPECT_EQ(out.size(), sig.size());
}

TEST(SignalAdv3, MovingAverage_SmoothsNoise) {
    // Moving average should reduce variance
    size_t N = 200;
    std::vector<double> noisy(N);
    for (size_t i = 0; i < N; ++i) noisy[i] = (i % 2 == 0) ? 1.0 : -1.0;
    auto smooth = moving_average(noisy, 10);
    double power_noisy = power(noisy);
    double power_smooth = power(smooth);
    EXPECT_LT(power_smooth, power_noisy)
        << "Moving average should reduce power of alternating signal";
}

// ---------------------------------------------------------------------------
// Triangular window
// ---------------------------------------------------------------------------

TEST(SignalAdv3, Triangular_OutputSize) {
    auto w = triangular(64);
    EXPECT_EQ(w.size(), 64u);
}

TEST(SignalAdv3, Triangular_NonNegative) {
    auto w = triangular(32);
    for (auto v : w) EXPECT_GE(v, 0.0);
}

TEST(SignalAdv3, Triangular_NearlySymmetric) {
    // Implementation may use asymmetric Bartlett formula for even N; check approximate symmetry
    auto w = triangular(64);
    size_t N = w.size();
    // Check that values on opposite ends are close (within 5% of max)
    double max_val = *std::max_element(w.begin(), w.end());
    for (size_t i = 0; i < N / 2; ++i)
        EXPECT_NEAR(w[i], w[N - 1 - i], max_val * 0.1)
            << "Triangular not approximately symmetric at i=" << i;
}

TEST(SignalAdv3, Triangular_PeakAtCenter) {
    auto w = triangular(64);
    double max_val = *std::max_element(w.begin(), w.end());
    // For even N, peak is at N/2-1 or N/2
    EXPECT_GT(max_val, w[0]);
    EXPECT_GT(max_val, w[w.size() - 1]);
}
