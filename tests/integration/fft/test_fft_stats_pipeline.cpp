// MathScript Integration Test: FFT + Stats frequency analysis pipeline (Wave 52)

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <complex>

#include "ms/fft/fft.hpp"
#include "ms/stats/stats.hpp"
#include "ms/signal/signal.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using CVector = std::vector<std::complex<double>>;

// ---------------------------------------------------------------------------
// FFT + Stats: frequency domain statistics
// ---------------------------------------------------------------------------

TEST(FftStatsPipeline, FFT_PowerSpectrum_SineWave_HasOnePeak) {
    const size_t N = 256;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 8.0 * i / N); // freq bin 8
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    const auto& X = Xr.value();
    // Power spectrum magnitudes over positive frequencies
    std::vector<double> power(N / 2);
    for (size_t i = 0; i < N / 2; ++i)
        power[i] = std::norm(X[i]);
    size_t peak_bin = std::distance(power.begin(), std::max_element(power.begin(), power.end()));
    EXPECT_EQ(peak_bin, 8u) << "FFT peak should be at frequency bin 8";
}

TEST(FftStatsPipeline, FFT_Parseval_HoldsTight) {
    // Parseval: sum|X[k]|^2 = N * sum|x[n]|^2
    const size_t N = 64;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::cos(2.0 * M_PI * 3.0 * i / N) + 0.5 * std::sin(2.0 * M_PI * 5.0 * i / N);
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    const auto& X = Xr.value();
    double time_energy = 0.0;
    for (double v : x) time_energy += v * v;
    double freq_energy = 0.0;
    for (auto& c : X) freq_energy += std::norm(c);
    EXPECT_NEAR(freq_energy, N * time_energy, N * time_energy * 0.01);
}

TEST(FftStatsPipeline, FFT_Impulse_FlatMagnitude) {
    // Impulse at index 0 has flat magnitude spectrum = 1
    const size_t N = 64;
    std::vector<double> x(N, 0.0);
    x[0] = 1.0;
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    const auto& X = Xr.value();
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(std::abs(X[i]), 1.0, 1e-10) << "Impulse FFT flat at bin " << i;
}

TEST(FftStatsPipeline, FFT_DC_Component_EqualsMeanTimesN) {
    const size_t N = 32;
    std::vector<double> x(N, 3.0);
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    const auto& X = Xr.value();
    EXPECT_NEAR(X[0].real(), N * 3.0, 1e-8);
    EXPECT_NEAR(X[0].imag(), 0.0, 1e-8);
}

TEST(FftStatsPipeline, FFT_Linearity_ScaledSignal) {
    const size_t N = 64;
    std::vector<double> x(N), x2(N);
    for (size_t i = 0; i < N; ++i) {
        x[i]  = std::sin(2.0 * M_PI * 4.0 * i / N);
        x2[i] = 3.0 * x[i];
    }
    auto X1r = fft(x), X2r = fft(x2);
    ASSERT_TRUE(X1r.has_value()); ASSERT_TRUE(X2r.has_value());
    const auto& X1 = X1r.value(); const auto& X2 = X2r.value();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(X2[i].real(), 3.0 * X1[i].real(), 1e-8);
        EXPECT_NEAR(X2[i].imag(), 3.0 * X1[i].imag(), 1e-8);
    }
}

TEST(FftStatsPipeline, IFFT_FFT_Roundtrip) {
    const size_t N = 64;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 5.0 * i / N) + std::cos(2.0 * M_PI * 11.0 * i / N);
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    auto xrecr = ifft(Xr.value());
    ASSERT_TRUE(xrecr.has_value());
    const auto& xrec = xrecr.value();
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(xrec[i], x[i], 1e-8) << "IFFT(FFT(x)) != x at i=" << i;
}

// ---------------------------------------------------------------------------
// FFT + Stats: spectral statistics
// ---------------------------------------------------------------------------

TEST(FftStatsPipeline, FFT_SineWave_MeanNearZero_StddevExpected) {
    const size_t N = 128;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 7.0 * i / N);
    double m = mean(x);
    double s = stddev(x);
    EXPECT_NEAR(m, 0.0, 0.01);
    EXPECT_NEAR(s, 1.0 / std::sqrt(2.0), 0.05);
}

TEST(FftStatsPipeline, FFT_MixedFreqs_DominantPeak) {
    const size_t N = 128;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = 3.0 * std::sin(2.0 * M_PI * 10.0 * i / N)
               + 0.5 * std::sin(2.0 * M_PI * 20.0 * i / N);
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    const auto& X = Xr.value();
    std::vector<double> mag(N / 2);
    for (size_t i = 0; i < N / 2; ++i) mag[i] = std::abs(X[i]);
    size_t peak = std::distance(mag.begin(), std::max_element(mag.begin(), mag.end()));
    EXPECT_EQ(peak, 10u) << "Dominant frequency should be at bin 10";
}

TEST(FftStatsPipeline, FFT_ZeroSignal_ZeroSpectrum) {
    const size_t N = 32;
    std::vector<double> x(N, 0.0);
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    const auto& X = Xr.value();
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(std::abs(X[i]), 0.0, 1e-12) << "Zero signal should have zero spectrum";
}

// ---------------------------------------------------------------------------
// FFT + Signal: lowpass filter reduces high-frequency content
// ---------------------------------------------------------------------------

TEST(FftStatsPipeline, Lowpass_Output_AllFinite_And_CorrectSize) {
    const size_t N = 128;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 5.0 * i / N)
               + std::sin(2.0 * M_PI * 50.0 * i / N);
    auto xf = lowpass(x, 0.2, 32);
    EXPECT_EQ(xf.size(), N);
    for (double v : xf) EXPECT_TRUE(std::isfinite(v));
    // After lowpass, FFT should be valid
    auto Xfr = fft(xf);
    ASSERT_TRUE(Xfr.has_value());
    for (auto& c : Xfr.value()) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }
}

TEST(FftStatsPipeline, FFT_AllFinite) {
    const size_t N = 128;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 3.0 * i / N) * std::exp(-0.01 * i);
    auto Xr = fft(x);
    ASSERT_TRUE(Xr.has_value());
    for (auto& c : Xr.value()) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }
}
