// MathScript: Advanced FFT Energy Conservation and Phase Tests (Wave 49)
// Note: fft() takes vector<double>, ifft() returns vector<double>

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <vector>
#include <numeric>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/fft/fft.hpp"

using namespace ms;
using CVector = std::vector<std::complex<double>>;

// Time-domain signal energy (real)
static double signal_energy(const std::vector<double>& x) {
    double e = 0.0;
    for (auto v : x) e += v * v;
    return e;
}

// Frequency-domain energy
static double freq_energy(const CVector& X) {
    double e = 0.0;
    for (auto& c : X) e += std::norm(c);
    return e;
}

// ---------------------------------------------------------------------------
// Parseval's theorem: ||x||^2 = (1/N) * ||X||^2
// ---------------------------------------------------------------------------

TEST(FFTPhaseEnergy, Parseval_Sine_Energy) {
    int N = 64;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 3.0 * i / N);
    auto X_res = fft(x);
    ASSERT_TRUE(X_res.has_value());
    auto& X = X_res.value();
    double time_e = signal_energy(x);
    double freq_e = freq_energy(X) / N;
    EXPECT_NEAR(time_e, freq_e, 1e-8) << "Parseval's theorem violated for sine";
}

TEST(FFTPhaseEnergy, Parseval_Random_Signal) {
    int N = 32;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(i * 0.7 + 1.3) + std::cos(i * 0.5);
    auto X_res = fft(x);
    ASSERT_TRUE(X_res.has_value());
    auto& X = X_res.value();
    double time_e = signal_energy(x);
    double freq_e = freq_energy(X) / N;
    EXPECT_NEAR(time_e, freq_e, 1e-8) << "Parseval's theorem violated for random signal";
}

TEST(FFTPhaseEnergy, Parseval_ImpulseSignal) {
    int N = 16;
    std::vector<double> x(N, 0.0);
    x[0] = 1.0;
    auto X_res = fft(x);
    ASSERT_TRUE(X_res.has_value());
    auto& X = X_res.value();
    double freq_e = freq_energy(X) / N;
    EXPECT_NEAR(freq_e, 1.0, 1e-10) << "Impulse energy should be 1";
}

// ---------------------------------------------------------------------------
// Linearity: FFT(alpha * x) = alpha * FFT(x)
// ---------------------------------------------------------------------------

TEST(FFTPhaseEnergy, Linearity_ScaledSine) {
    int N = 32;
    double alpha = 3.0;
    std::vector<double> x(N), ax(N);
    for (int i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * i / N);
        ax[i] = alpha * x[i];
    }
    auto X_res = fft(x);
    auto aX_res = fft(ax);
    ASSERT_TRUE(X_res.has_value());
    ASSERT_TRUE(aX_res.has_value());
    auto& X = X_res.value();
    auto& aX = aX_res.value();
    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(std::abs(aX[k]), alpha * std::abs(X[k]), 1e-8)
            << "Linearity violated at k=" << k;
    }
}

// ---------------------------------------------------------------------------
// Phase shift: circular shift in time → magnitude preserved in frequency
// ---------------------------------------------------------------------------

TEST(FFTPhaseEnergy, PhaseShift_MagnitudePreserved) {
    int N = 16, shift = 3;
    std::vector<double> x(N), x_shifted(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * M_PI * 2.0 * i / N);
    for (int i = 0; i < N; ++i)
        x_shifted[i] = x[(i + shift) % N];

    auto X_res = fft(x);
    auto Xs_res = fft(x_shifted);
    ASSERT_TRUE(X_res.has_value());
    ASSERT_TRUE(Xs_res.has_value());
    auto& X = X_res.value();
    auto& Xs = Xs_res.value();
    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(std::abs(Xs[k]), std::abs(X[k]), 1e-8)
            << "Magnitude changed after circular shift at k=" << k;
    }
}

// ---------------------------------------------------------------------------
// DC signal: all energy at k=0
// ---------------------------------------------------------------------------

TEST(FFTPhaseEnergy, FFT_DC_Signal_HasSinglePeak) {
    int N = 16;
    double dc = 5.0;
    std::vector<double> x(N, dc);
    auto X_res = fft(x);
    ASSERT_TRUE(X_res.has_value());
    auto& X = X_res.value();
    EXPECT_NEAR(std::abs(X[0]), N * dc, 1e-8);
    for (int k = 1; k < N; ++k)
        EXPECT_NEAR(std::abs(X[k]), 0.0, 1e-8) << "DC signal should zero at k=" << k;
}

// ---------------------------------------------------------------------------
// RFFT properties
// ---------------------------------------------------------------------------

TEST(FFTPhaseEnergy, RFFT_OutputSize_Is_N_over_2_Plus1) {
    int N = 64;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = std::sin(2.0 * M_PI * 5.0 * i / N);
    auto X_res = rfft(x);
    ASSERT_TRUE(X_res.has_value());
    EXPECT_EQ(X_res.value().size(), static_cast<size_t>(N / 2 + 1));
}

TEST(FFTPhaseEnergy, RFFT_Roundtrip_ViaIRFFT) {
    int N = 32;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = std::cos(2.0 * M_PI * 3.0 * i / N);
    auto X_res = rfft(x);
    ASSERT_TRUE(X_res.has_value());
    auto back_res = irfft(X_res.value(), N);
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    ASSERT_EQ(back.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], x[i], 1e-8) << "RFFT roundtrip error at i=" << i;
}

TEST(FFTPhaseEnergy, RFFT_Parseval_Energy) {
    int N = 32;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = std::sin(2.0 * M_PI * 4.0 * i / N) + 0.5;
    auto X_res = rfft(x);
    ASSERT_TRUE(X_res.has_value());
    auto& X = X_res.value();
    double time_e = signal_energy(x);
    // RFFT Parseval: ||x||^2 = (|X[0]|^2 + 2*sum|X[k]|^2_{k=1..N/2-1} + |X[N/2]|^2) / N
    double freq_e = std::norm(X[0]);
    for (size_t k = 1; k + 1 < X.size(); ++k) freq_e += 2.0 * std::norm(X[k]);
    freq_e += std::norm(X.back());
    freq_e /= N;
    EXPECT_NEAR(time_e, freq_e, 1e-7) << "RFFT Parseval's theorem violated";
}

// ---------------------------------------------------------------------------
// IFFT roundtrip
// ---------------------------------------------------------------------------

TEST(FFTPhaseEnergy, IFFT_Roundtrip_Cosine) {
    int N = 32;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::cos(2.0 * M_PI * 5.0 * i / N);
    auto X_res = fft(x);
    ASSERT_TRUE(X_res.has_value());
    auto back_res = ifft(X_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    ASSERT_EQ(back.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], x[i], 1e-8) << "IFFT roundtrip error at i=" << i;
}

TEST(FFTPhaseEnergy, IFFT_Roundtrip_AllFinite) {
    int N = 64;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = std::sin(i * 0.4) + std::cos(i * 0.3);
    auto X_res = fft(x);
    ASSERT_TRUE(X_res.has_value());
    auto back_res = ifft(X_res.value());
    ASSERT_TRUE(back_res.has_value());
    for (auto v : back_res.value())
        EXPECT_TRUE(std::isfinite(v));
}
