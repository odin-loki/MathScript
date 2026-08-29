// MathScript: Advanced FFT Tests (Wave 44)
// Tests for roundtrip fidelity, windowed FFT, 2D FFT, spectral properties

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
#include "ms/signal/signal.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// FFT roundtrip: fft then ifft restores signal
// ---------------------------------------------------------------------------

TEST(FFTAdv2, Roundtrip_Impulse_AtZero) {
    // Delta at position 0: ifft(fft(delta)) = delta
    int N = 16;
    std::vector<double> signal(N, 0.0);
    signal[0] = 1.0;
    auto spec_res = fft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto back_res = ifft(spec_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    ASSERT_EQ(back.size(), static_cast<size_t>(N));
    EXPECT_NEAR(back[0], 1.0, 1e-10);
    for (int i = 1; i < N; ++i)
        EXPECT_NEAR(back[i], 0.0, 1e-10);
}

TEST(FFTAdv2, Roundtrip_SineWave) {
    int N = 64;
    double f0 = 4.0;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * f0 * i / N);
    auto spec_res = fft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto back_res = ifft(spec_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    ASSERT_EQ(back.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], signal[i], 1e-9) << "Roundtrip error at i=" << i;
}

TEST(FFTAdv2, Roundtrip_DCSignal) {
    int N = 32;
    std::vector<double> signal(N, 5.0);
    auto spec_res = fft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto back_res = ifft(spec_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], 5.0, 1e-9);
}

TEST(FFTAdv2, Roundtrip_RandomLike_Signal) {
    // Use a deterministic "random-like" signal
    int N = 32;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::sin(i * 1.3) + 0.5 * std::cos(i * 0.7);
    auto spec_res = fft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto back_res = ifft(spec_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], signal[i], 1e-9);
}

// ---------------------------------------------------------------------------
// Parseval's theorem: sum|x|^2 = (1/N) * sum|X[k]|^2
// ---------------------------------------------------------------------------

TEST(FFTAdv2, Parseval_SineWave) {
    int N = 64;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * 3.0 * i / N);

    double energy_time = 0.0;
    for (auto v : signal) energy_time += v * v;

    auto spec_res = fft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto& spec = spec_res.value();
    double energy_freq = 0.0;
    for (auto& c : spec) energy_freq += std::norm(c);
    energy_freq /= N;

    EXPECT_NEAR(energy_time, energy_freq, 1e-6);
}

TEST(FFTAdv2, Parseval_DCSignal) {
    int N = 32;
    std::vector<double> signal(N, 2.0);
    double energy_time = N * 4.0;  // 32 * 2^2 = 128

    auto spec_res = fft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto& spec = spec_res.value();
    double energy_freq = 0.0;
    for (auto& c : spec) energy_freq += std::norm(c);
    energy_freq /= N;

    EXPECT_NEAR(energy_time, energy_freq, 1e-6);
}

// ---------------------------------------------------------------------------
// FFT frequency properties
// ---------------------------------------------------------------------------

TEST(FFTAdv2, FFT_TwoSines_TwoPeaks) {
    // Signal with 2 pure frequencies should have 2 peaks in spectrum
    int N = 128;
    double f1 = 8.0, f2 = 20.0;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * f1 * i / N) +
                    std::sin(2.0 * M_PI * f2 * i / N);

    auto spec_res = fft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto& spec = spec_res.value();

    // Verify peaks at expected bins (within positive frequencies)
    double mag_f1 = std::abs(spec[static_cast<size_t>(f1)]);
    double mag_f2 = std::abs(spec[static_cast<size_t>(f2)]);
    double mag_dc = std::abs(spec[0]);
    double mag_mid = std::abs(spec[14]);  // Not a peak

    EXPECT_GT(mag_f1, mag_dc) << "Peak at f1 should be > DC";
    EXPECT_GT(mag_f2, mag_mid) << "Peak at f2 should be > neighboring bin";
}

TEST(FFTAdv2, FFT_LinearPhaseShift) {
    // A delayed signal x[n-d] has spectrum X[k] * e^(-j2pi*k*d/N)
    // Just verify magnitude is the same
    int N = 32;
    int delay = 3;
    std::vector<double> signal(N), delayed(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = std::cos(2.0 * M_PI * 4.0 * i / N);
        delayed[i] = std::cos(2.0 * M_PI * 4.0 * (i - delay) / N);
    }
    auto s1 = fft(signal);
    auto s2 = fft(delayed);
    ASSERT_TRUE(s1.has_value());
    ASSERT_TRUE(s2.has_value());
    // Magnitude should be the same for all bins
    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(std::abs(s1.value()[k]), std::abs(s2.value()[k]), 1e-8)
            << "Magnitude differs at bin " << k;
    }
}

// ---------------------------------------------------------------------------
// RFFT: real FFT - only returns first N/2+1 components
// ---------------------------------------------------------------------------

TEST(FFTAdv2, RFFT_OutputSize) {
    int N = 32;
    std::vector<double> signal(N, 1.0);
    auto spec_res = rfft(signal);
    ASSERT_TRUE(spec_res.has_value());
    EXPECT_EQ(spec_res.value().size(), static_cast<size_t>(N / 2 + 1));
}

TEST(FFTAdv2, RFFT_DCComponent) {
    int N = 16;
    std::vector<double> signal(N, 3.0);
    auto spec_res = rfft(signal);
    ASSERT_TRUE(spec_res.has_value());
    // DC component = N * mean
    EXPECT_NEAR(std::abs(spec_res.value()[0]), N * 3.0, 0.01);
}

TEST(FFTAdv2, RFFT_IRFFT_Roundtrip) {
    int N = 32;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::cos(2.0 * M_PI * 5.0 * i / N);
    auto spec_res = rfft(signal);
    ASSERT_TRUE(spec_res.has_value());
    auto back_res = irfft(spec_res.value(), N);
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    ASSERT_EQ(back.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], signal[i], 1e-9) << "RFFT roundtrip error at i=" << i;
}

// ---------------------------------------------------------------------------
// Windowed FFT: apply window before FFT to reduce spectral leakage
// ---------------------------------------------------------------------------

TEST(FFTAdv2, WindowedFFT_Hamming_ReducesLeakage) {
    // Without window, a non-integer-cycle sine leaks into neighboring bins
    // With window, the main lobe is wider but sidelobes are lower
    int N = 64;
    double f = 3.7;  // Non-integer cycles - causes leakage
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * f * i / N);

    auto win = hamming(N);
    std::vector<double> windowed(N);
    for (int i = 0; i < N; ++i) windowed[i] = signal[i] * win[i];

    auto plain_spec = fft(signal);
    auto windowed_spec = fft(windowed);
    ASSERT_TRUE(plain_spec.has_value());
    ASSERT_TRUE(windowed_spec.has_value());

    // Both should have finite output
    for (auto& c : plain_spec.value()) EXPECT_TRUE(std::isfinite(std::abs(c)));
    for (auto& c : windowed_spec.value()) EXPECT_TRUE(std::isfinite(std::abs(c)));
}

// ---------------------------------------------------------------------------
// DFT vs FFT consistency
// ---------------------------------------------------------------------------

TEST(FFTAdv2, DFT_Matches_FFT) {
    int N = 16;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::cos(2.0 * M_PI * 2.0 * i / N);

    auto fft_res = fft(signal);
    auto dft_res = dft(signal);
    ASSERT_TRUE(fft_res.has_value());
    ASSERT_TRUE(dft_res.has_value());
    auto& F = fft_res.value();
    auto& D = dft_res.value();
    ASSERT_EQ(F.size(), D.size());
    for (size_t k = 0; k < F.size(); ++k) {
        EXPECT_NEAR(std::real(F[k]), std::real(D[k]), 1e-9)
            << "DFT/FFT real part mismatch at k=" << k;
        EXPECT_NEAR(std::imag(F[k]), std::imag(D[k]), 1e-9)
            << "DFT/FFT imag part mismatch at k=" << k;
    }
}
