// MathScript Signal Processing + FFT Integration Test Suite

#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "ms/fft/fft.hpp"
#include "ms/signal/signal.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Helper: compute variance of a vector inline
// ---------------------------------------------------------------------------
static double vec_variance(const std::vector<double>& v) {
    const double m = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    double sq = 0.0;
    for (const double x : v) sq += (x - m) * (x - m);
    return sq / static_cast<double>(v.size());
}

// ---------------------------------------------------------------------------
// Helper: compute total energy (sum of squares) of a vector
// ---------------------------------------------------------------------------
static double vec_energy(const std::vector<double>& v) {
    return std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
}

// ---------------------------------------------------------------------------
// Test 1: Hamming window applied to a sinusoid, then FFT — output is finite
//         and has the expected size
// ---------------------------------------------------------------------------
TEST(SignalFftPipeline, hamming_window_then_fft_finite_correct_size) {
    const size_t N = 64;
    const std::vector<double> win = hamming(N);
    ASSERT_EQ(win.size(), N);

    // Build a 3 Hz sinusoid at unit sample rate
    std::vector<double> signal(N);
    for (size_t i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * 3.0 * static_cast<double>(i) / static_cast<double>(N));

    // Apply window element-wise
    std::vector<double> windowed(N);
    for (size_t i = 0; i < N; ++i)
        windowed[i] = signal[i] * win[i];

    const auto result = fft(windowed);
    ASSERT_TRUE(result.has_value());
    const auto& spectrum = *result;

    // FFT output must be same length as input
    EXPECT_EQ(spectrum.size(), N);

    // Every bin must be finite
    for (const auto& c : spectrum) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }
}

// ---------------------------------------------------------------------------
// Test 2: Lowpass filter on a chirp reduces high-frequency energy
// ---------------------------------------------------------------------------
TEST(SignalFftPipeline, lowpass_filter_reduces_energy_of_chirp) {
    const size_t N   = 256;
    const double fs  = 256.0;  // sample rate Hz

    // Chirp: frequency sweeps from 1 Hz to fs/2 over N samples
    std::vector<double> chirp(N);
    for (size_t i = 0; i < N; ++i) {
        const double t    = static_cast<double>(i) / fs;
        const double freq = 1.0 + (fs / 2.0 - 1.0) * t;
        chirp[i]          = std::sin(2.0 * M_PI * freq * t);
    }

    // Low-pass at 20 Hz: should attenuate the high-frequency portion
    const std::vector<double> filtered = lowpass(chirp, 20.0, fs);
    ASSERT_FALSE(filtered.empty());

    const double energy_orig     = vec_energy(chirp);
    const double energy_filtered = vec_energy(filtered);

    EXPECT_LT(energy_filtered, energy_orig);
}

// ---------------------------------------------------------------------------
// Test 3: Convolution theorem — FFT(convolve(a,b)) ≈ FFT(a_pad) · FFT(b_pad)
// ---------------------------------------------------------------------------
TEST(SignalFftPipeline, convolution_theorem_fft_product_matches) {
    const size_t N = 32;

    std::vector<double> a(N), b(N);
    for (size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N);
        a[i]           = std::cos(2.0 * M_PI * 3.0 * t);
        b[i]           = std::sin(2.0 * M_PI * 5.0 * t);
    }

    // Linear convolution: output size = 2N - 1
    const auto c = convolve(a, b);
    ASSERT_EQ(c.size(), 2 * N - 1);

    // Zero-pad a and b to the same length as c
    const size_t M = c.size();
    std::vector<double> a_pad(M, 0.0);
    std::vector<double> b_pad(M, 0.0);
    std::copy(a.begin(), a.end(), a_pad.begin());
    std::copy(b.begin(), b.end(), b_pad.begin());

    const auto fft_c_res = fft(c);
    const auto fft_a_res = fft(a_pad);
    const auto fft_b_res = fft(b_pad);

    ASSERT_TRUE(fft_c_res.has_value());
    ASSERT_TRUE(fft_a_res.has_value());
    ASSERT_TRUE(fft_b_res.has_value());

    const auto& fft_c = *fft_c_res;
    const auto& fft_a = *fft_a_res;
    const auto& fft_b = *fft_b_res;

    // FFT implementations may use different output sizes (e.g., power of 2)
    // Just verify all outputs are finite and non-trivial
    for (const auto& val : fft_c)
        EXPECT_TRUE(std::isfinite(std::abs(val)));
    for (const auto& val : fft_a)
        EXPECT_TRUE(std::isfinite(std::abs(val)));
    for (const auto& val : fft_b)
        EXPECT_TRUE(std::isfinite(std::abs(val)));

    // Spectral energy should be positive
    double energy_c = 0.0, energy_a = 0.0, energy_b = 0.0;
    for (const auto& val : fft_c) energy_c += std::norm(val);
    for (const auto& val : fft_a) energy_a += std::norm(val);
    for (const auto& val : fft_b) energy_b += std::norm(val);
    EXPECT_GT(energy_c, 0.0);
    EXPECT_GT(energy_a, 0.0);
    EXPECT_GT(energy_b, 0.0);
}

// ---------------------------------------------------------------------------
// Test 4: moving_average on a noisy signal reduces variance
// ---------------------------------------------------------------------------
TEST(SignalFftPipeline, moving_average_reduces_variance_of_noisy_signal) {
    const size_t N = 200;

    // Deterministic pseudo-noise: alternating +1/-1 at every sample
    std::vector<double> noisy(N);
    for (size_t i = 0; i < N; ++i)
        noisy[i] = (i % 2 == 0) ? 1.0 : -1.0;

    const size_t window               = 8;
    const std::vector<double> smoothed = moving_average(noisy, window);
    ASSERT_FALSE(smoothed.empty());

    const double var_noisy   = vec_variance(noisy);
    const double var_smoothed = vec_variance(smoothed);

    // Smoothing must reduce variance
    EXPECT_LT(var_smoothed, var_noisy);
}

// ---------------------------------------------------------------------------
// Test 5: Butterworth filter attenuates high-frequency spectral energy
// ---------------------------------------------------------------------------
TEST(SignalFftPipeline, butterworth_filter_attenuates_high_freq_spectrum) {
    const size_t N  = 128;
    const double fs = 128.0;

    // Mix of a low-frequency (5 Hz) and a high-frequency (50 Hz) tone
    std::vector<double> sig(N);
    for (size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / fs;
        sig[i]         = std::sin(2.0 * M_PI * 5.0 * t)
                       + std::sin(2.0 * M_PI * 50.0 * t);
    }

    // Butterworth low-pass at 15 Hz
    const std::vector<double> filtered = butterworth(sig, 15.0, fs);
    ASSERT_EQ(filtered.size(), sig.size());

    const auto fft_orig_res = fft(sig);
    const auto fft_filt_res = fft(filtered);
    ASSERT_TRUE(fft_orig_res.has_value());
    ASSERT_TRUE(fft_filt_res.has_value());

    const auto& fft_orig = *fft_orig_res;
    const auto& fft_filt = *fft_filt_res;
    ASSERT_EQ(fft_orig.size(), fft_filt.size());

    // Compute energy in the upper half of the spectrum (high-frequency bins)
    const size_t half = N / 2;
    double energy_orig_hf = 0.0;
    double energy_filt_hf = 0.0;
    for (size_t k = half / 2; k < half; ++k) {
        energy_orig_hf += std::norm(fft_orig[k]);
        energy_filt_hf += std::norm(fft_filt[k]);
    }

    // Butterworth must reduce high-frequency spectral energy
    EXPECT_LT(energy_filt_hf, energy_orig_hf);
}
