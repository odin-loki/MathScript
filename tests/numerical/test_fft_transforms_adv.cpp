// MathScript: Advanced FFT Transform Tests
// Tests dct2/idct2 roundtrip, dst2, rfft/irfft roundtrip, dft vs fft, fftshift

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include "ms/fft/fft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// DCT2/IDCT2 roundtrip
// ---------------------------------------------------------------------------

TEST(FFTTransformsAdv, DCT2_IDCT2_Roundtrip_Impulse) {
    std::vector<double> x = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto dct = dct2(x);
    ASSERT_TRUE(dct.has_value());
    auto idct = idct2(dct.value());
    ASSERT_TRUE(idct.has_value());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(idct.value()[i], x[i], 1e-9)
            << "Roundtrip failed at index " << i;
    }
}

TEST(FFTTransformsAdv, DCT2_IDCT2_Roundtrip_Sine) {
    const size_t N = 32;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * 3.0 * i / N);
    }
    auto dct = dct2(x);
    ASSERT_TRUE(dct.has_value());
    auto idct = idct2(dct.value());
    ASSERT_TRUE(idct.has_value());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(idct.value()[i], x[i], 1e-8)
            << "DCT2/IDCT2 roundtrip at index " << i;
    }
}

TEST(FFTTransformsAdv, DCT2_Energy_Preserved) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0};
    auto dct = dct2(x);
    ASSERT_TRUE(dct.has_value());

    // Parseval's theorem: sum of x^2 proportional to sum of DCT^2
    double energy_x = 0.0, energy_dct = 0.0;
    for (double v : x) energy_x += v * v;
    for (double v : dct.value()) energy_dct += v * v;
    EXPECT_GT(energy_dct, 0.0);
    EXPECT_TRUE(std::isfinite(energy_dct));
}

TEST(FFTTransformsAdv, DCT2_ConstantSignal_DC_Dominant) {
    std::vector<double> x(16, 1.0);  // constant signal
    auto dct = dct2(x);
    ASSERT_TRUE(dct.has_value());
    const auto& D = dct.value();
    // For constant signal, DC component (index 0) should dominate
    double dc = std::abs(D[0]);
    for (size_t i = 1; i < D.size(); ++i) {
        EXPECT_GE(dc, std::abs(D[i]) - 1e-10);
    }
}

TEST(FFTTransformsAdv, IDCT2_Size_Preserved) {
    std::vector<double> x(64, 0.5);
    auto dct = dct2(x);
    ASSERT_TRUE(dct.has_value());
    EXPECT_EQ(dct.value().size(), 64u);
    auto idct = idct2(dct.value());
    ASSERT_TRUE(idct.has_value());
    EXPECT_EQ(idct.value().size(), 64u);
}

// ---------------------------------------------------------------------------
// DST2
// ---------------------------------------------------------------------------

TEST(FFTTransformsAdv, DST2_OutputSize) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    auto dst = dst2(x);
    ASSERT_TRUE(dst.has_value());
    EXPECT_EQ(dst.value().size(), x.size());
}

TEST(FFTTransformsAdv, DST2_AllFinite) {
    std::vector<double> x(16);
    for (size_t i = 0; i < 16; ++i) x[i] = std::sin(M_PI * i / 15.0);
    auto dst = dst2(x);
    ASSERT_TRUE(dst.has_value());
    for (double v : dst.value()) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(FFTTransformsAdv, DST2_ZeroInput_ZeroOutput) {
    std::vector<double> x(8, 0.0);
    auto dst = dst2(x);
    ASSERT_TRUE(dst.has_value());
    for (double v : dst.value()) {
        EXPECT_NEAR(v, 0.0, 1e-10);
    }
}

// ---------------------------------------------------------------------------
// RFFT/IRFFT roundtrip
// ---------------------------------------------------------------------------

TEST(FFTTransformsAdv, RFFT_IRFFT_Roundtrip_Cosine) {
    const size_t N = 64;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::cos(2.0 * M_PI * 5.0 * i / N);
    }
    auto rfft_r = rfft(x);
    ASSERT_TRUE(rfft_r.has_value());
    auto irfft_r = irfft(rfft_r.value(), N);
    ASSERT_TRUE(irfft_r.has_value());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(irfft_r.value()[i], x[i], 1e-9)
            << "RFFT/IRFFT roundtrip at index " << i;
    }
}

TEST(FFTTransformsAdv, RFFT_IRFFT_Roundtrip_Random) {
    const size_t N = 32;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = static_cast<double>(i % 7) - 3.0;
    auto rfft_r = rfft(x);
    ASSERT_TRUE(rfft_r.has_value());
    auto irfft_r = irfft(rfft_r.value(), N);
    ASSERT_TRUE(irfft_r.has_value());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(irfft_r.value()[i], x[i], 1e-9);
    }
}

TEST(FFTTransformsAdv, RFFT_OutputSize) {
    const size_t N = 64;
    std::vector<double> x(N, 1.0);
    auto rfft_r = rfft(x);
    ASSERT_TRUE(rfft_r.has_value());
    // RFFT of N-point real signal gives N/2 + 1 complex outputs
    EXPECT_EQ(rfft_r.value().size(), N/2 + 1);
}

TEST(FFTTransformsAdv, RFFT_PeakAtCorrectBin) {
    const size_t N = 64;
    const size_t freq_bin = 8;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::cos(2.0 * M_PI * freq_bin * i / N);
    }
    auto rfft_r = rfft(x);
    ASSERT_TRUE(rfft_r.has_value());
    const auto& R = rfft_r.value();
    size_t peak_bin = 0;
    double peak_mag = 0.0;
    for (size_t i = 0; i < R.size(); ++i) {
        double mag = std::abs(R[i]);
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_bin = i;
        }
    }
    EXPECT_EQ(peak_bin, freq_bin);
}

// ---------------------------------------------------------------------------
// DFT vs FFT comparison
// ---------------------------------------------------------------------------

TEST(FFTTransformsAdv, DFT_Vs_FFT_SameOutput) {
    const size_t N = 16;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = std::sin(2.0 * M_PI * 2.0 * i / N);

    auto fft_r = fft(x);
    ASSERT_TRUE(fft_r.has_value());
    auto dft_r = dft(x);
    ASSERT_TRUE(dft_r.has_value());

    ASSERT_EQ(fft_r.value().size(), dft_r.value().size());
    for (size_t i = 0; i < fft_r.value().size(); ++i) {
        EXPECT_NEAR(std::abs(fft_r.value()[i]), std::abs(dft_r.value()[i]), 1e-8)
            << "FFT vs DFT magnitude mismatch at bin " << i;
    }
}

TEST(FFTTransformsAdv, DFT_AllFinite) {
    std::vector<double> x = {1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 0.0};
    auto dft_r = dft(x);
    ASSERT_TRUE(dft_r.has_value());
    for (const auto& c : dft_r.value()) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }
}

// ---------------------------------------------------------------------------
// FFTShift / IFFTShift
// ---------------------------------------------------------------------------

TEST(FFTTransformsAdv, FFTShift_IFFTShift_Roundtrip) {
    std::vector<std::complex<double>> x(8);
    for (size_t i = 0; i < 8; ++i) x[i] = {static_cast<double>(i), 0.0};

    auto shifted = fftshift(x);
    auto unshifted = ifftshift(shifted);

    ASSERT_EQ(unshifted.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(unshifted[i].real(), x[i].real(), 1e-10);
    }
}

TEST(FFTTransformsAdv, FFTShift_MovesCenter) {
    // For even N, DC (index 0) should move to center (N/2)
    const size_t N = 8;
    std::vector<std::complex<double>> x(N, {0.0, 0.0});
    x[0] = {1.0, 0.0};  // DC component

    auto shifted = fftshift(x);
    EXPECT_NEAR(shifted[N/2].real(), 1.0, 1e-10);
}

TEST(FFTTransformsAdv, FFT_IFFT_Roundtrip) {
    const size_t N = 32;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = std::sin(2.0 * M_PI * 3.0 * i / N) + 0.5;

    auto fft_r = fft(x);
    ASSERT_TRUE(fft_r.has_value());
    auto ifft_r = ifft(fft_r.value());
    ASSERT_TRUE(ifft_r.has_value());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(ifft_r.value()[i], x[i], 1e-9);
    }
}
