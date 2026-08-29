// MathScript FFT Extended Numerical Reference Tests
// Tests: fft2, rfft/irfft roundtrip, and additional FFT properties

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <complex>
#include <numeric>

#include "ms/fft/fft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// rfft / irfft roundtrip
// ---------------------------------------------------------------------------

TEST(FFT_RealFFT, RFFT_SizePreservation) {
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8};
    auto result = rfft(x);
    ASSERT_TRUE(result.has_value());
    // rfft of N-point real signal gives N/2+1 complex values
    EXPECT_EQ(result->size(), x.size() / 2 + 1);
}

TEST(FFT_RealFFT, RFFT_IRFFT_Roundtrip) {
    std::vector<double> x = {1.0, 2.0, -1.0, 0.5, 3.0, -0.5, 2.5, 1.0};
    const size_t N = x.size();
    auto spectrum = rfft(x);
    ASSERT_TRUE(spectrum.has_value());
    auto recovered = irfft(*spectrum, N);
    ASSERT_TRUE(recovered.has_value());
    ASSERT_EQ(recovered->size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR((*recovered)[i], x[i], 1e-8);
}

TEST(FFT_RealFFT, RFFT_FiniteOutput) {
    std::vector<double> x = {3.0, 1.0, -2.0, 0.0, 4.0, -1.0, 2.0, 0.5};
    auto r = rfft(x);
    ASSERT_TRUE(r.has_value());
    for (auto c : *r) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }
}

TEST(FFT_RealFFT, RFFT_ZeroInput_ZeroOutput) {
    std::vector<double> x(8, 0.0);
    auto r = rfft(x);
    ASSERT_TRUE(r.has_value());
    for (auto c : *r) {
        EXPECT_NEAR(c.real(), 0.0, 1e-10);
        EXPECT_NEAR(c.imag(), 0.0, 1e-10);
    }
}

TEST(FFT_RealFFT, RFFT_Constant_First_Coeff_Is_Sum) {
    // For constant signal x[n]=1, rfft[0] = N (sum of all values)
    std::vector<double> x(8, 1.0);
    auto r = rfft(x);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR((*r)[0].real(), static_cast<double>(x.size()), 1e-8);
    EXPECT_NEAR((*r)[0].imag(), 0.0, 1e-10);
}

TEST(FFT_RealFFT, IRFFT_SizePreservation) {
    std::vector<double> x = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    auto spectrum = rfft(x);
    ASSERT_TRUE(spectrum.has_value());
    auto r = irfft(*spectrum, x.size());
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->size(), x.size());
}

// ---------------------------------------------------------------------------
// fft2 (2D FFT, but input is 1D vector; result is same size)
// ---------------------------------------------------------------------------

TEST(FFT_2D, FFT2_SizePreservation) {
    // fft2 may treat the input as a 1D complex signal
    std::vector<std::complex<double>> x = {
        {1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0}
    };
    auto result = fft2(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), x.size());
}

TEST(FFT_2D, FFT2_FiniteOutput) {
    std::vector<std::complex<double>> x(8);
    for (int i = 0; i < 8; ++i) x[i] = {static_cast<double>(i), 0.0};
    auto result = fft2(x);
    ASSERT_TRUE(result.has_value());
    for (auto c : *result) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }
}

TEST(FFT_2D, FFT2_Zero_Input_Zero_Output) {
    std::vector<std::complex<double>> x(8, {0.0, 0.0});
    auto result = fft2(x);
    ASSERT_TRUE(result.has_value());
    for (auto c : *result) {
        EXPECT_NEAR(c.real(), 0.0, 1e-10);
        EXPECT_NEAR(c.imag(), 0.0, 1e-10);
    }
}

// ---------------------------------------------------------------------------
// FFT energy conservation (Parseval's theorem)
// fft(real) -> complex, sum |X[k]|^2 = N * sum |x[n]|^2
// ---------------------------------------------------------------------------

TEST(FFT_Parseval, Energy_Conservation) {
    std::vector<double> x = {1.0, 2.0, -1.0, 0.5, 3.0, -0.5, 2.5, 1.0};
    const size_t N = x.size();
    
    double signal_energy = 0.0;
    for (double v : x) signal_energy += v * v;
    
    auto spectrum = fft(x);  // takes real vector
    ASSERT_TRUE(spectrum.has_value());
    
    double spectral_energy = 0.0;
    for (auto c : *spectrum) spectral_energy += std::norm(c);  // norm = |c|^2
    
    // Parseval: spectral_energy = N * signal_energy
    EXPECT_NEAR(spectral_energy, N * signal_energy, 1e-6);
}

TEST(FFT_Parseval, RFFT_Energy_Conservation) {
    std::vector<double> x = {1.0, 0.5, -1.0, 0.25, 2.0, -0.5, 1.5, 0.75};
    const size_t N = x.size();
    
    double signal_energy = 0.0;
    for (double v : x) signal_energy += v * v;
    
    auto spectrum = rfft(x);
    ASSERT_TRUE(spectrum.has_value());
    
    // For rfft with N=8 (even), N/2+1=5 coefficients
    // Parseval: E = (|X[0]|^2 + 2*sum_{k=1}^{N/2-1}|X[k]|^2 + |X[N/2]|^2) / N
    double spectral_energy = std::norm((*spectrum)[0]);
    for (size_t k = 1; k < spectrum->size() - 1; ++k)
        spectral_energy += 2.0 * std::norm((*spectrum)[k]);
    spectral_energy += std::norm(spectrum->back());
    spectral_energy /= N;
    
    EXPECT_NEAR(spectral_energy, signal_energy, 1e-6);
}

// ---------------------------------------------------------------------------
// FFT: linearity property
// FFT(a*x + b*y) = a*FFT(x) + b*FFT(y) (real inputs)
// ---------------------------------------------------------------------------

TEST(FFT_Properties, Linearity) {
    const int N = 8;
    std::vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = std::sin(2 * M_PI * i / N);
        y[i] = std::cos(2 * M_PI * i / N);
    }
    double a = 2.0, b = 3.0;
    
    auto Fx = fft(x);
    auto Fy = fft(y);
    ASSERT_TRUE(Fx.has_value());
    ASSERT_TRUE(Fy.has_value());
    
    // Compute a*x + b*y
    std::vector<double> xy(N);
    for (int i = 0; i < N; ++i) xy[i] = a * x[i] + b * y[i];
    auto Fxy = fft(xy);
    ASSERT_TRUE(Fxy.has_value());
    
    for (int i = 0; i < N; ++i) {
        std::complex<double> expected = a * (*Fx)[i] + b * (*Fy)[i];
        EXPECT_NEAR((*Fxy)[i].real(), expected.real(), 1e-8);
        EXPECT_NEAR((*Fxy)[i].imag(), expected.imag(), 1e-8);
    }
}
