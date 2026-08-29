// MathScript FFT Extended Numerical Reference Tests

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <vector>
#include <numeric>

#include "ms/fft/fft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using CplxVec = std::vector<std::complex<double>>;

// ---------------------------------------------------------------------------
// DFT: direct Discrete Fourier Transform
// ---------------------------------------------------------------------------

TEST(NumericalFftExt, DFT_ImpulseAllOnes) {
    // DFT of a unit impulse at index 0: all bins = 1
    const std::vector<double> x(8, 0.0);
    // DFT of constant: bin 0 = N
    const std::vector<double> ones(8, 1.0);
    const auto result = dft(ones);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 8u);
    // Bin 0 should have magnitude 8
    EXPECT_NEAR(std::abs((*result)[0]), 8.0, 1e-9);
    // Other bins should be 0
    for (size_t i = 1; i < 8; ++i)
        EXPECT_NEAR(std::abs((*result)[i]), 0.0, 1e-9);
}

TEST(NumericalFftExt, DFT_SingleToneRealPart) {
    // DFT of cos(2pi*k/N*n) should have peaks at bins k and N-k
    const size_t N = 8;
    const size_t k = 1;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::cos(2.0 * M_PI * k * i / N);
    const auto result = dft(x);
    ASSERT_TRUE(result.has_value());
    // Bin k should have large magnitude
    EXPECT_GT(std::abs((*result)[k]), 3.0);
}

TEST(NumericalFftExt, DFT_AgreesWithFFT) {
    // DFT and FFT should give the same result
    const size_t N = 8;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i) + 0.5;
    const auto fft_result = fft(x);
    const auto dft_result = dft(x);
    ASSERT_TRUE(fft_result.has_value());
    ASSERT_TRUE(dft_result.has_value());
    ASSERT_EQ(fft_result->size(), dft_result->size());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(std::abs((*fft_result)[i] - (*dft_result)[i]), 0.0, 1e-8)
            << "DFT and FFT disagree at bin " << i;
    }
}

// ---------------------------------------------------------------------------
// RFFT / IRFFT roundtrip
// ---------------------------------------------------------------------------

TEST(NumericalFftExt, RFFT_IRFFT_Roundtrip_N8) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto rfft_result = rfft(x);
    ASSERT_TRUE(rfft_result.has_value());
    const auto irfft_result = irfft(*rfft_result, x.size());
    ASSERT_TRUE(irfft_result.has_value());
    ASSERT_EQ(irfft_result->size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR((*irfft_result)[i], x[i], 1e-10);
}

TEST(NumericalFftExt, RFFT_OutputSize) {
    // rfft of N-point real signal returns N/2+1 complex bins
    const std::vector<double> x(16, 1.0);
    const auto result = rfft(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 9u);  // N/2 + 1 = 9
}

TEST(NumericalFftExt, RFFT_DC_Component) {
    // DC component (bin 0) of rfft of ones = N
    const size_t N = 16;
    const std::vector<double> x(N, 1.0);
    const auto result = rfft(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(std::abs((*result)[0]), static_cast<double>(N), 1e-9);
}

// ---------------------------------------------------------------------------
// FFTShift / IFFTShift
// ---------------------------------------------------------------------------

TEST(NumericalFftExt, FFTShift_SwapsHalves) {
    // fftshift moves zero frequency to center
    const CplxVec x = {{1,0}, {2,0}, {3,0}, {4,0}};
    const auto shifted = fftshift(x);
    ASSERT_EQ(shifted.size(), 4u);
    // For N=4: fftshift swaps [0,1,2,3] -> [2,3,0,1]
    EXPECT_NEAR(shifted[0].real(), 3.0, 1e-12);
    EXPECT_NEAR(shifted[1].real(), 4.0, 1e-12);
}

TEST(NumericalFftExt, FFTShift_IFFTShift_Roundtrip) {
    // ifftshift(fftshift(x)) = x
    const CplxVec x = {{1,0}, {2,1}, {3,-1}, {4,2}, {5,0}, {6,-2}};
    const auto shifted   = fftshift(x);
    const auto unshifted = ifftshift(shifted);
    ASSERT_EQ(unshifted.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(unshifted[i].real(), x[i].real(), 1e-12);
}

// ---------------------------------------------------------------------------
// DCT2 / IDCT2 roundtrip
// ---------------------------------------------------------------------------

TEST(NumericalFftExt, DCT2_IDCT2_Roundtrip) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto dct_result = dct2(x);
    ASSERT_TRUE(dct_result.has_value());
    const auto idct_result = idct2(*dct_result);
    ASSERT_TRUE(idct_result.has_value());
    ASSERT_EQ(idct_result->size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR((*idct_result)[i], x[i], 1e-8);
}

TEST(NumericalFftExt, DCT2_Constant_DC_Dominant) {
    // DCT2 of constant input: all energy in DC bin
    const size_t N = 8;
    const std::vector<double> x(N, 3.0);
    const auto result = dct2(x);
    ASSERT_TRUE(result.has_value());
    // DC bin should dominate
    const double dc = std::abs((*result)[0]);
    for (size_t i = 1; i < N; ++i)
        EXPECT_LT(std::abs((*result)[i]), dc + 1e-8);
}

// ---------------------------------------------------------------------------
// DST2
// ---------------------------------------------------------------------------

TEST(NumericalFftExt, DST2_OutputSize) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const auto result = dst2(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 4u);
}

TEST(NumericalFftExt, DST2_FiniteOutputs) {
    const std::vector<double> x = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
    const auto result = dst2(x);
    ASSERT_TRUE(result.has_value());
    for (double v : *result)
        EXPECT_TRUE(std::isfinite(v));
}
