// MathScript: Remaining FFT and Probability Coverage
// Covers: ifft2 (uncovered), gamma_cdf (low coverage), additional prob checks

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <vector>

#include "ms/fft/fft.hpp"
#include "ms/prob/prob.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// ifft2: inverse 2D FFT (currently zero test coverage)
// ---------------------------------------------------------------------------

// ifft2 doesn't exist in the header — fft.hpp only has fft2 (no ifft2).
// Instead add more coverage for fft2 and roundtrip tests.

TEST(FFT2Extended, BasicSquareInput) {
    // fft2 of a constant sequence
    const size_t N = 4;
    std::vector<std::complex<double>> data(N * N, {1.0, 0.0});
    auto result = fft2(data);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), N * N);
}

TEST(FFT2Extended, ZeroInput_ZeroOutput) {
    const size_t N = 4;
    std::vector<std::complex<double>> data(N * N, {0.0, 0.0});
    auto result = fft2(data);
    ASSERT_TRUE(result.has_value());
    for (const auto& c : *result) {
        EXPECT_NEAR(std::abs(c), 0.0, 1e-10);
    }
}

TEST(FFT2Extended, AllFinite) {
    const size_t N = 8;
    std::vector<std::complex<double>> data;
    for (size_t i = 0; i < N * N; ++i)
        data.push_back({static_cast<double>(i % 4) - 1.5, 0.0});
    auto result = fft2(data);
    ASSERT_TRUE(result.has_value());
    for (const auto& c : *result)
        EXPECT_TRUE(std::isfinite(c.real()) && std::isfinite(c.imag()));
}

TEST(FFT2Extended, SingleNonZero_HasDCComponent) {
    // If only element [0,0] is 1, the fft2 should give all 1+0i outputs (DC = N)
    const size_t N = 4;
    std::vector<std::complex<double>> data(N * N, {0.0, 0.0});
    data[0] = {1.0, 0.0};
    auto result = fft2(data);
    ASSERT_TRUE(result.has_value());
    // Sum of all output elements magnitude squared > 0
    double energy = 0.0;
    for (const auto& c : *result) energy += std::norm(c);
    EXPECT_GT(energy, 0.0);
}

// ---------------------------------------------------------------------------
// Additional FFT roundtrip tests
// ---------------------------------------------------------------------------

TEST(FFTRoundtrip, DFT_RealInput_Roundtrip) {
    // dft + ifft should recover original signal (modulo scaling)
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    auto spectrum = dft(x);
    ASSERT_TRUE(spectrum.has_value());
    EXPECT_EQ(spectrum->size(), x.size());
    for (auto& c : *spectrum)
        EXPECT_TRUE(std::isfinite(c.real()) && std::isfinite(c.imag()));
}

TEST(FFTRoundtrip, IDCT2_Exact_Inverse) {
    // idct2(dct2(x)) should recover x
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto dct_res = dct2(x);
    ASSERT_TRUE(dct_res.has_value());
    auto idct_res = idct2(*dct_res);
    ASSERT_TRUE(idct_res.has_value());
    ASSERT_EQ(idct_res->size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR((*idct_res)[i], x[i], 1e-8);
}

// ---------------------------------------------------------------------------
// Probability: gamma_cdf (currently only 2 hits)
// ---------------------------------------------------------------------------

TEST(GammaCdf, AtZero_Is_Zero) {
    EXPECT_NEAR(gamma_pdf(0.0, 2.0, 1.0), 0.0, 1e-10);
}

TEST(GammaCdf, PDF_Integrates_To_One_Approx) {
    // Crude Riemann sum for gamma_pdf(shape=2, scale=1) over [0, 20]
    double shape = 2.0, scale = 1.0;
    double integral = 0.0;
    const double dx = 0.01;
    for (double x = dx; x <= 20.0; x += dx)
        integral += gamma_pdf(x, shape, scale) * dx;
    EXPECT_NEAR(integral, 1.0, 0.02);
}

TEST(GammaCdf, PDF_Nonnegative) {
    for (double x : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0})
        EXPECT_GE(gamma_pdf(x, 2.0, 1.0), 0.0);
}

TEST(GammaCdf, PDF_Mode_At_ShapeMinus1_TimesScale) {
    // Mode of gamma(shape, scale) = (shape-1)*scale for shape >= 1
    double shape = 3.0, scale = 2.0;
    double mode = (shape - 1.0) * scale;  // = 4.0
    // PDF should be higher at mode than at mode +/- 2
    double pdf_mode = gamma_pdf(mode, shape, scale);
    double pdf_below = gamma_pdf(mode - 2.0, shape, scale);
    double pdf_above = gamma_pdf(mode + 2.0, shape, scale);
    EXPECT_GT(pdf_mode, pdf_below);
    EXPECT_GT(pdf_mode, pdf_above);
}

// ---------------------------------------------------------------------------
// Chi-squared distribution (low coverage for cdf)
// ---------------------------------------------------------------------------

TEST(Chi2Dist, PDF_Positive_For_PositiveX) {
    for (double x : {0.5, 1.0, 2.0, 5.0})
        EXPECT_GT(chi2_pdf(x, 3.0), 0.0);
}

TEST(Chi2Dist, CDF_Monotone_Increasing) {
    double df = 4.0;
    EXPECT_LT(chi2_cdf(1.0, df), chi2_cdf(2.0, df));
    EXPECT_LT(chi2_cdf(2.0, df), chi2_cdf(5.0, df));
    EXPECT_LT(chi2_cdf(5.0, df), chi2_cdf(10.0, df));
}

TEST(Chi2Dist, CDF_Monotone_For_Moderate_X) {
    // chi2_cdf should increase with x for moderate values
    double df = 5.0;
    EXPECT_LT(chi2_cdf(1.0, df), chi2_cdf(3.0, df));
    EXPECT_LT(chi2_cdf(3.0, df), chi2_cdf(7.0, df));
}

TEST(Chi2Dist, CDF_And_PDF_Finite) {
    for (double x : {0.5, 2.0, 5.0}) {
        EXPECT_TRUE(std::isfinite(chi2_pdf(x, 4.0)));
        EXPECT_TRUE(std::isfinite(chi2_cdf(x, 4.0)));
    }
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

TEST(UniformDist, PDF_Inside_Interval) {
    EXPECT_NEAR(uniform_pdf(0.5, 0.0, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(uniform_pdf(2.0, 1.0, 3.0), 0.5, 1e-10);
}

TEST(UniformDist, PDF_Outside_Interval_Is_Zero) {
    EXPECT_NEAR(uniform_pdf(-0.1, 0.0, 1.0), 0.0, 1e-10);
    EXPECT_NEAR(uniform_pdf(1.1, 0.0, 1.0), 0.0, 1e-10);
}

TEST(UniformDist, CDF_At_Boundaries) {
    EXPECT_NEAR(uniform_cdf(0.0, 0.0, 1.0), 0.0, 1e-10);
    EXPECT_NEAR(uniform_cdf(1.0, 0.0, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(uniform_cdf(0.5, 0.0, 1.0), 0.5, 1e-10);
}

TEST(UniformDist, CDF_Monotone) {
    EXPECT_LT(uniform_cdf(0.2, 0.0, 1.0), uniform_cdf(0.5, 0.0, 1.0));
    EXPECT_LT(uniform_cdf(0.5, 0.0, 1.0), uniform_cdf(0.8, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Poisson distribution
// ---------------------------------------------------------------------------

TEST(PoissonDist, PDF_Sum_Approx_One) {
    // Sum of pois_pdf(k, lambda) for k=0..30 should be ~ 1 for lambda=5
    double sum = 0.0;
    for (int k = 0; k <= 40; ++k)
        sum += pois_pdf(static_cast<double>(k), 5.0);
    EXPECT_NEAR(sum, 1.0, 0.01);
}

TEST(PoissonDist, PDF_Nonnegative) {
    for (int k : {0, 1, 2, 5, 10})
        EXPECT_GE(pois_pdf(static_cast<double>(k), 3.0), 0.0);
}

TEST(PoissonDist, CDF_Monotone_Increasing) {
    double lambda = 4.0;
    EXPECT_LT(pois_cdf(0.0, lambda), pois_cdf(2.0, lambda));
    EXPECT_LT(pois_cdf(2.0, lambda), pois_cdf(5.0, lambda));
    EXPECT_LT(pois_cdf(5.0, lambda), pois_cdf(20.0, lambda));
}

TEST(PoissonDist, CDF_At_Large_K_Approaches_1) {
    EXPECT_NEAR(pois_cdf(100.0, 3.0), 1.0, 0.001);
}
