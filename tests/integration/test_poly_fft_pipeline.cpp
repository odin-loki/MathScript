// MathScript Integration: Poly + FFT Spectral Analysis Pipeline (Wave 54)

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>

#include "ms/poly/poly.hpp"
#include "ms/fft/fft.hpp"
#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Poly → sample → FFT
// ---------------------------------------------------------------------------

TEST(PolyFftPipeline, Poly_Sample_FFT_AllFinite) {
    // Sample p(t) = t^2 - 2t + 1 on N points and compute FFT
    std::vector<double> coeffs = {1.0, -2.0, 1.0}; // 1 - 2t + t^2
    const size_t N = 64;
    std::vector<double> samples(N);
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / N;
        auto pv = poly_eval(coeffs, t);
        ASSERT_FALSE(pv.empty());
        samples[i] = pv[0];
    }
    auto Xr = fft(samples);
    ASSERT_TRUE(Xr.has_value());
    for (auto& c : Xr.value()) {
        EXPECT_TRUE(std::isfinite(c.real()));
        EXPECT_TRUE(std::isfinite(c.imag()));
    }
}

TEST(PolyFftPipeline, ConstantPoly_FFT_OnlyDCNonzero) {
    // p(t) = 5.0 (constant) → FFT has only DC component
    std::vector<double> coeffs = {5.0};
    const size_t N = 32;
    std::vector<double> samples(N);
    for (size_t i = 0; i < N; ++i) {
        auto pv = poly_eval(coeffs, static_cast<double>(i));
        ASSERT_FALSE(pv.empty());
        samples[i] = pv[0];
    }
    auto Xr = fft(samples);
    ASSERT_TRUE(Xr.has_value());
    const auto& X = Xr.value();
    // DC component = N * 5
    EXPECT_NEAR(X[0].real(), N * 5.0, 1e-8);
    // All other components should be (near) zero
    for (size_t i = 1; i < N; ++i)
        EXPECT_NEAR(std::abs(X[i]), 0.0, 1e-6) << "Non-DC bin should be zero at i=" << i;
}

TEST(PolyFftPipeline, PolyMul_Samples_FFT_Parseval) {
    // Multiply two polynomials, sample result, check Parseval's theorem
    std::vector<double> p = {1.0, 1.0}; // 1 + x
    std::vector<double> q = {1.0, -1.0}; // 1 - x
    auto pq = poly_mul(p, q); // = 1 - x^2
    const size_t N = 64;
    std::vector<double> samples(N);
    for (size_t i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N;
        auto pv = poly_eval(pq, x);
        ASSERT_FALSE(pv.empty());
        samples[i] = pv[0];
    }
    auto Xr = fft(samples);
    ASSERT_TRUE(Xr.has_value());
    // Parseval: sum|x|^2 = sum|X|^2 / N (for DFT)
    double time_energy = 0.0;
    for (double v : samples) time_energy += v * v;
    double freq_energy = 0.0;
    for (auto& c : Xr.value()) freq_energy += std::norm(c);
    EXPECT_NEAR(freq_energy, N * time_energy, N * time_energy * 0.01);
}

TEST(PolyFftPipeline, PolyAdd_Samples_FFT_Linearity) {
    // fft(p(t) + q(t)) = fft(p(t)) + fft(q(t))
    std::vector<double> p = {1.0, 0.0, 1.0}; // 1 + t^2
    std::vector<double> q = {0.0, 1.0, 0.0}; // t
    auto pq = poly_add(p, q); // 1 + t + t^2
    const size_t N = 32;
    std::vector<double> sp(N), sq(N), spq(N);
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / N;
        auto vp  = poly_eval(p, t);  ASSERT_FALSE(vp.empty());  sp[i]  = vp[0];
        auto vq  = poly_eval(q, t);  ASSERT_FALSE(vq.empty());  sq[i]  = vq[0];
        auto vpq = poly_eval(pq, t); ASSERT_FALSE(vpq.empty()); spq[i] = vpq[0];
    }
    auto Xpr  = fft(sp);
    auto Xqr  = fft(sq);
    auto Xpqr = fft(spq);
    ASSERT_TRUE(Xpr.has_value()); ASSERT_TRUE(Xqr.has_value()); ASSERT_TRUE(Xpqr.has_value());
    const auto& Xp = Xpr.value(); const auto& Xq = Xqr.value(); const auto& Xpq = Xpqr.value();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR((Xp[i] + Xq[i]).real(), Xpq[i].real(), 1e-8);
        EXPECT_NEAR((Xp[i] + Xq[i]).imag(), Xpq[i].imag(), 1e-8);
    }
}

// ---------------------------------------------------------------------------
// Poly derivative → sample → Stats
// ---------------------------------------------------------------------------

TEST(PolyFftPipeline, PolyDeriv_Sample_Stats_Mean) {
    // p(t) = t^2, dp/dt = 2t
    // Mean of dp/dt over [0,1] = integral(2t dt, 0, 1) = [t^2]_0^1 = 1
    std::vector<double> p = {0.0, 0.0, 1.0}; // t^2
    auto dp = poly_deriv(p); // = [0, 2] → 2t
    const size_t N = 1001;
    std::vector<double> samples(N);
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / (N - 1);
        auto pv = poly_eval(dp, t);
        ASSERT_FALSE(pv.empty());
        samples[i] = pv[0];
    }
    double m = mean(samples);
    EXPECT_NEAR(m, 1.0, 0.01) << "Mean of 2t on [0,1] should be ~1";
}

TEST(PolyFftPipeline, PolyDeriv_Zero_At_Minimum) {
    // p(t) = (t-2)^2 = t^2 - 4t + 4
    // dp/dt = 2t - 4 → zero at t=2
    std::vector<double> p = {4.0, -4.0, 1.0};
    auto dp = poly_deriv(p);
    auto v = poly_eval(dp, 2.0);
    ASSERT_FALSE(v.empty());
    EXPECT_NEAR(v[0], 0.0, 1e-10) << "Derivative at minimum should be 0";
}

// ---------------------------------------------------------------------------
// Poly subtraction → residual → Stats
// ---------------------------------------------------------------------------

TEST(PolyFftPipeline, PolyFit_Residual_Stats) {
    // p(t) = t^2 approximately ~ t^2 (exact for true polynomial)
    // Residual p(t) - fitted(t) should be near zero
    std::vector<double> p = {1.0, -2.0, 1.0}; // (1-t)^2
    std::vector<double> residuals;
    for (int i = 0; i <= 10; ++i) {
        double t = 0.1 * i;
        auto vp = poly_eval(p, t); ASSERT_FALSE(vp.empty());
        double exact = (1.0 - t) * (1.0 - t);
        residuals.push_back(std::abs(vp[0] - exact));
    }
    double max_res = max_value(residuals);
    EXPECT_NEAR(max_res, 0.0, 1e-10) << "Poly_eval should match exact analytic expression";
}

// ---------------------------------------------------------------------------
// IFFT(FFT(samples)) roundtrip for polynomial samples
// ---------------------------------------------------------------------------

TEST(PolyFftPipeline, Poly_FFT_IFFT_Roundtrip) {
    std::vector<double> coeffs = {2.0, -1.0, 3.0}; // 2 - x + 3x^2
    const size_t N = 32;
    std::vector<double> samples(N);
    for (size_t i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N;
        auto pv = poly_eval(coeffs, x);
        ASSERT_FALSE(pv.empty());
        samples[i] = pv[0];
    }
    auto Xr = fft(samples);
    ASSERT_TRUE(Xr.has_value());
    auto xrecr = ifft(Xr.value());
    ASSERT_TRUE(xrecr.has_value());
    const auto& xrec = xrecr.value();
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(xrec[i], samples[i], 1e-8)
            << "FFT/IFFT roundtrip failed at i=" << i;
}
