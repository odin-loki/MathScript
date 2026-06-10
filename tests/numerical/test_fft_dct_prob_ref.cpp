// MathScript FFT DCT/DST, RNG, and Probability Distribution Numerical Reference Tests

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

#include "ms/fft/fft.hpp"
#include "ms/core/rng.hpp"
#include "ms/prob/prob.hpp"
#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// DCT-2 / IDCT-2 roundtrip
// ---------------------------------------------------------------------------

TEST(FFT_DCT, DCT2_SizePreservation) {
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto result = dct2(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), x.size());
}

TEST(FFT_DCT, IDCT2_SizePreservation) {
    std::vector<double> x = {1, 2, 3, 4};
    const auto c = dct2(x);
    ASSERT_TRUE(c.has_value());
    const auto r = idct2(*c);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->size(), x.size());
}

TEST(FFT_DCT, DCT2_IDCT2_Roundtrip) {
    std::vector<double> x = {1.0, 2.0, -1.0, 0.5, 3.0, -0.5, 2.5, 1.0};
    const auto c = dct2(x);
    ASSERT_TRUE(c.has_value());
    const auto r = idct2(*c);
    ASSERT_TRUE(r.has_value());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR((*r)[i], x[i], 1e-8);
}

TEST(FFT_DCT, DCT2_Constant_Signal) {
    // DCT-2 of constant signal: first coefficient = N*val*sqrt(2), rest ~0
    std::vector<double> x(8, 1.0);
    const auto c = dct2(x);
    ASSERT_TRUE(c.has_value());
    // First coeff should be largest in magnitude
    double first = std::abs((*c)[0]);
    for (size_t i = 1; i < c->size(); ++i)
        EXPECT_GT(first, std::abs((*c)[i]) - 1e-8);
}

TEST(FFT_DCT, DCT2_Finite_Output) {
    std::vector<double> x = {3.0, 1.0, -2.0, 0.0, 4.0, -1.0, 2.0, 0.5};
    const auto c = dct2(x);
    ASSERT_TRUE(c.has_value());
    for (auto v : *c) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// DST-2
// ---------------------------------------------------------------------------

TEST(FFT_DST, DST2_SizePreservation) {
    std::vector<double> x = {1, 2, 3, 4};
    const auto result = dst2(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), x.size());
}

TEST(FFT_DST, DST2_Finite_Output) {
    std::vector<double> x = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0};
    const auto c = dst2(x);
    ASSERT_TRUE(c.has_value());
    for (auto v : *c) EXPECT_TRUE(std::isfinite(v));
}

TEST(FFT_DST, DST2_Zero_Input_Is_Zero_Output) {
    std::vector<double> x(8, 0.0);
    const auto c = dst2(x);
    ASSERT_TRUE(c.has_value());
    for (auto v : *c) EXPECT_NEAR(v, 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// fftshift / ifftshift (operate on complex vectors)
// ---------------------------------------------------------------------------

TEST(FFT_Shift, Ifftshift_Undoes_Fftshift) {
    std::vector<std::complex<double>> x = {{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0}};
    const auto shifted = fftshift(x);
    const auto back = ifftshift(shifted);
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(back[i].real(), x[i].real(), 1e-12);
}

TEST(FFT_Shift, Ifftshift_Size_Preserved) {
    std::vector<std::complex<double>> x = {{1,0},{2,0},{3,0},{4,0}};
    const auto shifted = fftshift(x);
    const auto back = ifftshift(shifted);
    EXPECT_EQ(back.size(), x.size());
}

// ---------------------------------------------------------------------------
// RNG: session_uniform and session_normal
// Must call set_session_rng(uniformFn, normalFn) before using session_* functions.
// ---------------------------------------------------------------------------

static double uniform_impl() {
    thread_local unsigned int s = 12345;
    s = s * 1664525u + 1013904223u;
    return (s >> 8) / double(1 << 24);
}

static double normal_impl() {
    // Box-Muller
    double u1 = std::max(uniform_impl(), 1e-15);
    double u2 = uniform_impl();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

struct RngFixture : public ::testing::Test {
    void SetUp() override { set_session_rng(uniform_impl, normal_impl); }
    void TearDown() override { clear_session_rng(); }
};

TEST_F(RngFixture, SessionRngActive_AfterSet) {
    EXPECT_TRUE(session_rng_active());
}

TEST_F(RngFixture, SessionUniform_InUnitRange) {
    for (int i = 0; i < 50; ++i) {
        double v = session_uniform();
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST_F(RngFixture, SessionNormal_ReturnsFiniteValues) {
    for (int i = 0; i < 50; ++i) {
        double v = session_normal();
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST_F(RngFixture, SessionNormal_Mean_Reasonable) {
    std::vector<double> samples;
    for (int i = 0; i < 500; ++i)
        samples.push_back(session_normal());
    double m = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    EXPECT_NEAR(m, 0.0, 0.5);  // loose tolerance
}

TEST(RNG_Session, SessionRngInactive_BeforeSet) {
    clear_session_rng();
    // After clearing, rng should be inactive
    EXPECT_FALSE(session_rng_active());
}

// ---------------------------------------------------------------------------
// Probability distributions: norm_pdf, exp_pdf, binom_pdf, t_pdf, t_cdf
// ---------------------------------------------------------------------------

TEST(ProbDist, NormPdf_AtMean_MaxValue) {
    // norm_pdf peaks at mean
    EXPECT_GT(norm_pdf(0.0, 0.0, 1.0), norm_pdf(1.0, 0.0, 1.0));
    EXPECT_GT(norm_pdf(0.0, 0.0, 1.0), norm_pdf(-1.0, 0.0, 1.0));
}

TEST(ProbDist, NormPdf_Standard_AtZero) {
    // Standard normal: pdf(0) = 1/sqrt(2*pi) ≈ 0.3989
    EXPECT_NEAR(norm_pdf(0.0, 0.0, 1.0), 1.0 / std::sqrt(2.0 * M_PI), 1e-8);
}

TEST(ProbDist, NormPdf_Symmetric) {
    EXPECT_NEAR(norm_pdf(1.0, 0.0, 1.0), norm_pdf(-1.0, 0.0, 1.0), 1e-12);
}

TEST(ProbDist, NormPdf_Always_Positive) {
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0})
        EXPECT_GT(norm_pdf(x, 0.0, 1.0), 0.0);
}

TEST(ProbDist, ExpPdf_AtZero_IsLambda) {
    // exp_pdf(0, lambda) = lambda
    EXPECT_NEAR(exp_pdf(0.0, 2.0), 2.0, 1e-10);
    EXPECT_NEAR(exp_pdf(0.0, 0.5), 0.5, 1e-10);
}

TEST(ProbDist, ExpPdf_Decreasing) {
    EXPECT_GT(exp_pdf(0.0, 1.0), exp_pdf(1.0, 1.0));
    EXPECT_GT(exp_pdf(1.0, 1.0), exp_pdf(2.0, 1.0));
}

TEST(ProbDist, ExpPdf_Negative_Returns_Zero) {
    // PDF is 0 for x < 0
    EXPECT_EQ(exp_pdf(-1.0, 1.0), 0.0);
}

TEST(ProbDist, BinomPdf_Sum_To_One) {
    // sum over k=0..n of binom_pdf(k,n,p) = 1
    double total = 0.0;
    for (int k = 0; k <= 10; ++k)
        total += binom_pdf(k, 10, 0.4);
    EXPECT_NEAR(total, 1.0, 1e-8);
}

TEST(ProbDist, BinomPdf_Zero_And_N) {
    // binom_pdf(0, n, 0) = 1
    EXPECT_NEAR(binom_pdf(0, 5, 0.0), 1.0, 1e-12);
    // binom_pdf(n, n, 1) = 1
    EXPECT_NEAR(binom_pdf(5, 5, 1.0), 1.0, 1e-12);
}

TEST(ProbDist, BinomPdf_Peak_Near_Mode) {
    // For n=10, p=0.5, mode is k=5
    int n = 10;
    double p = 0.5;
    double pk5 = binom_pdf(5, n, p);
    EXPECT_GT(pk5, binom_pdf(0, n, p));
    EXPECT_GT(pk5, binom_pdf(10, n, p));
}

TEST(ProbDist, TPdf_Symmetric) {
    EXPECT_NEAR(t_pdf(1.0, 5.0), t_pdf(-1.0, 5.0), 1e-12);
}

TEST(ProbDist, TPdf_Peak_At_Zero) {
    EXPECT_GT(t_pdf(0.0, 5.0), t_pdf(1.0, 5.0));
    EXPECT_GT(t_pdf(0.0, 5.0), t_pdf(-1.0, 5.0));
}

TEST(ProbDist, TPdf_Positive) {
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0})
        EXPECT_GT(t_pdf(x, 10.0), 0.0);
}

TEST(ProbDist, TCdf_At_Zero_Is_Half) {
    // t distribution is symmetric around 0
    EXPECT_NEAR(t_cdf(0.0, 5.0), 0.5, 1e-8);
}

TEST(ProbDist, TCdf_Monotone) {
    EXPECT_LT(t_cdf(-1.0, 10.0), t_cdf(0.0, 10.0));
    EXPECT_LT(t_cdf(0.0, 10.0), t_cdf(1.0, 10.0));
}
