#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// mode
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, Mode_SingleMostFrequent) {
    // {1,2,2,3,3,3} => mode = 3
    const std::vector<double> data = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0};
    EXPECT_NEAR(mode(data), 3.0, 1e-12);
}

TEST(StatsGapsTest, Mode_AllEqual) {
    // All equal: mode = that value
    const std::vector<double> data = {5.0, 5.0, 5.0, 5.0};
    EXPECT_NEAR(mode(data), 5.0, 1e-12);
}

TEST(StatsGapsTest, Mode_TwoValues) {
    // {7,7,3} => mode = 7
    const std::vector<double> data = {7.0, 7.0, 3.0};
    EXPECT_NEAR(mode(data), 7.0, 1e-12);
}

TEST(StatsGapsTest, Mode_SingleElement) {
    const std::vector<double> data = {42.0};
    EXPECT_NEAR(mode(data), 42.0, 1e-12);
}

// ---------------------------------------------------------------------------
// correlation
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, Correlation_PerfectPositive) {
    // y = x: correlation should be 1.0
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(correlation(x, y), 1.0, 1e-10);
}

TEST(StatsGapsTest, Correlation_PerfectNegative) {
    // y = -x: correlation should be -1.0
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> y = {-1.0, -2.0, -3.0, -4.0, -5.0};
    EXPECT_NEAR(correlation(x, y), -1.0, 1e-10);
}

TEST(StatsGapsTest, Correlation_Orthogonal) {
    // Constant y has zero variance => correlation undefined, but check finite result
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y = {1.0, 1.0, 1.0};
    const double r = correlation(x, y);
    EXPECT_TRUE(std::isfinite(r) || std::isnan(r));  // implementation may return NaN/0
}

TEST(StatsGapsTest, Correlation_Known) {
    // Pearson r for ((1,3),(2,4),(3,5),(4,6)) = 1.0
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y = {3.0, 4.0, 5.0, 6.0};
    EXPECT_NEAR(correlation(x, y), 1.0, 1e-10);
}

TEST(StatsGapsTest, Correlation_Range) {
    // Valid Pearson correlation is in [-1, 1]
    const std::vector<double> x = {1.0, 3.0, 2.0, 5.0, 4.0};
    const std::vector<double> y = {2.0, 5.0, 1.0, 8.0, 3.0};
    const double r = correlation(x, y);
    if (std::isfinite(r)) {
        EXPECT_GE(r, -1.0 - 1e-10);
        EXPECT_LE(r,  1.0 + 1e-10);
    }
}

// ---------------------------------------------------------------------------
// binom_cdf: cumulative distribution function
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, BinomCDF_All_K_Leq_N_Is_One) {
    // P(X <= n) = 1
    const double cdf = binom_cdf(4, 4, 0.5);
    EXPECT_NEAR(cdf, 1.0, 1e-10);
}

TEST(StatsGapsTest, BinomCDF_Zero_K_Is_Prob_Zero) {
    // P(X <= 0) = (1-p)^n for fair coin: 0.5^4 = 0.0625
    const double cdf = binom_cdf(0, 4, 0.5);
    EXPECT_NEAR(cdf, 0.0625, 1e-10);
}

TEST(StatsGapsTest, BinomCDF_Monotone) {
    // CDF is non-decreasing
    const double c0 = binom_cdf(0, 5, 0.5);
    const double c1 = binom_cdf(1, 5, 0.5);
    const double c2 = binom_cdf(2, 5, 0.5);
    const double c5 = binom_cdf(5, 5, 0.5);
    EXPECT_LE(c0, c1);
    EXPECT_LE(c1, c2);
    EXPECT_LE(c2, c5);
    EXPECT_NEAR(c5, 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// gamma_cdf (stub — check it doesn't crash)
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, GammaCDF_Not_Tested_Before_Smoke) {
    // gamma_cdf is not in prob.hpp (only gamma_pdf is), so just test gamma_pdf more
    EXPECT_GT(gamma_pdf(1.0, 2.0, 1.0), 0.0);
    EXPECT_NEAR(gamma_pdf(0.0, 1.0, 1.0), 0.0, 1e-12);  // gamma_pdf(0) = 0 for shape>0
}

TEST(StatsGapsTest, GammaPDF_Exponential_Special_Case) {
    // gamma(x; shape=1, scale=1) = exp(-x) = exp_pdf(x, lambda=1)
    const double x = 2.0;
    const double gamma_val = gamma_pdf(x, 1.0, 1.0);
    const double exp_val   = exp_pdf(x, 1.0);
    EXPECT_NEAR(gamma_val, exp_val, 1e-10);
}

// ---------------------------------------------------------------------------
// simd: exp_map (zero coverage)
// ---------------------------------------------------------------------------
#include "ms/simd/simd.hpp"

TEST(StatsGapsTest, SimdExpMap_KnownValues) {
    const std::vector<double> in = {0.0, 1.0, 2.0};
    std::vector<double> out(3, 0.0);
    ms::simd::exp_map(in, out);
    EXPECT_NEAR(out[0], 1.0, 1e-10);
    EXPECT_NEAR(out[1], std::exp(1.0), 1e-10);
    EXPECT_NEAR(out[2], std::exp(2.0), 1e-10);
}

TEST(StatsGapsTest, SimdExpMap_NegativeInput) {
    const std::vector<double> in = {-1.0, -2.0};
    std::vector<double> out(2, 0.0);
    ms::simd::exp_map(in, out);
    EXPECT_NEAR(out[0], std::exp(-1.0), 1e-10);
    EXPECT_NEAR(out[1], std::exp(-2.0), 1e-10);
}

TEST(StatsGapsTest, SimdDot_KnownValue) {
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {4.0, 5.0, 6.0};
    const double result = ms::simd::dot(a, b);
    EXPECT_NEAR(result, 32.0, 1e-10);
}

TEST(StatsGapsTest, SimdScale_KnownValue) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> out(3, 0.0);
    ms::simd::scale(2.5, x, out);
    EXPECT_NEAR(out[0], 2.5, 1e-12);
    EXPECT_NEAR(out[1], 5.0, 1e-12);
    EXPECT_NEAR(out[2], 7.5, 1e-12);
}

TEST(StatsGapsTest, SimdAxpy_InPlace) {
    // y = alpha*x + y
    std::vector<double> y = {1.0, 2.0, 3.0};
    const std::vector<double> x = {1.0, 1.0, 1.0};
    ms::simd::axpy(2.0, x, y);
    EXPECT_NEAR(y[0], 3.0, 1e-12);
    EXPECT_NEAR(y[1], 4.0, 1e-12);
    EXPECT_NEAR(y[2], 5.0, 1e-12);
}
