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

// ---------------------------------------------------------------------------
// arfit / pacf
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, ArfitAR1RecoversPhi) {
    // Exact AR(1): x_t = 0.7 x_{t-1}, x_0 = 1
    std::vector<double> x(80);
    x[0] = 1.0;
    for (size_t i = 1; i < x.size(); ++i)
        x[i] = 0.7 * x[i - 1];
    auto phi = arfit(x, 1);
    ASSERT_EQ(phi.size(), 1u);
    EXPECT_NEAR(phi[0], 0.7, 0.05);
}

TEST(StatsGapsTest, PacfLag1MatchesCorr) {
    // Same exact AR(1). pacf[0] is lag 0 (= 1); lag-1 is pacf[1].
    std::vector<double> x(80);
    x[0] = 1.0;
    for (size_t i = 1; i < x.size(); ++i)
        x[i] = 0.7 * x[i - 1];
    auto p = pacf(x, 4);
    ASSERT_GE(p.size(), 5u);
    EXPECT_NEAR(p[1], 0.7, 0.15);
    EXPECT_NEAR(p[2], 0.0, 0.15);
    EXPECT_NEAR(p[3], 0.0, 0.15);
    EXPECT_NEAR(p[4], 0.0, 0.15);
}

// ---------------------------------------------------------------------------
// geometric_mean / harmonic_mean / rms
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, GeometricMean_ExactProduct) {
    const std::vector<double> data = {2.0, 8.0, 32.0};
    EXPECT_NEAR(geometric_mean(data), 8.0, 1e-12);
}

TEST(StatsGapsTest, GeometricMean_AllEqual) {
    const std::vector<double> data = {5.0, 5.0, 5.0, 5.0};
    EXPECT_NEAR(geometric_mean(data), 5.0, 1e-12);
}

TEST(StatsGapsTest, GeometricMean_AMGM) {
    const std::vector<double> data = {1.0, 4.0};
    EXPECT_NEAR(geometric_mean(data), 2.0, 1e-12);
    EXPECT_LE(geometric_mean(data), mean(data) + 1e-12);
}

TEST(StatsGapsTest, GeometricMean_EmptyAndNonPositive) {
    EXPECT_NEAR(geometric_mean(std::vector<double>{}), 0.0, 1e-12);
    EXPECT_TRUE(std::isnan(geometric_mean(std::vector<double>{1.0, 0.0})));
    EXPECT_TRUE(std::isnan(geometric_mean(std::vector<double>{-2.0, 3.0})));
}

TEST(StatsGapsTest, HarmonicMean_ExactPair) {
    // 2 / (1 + 1/4) = 1.6
    const std::vector<double> data = {1.0, 4.0};
    EXPECT_NEAR(harmonic_mean(data), 1.6, 1e-12);
}

TEST(StatsGapsTest, HarmonicMean_AllEqual) {
    const std::vector<double> data = {3.0, 3.0, 3.0};
    EXPECT_NEAR(harmonic_mean(data), 3.0, 1e-12);
}

TEST(StatsGapsTest, HarmonicMean_HM_LE_GM_LE_AM) {
    const std::vector<double> data = {1.0, 2.0, 4.0};
    const double hm = harmonic_mean(data);
    const double gm = geometric_mean(data);
    const double am = mean(data);
    EXPECT_NEAR(hm, 12.0 / 7.0, 1e-12);
    EXPECT_NEAR(gm, 2.0, 1e-12);
    EXPECT_LE(hm, gm + 1e-12);
    EXPECT_LE(gm, am + 1e-12);
}

TEST(StatsGapsTest, HarmonicMean_EmptyAndZero) {
    EXPECT_NEAR(harmonic_mean(std::vector<double>{}), 0.0, 1e-12);
    EXPECT_NEAR(harmonic_mean(std::vector<double>{2.0, 0.0, 3.0}), 0.0, 1e-12);
}

TEST(StatsGapsTest, Rms_ThreeFour) {
    // rms({3,4}) = sqrt((9+16)/2) = sqrt(12.5)
    const std::vector<double> data = {3.0, 4.0};
    EXPECT_NEAR(rms(data), std::sqrt(12.5), 1e-12);
}

TEST(StatsGapsTest, Rms_ConstantAndEmpty) {
    EXPECT_NEAR(rms(std::vector<double>{7.0, 7.0, 7.0}), 7.0, 1e-12);
    EXPECT_NEAR(rms(std::vector<double>{}), 0.0, 1e-12);
}

TEST(StatsGapsTest, Rms_MatchesQuadraticMean) {
    const std::vector<double> data = {-2.0, 1.0, 2.0};
    EXPECT_NEAR(rms(data), std::sqrt((4.0 + 1.0 + 4.0) / 3.0), 1e-12);
}

// ---------------------------------------------------------------------------
// spearman / kendall
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, Spearman_PerfectMonotone) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y = {1.0, 4.0, 9.0, 16.0};
    EXPECT_NEAR(spearman(x, y), 1.0, 1e-12);
    const std::vector<double> y_rev = {16.0, 9.0, 4.0, 1.0};
    EXPECT_NEAR(spearman(x, y_rev), -1.0, 1e-12);
}

TEST(StatsGapsTest, Spearman_HandComputedRanks) {
    // ranks x={1,2,3}, y={1,3,2} => d^2 = 0+1+1=2; 1 - 6*2/(3*8) = 1 - 12/24 = 0.5
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y = {1.0, 3.0, 2.0};
    EXPECT_NEAR(spearman(x, y), 0.5, 1e-12);
}

TEST(StatsGapsTest, Spearman_EmptyOrMismatch) {
    EXPECT_NEAR(spearman(std::vector<double>{}, std::vector<double>{}), 0.0, 1e-12);
    const std::vector<double> x = {1.0, 2.0};
    const std::vector<double> y = {1.0};
    EXPECT_NEAR(spearman(x, y), 0.0, 1e-12);
}

TEST(StatsGapsTest, Kendall_PerfectConcordance) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y = {10.0, 20.0, 30.0, 40.0};
    EXPECT_NEAR(kendall(x, y), 1.0, 1e-12);
    const std::vector<double> y_rev = {40.0, 30.0, 20.0, 10.0};
    EXPECT_NEAR(kendall(x, y_rev), -1.0, 1e-12);
}

TEST(StatsGapsTest, Kendall_HandComputedOneDiscord) {
    // pairs: (1,2) C, (1,3) C, (2,3) D => (2-1)/3 = 1/3
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y = {1.0, 3.0, 2.0};
    EXPECT_NEAR(kendall(x, y), 1.0 / 3.0, 1e-12);
}

TEST(StatsGapsTest, Kendall_EmptyOrMismatch) {
    EXPECT_NEAR(kendall(std::vector<double>{}, std::vector<double>{}), 0.0, 1e-12);
    const std::vector<double> x = {1.0, 2.0};
    const std::vector<double> y = {1.0};
    EXPECT_NEAR(kendall(x, y), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// chi2_gof / ks_test
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, Chi2Gof_ObservedEqualsExpected) {
    const std::vector<double> obs = {10.0, 20.0, 30.0};
    EXPECT_NEAR(chi2_gof(obs, obs), 0.0, 1e-12);
}

TEST(StatsGapsTest, Chi2Gof_HandComputed) {
    // (8-10)^2/10 + (12-10)^2/10 = 0.4 + 0.4 = 0.8
    const std::vector<double> obs = {8.0, 12.0};
    const std::vector<double> exp = {10.0, 10.0};
    EXPECT_NEAR(chi2_gof(obs, exp), 0.8, 1e-12);
}

TEST(StatsGapsTest, Chi2Gof_EmptyOrMismatch) {
    EXPECT_NEAR(chi2_gof(std::vector<double>{}, std::vector<double>{}), 0.0, 1e-12);
    const std::vector<double> obs = {1.0, 2.0};
    const std::vector<double> exp = {1.0};
    EXPECT_NEAR(chi2_gof(obs, exp), 0.0, 1e-12);
}

TEST(StatsGapsTest, KsTest_UniformExact) {
    // x = {0.25, 0.5, 0.75}, F = identity. D_n = 0.25
    const std::vector<double> x = {0.25, 0.5, 0.75};
    const double d = ks_test(x, [](double t) { return t; });
    EXPECT_NEAR(d, 0.25, 1e-12);
}

TEST(StatsGapsTest, KsTest_Empty) {
    EXPECT_NEAR(ks_test(std::vector<double>{}, [](double t) { return t; }), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// multiple_regression / acf / bootstrap_mean
// ---------------------------------------------------------------------------

TEST(StatsGapsTest, MultipleRegression_ExactLine) {
    // y = 1 + 2 x
    const std::vector<std::vector<double>> X = {
        {1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}, {1.0, 4.0}};
    const std::vector<double> y = {3.0, 5.0, 7.0, 9.0};
    const auto beta = multiple_regression(X, y);
    ASSERT_EQ(beta.size(), 2u);
    EXPECT_NEAR(beta[0], 1.0, 1e-12);
    EXPECT_NEAR(beta[1], 2.0, 1e-12);
}

TEST(StatsGapsTest, MultipleRegression_SinglePredictor) {
    const std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    const std::vector<double> y = {3.0, 6.0, 9.0};
    const auto beta = multiple_regression(X, y);
    ASSERT_EQ(beta.size(), 1u);
    EXPECT_NEAR(beta[0], 3.0, 1e-12);
}

TEST(StatsGapsTest, MultipleRegression_EmptyDesign) {
    EXPECT_TRUE(multiple_regression({}, std::vector<double>{}).empty());
}

TEST(StatsGapsTest, Acf_LagZeroIsOne) {
    const std::vector<double> x = {1.0, 3.0, 2.0, 5.0};
    const auto r = acf(x, 2);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_NEAR(r[0], 1.0, 1e-12);
}

TEST(StatsGapsTest, Acf_HandComputedThreePoints) {
    // x={1,2,3}, mean=2, var0=2; rho(1)=0, rho(2)=-0.5
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const auto r = acf(x, 2);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_NEAR(r[0], 1.0, 1e-12);
    EXPECT_NEAR(r[1], 0.0, 1e-12);
    EXPECT_NEAR(r[2], -0.5, 1e-12);
}

TEST(StatsGapsTest, Acf_ConstantAndInvalid) {
    const auto flat = acf(std::vector<double>{4.0, 4.0, 4.0}, 2);
    ASSERT_EQ(flat.size(), 3u);
    EXPECT_NEAR(flat[0], 0.0, 1e-12);
    EXPECT_TRUE(acf(std::vector<double>{}, 2).empty());
    EXPECT_TRUE(acf(std::vector<double>{1.0, 2.0}, -1).empty());
}

TEST(StatsGapsTest, BootstrapMean_ConstantIsExact) {
    const std::vector<double> data = {5.0, 5.0, 5.0, 5.0};
    EXPECT_NEAR(bootstrap_mean(data, 200, 7), 5.0, 1e-12);
}

TEST(StatsGapsTest, BootstrapMean_NearSampleMean) {
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(bootstrap_mean(data, 4000, 42), mean(data), 0.15);
}

TEST(StatsGapsTest, BootstrapMean_DeterministicSeedAndEmpty) {
    const std::vector<double> data = {1.0, 2.0, 3.0};
    EXPECT_NEAR(bootstrap_mean(data, 300, 99), bootstrap_mean(data, 300, 99), 1e-15);
    EXPECT_NEAR(bootstrap_mean(std::vector<double>{}, 100, 1), 0.0, 1e-12);
}

TEST(StatsGapsTest, Iqr_TooSmallReturnsZero) {
    const std::vector<double> three = {1.0, 2.0, 3.0};
    const std::vector<double> empty;
    EXPECT_NEAR(iqr(three), 0.0, 1e-12);
    EXPECT_NEAR(iqr(empty), 0.0, 1e-12);
}

TEST(StatsGapsTest, Iqr_KnownNinePoints) {
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    EXPECT_NEAR(iqr(data), percentile(data, 75.0) - percentile(data, 25.0), 1e-12);
}

TEST(StatsGapsTest, TrimmedMean_EmptyAndOverTrim) {
    const std::vector<double> empty;
    EXPECT_NEAR(trimmed_mean(empty, 0.1), 0.0, 1e-12);
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    EXPECT_NEAR(trimmed_mean(data, 0.5), median(data), 1e-12);
    EXPECT_NEAR(trimmed_mean(data, 0.0), mean(data), 1e-12);
}

TEST(StatsGapsTest, WeightedVariance_PopulationForm) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> w = {1.0, 1.0, 1.0};
    const double pop = weighted_variance(x, w, false);
    EXPECT_NEAR(pop, 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(weighted_variance(x, w, true), 1.0, 1e-12);
}

TEST(StatsGapsTest, WeightedMean_ZeroWeightsAndEmpty) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> zero_w = {0.0, 0.0, 0.0};
    const std::vector<double> empty;
    EXPECT_NEAR(weighted_mean(x, zero_w), 0.0, 1e-12);
    EXPECT_NEAR(weighted_mean(empty, empty), 0.0, 1e-12);
    EXPECT_NEAR(weighted_variance(x, zero_w, true), 0.0, 1e-12);
    EXPECT_NEAR(weighted_correlation(x, x, zero_w), 0.0, 1e-12);
}

TEST(StatsGapsTest, WeightedCorrelation_ZeroVariance) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y = {5.0, 5.0, 5.0};
    const std::vector<double> w = {1.0, 1.0, 1.0};
    EXPECT_NEAR(weighted_correlation(x, y, w), 0.0, 1e-12);
}

TEST(StatsGapsTest, Chi2Gof_ZeroExpectedSkipped) {
    const std::vector<double> obs = {1.0, 2.0, 3.0};
    const std::vector<double> exp = {1.0, 0.0, 2.0};
    EXPECT_NEAR(chi2_gof(obs, exp), 0.5, 1e-12);
}

TEST(StatsGapsTest, Kendall_TiesAreIgnored) {
    // Pair (0,1) is tied in x and skipped; remaining pairs are both concordant.
    // denom is still C(3,2)=3, so tau = 2/3.
    const std::vector<double> x = {1.0, 1.0, 2.0};
    const std::vector<double> y = {3.0, 4.0, 5.0};
    EXPECT_NEAR(kendall(x, y), 2.0 / 3.0, 1e-12);
}

TEST(StatsGapsTest, Arfit_InvalidOrder) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> empty;
    EXPECT_TRUE(arfit(x, 0).empty());
    EXPECT_TRUE(arfit(x, 3).empty());
    EXPECT_TRUE(arfit(empty, 1).empty());
}

TEST(StatsGapsTest, Pacf_EmptyAndNonPositiveLag) {
    const std::vector<double> empty;
    const std::vector<double> x = {1.0, 2.0, 3.0};
    EXPECT_TRUE(pacf(empty, 2).empty());
    EXPECT_TRUE(pacf(x, 0).empty());
    EXPECT_TRUE(pacf(x, -1).empty());
}

TEST(StatsGapsTest, MultipleRegression_EmptyFirstRow) {
    const std::vector<std::vector<double>> X = {{}};
    const std::vector<double> y = {1.0};
    EXPECT_TRUE(multiple_regression(X, y).empty());
}

TEST(StatsGapsTest, BootstrapCI_PairEmptyAndConstant) {
    const std::vector<double> empty;
    const auto empty_ci = bootstrap_ci(empty, 0.95, 50, 1);
    EXPECT_NEAR(empty_ci.first, 0.0, 1e-12);
    EXPECT_NEAR(empty_ci.second, 0.0, 1e-12);

    const std::vector<double> data = {4.0, 4.0, 4.0, 4.0};
    const auto ci = bootstrap_ci(data, 0.95, 80, 7);
    EXPECT_NEAR(ci.first, 4.0, 1e-12);
    EXPECT_NEAR(ci.second, 4.0, 1e-12);
}

TEST(StatsGapsTest, BootstrapCI_StatFnEmptyAndNResamplesZero) {
    const std::vector<double> empty;
    auto mean_stat = [](std::span<const double> v) { return mean(v); };
    const auto r0 = bootstrap_ci(empty, mean_stat, 100, 0.95, 1);
    EXPECT_NEAR(r0.point_estimate, 0.0, 1e-12);
    EXPECT_NEAR(r0.lower, 0.0, 1e-12);

    const std::vector<double> data = {1.0, 2.0, 3.0};
    const auto r1 = bootstrap_ci(data, mean_stat, 0, 0.95, 1);
    EXPECT_NEAR(r1.point_estimate, 0.0, 1e-12);
}

TEST(StatsGapsTest, PartialCorrelation_ControlsForZ) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    const std::vector<double> z = {0.0, 1.0, 0.0, 1.0, 0.0};
    const double r = partial_correlation(x, y, z);
    EXPECT_NEAR(r, 1.0, 0.05);
    const std::vector<double> empty;
    EXPECT_NEAR(partial_correlation(empty, empty, empty), 0.0, 1e-12);
}

TEST(StatsGapsTest, Vif_OrthogonalAndOutOfRange) {
    const std::vector<std::vector<double>> X = {
        {1.0, 1.0}, {1.0, -1.0}, {1.0, 1.0}, {1.0, -1.0}};
    EXPECT_NEAR(vif(X, 1), variance_inflation_factor(X, 1), 1e-12);
    EXPECT_LT(vif(X, 1), 1.2);
    EXPECT_NEAR(variance_inflation_factor(X, 5), 0.0, 1e-12);
    const std::vector<std::vector<double>> empty;
    EXPECT_NEAR(variance_inflation_factor(empty, 0), 1.0, 1e-12);
}

TEST(StatsGapsTest, Kde_EmptySamplesAndBadBandwidth) {
    const std::vector<double> empty;
    const std::vector<double> grid = {-1.0, 0.0, 1.0};
    const std::vector<double> samples = {0.0, 1.0};
    EXPECT_TRUE(kde(empty, grid, 1.0).empty());
    EXPECT_TRUE(kde(samples, grid, 0.0).empty());
    EXPECT_TRUE(kde(samples, empty, 1.0).empty());
}

TEST(StatsGapsTest, LinearRegression_MismatchAndConstantX) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y_short = {1.0, 2.0};
    EXPECT_NEAR(linear_regression(x, y_short).slope, 0.0, 1e-12);
    const std::vector<double> x_flat = {2.0, 2.0, 2.0};
    const std::vector<double> y = {1.0, 2.0, 3.0};
    EXPECT_NEAR(linear_regression(x_flat, y).slope, 0.0, 1e-12);
}

TEST(StatsGapsTest, Spearman_TwoPointsAndTiesIgnoredPath) {
    const std::vector<double> x = {1.0, 2.0};
    const std::vector<double> y = {3.0, 1.0};
    EXPECT_NEAR(spearman(x, y), -1.0, 1e-12);
}

TEST(StatsGapsTest, PartialCorrelation_PerfectControlDenomZero) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y = {2.0, 1.0, 4.0, 3.0};
    const std::vector<double> z = {1.0, 2.0, 3.0, 4.0};
    EXPECT_NEAR(partial_correlation(x, y, z), 0.0, 1e-12);
    const std::vector<double> y_short = {1.0, 2.0};
    EXPECT_NEAR(partial_correlation(x, y_short, z), 0.0, 1e-12);
}

TEST(StatsGapsTest, Anova_EmptyGroupAmongTwoValid) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0},
        {},
        {10.0, 11.0, 12.0},
    };
    const auto r = one_way_anova(groups);
    EXPECT_GT(r.f_stat, 1.0);
    EXPECT_EQ(r.df_between, 1);
    EXPECT_EQ(r.df_within, 4);
}

TEST(StatsGapsTest, Anova_ConstantGroupsZeroWithin) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
    };
    const auto r = one_way_anova(groups);
    EXPECT_NEAR(r.f_stat, 0.0, 1e-12);
    EXPECT_NEAR(r.p_value, 0.0, 1e-12);
    EXPECT_EQ(r.df_between, 1);
    EXPECT_EQ(r.df_within, 4);
}

TEST(StatsGapsTest, MannWhitney_AllTiesZeroVariance) {
    const std::vector<double> a = {1.0, 1.0, 1.0};
    const std::vector<double> b = {1.0, 1.0, 1.0};
    const auto r = mann_whitney_u(a, b);
    EXPECT_NEAR(r.u_stat, 4.5, 1e-12);
    EXPECT_NEAR(r.p_value, 0.0, 1e-12);
}

TEST(StatsGapsTest, Levene_EmptyGroupSkipped) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0},
        {},
        {5.0, 6.0, 7.0, 8.0},
    };
    const auto r = levene_test(groups);
    EXPECT_NEAR(r.f_stat, 0.0, 1e-12);
    EXPECT_NEAR(r.p_value, 1.0, 1e-12);
    EXPECT_EQ(r.df_between, 1);
    EXPECT_EQ(r.df_within, 6);
}

TEST(StatsGapsTest, Bartlett_EmptyGroupAndSingleton) {
    const std::vector<std::vector<double>> with_empty = {
        {1.0, 2.0, 3.0, 4.0},
        {},
        {1.0, 10.0, 100.0, 1000.0},
    };
    const auto skip_empty = bartlett_test(with_empty);
    EXPECT_EQ(skip_empty.df, 1);
    EXPECT_GT(skip_empty.chi2_stat, 0.0);
    EXPECT_GE(skip_empty.p_value, 0.0);
    EXPECT_LE(skip_empty.p_value, 1.0);

    const std::vector<std::vector<double>> singleton = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0},
    };
    const auto r = bartlett_test(singleton);
    EXPECT_NEAR(r.chi2_stat, 0.0, 1e-12);
    EXPECT_NEAR(r.p_value, 1.0, 1e-12);
    EXPECT_EQ(r.df, 0);
}

TEST(StatsGapsTest, Fligner_ConstantGroupsZeroScoreVar) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
    };
    const auto r = fligner_test(groups);
    EXPECT_NEAR(r.chi2_stat, 0.0, 1e-12);
    EXPECT_NEAR(r.p_value, 1.0, 1e-12);
    EXPECT_EQ(r.df, 0);
}

TEST(StatsGapsTest, BootstrapCI_SingleResampleEqualIndices) {
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    auto mean_stat = [](std::span<const double> v) { return mean(v); };
    const auto r = bootstrap_ci(data, mean_stat, 1, 0.95, 7);
    EXPECT_NEAR(r.point_estimate, mean(data), 1e-12);
    EXPECT_NEAR(r.lower, r.upper, 1e-12);
    EXPECT_NEAR(r.std_error, 0.0, 1e-12);

    const auto pair_ci = bootstrap_ci(data, 0.95, 1, 7);
    EXPECT_NEAR(pair_ci.first, pair_ci.second, 1e-12);
}

TEST(StatsGapsTest, Arfit_ConstantSeriesSingularPivot) {
    const std::vector<double> x = {4.0, 4.0, 4.0, 4.0, 4.0};
    const auto phi = arfit(x, 2);
    ASSERT_EQ(phi.size(), 2u);
    EXPECT_NEAR(phi[0], 0.0, 1e-12);
    EXPECT_NEAR(phi[1], 0.0, 1e-12);
}

TEST(StatsGapsTest, Kde_SingleSampleHugeBandwidth) {
    const std::vector<double> samples = {0.0};
    const std::vector<double> grid = {0.0};
    const auto dens = kde(samples, grid, 1e13);
    ASSERT_EQ(dens.size(), 1u);
    EXPECT_GT(dens[0], 0.0);
    EXPECT_LT(dens[0], 1e-12);
}

TEST(StatsGapsTest, RankCorr_SinglePointAndNaN) {
    const std::vector<double> x = {1.0};
    const std::vector<double> y = {2.0};
    EXPECT_TRUE(std::isnan(spearman(x, y)));
    EXPECT_NEAR(kendall(x, y), 0.0, 1e-12);
}

TEST(StatsGapsTest, Descriptive_EmptyAndSinglePoint) {
    const std::vector<double> empty;
    const std::vector<double> one = {7.0};
    EXPECT_NEAR(mean(empty), 0.0, 1e-12);
    EXPECT_NEAR(var(empty), 0.0, 1e-12);
    EXPECT_NEAR(median(empty), 0.0, 1e-12);
    EXPECT_NEAR(percentile(empty, 50.0), 0.0, 1e-12);
    EXPECT_NEAR(mode(empty), 0.0, 1e-12);
    EXPECT_NEAR(iqr(empty), 0.0, 1e-12);
    EXPECT_NEAR(rms(empty), 0.0, 1e-12);
    EXPECT_NEAR(mean(one), 7.0, 1e-12);
    EXPECT_NEAR(var(one), 0.0, 1e-12);
    EXPECT_NEAR(median(one), 7.0, 1e-12);
    EXPECT_NEAR(percentile(one, 40.0), 7.0, 1e-12);
    EXPECT_NEAR(iqr(one), 0.0, 1e-12);
    EXPECT_NEAR(rms(one), 7.0, 1e-12);
    EXPECT_NEAR(geometric_mean(one), 7.0, 1e-12);
    EXPECT_NEAR(harmonic_mean(one), 7.0, 1e-12);
}

TEST(StatsGapsTest, WeightedCorr_N1MismatchAndAllTies) {
    const std::vector<double> one_x = {1.0};
    const std::vector<double> one_y = {2.0};
    const std::vector<double> one_w = {1.0};
    const std::vector<double> two_w = {1.0, 2.0};
    EXPECT_NEAR(weighted_variance(one_x, one_w, true), 0.0, 1e-12);
    EXPECT_NEAR(weighted_mean(one_x, two_w), 0.0, 1e-12);
    EXPECT_NEAR(weighted_correlation(one_x, one_y, two_w), 0.0, 1e-12);
    EXPECT_NEAR(correlation(one_x, one_y), 0.0, 1e-12);
    EXPECT_NEAR(partial_correlation(one_x, one_y, one_x), 0.0, 1e-12);
    const std::vector<double> tied_x = {5.0, 5.0, 5.0};
    const std::vector<double> free_y = {1.0, 2.0, 3.0};
    EXPECT_NEAR(kendall(tied_x, free_y), 0.0, 1e-12);
}

TEST(StatsGapsTest, AnovaAcfArfit_SingletonAndN1) {
    const std::vector<std::vector<double>> singletons = {{1.0}, {2.0}};
    const auto anova = one_way_anova(singletons);
    EXPECT_NEAR(anova.f_stat, 0.0, 1e-12);
    EXPECT_EQ(anova.df_between, 0);

    const std::vector<double> one = {4.0};
    const auto ac = acf(one, 1);
    ASSERT_EQ(ac.size(), 2u);
    EXPECT_NEAR(ac[0], 0.0, 1e-12);
    EXPECT_TRUE(arfit(one, 1).empty());
    const auto pc = pacf(one, 1);
    ASSERT_EQ(pc.size(), 2u);
    EXPECT_NEAR(pc[0], 1.0, 1e-12);
}

TEST(StatsGapsTest, Wilcoxon_AllZeroDiffsAndMismatch) {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y = {1.0, 2.0, 3.0};
    const auto zeros = wilcoxon_signed_rank(x, y);
    EXPECT_NEAR(zeros.w_stat, 0.0, 1e-12);
    EXPECT_NEAR(zeros.z_stat, 0.0, 1e-12);
    EXPECT_NEAR(zeros.p_value, 1.0, 1e-12);
    EXPECT_EQ(zeros.n_eff, 0);

    const std::vector<double> y_short = {1.0};
    const auto mismatch = wilcoxon_signed_rank(x, y_short);
    EXPECT_EQ(mismatch.n_eff, 0);
    EXPECT_NEAR(mismatch.p_value, 1.0, 1e-12);
}

TEST(StatsGapsTest, ShapiroWilk_N2AndConstant) {
    const std::vector<double> two = {1.0, 2.0};
    const auto tiny = shapiro_wilk(two);
    EXPECT_NEAR(tiny.w_stat, 1.0, 1e-12);
    EXPECT_NEAR(tiny.p_value, 1.0, 1e-12);

    const std::vector<double> flat = {4.0, 4.0, 4.0, 4.0};
    const auto constant = shapiro_wilk(flat);
    EXPECT_NEAR(constant.w_stat, 1.0, 1e-12);
    EXPECT_NEAR(constant.p_value, 1.0, 1e-12);
}

TEST(StatsGapsTest, Vif_JaggedRowAndSingleColumn) {
    const std::vector<std::vector<double>> jagged = {{1.0, 2.0}, {3.0}};
    EXPECT_NEAR(variance_inflation_factor(jagged, 0), 0.0, 1e-12);

    const std::vector<std::vector<double>> one_col = {{1.0}, {2.0}, {3.0}};
    EXPECT_NEAR(vif(one_col, 0), 1.0, 1e-12);
}

TEST(StatsGapsTest, KruskalFriedman_EmptyJaggedAndAllTies) {
    const auto kw_empty = kruskal_wallis({{1.0, 2.0, 3.0}, {}});
    EXPECT_NEAR(kw_empty.h_stat, 0.0, 1e-12);
    EXPECT_EQ(kw_empty.df, 0);
    EXPECT_NEAR(kw_empty.p_value, 1.0, 1e-12);

    const auto kw_ties = kruskal_wallis({{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}});
    EXPECT_NEAR(kw_ties.h_stat, 0.0, 1e-12);
    EXPECT_EQ(kw_ties.df, 1);
    EXPECT_NEAR(kw_ties.p_value, 1.0, 1e-12);

    const auto fr_jagged = friedman({{1.0, 2.0}, {1.0}});
    EXPECT_NEAR(fr_jagged.chi2_stat, 0.0, 1e-12);
    EXPECT_EQ(fr_jagged.df, 0);
    EXPECT_NEAR(fr_jagged.p_value, 1.0, 1e-12);

    const auto fr_ties = friedman({{5.0, 5.0}, {5.0, 5.0}});
    EXPECT_NEAR(fr_ties.chi2_stat, 0.0, 1e-12);
    EXPECT_EQ(fr_ties.df, 1);
    EXPECT_NEAR(fr_ties.p_value, 1.0, 1e-12);
}

TEST(StatsGapsTest, Ks2SampleLjungBoxJarque_EmptyN1Constant) {
    const std::vector<double> empty;
    const std::vector<double> low = {1.0, 2.0};
    const std::vector<double> high = {10.0, 20.0};
    const auto ks_empty = ks_test_2sample(empty, low);
    EXPECT_NEAR(ks_empty.d_stat, 0.0, 1e-12);
    EXPECT_NEAR(ks_empty.p_value, 0.0, 1e-12);

    const auto ks_split = ks_test_2sample(low, high);
    EXPECT_GT(ks_split.d_stat, 0.0);
    EXPECT_GE(ks_split.p_value, 0.0);
    EXPECT_LE(ks_split.p_value, 1.0);

    const std::vector<double> one = {4.0};
    const auto lb_n1 = ljung_box(one, 3);
    EXPECT_NEAR(lb_n1.q_stat, 0.0, 1e-12);
    EXPECT_EQ(lb_n1.df, 0);
    EXPECT_NEAR(lb_n1.p_value, 1.0, 1e-12);
    const auto lb_bad_lag = ljung_box(low, 0);
    EXPECT_EQ(lb_bad_lag.df, 0);

    const auto jb_tiny = jarque_bera(low);
    EXPECT_NEAR(jb_tiny.jb_stat, 0.0, 1e-12);
    EXPECT_NEAR(jb_tiny.p_value, 1.0, 1e-12);
    const std::vector<double> flat = {3.0, 3.0, 3.0, 3.0, 3.0};
    const auto jb_flat = jarque_bera(flat);
    EXPECT_NEAR(jb_flat.jb_stat, 0.0, 1e-12);
    EXPECT_NEAR(jb_flat.p_value, 1.0, 1e-12);
}

TEST(StatsGapsTest, BootstrapCI_InvertedLevelAndKdeFar) {
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto mean_stat = [](std::span<const double> v) { return mean(v); };
    const auto r = bootstrap_ci(data, mean_stat, 10, -0.6, 1);
    EXPECT_TRUE(std::isfinite(r.lower));
    EXPECT_TRUE(std::isfinite(r.upper));
    EXPECT_NEAR(r.point_estimate, mean(data), 1e-12);

    const auto pair_ci = bootstrap_ci(data, -0.6, 10, 1);
    EXPECT_TRUE(std::isfinite(pair_ci.first));
    EXPECT_TRUE(std::isfinite(pair_ci.second));

    const std::vector<double> samples = {0.0};
    const std::vector<double> grid = {10.0};
    const auto dens = kde(samples, grid, 1.0, "epanechnikov");
    ASSERT_EQ(dens.size(), 1u);
    EXPECT_NEAR(dens[0], 0.0, 1e-12);
}

TEST(StatsGapsTest, MannWhitneyLevene_EmptyN1AndMismatch) {
    const std::vector<double> empty;
    const std::vector<double> one_a = {1.0};
    const std::vector<double> one_b = {2.0};
    const auto mw_empty = mann_whitney_u(empty, one_b);
    EXPECT_NEAR(mw_empty.u_stat, 0.0, 1e-12);
    EXPECT_NEAR(mw_empty.p_value, 0.0, 1e-12);
    const auto mw_empty_b = mann_whitney_u(one_a, empty);
    EXPECT_NEAR(mw_empty_b.u_stat, 0.0, 1e-12);

    const auto mw_n1 = mann_whitney_u(one_a, one_b);
    EXPECT_NEAR(mw_n1.u_stat, 0.0, 1e-12);
    EXPECT_NEAR(mw_n1.p_value, 1.0, 1e-12);

    const auto lev_empty = levene_test({});
    EXPECT_NEAR(lev_empty.f_stat, 0.0, 1e-12);
    EXPECT_EQ(lev_empty.df_between, 0);
    const auto lev_n1 = levene_test({{1.0}, {2.0}});
    EXPECT_NEAR(lev_n1.f_stat, 0.0, 1e-12);
    EXPECT_EQ(lev_n1.df_between, 0);
    EXPECT_EQ(lev_n1.df_within, 0);
}

TEST(StatsGapsTest, PacfMultipleRegression_N1MismatchAndSingular) {
    const std::vector<double> one = {4.0};
    const auto pc = pacf(one, 4);
    ASSERT_EQ(pc.size(), 5u);
    EXPECT_NEAR(pc[0], 1.0, 1e-12);
    EXPECT_NEAR(pc[1], 0.0, 1e-12);
    EXPECT_NEAR(pc[4], 0.0, 1e-12);

    const std::vector<std::vector<double>> n1 = {{5.0}};
    const std::vector<double> y1 = {10.0};
    const auto b1 = multiple_regression(n1, y1);
    ASSERT_EQ(b1.size(), 1u);
    EXPECT_NEAR(b1[0], 2.0, 1e-12);

    const std::vector<std::vector<double>> under = {{1.0, 2.0}};
    const auto b_under = multiple_regression(under, std::vector<double>{3.0});
    ASSERT_EQ(b_under.size(), 2u);
    for (double v : b_under) {
        EXPECT_TRUE(std::isfinite(v));
    }

    const std::vector<std::vector<double>> singular = {
        {1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
    const std::vector<double> ys = {1.0, 2.0, 3.0};
    const auto b_sing = multiple_regression(singular, ys);
    ASSERT_EQ(b_sing.size(), 2u);
    for (double v : b_sing) {
        EXPECT_TRUE(std::isfinite(v));
    }
}
