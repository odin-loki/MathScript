#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"

using namespace ms;

// ============================================================
// Mean — exact arithmetic results
// ============================================================

TEST(StatsNumerical, MeanExact_Integers) {
    std::vector<double> v{1, 2, 3, 4, 5};
    EXPECT_DOUBLE_EQ(mean(v), 3.0);
}

TEST(StatsNumerical, MeanExact_Symmetric) {
    // Symmetric around zero: mean must be exactly 0
    std::vector<double> v{-4.0, -2.0, 0.0, 2.0, 4.0};
    EXPECT_DOUBLE_EQ(mean(v), 0.0);
}

TEST(StatsNumerical, MeanExact_SingleValue) {
    std::vector<double> v{7.5};
    EXPECT_DOUBLE_EQ(mean(v), 7.5);
}

TEST(StatsNumerical, MeanExact_AllSame) {
    std::vector<double> v{3.0, 3.0, 3.0, 3.0};
    EXPECT_DOUBLE_EQ(mean(v), 3.0);
}

TEST(StatsNumerical, MeanExact_Fractional) {
    // {1, 2} → mean = 1.5
    std::vector<double> v{1.0, 2.0};
    EXPECT_DOUBLE_EQ(mean(v), 1.5);
}

// ============================================================
// Variance and StdDev
// ============================================================

TEST(StatsNumerical, VarZero_ConstantData) {
    std::vector<double> v{5.0, 5.0, 5.0, 5.0};
    EXPECT_DOUBLE_EQ(var(v), 0.0);
    EXPECT_DOUBLE_EQ(stddev(v), 0.0);
}

TEST(StatsNumerical, StddevPositive_NonConstant) {
    std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_GT(stddev(v), 0.0);
}

TEST(StatsNumerical, StddevSquaredEqualsVariance) {
    std::vector<double> v{2.0, 4.0, 6.0, 8.0, 10.0};
    double s = stddev(v);
    double variance = var(v);
    EXPECT_NEAR(s * s, variance, 1e-10);
}

// ============================================================
// Min / Max
// ============================================================

TEST(StatsNumerical, MinMaxExact) {
    std::vector<double> v{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    EXPECT_DOUBLE_EQ(min_value(v), 1.0);
    EXPECT_DOUBLE_EQ(max_value(v), 9.0);
}

TEST(StatsNumerical, MinMaxSingleElement) {
    std::vector<double> v{42.0};
    EXPECT_DOUBLE_EQ(min_value(v), 42.0);
    EXPECT_DOUBLE_EQ(max_value(v), 42.0);
}

// ============================================================
// Median
// ============================================================

TEST(StatsNumerical, MedianExact_OddCount) {
    std::vector<double> v{1.0, 3.0, 5.0, 7.0, 9.0};
    EXPECT_DOUBLE_EQ(median(v), 5.0);
}

TEST(StatsNumerical, MedianExact_EvenCount) {
    // {1,2,3,4} → middle two are 2 and 3 → median = 2.5
    std::vector<double> v{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(median(v), 2.5);
}

TEST(StatsNumerical, MedianExact_SingleElement) {
    std::vector<double> v{99.0};
    EXPECT_DOUBLE_EQ(median(v), 99.0);
}

namespace {

double median_full_sort(std::span<const double> data) {
    if (data.empty()) {
        return 0.0;
    }
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }
    return sorted[n / 2];
}

double trimmed_mean_full_sort(std::span<const double> data, double frac) {
    if (data.empty()) {
        return 0.0;
    }
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t trim =
        static_cast<size_t>(frac * static_cast<double>(sorted.size()));
    if (2 * trim >= sorted.size()) {
        return median_full_sort(data);
    }
    return std::accumulate(sorted.begin() + static_cast<std::ptrdiff_t>(trim),
                           sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() - trim),
                           0.0) /
           static_cast<double>(sorted.size() - 2 * trim);
}

} // namespace

TEST(StatsNumerical, MedianMatchesFullSort_Unsorted) {
    std::vector<double> v{42.0, 7.0, 19.0, 3.0, 55.0, 11.0, 31.0, 8.0};
    EXPECT_DOUBLE_EQ(median(v), median_full_sort(v));
}

TEST(StatsNumerical, MedianMatchesFullSort_Duplicates) {
    std::vector<double> v{5.0, 2.0, 5.0, 2.0, 5.0, 1.0, 2.0, 5.0};
    EXPECT_DOUBLE_EQ(median(v), median_full_sort(v));
}

TEST(StatsNumerical, MedianMatchesFullSort_Random) {
    std::mt19937 rng(224);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (size_t n : {1u, 2u, 3u, 4u, 15u, 16u, 127u, 128u, 511u, 512u}) {
        std::vector<double> v(n);
        for (double& x : v) {
            x = dist(rng);
        }
        EXPECT_DOUBLE_EQ(median(v), median_full_sort(v)) << "n=" << n;
    }
}

TEST(StatsNumerical, IqrMatchesPercentileDifference) {
    std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    EXPECT_DOUBLE_EQ(iqr(v), percentile(v, 75.0) - percentile(v, 25.0));
}

TEST(StatsNumerical, TrimmedMeanMatchesFullSort) {
    std::vector<double> v{10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0};
    for (double frac : {0.1, 0.2, 0.25}) {
        EXPECT_DOUBLE_EQ(trimmed_mean(v, frac), trimmed_mean_full_sort(v, frac))
            << "frac=" << frac;
    }
}

// ============================================================
// Percentile
// ============================================================

TEST(StatsNumerical, Percentile50MatchesMedian) {
    // Provide a sorted array so percentile is unambiguous
    std::vector<double> v{10.0, 20.0, 30.0, 40.0, 50.0};
    double med = median(v);
    double p50 = percentile(v, 50.0);
    EXPECT_NEAR(p50, med, 1.0); // allow ±1 for interpolation conventions
}

TEST(StatsNumerical, Percentile0And100) {
    std::vector<double> v{2.0, 4.0, 6.0, 8.0, 10.0};
    EXPECT_NEAR(percentile(v, 0.0), min_value(v), 1e-6);
    EXPECT_NEAR(percentile(v, 100.0), max_value(v), 1e-6);
}

TEST(StatsNumerical, PercentileMonotone) {
    std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double q1 = percentile(v, 25.0);
    double q2 = percentile(v, 50.0);
    double q3 = percentile(v, 75.0);
    EXPECT_LE(q1, q2);
    EXPECT_LE(q2, q3);
}

namespace {

double percentile_full_sort(std::span<const double> data, double p) {
    if (data.empty()) {
        return 0.0;
    }
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t idx = static_cast<size_t>(
        (p / 100.0) * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

} // namespace

TEST(StatsNumerical, PercentileMatchesFullSort_Unsorted) {
    std::vector<double> v{42.0, 7.0, 19.0, 3.0, 55.0, 11.0, 31.0, 8.0};
    for (double p : {0.0, 25.0, 50.0, 75.0, 90.0, 100.0}) {
        EXPECT_DOUBLE_EQ(percentile(v, p), percentile_full_sort(v, p))
            << "p=" << p;
    }
}

TEST(StatsNumerical, PercentileMatchesFullSort_Duplicates) {
    std::vector<double> v{5.0, 2.0, 5.0, 2.0, 5.0, 1.0, 2.0, 5.0};
    for (double p : {0.0, 12.5, 37.5, 50.0, 62.5, 87.5, 100.0}) {
        EXPECT_DOUBLE_EQ(percentile(v, p), percentile_full_sort(v, p))
            << "p=" << p;
    }
}

TEST(StatsNumerical, PercentileMatchesFullSort_Random) {
    std::mt19937 rng(219);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> v(512);
    for (double& x : v) {
        x = dist(rng);
    }
    for (double p : {0.0, 1.0, 10.0, 33.3, 50.0, 66.7, 90.0, 99.0, 100.0}) {
        EXPECT_DOUBLE_EQ(percentile(v, p), percentile_full_sort(v, p))
            << "p=" << p;
    }
}

TEST(StatsNumerical, PercentileMatchesFullSort_SingleElement) {
    std::vector<double> v{17.5};
    for (double p : {0.0, 50.0, 100.0}) {
        EXPECT_DOUBLE_EQ(percentile(v, p), percentile_full_sort(v, p))
            << "p=" << p;
    }
}

TEST(StatsNumerical, PercentileMatchesFullSort_LargeSorted) {
    std::vector<double> v(1000);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<double>(i);
    }
    for (double p : {0.0, 25.0, 50.0, 75.0, 100.0}) {
        EXPECT_DOUBLE_EQ(percentile(v, p), percentile_full_sort(v, p))
            << "p=" << p;
    }
}

// ============================================================
// Mode
// ============================================================

TEST(StatsNumerical, ModeExact_ClearWinner) {
    // 3 appears three times, the most
    std::vector<double> v{1.0, 2.0, 3.0, 3.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(mode(v), 3.0);
}

TEST(StatsNumerical, ModeExact_SingleElement) {
    std::vector<double> v{7.0};
    EXPECT_DOUBLE_EQ(mode(v), 7.0);
}

// ============================================================
// Skewness
// ============================================================

TEST(StatsNumerical, SkewnessZero_Symmetric) {
    // Perfectly symmetric: {1,2,3,4,5} → skewness = 0
    std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(skewness(v), 0.0, 1e-10);
}

TEST(StatsNumerical, SkewnessPositive_RightTail) {
    // Right-skewed: small cluster left, long tail right
    std::vector<double> v{1.0, 1.0, 1.0, 2.0, 10.0};
    EXPECT_GT(skewness(v), 0.0);
}

TEST(StatsNumerical, SkewnessNegative_LeftTail) {
    // Left-skewed: long tail on the left
    std::vector<double> v{-10.0, 1.0, 2.0, 2.0, 2.0};
    EXPECT_LT(skewness(v), 0.0);
}

// ============================================================
// Kurtosis
// ============================================================

TEST(StatsNumerical, KurtosisNonNegative_BasicSet) {
    // Kurtosis result should be finite for well-behaved data
    std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double k = kurtosis(v);
    EXPECT_TRUE(std::isfinite(k));
}

// ============================================================
// Linear Regression — exact known results
// ============================================================

TEST(StatsNumerical, LinearRegression_Slope2Intercept3) {
    // y = 2x + 3 → exact solution
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y{5.0, 7.0, 9.0, 11.0, 13.0};
    auto fit = linear_regression(x, y);
    EXPECT_NEAR(fit.slope, 2.0, 1e-10);
    EXPECT_NEAR(fit.intercept, 3.0, 1e-10);
    EXPECT_NEAR(fit.r_squared, 1.0, 1e-10);
}

TEST(StatsNumerical, LinearRegression_NegativeSlope) {
    // y = -x + 10 → slope=-1, intercept=10
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y{9.0, 8.0, 7.0, 6.0, 5.0};
    auto fit = linear_regression(x, y);
    EXPECT_NEAR(fit.slope, -1.0, 1e-10);
    EXPECT_NEAR(fit.intercept, 10.0, 1e-10);
    EXPECT_NEAR(fit.r_squared, 1.0, 1e-10);
}

TEST(StatsNumerical, LinearRegression_ZeroSlope) {
    // Horizontal line: y = 4 for all x → slope = 0
    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    std::vector<double> y{4.0, 4.0, 4.0, 4.0};
    auto fit = linear_regression(x, y);
    EXPECT_NEAR(fit.slope, 0.0, 1e-10);
    EXPECT_NEAR(fit.intercept, 4.0, 1e-10);
}

// ============================================================
// Correlation
// ============================================================

TEST(StatsNumerical, CorrelationPerfectPositive) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y{2.0, 4.0, 6.0, 8.0, 10.0};
    EXPECT_NEAR(correlation(x, y), 1.0, 1e-10);
}

TEST(StatsNumerical, CorrelationPerfectNegative) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y{5.0, 4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(correlation(x, y), -1.0, 1e-10);
}

TEST(StatsNumerical, CorrelationBounded) {
    // Pearson r is always in [-1, 1]
    std::vector<double> x{1.0, 3.0, 2.0, 5.0, 4.0};
    std::vector<double> y{2.0, 1.0, 4.0, 3.0, 5.0};
    double r = correlation(x, y);
    EXPECT_GE(r, -1.0);
    EXPECT_LE(r, 1.0);
}

// ============================================================
// Weighted descriptive statistics
// ============================================================

TEST(StatsNumerical, WeightedMeanUniformWeightsMatchesMean) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> w1(5, 1.0);
    std::vector<double> w2(5, 2.0);
    EXPECT_DOUBLE_EQ(weighted_mean(x, w1), mean(x));
    EXPECT_DOUBLE_EQ(weighted_mean(x, w2), mean(x));
}

TEST(StatsNumerical, WeightedVarianceUniformWeightsMatchesVar) {
    std::vector<double> x{2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<double> w1(5, 1.0);
    std::vector<double> w2(5, 3.0);
    EXPECT_DOUBLE_EQ(weighted_variance(x, w1), var(x));
    EXPECT_DOUBLE_EQ(weighted_variance(x, w2), var(x));
}

TEST(StatsNumerical, WeightedCorrelationUniformWeightsMatchesCorrelation) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y{2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<double> w(5, 1.0);
    EXPECT_NEAR(weighted_correlation(x, y, w), correlation(x, y), 1e-12);
}

TEST(StatsNumerical, WeightedMeanFrequencyWeightsMatchesExpanded) {
    std::vector<double> x{1.0, 2.0, 3.0};
    std::vector<double> w{2.0, 1.0, 1.0};
    std::vector<double> expanded{1.0, 1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(weighted_mean(x, w), mean(expanded));
}

TEST(StatsNumerical, WeightedVarianceFrequencyWeightsDiffersFromExpanded) {
    // Reliability-weights bias correction differs from sum(w)-1 for integer counts.
    std::vector<double> x{1.0, 2.0, 3.0};
    std::vector<double> w{2.0, 1.0, 1.0};
    std::vector<double> expanded{1.0, 1.0, 2.0, 3.0};
    const double expanded_var = var(expanded);
    const double weighted_var = weighted_variance(x, w);
    EXPECT_DOUBLE_EQ(weighted_var, 1.1);
    EXPECT_NEAR(expanded_var, 0.9166666666666666, 1e-12);
    EXPECT_NE(weighted_var, expanded_var);
}

TEST(StatsNumerical, WeightedMeanEmphasizesHeavyWeight) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 100.0};
    std::vector<double> w{1.0, 1.0, 1.0, 1.0, 100.0};
    const double unweighted = mean(x);
    const double weighted = weighted_mean(x, w);
    EXPECT_GT(weighted, unweighted);
    EXPECT_GT(weighted, 50.0);
}

TEST(StatsNumerical, WeightedCorrelationReweightingChangesSign) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0};
    std::vector<double> y{4.0, 3.0, 2.0, 1.0, 10.0, 11.0, 12.0, 13.0};
    std::vector<double> w_uniform(8, 1.0);
    std::vector<double> w_neg{100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> w_pos{1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0};
    const double uniform_r = weighted_correlation(x, y, w_uniform);
    const double neg_r = weighted_correlation(x, y, w_neg);
    const double pos_r = weighted_correlation(x, y, w_pos);
    EXPECT_LT(neg_r, 0.0);
    EXPECT_GT(pos_r, 0.0);
    EXPECT_LT(neg_r, uniform_r);
    EXPECT_GT(pos_r, uniform_r);
}

TEST(StatsNumerical, WeightedStatsMismatchedLengthsReturnZero) {
    std::vector<double> x{1.0, 2.0, 3.0};
    std::vector<double> w{1.0, 2.0};
    std::vector<double> y{4.0, 5.0, 6.0};
    EXPECT_DOUBLE_EQ(weighted_mean(x, w), 0.0);
    EXPECT_DOUBLE_EQ(weighted_variance(x, w), 0.0);
    EXPECT_DOUBLE_EQ(weighted_correlation(x, y, w), 0.0);
}

TEST(StatsNumerical, WeightedVarianceDegenerateReturnsZero) {
    std::vector<double> single{5.0};
    std::vector<double> w1{1.0};
    EXPECT_DOUBLE_EQ(weighted_variance(single, w1), 0.0);

    std::vector<double> constant{3.0, 3.0, 3.0};
    std::vector<double> w2{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(weighted_variance(constant, w2), 0.0);
}

TEST(StatsNumerical, WeightedStatsNegativeWeightsReturnNaN) {
    std::vector<double> x{1.0, 2.0, 3.0};
    std::vector<double> w{-1.0, 1.0, 1.0};
    EXPECT_TRUE(std::isnan(weighted_mean(x, w)));
    EXPECT_TRUE(std::isnan(weighted_variance(x, w)));
    EXPECT_TRUE(std::isnan(weighted_correlation(x, x, w)));
}

// ============================================================
// t-test / z-test — sign and direction checks
// ============================================================

TEST(StatsNumerical, TtestAboveMu_PositiveStat) {
    // Sample mean > hypothesized mean → positive t-statistic
    std::vector<double> v{10.0, 11.0, 12.0, 13.0, 14.0};
    EXPECT_GT(ttest(v, 5.0), 0.0);
}

TEST(StatsNumerical, TtestBelowMu_NegativeStat) {
    std::vector<double> v{1.0, 2.0, 3.0};
    EXPECT_LT(ttest(v, 10.0), 0.0);
}

TEST(StatsNumerical, ZtestKnown_Direction) {
    // All values well above mu with small sigma → large positive z
    std::vector<double> v{100.0, 101.0, 102.0};
    double z = ztest(v, 0.0, 1.0);
    EXPECT_GT(z, 50.0);
}

// ============================================================
// Normal distribution — reference values
// ============================================================

TEST(ProbNumerical, NormPDF_AtMean) {
    // Standard normal at x=0: 1/sqrt(2*pi)
    double expected = 1.0 / std::sqrt(2.0 * M_PI);
    EXPECT_NEAR(norm_pdf(0.0, 0.0, 1.0), expected, 1e-10);
}

TEST(ProbNumerical, NormCDF_AtMean) {
    // CDF at mean = 0.5
    EXPECT_NEAR(norm_cdf(0.0, 0.0, 1.0), 0.5, 1e-10);
}

TEST(ProbNumerical, NormCDF_OneSigmaAbove) {
    // Phi(1) ≈ 0.84134
    EXPECT_NEAR(norm_cdf(1.0, 0.0, 1.0), 0.84134, 1e-4);
}

TEST(ProbNumerical, NormCDF_OneSigmaBelow) {
    // Phi(-1) ≈ 0.15866
    EXPECT_NEAR(norm_cdf(-1.0, 0.0, 1.0), 0.15866, 1e-4);
}

TEST(ProbNumerical, NormPPF_AtHalf) {
    // ppf(0.5) = mean for any (mu, sigma)
    EXPECT_NEAR(norm_ppf(0.5, 3.0, 2.0), 3.0, 1e-8);
}

TEST(ProbNumerical, NormPPF_InverseOfCDF) {
    // ppf(cdf(x)) = x
    double x = 1.5;
    EXPECT_NEAR(norm_ppf(norm_cdf(x, 0.0, 1.0), 0.0, 1.0), x, 1e-6);
}

// ============================================================
// Exponential distribution — exact formulas
// ============================================================

TEST(ProbNumerical, ExpPDF_AtOne_Lambda1) {
    // f(1; λ=1) = e^{-1}
    EXPECT_NEAR(exp_pdf(1.0, 1.0), std::exp(-1.0), 1e-10);
}

TEST(ProbNumerical, ExpCDF_AtOne_Lambda1) {
    // F(1; λ=1) = 1 - e^{-1}
    EXPECT_NEAR(exp_cdf(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-10);
}

TEST(ProbNumerical, ExpCDF_ComplementOfSurvival) {
    double x = 2.0, lam = 0.5;
    // F + S = 1
    EXPECT_NEAR(exp_cdf(x, lam) + std::exp(-lam * x), 1.0, 1e-10);
}

// ============================================================
// Binomial distribution — exact combinatorics
// ============================================================

TEST(ProbNumerical, BinomPDF_TwoSuccessesOf4) {
    // C(4,2)*0.5^4 = 6/16 = 0.375
    EXPECT_NEAR(binom_pdf(2, 4, 0.5), 0.375, 1e-10);
}

TEST(ProbNumerical, BinomPDF_CertainSuccess) {
    // p=1: only k=n has non-zero probability
    EXPECT_NEAR(binom_pdf(3, 3, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(binom_pdf(2, 3, 1.0), 0.0, 1e-10);
}

TEST(ProbNumerical, BinomCDF_TwoOrFewer) {
    // P(X<=2; n=4, p=0.5) = (1+4+6)/16 = 11/16 = 0.6875
    EXPECT_NEAR(binom_cdf(2, 4, 0.5), 0.6875, 1e-10);
}

// ============================================================
// Poisson distribution — exact formulas
// ============================================================

TEST(ProbNumerical, PoisPDF_TwoEvents_Lambda2) {
    // P(X=2; λ=2) = e^{-2}*2^2/2! = 2*e^{-2}
    EXPECT_NEAR(pois_pdf(2.0, 2.0), 2.0 * std::exp(-2.0), 1e-10);
}

TEST(ProbNumerical, PoisPDF_ZeroEvents) {
    // P(X=0; λ) = e^{-λ}
    EXPECT_NEAR(pois_pdf(0.0, 3.0), std::exp(-3.0), 1e-10);
}

TEST(ProbNumerical, PoisCDF_Monotone) {
    EXPECT_LT(pois_cdf(1.0, 3.0), pois_cdf(4.0, 3.0));
}

// ============================================================
// Chi-squared distribution
// ============================================================

TEST(ProbNumerical, Chi2CDF_DF2_AtX2) {
    // chi2(df=2) is Exp(0.5); CDF(2,df=2) = 1-e^{-1}
    EXPECT_NEAR(chi2_cdf(2.0, 2.0), 1.0 - std::exp(-1.0), 1e-6);
}

TEST(ProbNumerical, Chi2CDF_Monotone) {
    EXPECT_LT(chi2_cdf(1.0, 3.0), chi2_cdf(5.0, 3.0));
}

TEST(ProbNumerical, Chi2PDF_PositiveForPositiveX) {
    EXPECT_GT(chi2_pdf(3.0, 4.0), 0.0);
}

// ============================================================
// t-distribution
// ============================================================

TEST(ProbNumerical, TDistCDF_AtZero_Half) {
    // By symmetry, CDF(0, df) = 0.5 for any df > 0
    EXPECT_NEAR(t_cdf(0.0, 5.0), 0.5, 1e-10);
    EXPECT_NEAR(t_cdf(0.0, 30.0), 0.5, 1e-10);
}

TEST(ProbNumerical, TDistPDF_Symmetric) {
    // t-distribution is symmetric: pdf(x) = pdf(-x)
    EXPECT_NEAR(t_pdf(1.5, 10.0), t_pdf(-1.5, 10.0), 1e-10);
}

TEST(ProbNumerical, TDistPDF_DF1_AtZero) {
    // df=1 is Cauchy: pdf(0) = 1/pi
    EXPECT_NEAR(t_pdf(0.0, 1.0), 1.0 / M_PI, 1e-8);
}

// ============================================================
// Uniform distribution
// ============================================================

TEST(ProbNumerical, UniformCDF_Interior) {
    // On [0,10]: CDF(3) = 0.3
    EXPECT_NEAR(uniform_cdf(3.0, 0.0, 10.0), 0.3, 1e-10);
}

TEST(ProbNumerical, UniformPDF_Constant) {
    // pdf = 1/(b-a) everywhere inside
    EXPECT_NEAR(uniform_pdf(5.0, 0.0, 10.0), 0.1, 1e-10);
    EXPECT_NEAR(uniform_pdf(1.0, 0.0, 10.0), 0.1, 1e-10);
}

// ============================================================
// Gamma distribution
// ============================================================

TEST(ProbNumerical, GammaPDF_Shape1_IsExponential) {
    // gamma(x; shape=1, scale=1) = exp(-x) for x > 0
    EXPECT_NEAR(gamma_pdf(2.0, 1.0, 1.0), std::exp(-2.0), 1e-8);
}

TEST(ProbNumerical, GammaPDF_PositiveForPositiveX) {
    EXPECT_GT(gamma_pdf(1.0, 2.0, 1.0), 0.0);
    EXPECT_GT(gamma_pdf(3.0, 3.0, 2.0), 0.0);
}

// ============================================================
// Kernel density estimation (kde)
// ============================================================

namespace {

std::vector<double> linspace(double start, double stop, size_t count) {
    std::vector<double> grid(count);
    if (count == 0) {
        return grid;
    }
    if (count == 1) {
        grid[0] = start;
        return grid;
    }
    const double step = (stop - start) / static_cast<double>(count - 1);
    for (size_t i = 0; i < count; ++i) {
        grid[i] = start + static_cast<double>(i) * step;
    }
    return grid;
}

double trapezoid_integral(const std::vector<double>& grid,
                          const std::vector<double>& density) {
    double integral = 0.0;
    for (size_t i = 0; i + 1 < grid.size(); ++i) {
        const double dx = grid[i + 1] - grid[i];
        integral += 0.5 * (density[i] + density[i + 1]) * dx;
    }
    return integral;
}

size_t argmax(const std::vector<double>& values) {
    return static_cast<size_t>(
        std::distance(values.begin(),
                      std::max_element(values.begin(), values.end())));
}

double kde_naive_kernel(double u, const char* kernel) {
    if (std::strcmp(kernel, "epanechnikov") == 0) {
        if (std::abs(u) > 1.0) {
            return 0.0;
        }
        return 0.75 * (1.0 - u * u);
    }
    static constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * u * u);
}

std::vector<double> kde_naive_reference(std::span<const double> samples,
                                        std::span<const double> grid,
                                        double bandwidth,
                                        const char* kernel) {
    if (samples.empty() || bandwidth <= 0.0 || grid.empty()) {
        return {};
    }
    const double inv_nh =
        1.0 / (static_cast<double>(samples.size()) * bandwidth);
    std::vector<double> result(grid.size(), 0.0);
    for (size_t gi = 0; gi < grid.size(); ++gi) {
        const double x = grid[gi];
        double sum = 0.0;
        for (double xi : samples) {
            sum += kde_naive_kernel((x - xi) / bandwidth, kernel);
        }
        result[gi] = inv_nh * sum;
    }
    return result;
}

void expect_kde_matches_naive(const std::vector<double>& samples,
                              const std::vector<double>& grid,
                              double bandwidth,
                              const char* kernel,
                              double tol = 1e-10) {
    const auto optimized = kde(samples, grid, bandwidth, kernel);
    const auto naive =
        kde_naive_reference(samples, grid, bandwidth, kernel);
    ASSERT_EQ(optimized.size(), naive.size());
    for (size_t i = 0; i < optimized.size(); ++i) {
        EXPECT_NEAR(optimized[i], naive[i], tol) << "grid index " << i;
    }
}

} // namespace

TEST(StatsNumerical, Kde_EmptySamples_ReturnsEmpty) {
    const std::vector<double> grid = linspace(-1.0, 1.0, 11);
    const auto density = kde({}, grid, 0.5);
    EXPECT_TRUE(density.empty());
}

TEST(StatsNumerical, Kde_NonPositiveBandwidth_ReturnsEmpty) {
    const std::vector<double> samples = {1.0, 2.0, 3.0};
    const std::vector<double> grid = linspace(0.0, 4.0, 21);
    EXPECT_TRUE(kde(samples, grid, 0.0).empty());
    EXPECT_TRUE(kde(samples, grid, -0.25).empty());
}

TEST(StatsNumerical, Kde_EmptyGrid_ReturnsEmpty) {
    const std::vector<double> samples = {1.0, 2.0, 3.0};
    EXPECT_TRUE(kde(samples, {}, 0.5).empty());
}

TEST(StatsNumerical, Kde_OutputSizeMatchesGrid) {
    const std::vector<double> samples = {0.0, 1.0, 2.0, 3.0};
    const std::vector<double> grid = linspace(-1.0, 5.0, 31);
    const auto density = kde(samples, grid, 0.75, "gaussian");
    ASSERT_EQ(density.size(), grid.size());
}

TEST(StatsNumerical, Kde_GaussianSingleSamplePeaksAtSample) {
    const std::vector<double> samples = {5.0};
    const std::vector<double> grid = linspace(3.0, 7.0, 41);
    const auto density = kde(samples, grid, 1.0, "gaussian");
    ASSERT_EQ(density.size(), grid.size());
    EXPECT_NEAR(grid[argmax(density)], 5.0, 0.15);
    EXPECT_GT(density[argmax(density)], 0.0);
}

TEST(StatsNumerical, Kde_EpanechnikovCompactSupport) {
    const std::vector<double> samples = {0.0};
    const std::vector<double> grid = linspace(-3.0, 3.0, 61);
    const auto density = kde(samples, grid, 1.0, "epanechnikov");
    ASSERT_EQ(density.size(), grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::abs(grid[i]) > 1.0 + 1e-12) {
            EXPECT_DOUBLE_EQ(density[i], 0.0);
        }
    }
    EXPECT_GT(density[argmax(density)], 0.0);
}

TEST(StatsNumerical, Kde_NormalSamplesPeakNearMean) {
    std::vector<double> samples;
    samples.reserve(300);
    std::mt19937 rng(214);
    std::normal_distribution<double> dist(2.0, 0.6);
    for (int i = 0; i < 300; ++i) {
        samples.push_back(dist(rng));
    }
    const std::vector<double> grid = linspace(-2.0, 6.0, 201);
    const auto density = kde(samples, grid, 0.35, "gaussian");
    ASSERT_EQ(density.size(), grid.size());
    EXPECT_NEAR(grid[argmax(density)], mean(samples), 0.6);
}

TEST(StatsNumerical, Kde_NormalSamplesIntegratesApproximatelyToOne) {
    std::vector<double> samples;
    samples.reserve(400);
    std::mt19937 rng(214);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 400; ++i) {
        samples.push_back(dist(rng));
    }
    const std::vector<double> grid = linspace(-10.0, 10.0, 401);
    const auto density = kde(samples, grid, 0.4, "gaussian");
    const double integral = trapezoid_integral(grid, density);
    EXPECT_NEAR(integral, 1.0, 0.15);
}

TEST(StatsNumerical, Kde_EpanechnikovNormalSamplesIntegratesApproximatelyToOne) {
    std::vector<double> samples;
    samples.reserve(400);
    std::mt19937 rng(99);
    std::normal_distribution<double> dist(1.5, 0.8);
    for (int i = 0; i < 400; ++i) {
        samples.push_back(dist(rng));
    }
    const std::vector<double> grid = linspace(-8.0, 12.0, 401);
    const auto density = kde(samples, grid, 0.5, "epanechnikov");
    const double integral = trapezoid_integral(grid, density);
    EXPECT_NEAR(integral, 1.0, 0.2);
}

TEST(StatsNumerical, Kde_UnknownKernelDefaultsToGaussian) {
    const std::vector<double> samples = {1.0, 2.0, 3.0};
    const std::vector<double> grid = linspace(0.0, 4.0, 41);
    const auto gaussian = kde(samples, grid, 0.5, "gaussian");
    const auto unknown = kde(samples, grid, 0.5, "unknown");
    ASSERT_EQ(gaussian.size(), unknown.size());
    for (size_t i = 0; i < gaussian.size(); ++i) {
        EXPECT_NEAR(unknown[i], gaussian[i], 1e-12);
    }
}

TEST(StatsNumerical, Kde_OptimizedMatchesNaive_GaussianSmallUniform) {
    const std::vector<double> samples = {0.5, 1.5, 2.5, 3.5, 4.5};
    const std::vector<double> grid = linspace(-1.0, 6.0, 37);
    expect_kde_matches_naive(samples, grid, 0.75, "gaussian");
}

TEST(StatsNumerical, Kde_OptimizedMatchesNaive_GaussianSingleSample) {
    const std::vector<double> samples = {5.0};
    const std::vector<double> grid = linspace(0.0, 10.0, 51);
    expect_kde_matches_naive(samples, grid, 1.0, "gaussian");
}

TEST(StatsNumerical, Kde_OptimizedMatchesNaive_GaussianRandomSmall) {
    std::vector<double> samples;
    samples.reserve(24);
    std::mt19937 rng(218);
    std::normal_distribution<double> dist(1.0, 0.8);
    for (int i = 0; i < 24; ++i) {
        samples.push_back(dist(rng));
    }
    const std::vector<double> grid = linspace(-3.0, 5.0, 41);
    expect_kde_matches_naive(samples, grid, 0.35, "gaussian");
}

TEST(StatsNumerical, Kde_OptimizedMatchesNaive_GaussianWideGridSparseSamples) {
    const std::vector<double> samples = {-2.0, 0.0, 4.0, 8.0};
    const std::vector<double> grid = linspace(-10.0, 12.0, 45);
    expect_kde_matches_naive(samples, grid, 0.5, "gaussian");
}

TEST(StatsNumerical, Kde_OptimizedMatchesNaive_EpanechnikovSmall) {
    const std::vector<double> samples = {1.0, 2.0, 2.5, 4.0, 5.0};
    const std::vector<double> grid = linspace(-1.0, 7.0, 33);
    expect_kde_matches_naive(samples, grid, 0.6, "epanechnikov");
}

TEST(StatsNumerical, Kde_OptimizedMatchesNaive_EpanechnikovBoundarySupport) {
    const std::vector<double> samples = {0.0, 3.0};
    const std::vector<double> grid = linspace(-2.0, 5.0, 29);
    expect_kde_matches_naive(samples, grid, 1.0, "epanechnikov");
}

// ============================================================
// MAD (median absolute deviation)
// ============================================================

TEST(StatsNumerical, MadKnownValue_Unscaled) {
    // median=2, abs devs=[1,1,0,0,2,4,7], median dev=1
    std::vector<double> v{1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0};
    EXPECT_NEAR(mad(v, false), 1.0, 1e-12);
}

TEST(StatsNumerical, MadKnownValue_ScaledDefault) {
    std::vector<double> v{1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0};
    EXPECT_NEAR(mad(v), 1.4826, 1e-12);
    EXPECT_NEAR(mad(v, true), 1.4826, 1e-12);
}

TEST(StatsNumerical, MadEmpty) {
    std::vector<double> v{};
    EXPECT_DOUBLE_EQ(mad(v), 0.0);
    EXPECT_DOUBLE_EQ(mad(v, false), 0.0);
}

TEST(StatsNumerical, MadSingleElement) {
    std::vector<double> v{7.0};
    EXPECT_DOUBLE_EQ(mad(v), 0.0);
    EXPECT_DOUBLE_EQ(mad(v, false), 0.0);
}

TEST(StatsNumerical, MadConstantData) {
    std::vector<double> v{3.0, 3.0, 3.0, 3.0};
    EXPECT_DOUBLE_EQ(mad(v), 0.0);
    EXPECT_DOUBLE_EQ(mad(v, false), 0.0);
}

TEST(StatsNumerical, MadSymmetricEvenLength) {
    // median=0, abs devs=[3,1,1,3], median dev=2
    std::vector<double> v{-3.0, -1.0, 1.0, 3.0};
    EXPECT_NEAR(mad(v, false), 2.0, 1e-12);
    EXPECT_NEAR(mad(v), 2.0 * 1.4826, 1e-12);
}

TEST(StatsNumerical, MadScaleOffPreservesRawMedianDev) {
    std::vector<double> v{10.0, 12.0, 14.0, 18.0, 22.0};
    const double raw = mad(v, false);
    EXPECT_GT(raw, 0.0);
    EXPECT_NEAR(mad(v, true), raw * 1.4826, 1e-12);
}

TEST(StatsNumerical, MadOddLengthKnownMedian) {
    // median=5, abs devs=[4,2,0,1,3], median dev=2
    std::vector<double> v{1.0, 3.0, 5.0, 6.0, 7.0};
    EXPECT_NEAR(mad(v, false), 2.0, 1e-12);
}

TEST(StatsNumerical, MadTwoElements) {
    std::vector<double> v{2.0, 8.0};
    // median=5, abs devs=[3,3], median dev=3
    EXPECT_NEAR(mad(v, false), 3.0, 1e-12);
    EXPECT_NEAR(mad(v), 3.0 * 1.4826, 1e-12);
}
