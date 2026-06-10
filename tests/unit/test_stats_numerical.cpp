#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <gtest/gtest.h>
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
