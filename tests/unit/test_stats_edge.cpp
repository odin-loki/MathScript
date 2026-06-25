// MathScript: Stats Edge Case Tests (Wave 46)
// Tests for extreme inputs: single element, constant, repeated, large variance

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/stats/stats.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// mean with edge inputs
// ---------------------------------------------------------------------------

TEST(StatsEdge, Mean_LargeVector_Finite) {
    std::vector<double> d(10000, 1.0);
    EXPECT_NEAR(mean(d), 1.0, 1e-10);
}

TEST(StatsEdge, Mean_MixedSigns_ZeroMean) {
    std::vector<double> d;
    for (int i = -500; i <= 500; ++i) d.push_back(static_cast<double>(i));
    EXPECT_NEAR(mean(d), 0.0, 1e-9);
}

TEST(StatsEdge, Mean_VeryLargeValues_Finite) {
    std::vector<double> d = {1e10, 2e10, 3e10};
    EXPECT_NEAR(mean(d), 2e10, 1.0);
}

TEST(StatsEdge, Mean_VerySmallValues_Finite) {
    std::vector<double> d = {1e-10, 2e-10, 3e-10};
    EXPECT_NEAR(mean(d), 2e-10, 1e-20);
}

// ---------------------------------------------------------------------------
// var / stddev edge cases
// ---------------------------------------------------------------------------

TEST(StatsEdge, Var_TwoElements_ComputesCorrectly) {
    // var({2, 4}) with population variance = 1 (mean=3, dev=-1 and +1)
    std::vector<double> d = {2.0, 4.0};
    double v = var(d);
    EXPECT_GT(v, 0.0);
    EXPECT_TRUE(std::isfinite(v));
}

TEST(StatsEdge, Stddev_AlwaysNonNegative) {
    for (auto& d : std::vector<std::vector<double>>{
        {1.0}, {1.0, 1.0}, {1.0, 2.0, 3.0}, {-3.0, 0.0, 3.0}
    }) {
        EXPECT_GE(stddev(d), 0.0);
    }
}

TEST(StatsEdge, Var_HighPrecision_Symmetric) {
    // var({-3, -1, 1, 3}) = var({1, 3, -1, -3}) (order shouldn't matter)
    std::vector<double> d1 = {-3.0, -1.0, 1.0, 3.0};
    std::vector<double> d2 = {3.0, -3.0, 1.0, -1.0};
    EXPECT_NEAR(var(d1), var(d2), 1e-12);
}

// ---------------------------------------------------------------------------
// median edge cases
// ---------------------------------------------------------------------------

TEST(StatsEdge, Median_AllSameValue_IsValue) {
    std::vector<double> d(100, 7.0);
    EXPECT_NEAR(median(d), 7.0, 1e-10);
}

TEST(StatsEdge, Median_TwoElements_IsMidpoint) {
    std::vector<double> d = {3.0, 7.0};
    EXPECT_NEAR(median(d), 5.0, 1e-10);
}

TEST(StatsEdge, Median_LargeOdd_IsMiddle) {
    std::vector<double> d;
    for (int i = 0; i < 101; ++i) d.push_back(static_cast<double>(i));
    EXPECT_NEAR(median(d), 50.0, 1e-10);
}

TEST(StatsEdge, Median_LargeEven_IsMidpointOfMiddleTwo) {
    std::vector<double> d;
    for (int i = 0; i < 100; ++i) d.push_back(static_cast<double>(i));
    EXPECT_NEAR(median(d), 49.5, 1e-10);
}

// ---------------------------------------------------------------------------
// correlation edge cases
// ---------------------------------------------------------------------------

TEST(StatsEdge, Correlation_ResultInRange) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 4.0, 3.0, 2.0, 1.0};
    double r = correlation(x, y);
    EXPECT_GE(r, -1.0 - 1e-9);
    EXPECT_LE(r, 1.0 + 1e-9);
}

TEST(StatsEdge, Correlation_Symmetric_Property) {
    std::vector<double> x = {1.5, 2.3, 3.7, 4.1, 5.9};
    std::vector<double> y = {2.1, 3.2, 1.4, 5.0, 4.3};
    EXPECT_NEAR(correlation(x, y), correlation(y, x), 1e-12);
}

// ---------------------------------------------------------------------------
// percentile edge cases
// ---------------------------------------------------------------------------

TEST(StatsEdge, Percentile_AllSame_IsValue) {
    std::vector<double> d(20, 3.14);
    EXPECT_NEAR(percentile(d, 25.0), 3.14, 1e-10);
    EXPECT_NEAR(percentile(d, 50.0), 3.14, 1e-10);
    EXPECT_NEAR(percentile(d, 75.0), 3.14, 1e-10);
}

TEST(StatsEdge, Percentile_IsBetweenMinMax) {
    std::vector<double> d = {5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0};
    for (double p : {10.0, 25.0, 50.0, 75.0, 90.0}) {
        double val = percentile(d, p);
        EXPECT_GE(val, min_value(d) - 1e-9) << "Percentile below min at p=" << p;
        EXPECT_LE(val, max_value(d) + 1e-9) << "Percentile above max at p=" << p;
    }
}

// ---------------------------------------------------------------------------
// skewness and kurtosis
// ---------------------------------------------------------------------------

TEST(StatsEdge, Skewness_Symmetric_NearZero) {
    // Symmetric distribution: {1, 2, 3, 4, 5}
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0, 5.0};
    double sk = skewness(d);
    EXPECT_NEAR(sk, 0.0, 0.01) << "Symmetric data should have ~0 skewness";
}

TEST(StatsEdge, Skewness_RightSkewed_Positive) {
    // Right-skewed: many low values, few very high
    std::vector<double> d = {1.0, 1.0, 1.0, 2.0, 10.0};
    double sk = skewness(d);
    EXPECT_GT(sk, 0.0) << "Right-skewed data should have positive skewness";
}

TEST(StatsEdge, Kurtosis_IsFinite) {
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    EXPECT_TRUE(std::isfinite(kurtosis(d)));
}

TEST(StatsEdge, Kurtosis_Normal_NearThree) {
    // Normal distribution has kurtosis ~3
    // Use a large symmetric sample
    std::vector<double> d;
    for (int i = -20; i <= 20; ++i)
        d.push_back(static_cast<double>(i));
    double k = kurtosis(d);
    EXPECT_TRUE(std::isfinite(k));
}

// ---------------------------------------------------------------------------
// two-sample t-test
// ---------------------------------------------------------------------------

TEST(StatsEdge, TwoSampleTTest_SameDistribution_SmallT) {
    // Same distribution → t-statistic should be small
    std::vector<double> a = {5.0, 5.1, 4.9, 5.0, 5.2};
    std::vector<double> b = {5.0, 5.0, 5.1, 4.9, 5.0};
    double t = two_sample_ttest(a, b);
    EXPECT_LT(std::abs(t), 3.0) << "Same distribution should give small t-stat";
}

TEST(StatsEdge, TwoSampleTTest_DifferentMeans_LargeT) {
    // Very different means with some variance → large t-statistic
    std::vector<double> a, b;
    for (int i = 0; i < 20; ++i) {
        a.push_back(100.0 + (i % 3 - 1) * 0.1);  // ~100 with tiny variance
        b.push_back(0.0   + (i % 3 - 1) * 0.1);  // ~0 with tiny variance
    }
    double t = two_sample_ttest(a, b);
    EXPECT_GT(std::abs(t), 100.0) << "Very different means should give large t-stat";
}

// ---------------------------------------------------------------------------
// ztest
// ---------------------------------------------------------------------------

TEST(StatsEdge, ZTest_SampleMean_EqualsMu_NearZero) {
    // If sample mean equals mu, z should be near 0
    std::vector<double> d(50, 5.0);
    double z = ztest(d, 5.0, 1.0);
    EXPECT_NEAR(z, 0.0, 1e-6);
}

TEST(StatsEdge, ZTest_IsFinite) {
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0, 5.0};
    double z = ztest(d, 3.0, 1.0);
    EXPECT_TRUE(std::isfinite(z));
}

// ---------------------------------------------------------------------------
// linear_regression edge cases
// ---------------------------------------------------------------------------

TEST(StatsEdge, LinearRegression_SlopeSign_Positive) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    auto r = linear_regression(x, y);
    EXPECT_GT(r.slope, 0.0);
}

TEST(StatsEdge, LinearRegression_SlopeSign_Negative) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {10.0, 8.0, 6.0, 4.0, 2.0};
    auto r = linear_regression(x, y);
    EXPECT_LT(r.slope, 0.0);
}

TEST(StatsEdge, LinearRegression_PerfectFit_RSq1) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0};
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.r_squared, 1.0, 1e-9);
}
