// MathScript Advanced Stats: Linear Regression, Correlation, Skewness/Kurtosis (Wave 55)

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Linear regression
// ---------------------------------------------------------------------------

TEST(StatsRegressionAdv, Regression_PerfectFit_R2_Is1) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0}; // y = 2x + 1
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.r_squared, 1.0, 1e-10);
}

TEST(StatsRegressionAdv, Regression_PerfectFit_Slope) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {1.0, 4.0, 7.0, 10.0, 13.0}; // y = 3x + 1
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.slope, 3.0, 1e-10);
    EXPECT_NEAR(r.intercept, 1.0, 1e-10);
}

TEST(StatsRegressionAdv, Regression_Negative_Slope) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {10.0, 8.0, 6.0, 4.0, 2.0}; // y = -2x + 12
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.slope, -2.0, 1e-10);
    EXPECT_GT(r.r_squared, 0.99);
}

TEST(StatsRegressionAdv, Regression_HorizontalLine_ZeroSlope) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 5.0, 5.0, 5.0, 5.0}; // y = 5
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.slope, 0.0, 1e-10);
    EXPECT_NEAR(r.intercept, 5.0, 1e-10);
}

TEST(StatsRegressionAdv, Regression_AllFinite) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {2.0, 4.0, 6.0};
    auto r = linear_regression(x, y);
    EXPECT_TRUE(std::isfinite(r.slope));
    EXPECT_TRUE(std::isfinite(r.intercept));
    EXPECT_TRUE(std::isfinite(r.r_squared));
}

TEST(StatsRegressionAdv, Regression_R2_Between_0_And_1) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {1.1, 1.9, 3.2, 3.8, 5.1}; // noisy linear
    auto r = linear_regression(x, y);
    EXPECT_GE(r.r_squared, 0.0);
    EXPECT_LE(r.r_squared, 1.0 + 1e-10);
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

TEST(StatsRegressionAdv, Correlation_PerfectPositive) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0}; // y = 2x
    double r = correlation(x, y);
    EXPECT_NEAR(r, 1.0, 1e-10);
}

TEST(StatsRegressionAdv, Correlation_PerfectNegative) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 4.0, 3.0, 2.0, 1.0}; // inverse
    double r = correlation(x, y);
    EXPECT_NEAR(r, -1.0, 1e-10);
}

TEST(StatsRegressionAdv, Correlation_Symmetric) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {2.0, 1.5, 3.5, 2.5};
    double r_xy = correlation(x, y);
    double r_yx = correlation(y, x);
    EXPECT_NEAR(r_xy, r_yx, 1e-10);
}

TEST(StatsRegressionAdv, Correlation_InRange) {
    std::vector<double> x = {1.2, 3.4, 2.1, 5.6, 4.5};
    std::vector<double> y = {2.3, 1.1, 4.5, 3.2, 5.0};
    double r = correlation(x, y);
    EXPECT_GE(r, -1.0 - 1e-10);
    EXPECT_LE(r, 1.0 + 1e-10);
}

TEST(StatsRegressionAdv, Correlation_SelfCorrelation_Is1) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(correlation(x, x), 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Skewness and Kurtosis
// ---------------------------------------------------------------------------

TEST(StatsRegressionAdv, Skewness_SymmetricData_NearZero) {
    std::vector<double> x = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    EXPECT_NEAR(skewness(x), 0.0, 1e-10);
}

TEST(StatsRegressionAdv, Skewness_RightSkewed_Positive) {
    // Right-skewed: long right tail
    std::vector<double> x = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 10.0};
    EXPECT_GT(skewness(x), 0.0) << "Right-skewed data should have positive skewness";
}

TEST(StatsRegressionAdv, Skewness_LeftSkewed_Negative) {
    // Left-skewed: long left tail
    std::vector<double> x = {-10.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    EXPECT_LT(skewness(x), 0.0) << "Left-skewed data should have negative skewness";
}

TEST(StatsRegressionAdv, Skewness_AllFinite) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_TRUE(std::isfinite(skewness(x)));
}

TEST(StatsRegressionAdv, Kurtosis_Finite_For_SymmetricData) {
    // Kurtosis can be negative (platykurtic); just check it's finite
    std::vector<double> x = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    double k = kurtosis(x);
    EXPECT_TRUE(std::isfinite(k));
}

TEST(StatsRegressionAdv, Kurtosis_HeavyTail_HigherKurtosis) {
    // Heavy-tailed data should have higher kurtosis
    std::vector<double> x_norm = {-1.0, -0.5, 0.0, 0.5, 1.0};
    std::vector<double> x_heavy = {-10.0, -1.0, 0.0, 1.0, 10.0};
    double k_norm  = kurtosis(x_norm);
    double k_heavy = kurtosis(x_heavy);
    EXPECT_GT(k_heavy, k_norm)
        << "Heavy-tailed data should have higher kurtosis";
}

// ---------------------------------------------------------------------------
// Percentile and mode
// ---------------------------------------------------------------------------

TEST(StatsRegressionAdv, Percentile_Min_Is_P0) {
    std::vector<double> x = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    EXPECT_NEAR(percentile(x, 0.0), min_value(x), 1e-10);
}

TEST(StatsRegressionAdv, Percentile_P1_Finite_InRange) {
    std::vector<double> x = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    double p1 = percentile(x, 1.0);
    EXPECT_TRUE(std::isfinite(p1));
    EXPECT_GE(p1, min_value(x));
    EXPECT_LE(p1, max_value(x));
}

TEST(StatsRegressionAdv, Percentile_Within_DataRange) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    double p50 = percentile(x, 0.5);
    // Result should be within [min, max] regardless of definition
    EXPECT_GE(p50, min_value(x));
    EXPECT_LE(p50, max_value(x));
}

TEST(StatsRegressionAdv, Mode_Returns_MostFrequent) {
    std::vector<double> x = {1.0, 2.0, 2.0, 3.0, 2.0, 4.0};
    EXPECT_NEAR(mode(x), 2.0, 1e-10);
}
