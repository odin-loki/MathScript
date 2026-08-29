// MathScript: Advanced Statistics Tests (Wave 43)
// Tests for correlation, linear_regression, median, min/max, mode, percentile

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/stats/stats.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

TEST(StatsAdv2, Mean_AllSame) {
    std::vector<double> d = {5.0, 5.0, 5.0, 5.0};
    EXPECT_NEAR(mean(d), 5.0, 1e-10);
}

TEST(StatsAdv2, Mean_Positive_Negative) {
    std::vector<double> d = {-3.0, -1.0, 1.0, 3.0};
    EXPECT_NEAR(mean(d), 0.0, 1e-10);
}

TEST(StatsAdv2, Mean_SingleElement) {
    std::vector<double> d = {42.0};
    EXPECT_NEAR(mean(d), 42.0, 1e-10);
}

// ---------------------------------------------------------------------------
// median
// ---------------------------------------------------------------------------

TEST(StatsAdv2, Median_OddCount) {
    std::vector<double> d = {3.0, 1.0, 4.0, 1.0, 5.0};
    EXPECT_NEAR(median(d), 3.0, 1e-10);
}

TEST(StatsAdv2, Median_EvenCount) {
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0};
    EXPECT_NEAR(median(d), 2.5, 1e-10);
}

TEST(StatsAdv2, Median_AllSame) {
    std::vector<double> d = {7.0, 7.0, 7.0};
    EXPECT_NEAR(median(d), 7.0, 1e-10);
}

TEST(StatsAdv2, Median_TwoElements) {
    std::vector<double> d = {2.0, 8.0};
    EXPECT_NEAR(median(d), 5.0, 1e-10);
}

TEST(StatsAdv2, Median_LessThanOrEqualToMean_SymmetricData) {
    // For symmetric data: median == mean
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(median(d), mean(d), 1e-10);
}

// ---------------------------------------------------------------------------
// min_value / max_value
// ---------------------------------------------------------------------------

TEST(StatsAdv2, MinMax_Simple) {
    std::vector<double> d = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0};
    EXPECT_NEAR(min_value(d), 1.0, 1e-10);
    EXPECT_NEAR(max_value(d), 9.0, 1e-10);
}

TEST(StatsAdv2, MinMax_SingleElement) {
    std::vector<double> d = {42.0};
    EXPECT_NEAR(min_value(d), 42.0, 1e-10);
    EXPECT_NEAR(max_value(d), 42.0, 1e-10);
}

TEST(StatsAdv2, Min_AlwaysLeOrEqualMax) {
    std::vector<double> d = {5.0, 3.0, 8.0, 1.0, 9.0, 2.0};
    EXPECT_LE(min_value(d), max_value(d));
}

TEST(StatsAdv2, MinMax_Negative) {
    std::vector<double> d = {-5.0, -1.0, -3.0};
    EXPECT_NEAR(min_value(d), -5.0, 1e-10);
    EXPECT_NEAR(max_value(d), -1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// mode
// ---------------------------------------------------------------------------

TEST(StatsAdv2, Mode_ClearMode) {
    std::vector<double> d = {1.0, 2.0, 2.0, 3.0, 2.0};
    EXPECT_NEAR(mode(d), 2.0, 1e-10);
}

TEST(StatsAdv2, Mode_AllSame) {
    std::vector<double> d = {4.0, 4.0, 4.0};
    EXPECT_NEAR(mode(d), 4.0, 1e-10);
}

TEST(StatsAdv2, Mode_ReturnsFinite) {
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0};
    EXPECT_TRUE(std::isfinite(mode(d)));
}

// ---------------------------------------------------------------------------
// percentile
// ---------------------------------------------------------------------------

TEST(StatsAdv2, Percentile_50th_IsMedian) {
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(percentile(d, 50.0), median(d), 1e-6);
}

TEST(StatsAdv2, Percentile_0th_IsMin) {
    std::vector<double> d = {3.0, 1.0, 4.0, 1.0, 5.0};
    double p0 = percentile(d, 0.0);
    EXPECT_NEAR(p0, min_value(d), 1e-6);
}

TEST(StatsAdv2, Percentile_100th_IsMax) {
    std::vector<double> d = {3.0, 1.0, 4.0, 1.0, 5.0};
    double p100 = percentile(d, 100.0);
    EXPECT_NEAR(p100, max_value(d), 1e-6);
}

TEST(StatsAdv2, Percentile_Monotone) {
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double p25 = percentile(d, 25.0);
    double p50 = percentile(d, 50.0);
    double p75 = percentile(d, 75.0);
    EXPECT_LE(p25, p50);
    EXPECT_LE(p50, p75);
}

// ---------------------------------------------------------------------------
// correlation
// ---------------------------------------------------------------------------

TEST(StatsAdv2, Correlation_PerfectPositive) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};  // y = 2x
    double r = correlation(x, y);
    EXPECT_NEAR(r, 1.0, 1e-9);
}

TEST(StatsAdv2, Correlation_PerfectNegative) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 4.0, 3.0, 2.0, 1.0};
    double r = correlation(x, y);
    EXPECT_NEAR(r, -1.0, 1e-9);
}

TEST(StatsAdv2, Correlation_Uncorrelated_NearZero) {
    // For truly uncorrelated data: x = {-3,-1,1,3}, y = {1,-1,-1,1}
    // Dot product zero-mean x and y: (-3)(1)+(-1)(-1)+(1)(-1)+(3)(1) = -3+1-1+3 = 0
    std::vector<double> x = {-3.0, -1.0, 1.0, 3.0};
    std::vector<double> y = {1.0, -1.0, -1.0, 1.0};
    double r = correlation(x, y);
    EXPECT_NEAR(r, 0.0, 1e-9);
}

TEST(StatsAdv2, Correlation_Range_Minus1_To_1) {
    std::vector<double> x = {1.0, 3.0, 5.0, 7.0, 9.0};
    std::vector<double> y = {2.0, 1.0, 4.0, 3.0, 5.0};
    double r = correlation(x, y);
    EXPECT_GE(r, -1.0);
    EXPECT_LE(r, 1.0);
}

TEST(StatsAdv2, Correlation_Symmetric) {
    // corr(x, y) == corr(y, x)
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(correlation(x, y), correlation(y, x), 1e-10);
}

// ---------------------------------------------------------------------------
// linear_regression
// ---------------------------------------------------------------------------

TEST(StatsAdv2, LinearRegression_PerfectFit) {
    // y = 2x + 1, should give slope=2, intercept=1, r^2=1
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.slope, 2.0, 1e-9);
    EXPECT_NEAR(r.intercept, 1.0, 1e-9);
    EXPECT_NEAR(r.r_squared, 1.0, 1e-9);
}

TEST(StatsAdv2, LinearRegression_HorizontalLine) {
    // y = 5 (constant), slope=0, intercept=5
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 5.0, 5.0, 5.0, 5.0};
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.slope, 0.0, 1e-9);
    EXPECT_NEAR(r.intercept, 5.0, 1e-9);
}

TEST(StatsAdv2, LinearRegression_RSquared_Bounded) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1};
    auto r = linear_regression(x, y);
    EXPECT_GE(r.r_squared, 0.0);
    EXPECT_LE(r.r_squared, 1.0);
}

TEST(StatsAdv2, LinearRegression_NegativeSlope) {
    // y = -x + 10
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {9.0, 8.0, 7.0, 6.0, 5.0};
    auto r = linear_regression(x, y);
    EXPECT_NEAR(r.slope, -1.0, 1e-9);
    EXPECT_NEAR(r.intercept, 10.0, 1e-9);
    EXPECT_NEAR(r.r_squared, 1.0, 1e-9);
}

TEST(StatsAdv2, LinearRegression_AllValuesFinite) {
    std::vector<double> x = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> y = {1.0, 2.0, 2.5, 4.0};
    auto r = linear_regression(x, y);
    EXPECT_TRUE(std::isfinite(r.slope));
    EXPECT_TRUE(std::isfinite(r.intercept));
    EXPECT_TRUE(std::isfinite(r.r_squared));
}

// ---------------------------------------------------------------------------
// var / stddev consistency
// ---------------------------------------------------------------------------

TEST(StatsAdv2, Var_NonNegative) {
    std::vector<double> d = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_GE(var(d), 0.0);
}

TEST(StatsAdv2, Stddev_SquaredIsVar) {
    std::vector<double> d = {2.0, 4.0, 6.0, 8.0};
    double s = stddev(d);
    double v = var(d);
    EXPECT_NEAR(s * s, v, 1e-9);
}

TEST(StatsAdv2, Var_AllSame_IsZero) {
    std::vector<double> d = {3.0, 3.0, 3.0, 3.0};
    EXPECT_NEAR(var(d), 0.0, 1e-10);
}
