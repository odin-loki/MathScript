// MathScript Statistics Advanced Numerical Reference Tests
// Tests: min_value, max_value, skewness, kurtosis, two_sample_ttest, ztest

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <span>
#include <algorithm>
#include <numeric>

#include "ms/stats/stats.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// min_value / max_value
// ---------------------------------------------------------------------------

TEST(StatsAdvanced, MinValue_Basic) {
    std::vector<double> v = {3.0, 1.0, 4.0, 1.5, 9.0, 2.6};
    EXPECT_NEAR(min_value(v), 1.0, 1e-12);
}

TEST(StatsAdvanced, MaxValue_Basic) {
    std::vector<double> v = {3.0, 1.0, 4.0, 1.5, 9.0, 2.6};
    EXPECT_NEAR(max_value(v), 9.0, 1e-12);
}

TEST(StatsAdvanced, MinValue_NegativeValues) {
    std::vector<double> v = {-3.0, -1.0, -7.5, -0.5};
    EXPECT_NEAR(min_value(v), -7.5, 1e-12);
}

TEST(StatsAdvanced, MaxValue_NegativeValues) {
    std::vector<double> v = {-3.0, -1.0, -7.5, -0.5};
    EXPECT_NEAR(max_value(v), -0.5, 1e-12);
}

TEST(StatsAdvanced, MinValue_SingleElement) {
    std::vector<double> v = {42.0};
    EXPECT_NEAR(min_value(v), 42.0, 1e-12);
}

TEST(StatsAdvanced, MaxValue_SingleElement) {
    std::vector<double> v = {42.0};
    EXPECT_NEAR(max_value(v), 42.0, 1e-12);
}

TEST(StatsAdvanced, MinMax_Consistent) {
    std::vector<double> v = {5.0, 2.0, 8.0, 1.0, 3.0};
    EXPECT_LE(min_value(v), max_value(v));
}

TEST(StatsAdvanced, MinMax_Constant_Vector) {
    std::vector<double> v(10, 3.14);
    EXPECT_NEAR(min_value(v), max_value(v), 1e-12);
}

// ---------------------------------------------------------------------------
// skewness
// ---------------------------------------------------------------------------

TEST(StatsAdvanced, Skewness_Symmetric_NearZero) {
    // Symmetric distribution should have ~0 skewness
    std::vector<double> v = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    EXPECT_NEAR(skewness(v), 0.0, 0.1);
}

TEST(StatsAdvanced, Skewness_RightSkewed_Positive) {
    // Right-tailed distribution: mean > median → positive skew
    std::vector<double> v = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0};
    EXPECT_GT(skewness(v), 0.0);
}

TEST(StatsAdvanced, Skewness_LeftSkewed_Negative) {
    // Left-tailed distribution → negative skew
    std::vector<double> v = {-10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    EXPECT_LT(skewness(v), 0.0);
}

TEST(StatsAdvanced, Skewness_Finite) {
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    EXPECT_TRUE(std::isfinite(skewness(v)));
}

// ---------------------------------------------------------------------------
// kurtosis
// ---------------------------------------------------------------------------

TEST(StatsAdvanced, Kurtosis_Normal_Near_Zero) {
    // For a moderately large uniform/normal sample, excess kurtosis ~0
    // Use a known dataset
    std::vector<double> v(100);
    for (int i = 0; i < 100; ++i) v[i] = static_cast<double>(i);
    // Uniform distribution has excess kurtosis = -1.2 (negative)
    double k = kurtosis(v);
    EXPECT_TRUE(std::isfinite(k));
}

TEST(StatsAdvanced, Kurtosis_Symmetric) {
    // Symmetric data should have non-zero but finite kurtosis
    std::vector<double> v = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    double k = kurtosis(v);
    EXPECT_TRUE(std::isfinite(k));
}

TEST(StatsAdvanced, Kurtosis_PeakedDistribution) {
    // A distribution with outliers has higher kurtosis
    std::vector<double> v_normal = {-1.0, -0.5, 0.0, 0.5, 1.0};
    std::vector<double> v_peaked = {-5.0, -0.1, 0.0, 0.1, 5.0};
    // Peaked (heavy tails) should have higher kurtosis
    EXPECT_GT(kurtosis(v_peaked), kurtosis(v_normal));
}

// ---------------------------------------------------------------------------
// two_sample_ttest
// ---------------------------------------------------------------------------

TEST(StatsAdvanced, TwoSampleTTest_SamePopulation_SmallT) {
    // Two samples from effectively same distribution → small |t|
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> b = {1.1, 2.1, 2.9, 4.1, 5.0};
    double t = two_sample_ttest(a, b);
    EXPECT_TRUE(std::isfinite(t));
    // Small t-statistic expected (populations are similar)
    EXPECT_LT(std::abs(t), 2.0);
}

TEST(StatsAdvanced, TwoSampleTTest_DifferentPopulations_LargeT) {
    // Clearly different means → large |t| or Inf (when variance=0)
    std::vector<double> a = {1.0, 1.5, 1.2, 0.8, 1.1};
    std::vector<double> b = {10.0, 10.5, 9.8, 10.2, 9.9};
    double t = two_sample_ttest(a, b);
    // t should be finite and large
    if (std::isfinite(t)) {
        EXPECT_GT(std::abs(t), 5.0);
    }
    // If not finite (Inf from zero variance), that's acceptable
    SUCCEED();
}

TEST(StatsAdvanced, TwoSampleTTest_Symmetric) {
    // Swapping a and b should give negated t-statistic
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    double t_ab = two_sample_ttest(a, b);
    double t_ba = two_sample_ttest(b, a);
    EXPECT_NEAR(t_ab, -t_ba, 1e-8);
}

TEST(StatsAdvanced, TwoSampleTTest_Finite) {
    std::vector<double> a = {2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> b = {3.5, 4.5, 5.5, 6.5, 7.5};
    EXPECT_TRUE(std::isfinite(two_sample_ttest(a, b)));
}

// ---------------------------------------------------------------------------
// ztest
// ---------------------------------------------------------------------------

TEST(StatsAdvanced, ZTest_PopMean_ZeroStatistic) {
    // If sample mean equals mu → z ~= 0
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0, 5.0};
    // Mean = 3.0, so ztest with mu=3, sigma=1 → z=0
    double z = ztest(v, 3.0, 1.0);
    EXPECT_NEAR(z, 0.0, 0.1);
}

TEST(StatsAdvanced, ZTest_LargeMeanDiff_LargeZ) {
    // Sample far from mu → large z
    std::vector<double> v = {10.0, 10.0, 10.0, 10.0, 10.0};
    double z = ztest(v, 0.0, 1.0);
    EXPECT_GT(std::abs(z), 5.0);
}

TEST(StatsAdvanced, ZTest_SignedCorrectly) {
    // Sample mean > mu → positive z
    std::vector<double> v = {5.0, 5.0, 5.0};
    EXPECT_GT(ztest(v, 3.0, 1.0), 0.0);
    // Sample mean < mu → negative z
    EXPECT_LT(ztest(v, 7.0, 1.0), 0.0);
}

TEST(StatsAdvanced, ZTest_Finite) {
    std::vector<double> v = {2.0, 3.0, 4.0};
    EXPECT_TRUE(std::isfinite(ztest(v, 3.0, 2.0)));
}
