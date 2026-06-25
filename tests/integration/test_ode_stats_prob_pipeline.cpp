// MathScript Integration: ODE + Stats + Prob full system pipeline (Wave 55)

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/ode/ode.hpp"
#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// ODE solution statistics compared to analytical moments
// ---------------------------------------------------------------------------

TEST(OdeStatsProbPipeline, ODE_ExpDecay_MeanMatchesIntegral) {
    // y(t) = exp(-t): mean over [0, T] = (1 - e^{-T}) / T
    const double T = 2.0;
    const size_t N = 2001;
    const auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, T, N - 1);
    ASSERT_EQ(result.y.size(), N);
    double m = mean(result.y);
    double expected = (1.0 - std::exp(-T)) / T;
    EXPECT_NEAR(m, expected, 0.01);
}

TEST(OdeStatsProbPipeline, ODE_ExpGrowth_StddevFiniteAndPositive) {
    const auto f = [](double, double y) { return 0.5 * y; };
    auto result = ode_rk4(f, 0.0, 1.0, 2.0, 200);
    double s = stddev(result.y);
    EXPECT_TRUE(std::isfinite(s));
    EXPECT_GT(s, 0.0) << "Growing ODE should have positive stddev";
}

TEST(OdeStatsProbPipeline, ODE_ConstantODE_StddevNearZero) {
    // dy/dt = 0, y=5 → no variance
    const auto f = [](double, double) { return 0.0; };
    auto result = ode_euler(f, 0.0, 5.0, 3.0, 300);
    double s = stddev(result.y);
    EXPECT_NEAR(s, 0.0, 1e-10) << "Constant ODE should have zero stddev";
}

// ---------------------------------------------------------------------------
// ODE trajectory → prob distribution comparison
// ---------------------------------------------------------------------------

TEST(OdeStatsProbPipeline, ODE_ExpDecay_CDF_CompareNormPDF) {
    // ODE decaying exp: mean near 0.5 for t=[0,2], stddev finite
    const auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 2.0, 200);
    double m = mean(result.y);
    double s = stddev(result.y);
    // At mean, norm_pdf should be the maximum (peak of Gaussian)
    double pdf_at_mean = norm_pdf(m, m, s);
    double pdf_at_mean_plus_s = norm_pdf(m + s, m, s);
    EXPECT_GT(pdf_at_mean, pdf_at_mean_plus_s)
        << "Normal PDF should be larger at mean than at mean+sigma";
}

TEST(OdeStatsProbPipeline, ODE_Mean_Within_NormCDF_Interval) {
    // If ODE mean ~ 0.5, norm_cdf(mean, mean, s) = 0.5
    const auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 2.0, 200);
    double m = mean(result.y);
    double s = std::max(stddev(result.y), 1e-6);
    double cdf_at_mean = norm_cdf(m, m, s);
    EXPECT_NEAR(cdf_at_mean, 0.5, 1e-8);
}

// ---------------------------------------------------------------------------
// Stats of sampled distributions match theoretical
// ---------------------------------------------------------------------------

TEST(OdeStatsProbPipeline, BinomPDF_Sum_To_One) {
    // Sum of binom_pdf(k, n, p) for k=0..n should be 1
    int n = 10;
    double p = 0.3;
    double sum = 0.0;
    for (int k = 0; k <= n; ++k) sum += binom_pdf(k, n, p);
    EXPECT_NEAR(sum, 1.0, 1e-8);
}

TEST(OdeStatsProbPipeline, PoisPDF_Sum_To_One) {
    double lambda = 3.0;
    double sum = 0.0;
    for (int k = 0; k <= 50; ++k) sum += pois_pdf(static_cast<double>(k), lambda);
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(OdeStatsProbPipeline, Prob_NormCDF_Monotone_Statistics) {
    // Sorted CDF values → monotone increasing
    std::vector<double> xs = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    std::vector<double> cdfs;
    for (double x : xs) cdfs.push_back(norm_cdf(x, 0.0, 1.0));
    // Check monotonicity
    for (size_t i = 1; i < cdfs.size(); ++i)
        EXPECT_GE(cdfs[i], cdfs[i - 1]) << "norm_cdf not monotone at i=" << i;
    // Check min/max via stats
    EXPECT_LT(min_value(cdfs), 0.01);
    EXPECT_GT(max_value(cdfs), 0.99);
}

// ---------------------------------------------------------------------------
// ODE + stats correlation test
// ---------------------------------------------------------------------------

TEST(OdeStatsProbPipeline, ODE_TAndY_Correlation) {
    // For dy/dt = -y: y is monotone function of t → |corr(t, y)| near 1
    const auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 2.0, 200);
    double r = correlation(result.t, result.y);
    EXPECT_LT(r, -0.9) << "t and y should be strongly negatively correlated for decaying ODE";
}

TEST(OdeStatsProbPipeline, ODE_GrowthTAndY_PositiveCorrelation) {
    // For dy/dt = y: y grows → positive correlation with t
    const auto f = [](double, double y) { return y; };
    auto result = ode_rk4(f, 0.0, 1.0, 1.0, 100);
    double r = correlation(result.t, result.y);
    EXPECT_GT(r, 0.9);
}

// ---------------------------------------------------------------------------
// Ttest and ztest from ODE samples
// ---------------------------------------------------------------------------

TEST(OdeStatsProbPipeline, ODE_Samples_Ttest_MeanNearTruth) {
    // dy/dt = -y, y(0)=1: mean = (1-e^{-T})/T for T=1 ≈ 0.632
    const auto f = [](double, double y) { return -y; };
    auto result = ode_rk4(f, 0.0, 1.0, 1.0, 1000);
    double expected_mean = 1.0 - std::exp(-1.0); // ≈ 0.632
    double t_stat = ttest(result.y, expected_mean);
    // t-stat should be small (close to zero) when mean matches
    EXPECT_LT(std::abs(t_stat), 3.0)
        << "t-test vs true mean should give small t-statistic";
}
