#include <gtest/gtest.h>
#include <cmath>
#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"

using namespace ms;

TEST(StatsExtTest, linear_regression) {
    std::vector<double> x{1, 2, 3, 4, 5};
    std::vector<double> y{2, 4, 6, 8, 10};
    auto fit = linear_regression(x, y);
    EXPECT_NEAR(fit.slope, 2.0, 1e-12);
    EXPECT_NEAR(fit.intercept, 0.0, 1e-12);
    EXPECT_NEAR(fit.r_squared, 1.0, 1e-12);
}

TEST(StatsExtTest, two_sample_ttest) {
    std::vector<double> a{1, 2, 3};
    std::vector<double> b{4, 5, 6};
    EXPECT_LT(two_sample_ttest(a, b), 0.0);
}

TEST(ProbExtTest, uniform_cdf) {
    EXPECT_NEAR(uniform_cdf(0.5, 0.0, 1.0), 0.5, 1e-12);
    EXPECT_NEAR(uniform_pdf(0.5, 0.0, 1.0), 1.0, 1e-12);
}

TEST(ProbExtTest, chi2_cdf) {
    EXPECT_NEAR(chi2_cdf(1.0, 1.0), 0.682689492, 1e-3);
}

TEST(ProbExtTest, gamma_pdf) {
    EXPECT_NEAR(gamma_pdf(1.0, 2.0, 1.0), 0.367879441, 1e-6);
}

TEST(ProbExtTest, invalid_and_boundary_inputs) {
    EXPECT_DOUBLE_EQ(norm_pdf(0.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(norm_cdf(0.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(norm_ppf(0.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(norm_ppf(1.0, 0.0, 1.0), 0.0);

    EXPECT_DOUBLE_EQ(exp_pdf(-1.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(exp_pdf(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(exp_cdf(-1.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(exp_cdf(1.0, 0.0), 0.0);

    EXPECT_DOUBLE_EQ(binom_pdf(-1, 4, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(binom_pdf(5, 4, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(binom_pdf(2, 4, -0.1), 0.0);

    EXPECT_DOUBLE_EQ(pois_pdf(-1.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(pois_pdf(2.0, 0.0), 0.0);

    EXPECT_DOUBLE_EQ(chi2_pdf(0.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(chi2_cdf(0.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(chi2_pdf(1.0, 0.0), 0.0);

    EXPECT_DOUBLE_EQ(uniform_pdf(0.5, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(uniform_pdf(-1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(uniform_cdf(0.5, 1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(uniform_cdf(-1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(uniform_cdf(2.0, 0.0, 1.0), 1.0);

    EXPECT_DOUBLE_EQ(t_pdf(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(t_cdf(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(gamma_pdf(0.0, 2.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(gamma_pdf(1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(gamma_pdf(1.0, 2.0, 0.0), 0.0);
}

TEST(StatsExtTest, degenerate_hypothesis_and_regression) {
    const std::vector<double> one{5.0};
    const std::vector<double> constant{2.0, 2.0, 2.0};
    EXPECT_DOUBLE_EQ(ttest(one, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(ztest(one, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(correlation(constant, constant), 0.0);
    const std::vector<double> two_vals{1.0, 2.0};
    const std::vector<double> three_vals{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(skewness(two_vals), 0.0);
    EXPECT_DOUBLE_EQ(kurtosis(three_vals), 0.0);
    EXPECT_DOUBLE_EQ(two_sample_ttest(one, one), 0.0);

    const std::vector<double> x{1.0, 1.0, 1.0};
    const std::vector<double> y{2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(linear_regression(x, y).slope, 0.0);
}

TEST(StatsExtTest, zero_variance_sample_paths) {
    const std::vector<double> flat{3.0, 3.0, 3.0, 3.0};
    EXPECT_DOUBLE_EQ(ttest(flat, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(skewness(flat), 0.0);
    EXPECT_DOUBLE_EQ(kurtosis(flat), 0.0);
    const std::vector<double> also_flat{7.0, 7.0, 7.0, 7.0};
    EXPECT_DOUBLE_EQ(two_sample_ttest(flat, also_flat), 0.0);
}

TEST(ProbExtTest, norm_ppf_interior_and_binom_edges) {
    EXPECT_GT(std::abs(norm_ppf(0.25, 0.0, 1.0)), 0.0);
    EXPECT_GT(std::abs(norm_ppf(0.75, 0.0, 1.0)), 0.0);
    EXPECT_NEAR(binom_cdf(4, 4, 0.5), 1.0, 1e-12);
    EXPECT_GT(pois_cdf(3.0, 2.0), pois_cdf(1.0, 2.0));
}
