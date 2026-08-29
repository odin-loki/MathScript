#include <gtest/gtest.h>
#include <cmath>

#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"

using namespace ms;

TEST(StatsMoreTest, descriptive_statistics) {
    const std::vector<double> data{1, 2, 2, 3, 9};
    EXPECT_NEAR(mean(data), 3.4, 1e-12);
    EXPECT_NEAR(var(data), 10.3, 1e-12);
    EXPECT_NEAR(stddev(data), std::sqrt(10.3), 1e-12);
    EXPECT_NEAR(median(data), 2.0, 1e-12);
    EXPECT_DOUBLE_EQ(min_value(data), 1.0);
    EXPECT_DOUBLE_EQ(max_value(data), 9.0);
    EXPECT_DOUBLE_EQ(mode(data), 2.0);
    EXPECT_NEAR(percentile(data, 50.0), 2.0, 1e-12);
    EXPECT_GT(skewness(data), 0.0);
    EXPECT_LT(kurtosis(data), 0.0);
}

TEST(StatsMoreTest, hypothesis_tests) {
    const std::vector<double> sample{1, 2, 3, 4, 5};
    EXPECT_GT(ttest(sample, 0.0), 0.0);
    EXPECT_GT(ztest(sample, 0.0, 1.0), 0.0);
    const std::vector<double> x{1, 2, 3, 4, 5};
    const std::vector<double> y{2, 4, 6, 8, 10};
    EXPECT_NEAR(correlation(x, y), 1.0, 1e-12);
    const std::vector<double> a{1, 2, 3};
    const std::vector<double> b{1.1, 2.2, 2.9};
    EXPECT_GT(two_sample_ttest(a, b), -10.0);
    const auto fit = linear_regression(x, y);
    EXPECT_NEAR(fit.slope, 2.0, 1e-9);
    EXPECT_NEAR(fit.intercept, 0.0, 1e-9);
    EXPECT_NEAR(fit.r_squared, 1.0, 1e-9);
}

TEST(ProbMoreTest, normal_distribution) {
    EXPECT_NEAR(norm_pdf(0.0, 0.0, 1.0), 0.3989422804014327, 1e-6);
    EXPECT_NEAR(norm_cdf(0.0, 0.0, 1.0), 0.5, 1e-12);
    EXPECT_NEAR(norm_ppf(0.5, 0.0, 1.0), 0.0, 1e-6);
}

TEST(ProbMoreTest, discrete_and_other_distributions) {
    EXPECT_NEAR(exp_pdf(1.0, 1.0), std::exp(-1.0), 1e-12);
    EXPECT_NEAR(exp_cdf(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-12);
    EXPECT_NEAR(binom_pdf(2, 4, 0.5), 0.375, 1e-12);
    EXPECT_NEAR(binom_cdf(2, 4, 0.5), 0.6875, 1e-12);
    EXPECT_NEAR(pois_pdf(2.0, 1.0), std::exp(-1.0) / 2.0, 1e-12);
    EXPECT_NEAR(pois_cdf(2.0, 1.0), 0.9196986029280956, 1e-6);
    EXPECT_NEAR(t_pdf(0.0, 5.0), 0.379606689855647, 1e-6);
    EXPECT_GT(t_cdf(2.0, 5.0), 0.0);
}

TEST(ProbMoreTest, chi2_and_gamma) {
    EXPECT_GT(chi2_pdf(1.0, 2.0), 0.0);
    EXPECT_GT(chi2_cdf(2.0, 3.0), chi2_cdf(1.0, 3.0));
    EXPECT_GT(gamma_pdf(2.0, 3.0, 1.0), 0.0);
    EXPECT_NEAR(uniform_pdf(0.25, 0.0, 1.0), 1.0, 1e-12);
    EXPECT_NEAR(norm_ppf(0.5, 0.0, 1.0), 0.0, 1e-6);
    EXPECT_GT(norm_ppf(0.9, 0.0, 1.0), 0.0);
    EXPECT_GT(gamma_pdf(0.5, 0.5, 2.0), 0.0);
    EXPECT_GT(chi2_pdf(0.5, 4.0), 0.0);
    EXPECT_GT(t_pdf(2.0, 10.0), 0.0);
    EXPECT_TRUE(std::isfinite(t_cdf(-2.0, 10.0)));
    EXPECT_LT(t_cdf(-2.0, 10.0), t_cdf(2.0, 10.0));
}

TEST(StatsMoreTest, empty_input_guards) {
    const std::vector<double> empty;
    EXPECT_DOUBLE_EQ(mean(empty), 0.0);
    EXPECT_DOUBLE_EQ(var(empty), 0.0);
    EXPECT_DOUBLE_EQ(stddev(empty), 0.0);
    EXPECT_DOUBLE_EQ(median(empty), 0.0);
    EXPECT_DOUBLE_EQ(mode(empty), 0.0);
    EXPECT_DOUBLE_EQ(percentile(empty, 50.0), 0.0);
    EXPECT_DOUBLE_EQ(ttest(empty, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(correlation(empty, empty), 0.0);
    EXPECT_DOUBLE_EQ(skewness(empty), 0.0);
    EXPECT_DOUBLE_EQ(kurtosis(empty), 0.0);
    EXPECT_DOUBLE_EQ(two_sample_ttest(empty, empty), 0.0);
    EXPECT_DOUBLE_EQ(ztest(empty, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(linear_regression(empty, empty).slope, 0.0);
}

TEST(StatsMoreTest, even_length_median) {
    const std::vector<double> data{1, 2, 3, 4};
    EXPECT_NEAR(median(data), 2.5, 1e-12);
}

TEST(StatsMoreTest, remaining_branch_guards) {
    const std::vector<double> one{4.0};
    const std::vector<double> two{1.0, 2.0};
    const std::vector<double> x{1.0, 2.0, 3.0};
    const std::vector<double> y_short{1.0, 2.0};
    const std::vector<double> y_const{5.0, 5.0, 5.0};

    EXPECT_DOUBLE_EQ(var(one), 0.0);
    EXPECT_DOUBLE_EQ(correlation(x, y_short), 0.0);
    EXPECT_DOUBLE_EQ(correlation(x, y_const), 0.0);
    EXPECT_DOUBLE_EQ(linear_regression(one, one).slope, 0.0);
    EXPECT_DOUBLE_EQ(linear_regression(x, y_short).slope, 0.0);
    EXPECT_DOUBLE_EQ(two_sample_ttest(one, two), 0.0);
    EXPECT_DOUBLE_EQ(two_sample_ttest(two, one), 0.0);
    EXPECT_GT(ztest(one, 0.0, 1.0), 0.0);
}

TEST(StatsMoreTest, odd_median_and_percentile_bounds) {
    const std::vector<double> data{10.0, 20.0, 30.0, 40.0, 50.0};
    EXPECT_NEAR(median(data), 30.0, 1e-12);
    EXPECT_DOUBLE_EQ(percentile(data, 0.0), 10.0);
    EXPECT_DOUBLE_EQ(percentile(data, 100.0), 50.0);
}

TEST(StatsMoreTest, mode_and_regression_paths) {
    const std::vector<double> tied{1.0, 2.0, 2.0, 3.0, 3.0, 3.0};
    EXPECT_DOUBLE_EQ(mode(tied), 3.0);

    const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0};
    const auto fit = linear_regression(x, y);
    EXPECT_NEAR(fit.slope, 1.0, 1e-12);
    EXPECT_NEAR(fit.intercept, 1.0, 1e-12);
    EXPECT_NEAR(fit.r_squared, 1.0, 1e-12);
}

namespace {

double sample_covariance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return 0.0;
    }
    const double mx = mean(x);
    const double my = mean(y);
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - mx) * (y[i] - my);
    }
    return sum / static_cast<double>(x.size() - 1);
}

std::vector<double> zscore(const std::vector<double>& data) {
    const double m = mean(data);
    const double s = stddev(data);
    std::vector<double> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        out[i] = (data[i] - m) / s;
    }
    return out;
}

std::vector<size_t> histogram_bin_counts(const std::vector<double>& data, size_t bins) {
    if (data.empty() || bins == 0) {
        return {};
    }
    const double lo = min_value(data);
    const double hi = max_value(data);
    const double width = (hi == lo) ? 1.0 : (hi - lo) / static_cast<double>(bins);
    std::vector<size_t> counts(bins, 0);
    for (double v : data) {
        size_t idx = (width == 0.0) ? 0 : static_cast<size_t>((v - lo) / width);
        if (idx >= bins) {
            idx = bins - 1;
        }
        ++counts[idx];
    }
    return counts;
}

} // namespace

TEST(StatsMoreTest, mean_variance_reference) {
    const std::vector<double> data{1, 2, 3, 4, 5};
    EXPECT_NEAR(mean(data), 3.0, 1e-12);
    EXPECT_NEAR(var(data), 2.5, 1e-12);
}

TEST(StatsMoreTest, median_even_n) {
    const std::vector<double> data{1, 2, 3, 4};
    EXPECT_NEAR(median(data), 2.5, 1e-12);
}

TEST(StatsMoreTest, stddev_known) {
    const std::vector<double> data{2, 4, 4, 4, 5, 5, 7, 9};
    EXPECT_NEAR(stddev(data), std::sqrt(32.0 / 7.0), 1e-12);
}

TEST(StatsMoreTest, percentile_90th) {
    const std::vector<double> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_NEAR(percentile(data, 90.0), 9.0, 1e-12);
}

TEST(StatsMoreTest, correlation_perfect_positive) {
    const std::vector<double> x{1, 2, 3};
    const std::vector<double> y{1, 2, 3};
    EXPECT_NEAR(correlation(x, y), 1.0, 1e-12);
}

TEST(StatsMoreTest, correlation_perfect_negative) {
    const std::vector<double> x{1, 2, 3};
    const std::vector<double> y{3, 2, 1};
    EXPECT_NEAR(correlation(x, y), -1.0, 1e-12);
}

TEST(StatsMoreTest, covariance_identical_series) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(sample_covariance(x, x), var(x), 1e-12);
}

TEST(StatsMoreTest, histogram_bin_counts) {
    const std::vector<double> data{1, 1, 2, 3, 3, 3};
    const auto counts = histogram_bin_counts(data, 3);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts[0], 2u);
    EXPECT_EQ(counts[1], 1u);
    EXPECT_EQ(counts[2], 3u);
}

TEST(StatsMoreTest, zscore_standardizes) {
    const std::vector<double> data{1, 2, 3};
    const auto z = zscore(data);
    EXPECT_NEAR(mean(z), 0.0, 1e-12);
    EXPECT_NEAR(stddev(z), 1.0, 1e-12);
}

TEST(ProbMoreTest, normal_pdf_reference) {
    EXPECT_NEAR(norm_pdf(0.0, 0.0, 1.0), 0.3989422804014327, 1e-5);
}

TEST(ProbMoreTest, normal_cdf_reference) {
    EXPECT_NEAR(norm_cdf(0.0, 0.0, 1.0), 0.5, 1e-12);
    EXPECT_NEAR(norm_cdf(1.96, 0.0, 1.0), 0.975, 1e-2);
}

TEST(ProbMoreTest, poisson_pmf_reference) {
    EXPECT_NEAR(pois_pdf(3.0, 3.0), 0.224, 1e-2);
}

TEST(ProbMoreTest, binomial_pmf) {
    EXPECT_NEAR(binom_pdf(5, 10, 0.5), 0.246, 1e-2);
}
