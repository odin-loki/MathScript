#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// one_way_anova
// ---------------------------------------------------------------------------

TEST(StatsExtTest, one_way_anova_three_groups_differ) {
    const std::vector<std::vector<double>> groups = {
        {10.0, 11.0, 12.0},
        {20.0, 21.0, 22.0},
        {30.0, 31.0, 32.0},
    };
    const auto result = one_way_anova(groups);
    EXPECT_GT(result.f_stat, 100.0);
    EXPECT_LT(result.p_value, 0.001);
    EXPECT_EQ(result.df_between, 2);
    EXPECT_EQ(result.df_within, 6);
}

TEST(StatsExtTest, one_way_anova_three_groups_same) {
    const std::vector<std::vector<double>> groups = {
        {5.0, 6.0, 7.0},
        {5.1, 6.1, 7.1},
        {4.9, 5.9, 6.9},
    };
    const auto result = one_way_anova(groups);
    EXPECT_LT(result.f_stat, 5.0);
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_EQ(result.df_between, 2);
    EXPECT_EQ(result.df_within, 6);
}

TEST(StatsExtTest, one_way_anova_two_groups_edge_case) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
    };
    const auto result = one_way_anova(groups);
    EXPECT_GT(result.f_stat, 1.0);
    EXPECT_LT(result.p_value, 0.05);
    EXPECT_EQ(result.df_between, 1);
    EXPECT_EQ(result.df_within, 6);
}

TEST(StatsExtTest, one_way_anova_f_matches_t_squared_for_two_groups) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{2.5, 3.5, 4.5, 5.5};
    const auto anova = one_way_anova({a, b});
    const double t = two_sample_ttest(a, b);
    EXPECT_NEAR(anova.f_stat, t * t, 1e-6);
}

TEST(StatsExtTest, one_way_anova_degenerate_inputs) {
    const std::vector<std::vector<double>> one_group = {{1.0, 2.0, 3.0}};
    const auto too_few = one_way_anova(one_group);
    EXPECT_DOUBLE_EQ(too_few.f_stat, 0.0);
    EXPECT_DOUBLE_EQ(too_few.p_value, 0.0);

    const std::vector<std::vector<double>> with_empty = {{1.0, 2.0}, {}};
    const auto empty_group = one_way_anova(with_empty);
    EXPECT_DOUBLE_EQ(empty_group.f_stat, 0.0);
}

// ---------------------------------------------------------------------------
// mann_whitney_u
// ---------------------------------------------------------------------------

TEST(StatsExtTest, mann_whitney_u_stochastically_greater) {
    const std::vector<double> low{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> high{11.0, 12.0, 13.0, 14.0, 15.0};
    const auto result = mann_whitney_u(low, high);
    EXPECT_NEAR(result.u_stat, 0.0, 1e-12);
    EXPECT_LT(result.p_value, 0.05);
}

TEST(StatsExtTest, mann_whitney_u_nearly_identical) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b{1.1, 2.1, 3.1, 4.1, 5.1};
    const auto result = mann_whitney_u(a, b);
    EXPECT_GT(result.p_value, 0.1);
}

TEST(StatsExtTest, mann_whitney_u_known_small_example) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{5.0, 6.0, 7.0, 8.0};
    const auto result = mann_whitney_u(a, b);
    EXPECT_NEAR(result.u_stat, 0.0, 1e-12);
    EXPECT_LT(result.p_value, 0.05);
}

TEST(StatsExtTest, mann_whitney_u_tie_handling) {
    const std::vector<double> a{1.0, 1.0, 2.0, 3.0};
    const std::vector<double> b{2.0, 3.0, 4.0, 5.0};
    const auto result = mann_whitney_u(a, b);
    EXPECT_GE(result.u_stat, 0.0);
    EXPECT_LE(result.u_stat, 16.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

TEST(StatsExtTest, mann_whitney_u_empty_samples) {
    const std::vector<double> empty;
    const std::vector<double> data{1.0, 2.0};
    const auto result = mann_whitney_u(empty, data);
    EXPECT_DOUBLE_EQ(result.u_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 0.0);
}

// ---------------------------------------------------------------------------
// kruskal_wallis
// ---------------------------------------------------------------------------

TEST(StatsExtTest, kruskal_wallis_three_groups_differ) {
    const std::vector<std::vector<double>> groups = {
        {10.0, 11.0, 12.0},
        {20.0, 21.0, 22.0},
        {30.0, 31.0, 32.0},
    };
    const auto result = kruskal_wallis(groups);
    EXPECT_GT(result.h_stat, 5.0);
    EXPECT_LT(result.p_value, 0.05);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, kruskal_wallis_three_groups_same) {
    const std::vector<std::vector<double>> groups = {
        {5.0, 6.0, 7.0},
        {5.1, 6.1, 7.1},
        {4.9, 5.9, 6.9},
    };
    const auto result = kruskal_wallis(groups);
    EXPECT_LT(result.h_stat, 2.0);
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, kruskal_wallis_four_groups_differ) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0},
        {10.0, 11.0, 12.0, 13.0},
        {20.0, 21.0, 22.0, 23.0},
        {30.0, 31.0, 32.0, 33.0},
    };
    const auto result = kruskal_wallis(groups);
    EXPECT_GT(result.h_stat, 6.0);
    EXPECT_LT(result.p_value, 0.01);
    EXPECT_EQ(result.df, 3);
}

TEST(StatsExtTest, kruskal_wallis_two_groups_agrees_with_mann_whitney) {
    const std::vector<double> low{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> high{11.0, 12.0, 13.0, 14.0, 15.0};
    const auto kw = kruskal_wallis({low, high});
    const auto mw = mann_whitney_u(low, high);
    EXPECT_LT(kw.p_value, 0.05);
    EXPECT_LT(mw.p_value, 0.05);
    EXPECT_EQ(kw.df, 1);
}

TEST(StatsExtTest, kruskal_wallis_two_groups_non_significant_agreement) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b{1.1, 2.1, 3.1, 4.1, 5.1};
    const auto kw = kruskal_wallis({a, b});
    const auto mw = mann_whitney_u(a, b);
    EXPECT_GT(kw.p_value, 0.1);
    EXPECT_GT(mw.p_value, 0.1);
}

TEST(StatsExtTest, kruskal_wallis_tie_handling) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 1.0, 2.0, 3.0},
        {2.0, 3.0, 4.0, 5.0},
        {3.0, 4.0, 5.0, 6.0},
    };
    const auto result = kruskal_wallis(groups);
    EXPECT_TRUE(std::isfinite(result.h_stat));
    EXPECT_TRUE(std::isfinite(result.p_value));
    EXPECT_GE(result.h_stat, 0.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, kruskal_wallis_all_identical_values) {
    const std::vector<std::vector<double>> groups = {
        {5.0, 5.0, 5.0},
        {5.0, 5.0, 5.0},
        {5.0, 5.0, 5.0},
    };
    const auto result = kruskal_wallis(groups);
    EXPECT_NEAR(result.h_stat, 0.0, 1e-12);
    EXPECT_NEAR(result.p_value, 1.0, 1e-12);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, kruskal_wallis_degenerate_fewer_than_two_groups) {
    const std::vector<std::vector<double>> one_group = {{1.0, 2.0, 3.0}};
    const auto result = kruskal_wallis(one_group);
    EXPECT_DOUBLE_EQ(result.h_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.df, 0);
}

TEST(StatsExtTest, kruskal_wallis_empty_groups_mixed) {
    const std::vector<std::vector<double>> with_empty = {
        {1.0, 2.0, 3.0},
        {},
        {4.0, 5.0, 6.0},
    };
    const auto result = kruskal_wallis(with_empty);
    EXPECT_GT(result.h_stat, 0.0);
    EXPECT_LT(result.p_value, 0.05);
    EXPECT_EQ(result.df, 1);
}

TEST(StatsExtTest, kruskal_wallis_single_element_per_group) {
    const std::vector<std::vector<double>> groups = {
        {1.0},
        {2.0},
        {3.0},
    };
    const auto result = kruskal_wallis(groups);
    EXPECT_DOUBLE_EQ(result.h_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.df, 0);
}

TEST(StatsExtTest, kruskal_wallis_only_one_nonempty_group) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0},
        {},
    };
    const auto result = kruskal_wallis(groups);
    EXPECT_DOUBLE_EQ(result.h_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.df, 0);
}

// ---------------------------------------------------------------------------
// ks_test_2sample
// ---------------------------------------------------------------------------

TEST(StatsExtTest, ks_test_2sample_different_distributions) {
    const std::vector<double> low{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    const std::vector<double> high{10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};
    const auto result = ks_test_2sample(low, high);
    EXPECT_GT(result.d_stat, 0.9);
    EXPECT_LT(result.p_value, 0.05);
}

TEST(StatsExtTest, ks_test_2sample_same_distribution) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> b{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5};
    const auto result = ks_test_2sample(a, b);
    EXPECT_LT(result.d_stat, 0.3);
    EXPECT_GT(result.p_value, 0.1);
}

TEST(StatsExtTest, ks_test_2sample_d_stat_in_unit_interval) {
    const std::vector<double> a{3.0, 1.0, 4.0, 1.0, 5.0};
    const std::vector<double> b{9.0, 2.0, 6.0, 5.0, 3.0};
    const auto result = ks_test_2sample(a, b);
    EXPECT_GE(result.d_stat, 0.0);
    EXPECT_LE(result.d_stat, 1.0);
}

TEST(StatsExtTest, ks_test_2sample_empty_samples) {
    const std::vector<double> empty;
    const std::vector<double> data{1.0, 2.0, 3.0};
    const auto result = ks_test_2sample(empty, data);
    EXPECT_DOUBLE_EQ(result.d_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 0.0);
}

TEST(StatsExtTest, ks_test_2sample_identical_samples) {
    const std::vector<double> a{2.0, 4.0, 6.0, 8.0, 10.0};
    const std::vector<double> b{2.0, 4.0, 6.0, 8.0, 10.0};
    const auto result = ks_test_2sample(a, b);
    EXPECT_NEAR(result.d_stat, 0.0, 1e-12);
    EXPECT_NEAR(result.p_value, 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// bootstrap_ci (percentile method, arbitrary statistic)
// ---------------------------------------------------------------------------

namespace {

const auto mean_stat = [](std::span<const double> s) { return mean(s); };
const auto median_stat = [](std::span<const double> s) { return median(s); };
const auto stddev_stat = [](std::span<const double> s) { return stddev(s); };

} // namespace

TEST(StatsExtTest, bootstrap_ci_mean_point_estimate_matches_sample_mean) {
    const std::vector<double> data{8.0, 9.0, 10.0, 11.0, 12.0};
    const auto result = bootstrap_ci(data, mean_stat, 1000, 0.95, 42);
    EXPECT_NEAR(result.point_estimate, mean(data), 1e-12);
    EXPECT_NEAR(result.point_estimate, 10.0, 1e-12);
}

TEST(StatsExtTest, bootstrap_ci_mean_sane_bounds) {
    const std::vector<double> data{8.0, 9.0, 10.0, 11.0, 12.0};
    const auto result = bootstrap_ci(data, mean_stat, 1000, 0.95, 42);
    EXPECT_LT(result.lower, result.point_estimate);
    EXPECT_LT(result.point_estimate, result.upper);
    EXPECT_LT(result.lower, result.upper);
    EXPECT_GT(result.std_error, 0.0);
}

TEST(StatsExtTest, bootstrap_ci_mean_narrows_with_larger_sample) {
    std::vector<double> small;
    std::vector<double> large;
    for (int i = 0; i < 20; ++i) {
        small.push_back(10.0 + static_cast<double>((i % 5) - 2));
    }
    for (int i = 0; i < 200; ++i) {
        large.push_back(10.0 + static_cast<double>((i % 5) - 2));
    }
    const auto ci_small = bootstrap_ci(small, mean_stat, 1000, 0.95, 42);
    const auto ci_large = bootstrap_ci(large, mean_stat, 1000, 0.95, 42);
    const double width_small = ci_small.upper - ci_small.lower;
    const double width_large = ci_large.upper - ci_large.lower;
    EXPECT_LT(width_large, width_small);
}

TEST(StatsExtTest, bootstrap_ci_median_sane_bounds) {
    const std::vector<double> data{1.0, 2.0, 100.0, 3.0, 4.0};
    const auto result = bootstrap_ci(data, median_stat, 1000, 0.95, 7);
    EXPECT_NEAR(result.point_estimate, median(data), 1e-12);
    EXPECT_LT(result.lower, result.upper);
    EXPECT_LE(result.lower, result.point_estimate);
    EXPECT_GE(result.upper, result.point_estimate);
}

TEST(StatsExtTest, bootstrap_ci_stddev_sane_bounds) {
    const std::vector<double> data{2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    const auto result = bootstrap_ci(data, stddev_stat, 1000, 0.95, 99);
    EXPECT_NEAR(result.point_estimate, stddev(data), 1e-12);
    EXPECT_LT(result.lower, result.upper);
    EXPECT_LT(result.lower, result.point_estimate);
    EXPECT_LT(result.point_estimate, result.upper);
}

TEST(StatsExtTest, bootstrap_ci_deterministic_same_seed) {
    const std::vector<double> data{3.0, 5.0, 7.0, 9.0, 11.0};
    const auto a = bootstrap_ci(data, mean_stat, 500, 0.95, 12345);
    const auto b = bootstrap_ci(data, mean_stat, 500, 0.95, 12345);
    EXPECT_DOUBLE_EQ(a.point_estimate, b.point_estimate);
    EXPECT_DOUBLE_EQ(a.lower, b.lower);
    EXPECT_DOUBLE_EQ(a.upper, b.upper);
    EXPECT_DOUBLE_EQ(a.std_error, b.std_error);
}

TEST(StatsExtTest, bootstrap_ci_confidence_99_wider_than_90) {
    const std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    const auto ci90 = bootstrap_ci(data, mean_stat, 1000, 0.90, 42);
    const auto ci99 = bootstrap_ci(data, mean_stat, 1000, 0.99, 42);
    const double width90 = ci90.upper - ci90.lower;
    const double width99 = ci99.upper - ci99.lower;
    EXPECT_LT(width90, width99);
}

TEST(StatsExtTest, bootstrap_ci_confidence_95_between_90_and_99) {
    const std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    const auto ci90 = bootstrap_ci(data, mean_stat, 1000, 0.90, 42);
    const auto ci95 = bootstrap_ci(data, mean_stat, 1000, 0.95, 42);
    const auto ci99 = bootstrap_ci(data, mean_stat, 1000, 0.99, 42);
    const double width90 = ci90.upper - ci90.lower;
    const double width95 = ci95.upper - ci95.lower;
    const double width99 = ci99.upper - ci99.lower;
    EXPECT_LT(width90, width95);
    EXPECT_LT(width95, width99);
}

TEST(StatsExtTest, bootstrap_ci_small_dataset_three_points) {
    const std::vector<double> data{4.0, 5.0, 6.0};
    const auto result = bootstrap_ci(data, mean_stat, 500, 0.95, 1);
    EXPECT_NEAR(result.point_estimate, 5.0, 1e-12);
    EXPECT_LT(result.lower, result.upper);
    EXPECT_TRUE(std::isfinite(result.std_error));
}

TEST(StatsExtTest, bootstrap_ci_small_dataset_five_points) {
    const std::vector<double> data{7.0, 8.0, 9.0, 10.0, 11.0};
    const auto mean_result = bootstrap_ci(data, mean_stat, 500, 0.95, 3);
    const auto median_result = bootstrap_ci(data, median_stat, 500, 0.95, 3);
    EXPECT_LT(mean_result.lower, mean_result.upper);
    EXPECT_LT(median_result.lower, median_result.upper);
    EXPECT_TRUE(std::isfinite(mean_result.std_error));
    EXPECT_TRUE(std::isfinite(median_result.std_error));
}

TEST(StatsExtTest, bootstrap_ci_std_error_positive_for_spread_data) {
    const std::vector<double> data{0.0, 5.0, 10.0, 15.0, 20.0};
    const auto result = bootstrap_ci(data, mean_stat, 1000, 0.95, 42);
    EXPECT_GT(result.std_error, 0.0);
}

TEST(StatsExtTest, bootstrap_ci_empty_data_returns_zeros) {
    const std::vector<double> empty;
    const auto result = bootstrap_ci(empty, mean_stat, 1000, 0.95, 42);
    EXPECT_DOUBLE_EQ(result.point_estimate, 0.0);
    EXPECT_DOUBLE_EQ(result.lower, 0.0);
    EXPECT_DOUBLE_EQ(result.upper, 0.0);
    EXPECT_DOUBLE_EQ(result.std_error, 0.0);
}

// ---------------------------------------------------------------------------
// jarque_bera
// ---------------------------------------------------------------------------

namespace {

std::vector<double> make_norm_ppf_sample(int n) {
    std::vector<double> sample(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double p = (static_cast<double>(i) + 0.5) / static_cast<double>(n);
        sample[static_cast<size_t>(i)] = norm_ppf(p, 0.0, 1.0);
    }
    return sample;
}

} // namespace

TEST(StatsExtTest, jarque_bera_normal_like_sample) {
    const auto normal = make_norm_ppf_sample(100);
    const auto result = jarque_bera(normal);
    EXPECT_LT(result.jb_stat, 6.0);
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, jarque_bera_skewed_exponential_shape) {
    std::vector<double> skewed;
    for (int i = 1; i <= 60; ++i) {
        skewed.push_back(std::log(static_cast<double>(i)));
    }
    const auto result = jarque_bera(skewed);
    EXPECT_GT(result.jb_stat, 10.0);
    EXPECT_LT(result.p_value, 0.001);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, jarque_bera_bimodal_sample) {
    std::vector<double> bimodal;
    for (int i = 0; i < 30; ++i) {
        bimodal.push_back(-8.0);
    }
    for (int i = 0; i < 30; ++i) {
        bimodal.push_back(8.0);
    }
    const auto result = jarque_bera(bimodal);
    EXPECT_GT(result.jb_stat, 5.0);
    EXPECT_LT(result.p_value, 0.01);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, jarque_bera_heavy_right_skew) {
    std::vector<double> heavy_skew;
    for (int i = 1; i <= 80; ++i) {
        heavy_skew.push_back(std::exp(static_cast<double>(i) / 25.0));
    }
    const auto result = jarque_bera(heavy_skew);
    EXPECT_GT(result.jb_stat, 10.0);
    EXPECT_LT(result.p_value, 0.001);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, jarque_bera_too_small_sample) {
    const std::vector<double> tiny{1.0, 2.0, 3.0};
    const auto result = jarque_bera(tiny);
    EXPECT_DOUBLE_EQ(result.jb_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
}

TEST(StatsExtTest, jarque_bera_constant_sample) {
    const std::vector<double> flat{5.0, 5.0, 5.0, 5.0, 5.0};
    const auto result = jarque_bera(flat);
    EXPECT_DOUBLE_EQ(result.jb_stat, 0.0);
}

// ---------------------------------------------------------------------------
// ljung_box
// ---------------------------------------------------------------------------

TEST(StatsExtTest, ljung_box_uncorrelated_sequence) {
    std::vector<double> uncorrelated;
    uncorrelated.push_back(0.0);
    uint32_t state = 12345u;
    for (int i = 1; i < 500; ++i) {
        state = state * 1664525u + 1013904223u;
        uncorrelated.push_back(
            static_cast<double>(state) / 4294967296.0 - 0.5);
    }
    const auto result = ljung_box(uncorrelated, 8);
    const auto correlated = ljung_box(
        std::vector<double>{1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0},
        8);
    EXPECT_LT(result.q_stat, correlated.q_stat);
    EXPECT_GT(result.p_value, correlated.p_value);
    EXPECT_EQ(result.df, 8);
}

TEST(StatsExtTest, ljung_box_alternating_strong_autocorrelation) {
    std::vector<double> alternating;
    for (int i = 0; i < 100; ++i) {
        alternating.push_back((i % 2 == 0) ? 1.0 : -1.0);
    }
    const auto result = ljung_box(alternating, 5);
    EXPECT_NEAR(result.q_stat, 494.7, 5.0);
    EXPECT_EQ(result.df, 5);
}

TEST(StatsExtTest, ljung_box_ar1_autocorrelation) {
    std::vector<double> ar1(static_cast<size_t>(100));
    ar1[0] = 1.0;
    for (int i = 1; i < 100; ++i) {
        ar1[static_cast<size_t>(i)] =
            0.95 * ar1[static_cast<size_t>(i - 1)] +
            0.05 * static_cast<double>((i % 7) - 3);
    }
    const auto uncorrelated = ljung_box(
        std::vector<double>{0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.25, 0.05, -0.15,
                             0.12, -0.18, 0.22, -0.12, 0.18, -0.08, 0.28, -0.28, 0.08, -0.08},
        10);
    const auto result = ljung_box(ar1, 10);
    EXPECT_GT(result.q_stat, uncorrelated.q_stat);
    EXPECT_GT(result.q_stat, 5.0);
    EXPECT_EQ(result.df, 10);
}

TEST(StatsExtTest, ljung_box_cumulative_sum_autocorrelation) {
    std::vector<double> random_walk;
    double sum = 0.0;
    for (int i = 0; i < 80; ++i) {
        sum += ((i % 3) - 1) * 0.5;
        random_walk.push_back(sum);
    }
    const auto result = ljung_box(random_walk, 8);
    EXPECT_GT(result.q_stat, 30.0);
    EXPECT_LT(result.p_value, 0.01);
    EXPECT_EQ(result.df, 8);
}

TEST(StatsExtTest, ljung_box_degenerate_inputs) {
    const std::vector<double> one{1.0};
    const auto single = ljung_box(one, 5);
    EXPECT_DOUBLE_EQ(single.q_stat, 0.0);
    EXPECT_DOUBLE_EQ(single.p_value, 1.0);

    const std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto bad_lag = ljung_box(data, 0);
    EXPECT_DOUBLE_EQ(bad_lag.q_stat, 0.0);
    EXPECT_DOUBLE_EQ(bad_lag.p_value, 1.0);
}

// ---------------------------------------------------------------------------
// levene_test
// ---------------------------------------------------------------------------

TEST(StatsExtTest, levene_test_equal_variances) {
    const std::vector<std::vector<double>> groups = {
        {10.0, 11.0, 9.0, 10.5, 9.5},
        {20.0, 21.0, 19.0, 20.5, 19.5},
        {30.0, 31.0, 29.0, 30.5, 29.5},
    };
    const auto result = levene_test(groups);
    EXPECT_LT(result.f_stat, 3.0);
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_EQ(result.df_between, 2);
    EXPECT_EQ(result.df_within, 12);
}

TEST(StatsExtTest, levene_test_unequal_variances) {
    const std::vector<std::vector<double>> groups = {
        {10.0, 10.1, 9.9, 10.2, 9.8},
        {20.0, 20.1, 19.9, 20.2, 19.8},
        {0.0, 40.0, 10.0, 30.0, -10.0},
    };
    const auto result = levene_test(groups);
    EXPECT_GT(result.f_stat, 5.0);
    EXPECT_LT(result.p_value, 0.05);
    EXPECT_EQ(result.df_between, 2);
    EXPECT_EQ(result.df_within, 12);
}

TEST(StatsExtTest, levene_test_delegates_to_one_way_anova) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {6.0, 7.0, 8.0, 9.0, 10.0},
        {11.0, 12.0, 13.0, 14.0, 15.0},
    };
    std::vector<std::vector<double>> transformed;
    transformed.reserve(groups.size());
    for (const auto& group : groups) {
        const double med = median(group);
        std::vector<double> devs;
        devs.reserve(group.size());
        for (double v : group) {
            devs.push_back(std::abs(v - med));
        }
        transformed.push_back(std::move(devs));
    }
    const auto levene = levene_test(groups);
    const auto manual = one_way_anova(transformed);
    EXPECT_NEAR(levene.f_stat, manual.f_stat, 1e-12);
    EXPECT_NEAR(levene.p_value, manual.p_value, 1e-12);
    EXPECT_EQ(levene.df_between, manual.df_between);
    EXPECT_EQ(levene.df_within, manual.df_within);
}

TEST(StatsExtTest, levene_test_degenerate_inputs) {
    const std::vector<std::vector<double>> one_group = {{1.0, 2.0, 3.0}};
    const auto result = levene_test(one_group);
    EXPECT_DOUBLE_EQ(result.f_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 0.0);
}

// ---------------------------------------------------------------------------
// bartlett_test
// ---------------------------------------------------------------------------

TEST(StatsExtTest, bartlett_test_equal_variances) {
    const std::vector<std::vector<double>> groups = {
        {10.0, 11.0, 9.0, 10.5, 9.5},
        {20.0, 21.0, 19.0, 20.5, 19.5},
        {30.0, 31.0, 29.0, 30.5, 29.5},
    };
    const auto result = bartlett_test(groups);
    EXPECT_LT(result.chi2_stat, 6.0);
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, bartlett_test_unequal_variances) {
    const std::vector<std::vector<double>> groups = {
        {10.0, 10.1, 9.9, 10.2, 9.8},
        {20.0, 20.1, 19.9, 20.2, 19.8},
        {0.0, 40.0, 10.0, 30.0, -10.0},
    };
    const auto result = bartlett_test(groups);
    EXPECT_GT(result.chi2_stat, 10.0);
    EXPECT_LT(result.p_value, 0.01);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, bartlett_test_two_groups) {
    const std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {1.0, 1.5, 2.5, 3.5, 4.5},
    };
    const auto result = bartlett_test(groups);
    EXPECT_TRUE(std::isfinite(result.chi2_stat));
    EXPECT_TRUE(std::isfinite(result.p_value));
    EXPECT_GE(result.chi2_stat, 0.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
    EXPECT_EQ(result.df, 1);
}

TEST(StatsExtTest, bartlett_test_degenerate_inputs) {
    const std::vector<std::vector<double>> one_group = {{1.0, 2.0, 3.0}};
    const auto too_few = bartlett_test(one_group);
    EXPECT_DOUBLE_EQ(too_few.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(too_few.p_value, 1.0);

    const std::vector<std::vector<double>> zero_var = {{5.0, 5.0, 5.0}, {6.0, 7.0, 8.0}};
    const auto flat = bartlett_test(zero_var);
    EXPECT_DOUBLE_EQ(flat.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(flat.p_value, 1.0);
}
