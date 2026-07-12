#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
// friedman
// ---------------------------------------------------------------------------

// Hand-computed reference case (no ties). 4 blocks x 3 treatments.
// Row ranks (computed by hand from the raw values):
//   {10,20,30}  -> ranks (1,2,3)
//   {15,25,20}  -> ranks (1,3,2)
//   {5,8,6}     -> ranks (1,3,2)
//   {100,90,95} -> ranks (3,1,2)
// Rank sums: R1 = 1+1+1+3 = 6, R2 = 2+3+3+1 = 9, R3 = 3+2+2+2 = 9
// (check: 6+9+9 = 24 = n*k*(k+1)/2 = 4*3*4/2 = 24)
// chi2 = [12/(n*k*(k+1))] * sum(Rj^2) - 3*n*(k+1)
//      = [12/48] * (36+81+81) - 48 = 0.25*198 - 48 = 49.5 - 48 = 1.5
// No ties anywhere, so the tie-correction factor C = 1 exactly.
// df = k-1 = 2, and for df=2 the upper-tail chi-square probability has the closed form
// P(X > x) = exp(-x/2), so p_value = exp(-0.75).
TEST(StatsExtTest, friedman_reference_case_hand_computed) {
    const std::vector<std::vector<double>> data = {
        {10.0, 20.0, 30.0},
        {15.0, 25.0, 20.0},
        {5.0, 8.0, 6.0},
        {100.0, 90.0, 95.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 2);
    EXPECT_NEAR(result.chi2_stat, 1.5, 1e-9);
    EXPECT_NEAR(result.p_value, std::exp(-0.75), 1e-9);
}

// No-tie dataset (3 blocks x 4 treatments) where the tie-correction factor C = 1 exactly, so
// the function's output must match the naive uncorrected formula exactly.
//   {1,2,3,4} -> ranks (1,2,3,4)
//   {4,3,2,1} -> ranks (4,3,2,1)
//   {2,1,4,3} -> ranks (2,1,4,3)
// Rank sums: R1=1+4+2=7, R2=2+3+1=6, R3=3+2+4=9, R4=4+1+3=8 (sum=30=3*4*5/2)
// chi2 = [12/(3*4*5)] * (49+36+81+64) - 3*3*5 = 0.2*230 - 45 = 46 - 45 = 1.0
TEST(StatsExtTest, friedman_no_ties_matches_uncorrected_formula) {
    const std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0, 4.0},
        {4.0, 3.0, 2.0, 1.0},
        {2.0, 1.0, 4.0, 3.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 3);
    EXPECT_NEAR(result.chi2_stat, 1.0, 1e-9);
    EXPECT_NEAR(result.p_value, 1.0 - chi2_cdf(1.0, 3.0), 1e-12);
}

// Same shape as the reference case above, but block 2 has a tie ({15,20,20} instead of
// {15,25,20}), so the tie-correction factor C < 1 must kick in.
//   {10,20,30}  -> ranks (1,2,3)          (no tie)
//   {15,20,20}  -> ranks (1,2.5,2.5)      (tie group of size 2)
//   {5,8,6}     -> ranks (1,3,2)          (no tie)
//   {100,90,95} -> ranks (3,1,2)          (no tie)
// Rank sums: R1=1+1+1+3=6, R2=2+2.5+3+1=8.5, R3=3+2.5+2+2=9.5 (sum=24, as before)
// Uncorrected chi2 = 0.25*(36+72.25+90.25) - 48 = 0.25*198.5 - 48 = 49.625 - 48 = 1.625
// Tie correction: one tie group of size t=2 in one block, so sum(t^3-t) = 8-2 = 6.
// C = 1 - 6/(n*k*(k^3-k)) = 1 - 6/(4*3*24) = 1 - 6/288 = 1 - 1/48 = 47/48
// Corrected chi2 = 1.625 / (47/48) = 1.625 * 48/47 = 78/47 (~1.65957...)
TEST(StatsExtTest, friedman_tie_correction_inflates_statistic) {
    const std::vector<std::vector<double>> data = {
        {10.0, 20.0, 30.0},
        {15.0, 20.0, 20.0},
        {5.0, 8.0, 6.0},
        {100.0, 90.0, 95.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 2);
    EXPECT_NEAR(result.chi2_stat, 78.0 / 47.0, 1e-9);
    EXPECT_GT(result.chi2_stat, 1.625);  // corrected value must exceed the naive/uncorrected one
    EXPECT_NEAR(result.p_value, std::exp(-(78.0 / 47.0) / 2.0), 1e-9);
}

// Perfectly balanced (cyclic) ranks across 3 blocks x 3 treatments: every treatment gets each
// rank (1, 2, 3) exactly once. This is the textbook "no treatment effect" case, so the rank
// sums are all equal and chi2 must be exactly zero (p_value == 1.0).
TEST(StatsExtTest, friedman_null_effect_cyclic_balance) {
    const std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {2.0, 3.0, 1.0},
        {3.0, 1.0, 2.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 2);
    EXPECT_NEAR(result.chi2_stat, 0.0, 1e-12);
    EXPECT_NEAR(result.p_value, 1.0, 1e-12);
}

// Strong, consistent effect: 5 blocks x 3 treatments, every block ranks the treatments in the
// exact same order (treatment 1 lowest, treatment 3 highest). Rank sums: R1=5, R2=10, R3=15.
// chi2 = [12/(5*3*4)] * (25+100+225) - 3*5*4 = 0.2*350 - 60 = 70 - 60 = 10
// df=2, so p_value = exp(-10/2) = exp(-5), which is very small.
TEST(StatsExtTest, friedman_strong_effect_consistent_ranking) {
    const std::vector<std::vector<double>> data = {
        {10.0, 20.0, 30.0},
        {1.0, 5.0, 9.0},
        {100.0, 150.0, 200.0},
        {-5.0, 0.0, 5.0},
        {2.0, 2.5, 3.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 2);
    EXPECT_NEAR(result.chi2_stat, 10.0, 1e-9);
    EXPECT_NEAR(result.p_value, std::exp(-5.0), 1e-9);
    EXPECT_LT(result.p_value, 0.01);
}

TEST(StatsExtTest, friedman_degrees_of_freedom_various_k) {
    for (int k = 2; k <= 6; ++k) {
        std::vector<std::vector<double>> data;
        for (int block = 0; block < 3; ++block) {
            std::vector<double> row(static_cast<size_t>(k));
            for (int j = 0; j < k; ++j) {
                // Rotate the value order per block so there's some variation, without ties.
                row[static_cast<size_t>(j)] = static_cast<double>((j + block) % k);
            }
            data.push_back(row);
        }
        const auto result = friedman(data);
        EXPECT_EQ(result.df, k - 1) << "k=" << k;
    }
}

TEST(StatsExtTest, friedman_error_empty_input) {
    const std::vector<std::vector<double>> empty;
    const auto result = friedman(empty);
    EXPECT_DOUBLE_EQ(result.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.df, 0);
}

TEST(StatsExtTest, friedman_error_too_few_blocks) {
    const std::vector<std::vector<double>> one_block = {{1.0, 2.0, 3.0}};
    const auto result = friedman(one_block);
    EXPECT_DOUBLE_EQ(result.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.df, 0);
}

TEST(StatsExtTest, friedman_error_too_few_treatments) {
    const std::vector<std::vector<double>> one_treatment = {{1.0}, {2.0}, {3.0}};
    const auto result = friedman(one_treatment);
    EXPECT_DOUBLE_EQ(result.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.df, 0);

    const std::vector<std::vector<double>> zero_treatments = {{}, {}};
    const auto result2 = friedman(zero_treatments);
    EXPECT_DOUBLE_EQ(result2.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(result2.p_value, 1.0);
    EXPECT_EQ(result2.df, 0);
}

TEST(StatsExtTest, friedman_error_jagged_rows) {
    const std::vector<std::vector<double>> jagged = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0},
        {6.0, 7.0, 8.0},
    };
    const auto result = friedman(jagged);
    EXPECT_DOUBLE_EQ(result.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.df, 0);
}

// k=2 reduction: treatment B beats treatment A in every single block (a maximally consistent,
// extreme result -- the two-treatment analog of a sign-test result where every sign agrees).
// R_A = 6*1 = 6, R_B = 6*2 = 12. chi2 = [12/(6*2*3)]*(36+144) - 3*6*3 = (1/3)*180 - 54 = 6.
TEST(StatsExtTest, friedman_two_treatment_perfect_agreement) {
    const std::vector<std::vector<double>> data = {
        {1.0, 2.0}, {3.0, 9.0}, {-1.0, 0.0}, {5.0, 6.0}, {10.0, 20.0}, {0.0, 0.5},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 1);
    EXPECT_NEAR(result.chi2_stat, 6.0, 1e-9);
    EXPECT_NEAR(result.p_value, 1.0 - chi2_cdf(6.0, 1.0), 1e-9);
    EXPECT_LT(result.p_value, 0.02);
}

// k=2, mixed directions: A beats B in half the blocks and B beats A in the other half, so the
// rank sums come out exactly equal -> chi2 == 0 (fail to reject), unlike the perfect-agreement
// case above.
TEST(StatsExtTest, friedman_two_treatment_mixed_non_significant) {
    const std::vector<std::vector<double>> data = {
        {1.0, 2.0}, {3.0, 9.0}, {-1.0, 0.0}, {6.0, 5.0}, {20.0, 10.0}, {0.5, 0.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 1);
    EXPECT_NEAR(result.chi2_stat, 0.0, 1e-9);
    EXPECT_NEAR(result.p_value, 1.0, 1e-9);
}

// Every value within each row is identical (a full tie across all k treatments in every
// block), so every treatment gets the same average rank in every block and chi2 must be zero.
TEST(StatsExtTest, friedman_all_identical_values_within_rows) {
    const std::vector<std::vector<double>> data = {
        {5.0, 5.0, 5.0},
        {10.0, 10.0, 10.0},
        {1.0, 1.0, 1.0},
    };
    const auto result = friedman(data);
    EXPECT_NEAR(result.chi2_stat, 0.0, 1e-12);
    EXPECT_NEAR(result.p_value, 1.0, 1e-12);
    EXPECT_EQ(result.df, 2);
}

TEST(StatsExtTest, friedman_tie_handling_finite_and_bounded) {
    const std::vector<std::vector<double>> data = {
        {1.0, 1.0, 2.0, 3.0},
        {2.0, 3.0, 3.0, 5.0},
        {3.0, 4.0, 5.0, 5.0},
        {1.0, 2.0, 2.0, 4.0},
    };
    const auto result = friedman(data);
    EXPECT_TRUE(std::isfinite(result.chi2_stat));
    EXPECT_TRUE(std::isfinite(result.p_value));
    EXPECT_GE(result.chi2_stat, 0.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
    EXPECT_EQ(result.df, 3);
}

TEST(StatsExtTest, friedman_larger_dataset_sanity_bounds) {
    const std::vector<std::vector<double>> data = {
        {3.0, 1.0, 4.0, 1.0, 5.0},
        {9.0, 2.0, 6.0, 5.0, 3.0},
        {5.0, 8.0, 9.0, 7.0, 9.0},
        {3.0, 2.0, 3.0, 8.0, 4.0},
        {6.0, 2.0, 6.0, 4.0, 3.0},
        {3.0, 8.0, 3.0, 2.0, 7.0},
        {9.0, 5.0, 0.0, 2.0, 8.0},
        {8.0, 4.0, 1.0, 9.0, 7.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 4);
    EXPECT_GE(result.chi2_stat, 0.0);
    EXPECT_TRUE(std::isfinite(result.chi2_stat));
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

TEST(StatsExtTest, friedman_minimal_valid_input) {
    const std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 1.0},
    };
    const auto result = friedman(data);
    EXPECT_EQ(result.df, 1);
    EXPECT_GE(result.chi2_stat, 0.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
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

// ---------------------------------------------------------------------------
// fligner_test
// ---------------------------------------------------------------------------

TEST(StatsExtTest, fligner_test_p_value_in_valid_range) {
    // Three-group case with comparable (but not perfectly symmetric) spreads.
    const std::vector<std::vector<double>> three_groups = {
        {10.002, 10.597, 9.452, 8.219, 9.091, 8.017, 10.12, 12.68, 9.016, 8.759,
         10.98, 10.714, 10.211, 8.139, 9.941, 11.391, 7.312, 9.085, 6.198, 7.421},
        {16.317, 19.53, 17.465, 20.543, 20.314, 19.626, 14.966, 18.923, 19.903, 20.227,
         16.94, 19.044, 18.043, 18.382, 22.122, 18.385, 19.935, 21.769, 18.833, 19.777},
        {15.221, 15.128, 12.55, 15.152, 17.718, 11.906, 16.719, 15.239, 13.717, 19.001,
         16.525, 12.601, 15.149, 16.153, 14.622, 16.366, 14.867, 16.334, 17.877, 13.649},
    };
    const auto r3 = fligner_test(three_groups);
    EXPECT_GT(r3.p_value, 0.0);
    EXPECT_LT(r3.p_value, 1.0);
    EXPECT_EQ(r3.df, 2);

    // Three-group case with a genuine spread difference.
    const std::vector<std::vector<double>> unequal_groups = {
        {10.0, 10.1, 9.9, 10.2, 9.8},
        {20.0, 20.1, 19.9, 20.2, 19.8},
        {0.0, 40.0, 10.0, 30.0, -10.0},
    };
    const auto r_uneq = fligner_test(unequal_groups);
    EXPECT_GT(r_uneq.p_value, 0.0);
    EXPECT_LT(r_uneq.p_value, 1.0);
    EXPECT_EQ(r_uneq.df, 2);

    // Four-group case (mixed means/spreads).
    const std::vector<std::vector<double>> four_groups = {
        {5.051, 7.04, 6.837, 4.235, 4.553, 4.209, 5.855, 4.916, 6.12, 2.229,
         7.35, 4.855, 6.021, 4.795, 4.431},
        {9.389, 10.474, 7.392, 7.542, 10.057, 5.389, 3.457, 9.185, 5.988, 2.239,
         5.558, 6.597, 4.42, 3.523, 8.11},
        {3.794, 1.534, 0.513, 2.77, 3.434, 1.4, 3.089, 4.086, 1.586, 0.373,
         2.695, 2.495, 4.198, -0.569, 0.677},
        {-0.838, -1.734, 0.126, 0.528, -0.739, 1.386, 0.822, 0.627, 0.402, 0.956,
         -1.332, 0.614, 0.603, -1.768, 0.347},
    };
    const auto r4 = fligner_test(four_groups);
    EXPECT_GT(r4.p_value, 0.0);
    EXPECT_LT(r4.p_value, 1.0);
    EXPECT_EQ(r4.df, 3);
    EXPECT_NEAR(r4.chi2_stat, 13.2416415524, 1e-6);
    EXPECT_NEAR(r4.p_value, 0.0041421417, 1e-6);
}

TEST(StatsExtTest, fligner_test_equal_variances_large_p_value) {
    // Same underlying spread (sigma = 2) in all three groups, different means.
    // scipy.stats.fligner on this fixture gives statistic ~= 0.1463, p ~= 0.9295.
    const std::vector<std::vector<double>> groups = {
        {10.002, 10.597, 9.452, 8.219, 9.091, 8.017, 10.12, 12.68, 9.016, 8.759,
         10.98, 10.714, 10.211, 8.139, 9.941, 11.391, 7.312, 9.085, 6.198, 7.421},
        {16.317, 19.53, 17.465, 20.543, 20.314, 19.626, 14.966, 18.923, 19.903, 20.227,
         16.94, 19.044, 18.043, 18.382, 22.122, 18.385, 19.935, 21.769, 18.833, 19.777},
        {15.221, 15.128, 12.55, 15.152, 17.718, 11.906, 16.719, 15.239, 13.717, 19.001,
         16.525, 12.601, 15.149, 16.153, 14.622, 16.366, 14.867, 16.334, 17.877, 13.649},
    };
    const auto result = fligner_test(groups);
    EXPECT_EQ(result.df, 2);
    EXPECT_NEAR(result.chi2_stat, 0.1462656537, 1e-6);
    EXPECT_GT(result.p_value, 0.5);
    EXPECT_NEAR(result.p_value, 0.9294773623, 1e-6);
}

TEST(StatsExtTest, fligner_test_unequal_variances_small_p_value) {
    // Group 3 has a much larger spread (sigma = 8) than groups 1 and 2 (sigma = 1).
    // scipy.stats.fligner on this fixture gives statistic ~= 20.40, p ~= 3.71e-5.
    const std::vector<std::vector<double>> groups = {
        {10.305, 8.96, 10.75, 10.941, 8.049, 8.698, 10.128, 9.684, 9.983, 9.147,
         10.879, 10.778, 10.066, 11.127, 10.468, 9.141, 10.369, 9.041, 10.878, 9.95},
        {19.815, 19.319, 21.223, 19.845, 19.572, 19.648, 20.532, 20.365, 20.413, 20.431,
         22.142, 19.594, 19.488, 19.186, 20.616, 21.129, 19.886, 19.16, 19.176, 20.651},
        {20.946, 19.345, 9.676, 16.857, 15.933, 16.75, 21.971, 16.789, 20.431, 15.541,
         17.313, 20.05, 3.343, 12.443, 11.237, 9.889, 12.799, 26.96, 8.073, 22.746},
    };
    const auto result = fligner_test(groups);
    EXPECT_EQ(result.df, 2);
    EXPECT_NEAR(result.chi2_stat, 20.4022585431, 1e-6);
    EXPECT_LT(result.p_value, 0.01);
    EXPECT_NEAR(result.p_value, 0.0000371284, 1e-6);
}

TEST(StatsExtTest, fligner_test_robust_to_outliers_unlike_bartlett) {
    // Same three groups as the equal-variance fixture above, except group 3's
    // first two values are replaced with two extreme outliers (heavy tail) while
    // the rest of the group keeps its original, comparable spread. Bartlett's
    // test assumes normality and is known to be very sensitive to outliers/heavy
    // tails, so it sees the wildly inflated sample variance of group 3 and
    // (falsely) rejects equal variances with a vanishingly small p-value, even
    // though the bulk of the data across all three groups has comparable spread.
    // Fligner-Killeen, being rank/normal-score-based rather than working directly
    // with squared deviations, is far less swayed by just two extreme points and
    // remains comfortably non-significant. scipy.stats.bartlett / scipy.stats.fligner
    // on this fixture give chi2 ~= 120.25 (p ~= 7.7e-27) and FK ~= 3.16 (p ~= 0.206)
    // respectively -- a striking illustration of Bartlett's outlier-sensitivity.
    const std::vector<std::vector<double>> groups = {
        {10.002, 10.597, 9.452, 8.219, 9.091, 8.017, 10.12, 12.68, 9.016, 8.759,
         10.98, 10.714, 10.211, 8.139, 9.941, 11.391, 7.312, 9.085, 6.198, 7.421},
        {16.317, 19.53, 17.465, 20.543, 20.314, 19.626, 14.966, 18.923, 19.903, 20.227,
         16.94, 19.044, 18.043, 18.382, 22.122, 18.385, 19.935, 21.769, 18.833, 19.777},
        {75.221, -39.872, 12.55, 15.152, 17.718, 11.906, 16.719, 15.239, 13.717, 19.001,
         16.525, 12.601, 15.149, 16.153, 14.622, 16.366, 14.867, 16.334, 17.877, 13.649},
    };
    const auto bartlett = bartlett_test(groups);
    const auto fligner = fligner_test(groups);

    EXPECT_LT(bartlett.p_value, 1e-10);   // Bartlett: falsely "detects" a huge difference.
    EXPECT_GT(fligner.p_value, 0.1);      // Fligner-Killeen: robust, fails to reject at alpha=0.05.
    EXPECT_GT(fligner.p_value, bartlett.p_value * 1e6);
}

TEST(StatsExtTest, fligner_test_degenerate_inputs) {
    // Fewer than 2 (non-empty) groups.
    const std::vector<std::vector<double>> one_group = {{1.0, 2.0, 3.0}};
    const auto too_few = fligner_test(one_group);
    EXPECT_DOUBLE_EQ(too_few.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(too_few.p_value, 1.0);
    EXPECT_EQ(too_few.df, 0);

    const std::vector<std::vector<double>> empty_input = {};
    const auto empty_result = fligner_test(empty_input);
    EXPECT_DOUBLE_EQ(empty_result.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(empty_result.p_value, 1.0);

    // Empty groups mixed in with valid ones should simply be skipped.
    const std::vector<std::vector<double>> with_empty_group = {
        {1.0, 2.0, 3.0, 4.0}, {}, {5.0, 6.0, 7.0, 8.0}};
    const auto skips_empty = fligner_test(with_empty_group);
    EXPECT_EQ(skips_empty.df, 1);
    EXPECT_GE(skips_empty.p_value, 0.0);
    EXPECT_LE(skips_empty.p_value, 1.0);

    // All groups empty.
    const std::vector<std::vector<double>> all_empty = {{}, {}, {}};
    const auto all_empty_result = fligner_test(all_empty);
    EXPECT_DOUBLE_EQ(all_empty_result.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(all_empty_result.p_value, 1.0);
    EXPECT_EQ(all_empty_result.df, 0);

    // Groups with only a single data point each: N == k, so the test degenerates
    // gracefully to a zeroed result (mirrors levene_test's behavior in this case,
    // since one_way_anova / this test both require N > k).
    const std::vector<std::vector<double>> singletons = {{1.0}, {2.0}, {3.0}};
    const auto singleton_result = fligner_test(singletons);
    EXPECT_DOUBLE_EQ(singleton_result.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(singleton_result.p_value, 1.0);
    EXPECT_EQ(singleton_result.df, 0);
}

// ---------------------------------------------------------------------------
// shapiro_wilk
// ---------------------------------------------------------------------------

TEST(StatsExtTest, shapiro_wilk_normal_like_sample_high_w_and_p) {
    // make_norm_ppf_sample builds a sample from evenly-spaced normal quantiles, i.e. a
    // textbook-clean "normal-looking" dataset.
    const auto normal = make_norm_ppf_sample(25);
    const auto result = shapiro_wilk(normal);
    EXPECT_GT(result.w_stat, 0.9);
    EXPECT_LE(result.w_stat, 1.0);
    EXPECT_GT(result.p_value, 0.1);
}

TEST(StatsExtTest, shapiro_wilk_skewed_exponential_shape_low_w_and_p) {
    std::vector<double> skewed;
    for (int i = 1; i <= 30; ++i) {
        skewed.push_back(std::exp(static_cast<double>(i) / 8.0));
    }
    const auto result = shapiro_wilk(skewed);
    EXPECT_GT(result.w_stat, 0.0);
    EXPECT_LT(result.w_stat, 0.9);
    EXPECT_LT(result.p_value, 0.05);
}

TEST(StatsExtTest, shapiro_wilk_bimodal_sample_low_w) {
    std::vector<double> bimodal;
    for (int i = 0; i < 15; ++i) {
        bimodal.push_back(-8.0);
    }
    for (int i = 0; i < 15; ++i) {
        bimodal.push_back(8.0);
    }
    const auto normal = make_norm_ppf_sample(30);
    const auto bimodal_result = shapiro_wilk(bimodal);
    const auto normal_result = shapiro_wilk(normal);
    EXPECT_GT(bimodal_result.w_stat, 0.0);
    EXPECT_LT(bimodal_result.w_stat, 1.0);
    EXPECT_LT(bimodal_result.w_stat, normal_result.w_stat);
    EXPECT_LT(bimodal_result.p_value, 0.05);
}

TEST(StatsExtTest, shapiro_wilk_uniform_linear_sequence_sane_bounds) {
    std::vector<double> linear;
    for (int i = 1; i <= 20; ++i) {
        linear.push_back(static_cast<double>(i));
    }
    const auto result = shapiro_wilk(linear);
    // A perfectly evenly-spaced (uniform) sequence is not too far from normal in small
    // samples, so we only assert it's a valid, reasonably high W -- not a precise value.
    EXPECT_GT(result.w_stat, 0.8);
    EXPECT_LE(result.w_stat, 1.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

TEST(StatsExtTest, shapiro_wilk_degenerate_empty_input) {
    const std::vector<double> empty_input{};
    const auto result = shapiro_wilk(empty_input);
    EXPECT_DOUBLE_EQ(result.w_stat, 1.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
}

TEST(StatsExtTest, shapiro_wilk_degenerate_single_value) {
    const std::vector<double> single{42.0};
    const auto result = shapiro_wilk(single);
    EXPECT_DOUBLE_EQ(result.w_stat, 1.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
}

TEST(StatsExtTest, shapiro_wilk_degenerate_two_values) {
    const std::vector<double> pair{1.0, 2.0};
    const auto result = shapiro_wilk(pair);
    EXPECT_DOUBLE_EQ(result.w_stat, 1.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
}

TEST(StatsExtTest, shapiro_wilk_constant_sample_zero_variance) {
    const std::vector<double> flat{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    const auto result = shapiro_wilk(flat);
    EXPECT_DOUBLE_EQ(result.w_stat, 1.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
}

TEST(StatsExtTest, shapiro_wilk_w_stat_in_valid_range_across_vectors) {
    const std::vector<std::vector<double>> vectors = {
        make_norm_ppf_sample(10),
        make_norm_ppf_sample(50),
        {1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0},
        {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
        {0.1, 0.2, 0.15, 0.9, 0.85, 0.4, 0.5, 0.55},
    };
    for (const auto& v : vectors) {
        const auto result = shapiro_wilk(v);
        EXPECT_GT(result.w_stat, 0.0);
        EXPECT_LE(result.w_stat, 1.0);
        EXPECT_GE(result.p_value, 0.0);
        EXPECT_LE(result.p_value, 1.0);
    }
}

TEST(StatsExtTest, shapiro_wilk_skewed_sample_has_lower_w_than_symmetric_sample) {
    // Same sample size for both: a symmetric, roughly bell-shaped sample built from normal
    // quantiles vs. a heavily right-skewed exponential-like sample. The skewed sample should
    // score noticeably lower on W (relative-ordering check, more robust than absolute
    // thresholds tied to a particular null-distribution approximation).
    const auto symmetric = make_norm_ppf_sample(24);
    std::vector<double> skewed;
    for (int i = 1; i <= 24; ++i) {
        skewed.push_back(std::exp(static_cast<double>(i) / 6.0));
    }
    const auto symmetric_result = shapiro_wilk(symmetric);
    const auto skewed_result = shapiro_wilk(skewed);
    EXPECT_LT(skewed_result.w_stat, symmetric_result.w_stat);
}

TEST(StatsExtTest, shapiro_wilk_outlier_laden_sample_lower_w_than_clean_normal) {
    auto with_outliers = make_norm_ppf_sample(30);
    // Inject two extreme outliers into an otherwise normal-quantile sample.
    with_outliers[0] = -500.0;
    with_outliers[29] = 500.0;
    const auto clean = make_norm_ppf_sample(30);
    const auto outlier_result = shapiro_wilk(with_outliers);
    const auto clean_result = shapiro_wilk(clean);
    EXPECT_LT(outlier_result.w_stat, clean_result.w_stat);
    EXPECT_LT(outlier_result.p_value, clean_result.p_value);
}

TEST(StatsExtTest, shapiro_wilk_negative_and_shifted_values_still_sane) {
    // Same normal-quantile shape as the "clean" normal test, just shifted and scaled to
    // negative values -- W should be invariant to location/scale (affine-invariant statistic).
    auto shifted = make_norm_ppf_sample(20);
    for (double& v : shifted) {
        v = v * 3.0 - 100.0;
    }
    const auto result = shapiro_wilk(shifted);
    const auto baseline = shapiro_wilk(make_norm_ppf_sample(20));
    EXPECT_NEAR(result.w_stat, baseline.w_stat, 1e-9);
    EXPECT_NEAR(result.p_value, baseline.p_value, 1e-9);
}

TEST(StatsExtTest, shapiro_wilk_large_normal_like_sample_very_high_w) {
    const auto normal = make_norm_ppf_sample(200);
    const auto result = shapiro_wilk(normal);
    EXPECT_GT(result.w_stat, 0.95);
    EXPECT_GT(result.p_value, 0.1);
}

// ---------------------------------------------------------------------------
// wilcoxon_signed_rank
// ---------------------------------------------------------------------------

TEST(StatsExtTest, wilcoxon_signed_rank_clear_systematic_difference_significant) {
    // Every pair has y[i] > x[i] by a growing margin, so every difference is negative and the
    // ranking has no ties -- the most one-sided possible outcome (W+ = 0, W- = n(n+1)/2).
    const std::vector<double> x{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
    const std::vector<double> y{15.0, 26.0, 38.0, 51.0, 65.0, 80.0, 96.0, 113.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 8);
    EXPECT_NEAR(result.w_stat, 0.0, 1e-12);
    EXPECT_LT(result.p_value, 0.02);
    EXPECT_LT(result.z_stat, 0.0);
}

TEST(StatsExtTest, wilcoxon_signed_rank_balanced_no_effect_not_significant) {
    // Differences alternate sign with increasing magnitude (-1, 2, -3, 4, ..., 10), so positive
    // and negative rank-sums are close (W+ = 30, W- = 25 out of a total of 55) -- no real
    // systematic paired effect.
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    const std::vector<double> y{2.0, 0.0, 6.0, 0.0, 10.0, 0.0, 14.0, 0.0, 18.0, 0.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 10);
    EXPECT_NEAR(result.w_stat, 25.0, 1e-9);
    EXPECT_GT(result.p_value, 0.5);
}

TEST(StatsExtTest, wilcoxon_signed_rank_zero_diff_pairs_excluded) {
    // 5 of the 10 pairs are exactly equal (zero difference) and must be discarded; the
    // remaining 5 all share the same non-zero difference (-5).
    const std::vector<double> x{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    const std::vector<double> y{10.0, 25.0, 30.0, 45.0, 50.0, 65.0, 70.0, 85.0, 90.0, 105.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 5);  // 10 total - 5 zero-diff pairs
    EXPECT_NEAR(result.w_stat, 0.0, 1e-12);  // all remaining diffs are negative
    EXPECT_LT(result.p_value, 0.1);
}

TEST(StatsExtTest, wilcoxon_signed_rank_mismatched_lengths_returns_default) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    const std::vector<double> y{1.0, 2.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_DOUBLE_EQ(result.w_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.z_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
    EXPECT_EQ(result.n_eff, 0);
}

TEST(StatsExtTest, wilcoxon_signed_rank_all_zero_differences_degenerate) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 0);
    EXPECT_DOUBLE_EQ(result.w_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.z_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
}

TEST(StatsExtTest, wilcoxon_signed_rank_empty_inputs_returns_default) {
    const std::vector<double> empty;
    const auto result = wilcoxon_signed_rank(empty, empty);
    EXPECT_EQ(result.n_eff, 0);
    EXPECT_DOUBLE_EQ(result.w_stat, 0.0);
    EXPECT_DOUBLE_EQ(result.p_value, 1.0);
}

// Hand-computed reference case, no ties. Differences d = x - y = {3, -7, 5, -2, 9, -4}.
// Sorted |d| = {2, 3, 4, 5, 7, 9} -> ranks 1..6 respectively (no ties, so ranks are just the
// position in sorted order):
//   |d|=3 -> rank 2 (positive)      |d|=7 -> rank 5 (negative)
//   |d|=5 -> rank 4 (positive)      |d|=2 -> rank 1 (negative)
//   |d|=9 -> rank 6 (positive)      |d|=4 -> rank 3 (negative)
// W+ = 2 + 4 + 6 = 12, W- = 5 + 1 + 3 = 9 (check: 12 + 9 = 21 = 6*7/2). W = min(12, 9) = 9.
TEST(StatsExtTest, wilcoxon_signed_rank_hand_computed_no_ties) {
    const std::vector<double> x{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    const std::vector<double> y{7.0, 27.0, 25.0, 42.0, 41.0, 64.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 6);
    EXPECT_NEAR(result.w_stat, 9.0, 1e-9);
}

// Hand-computed reference case with ties in |d|. Differences d = x - y = {4, -4, 6, -6, 4, 2}.
// |d| values: {4, 4, 6, 6, 4, 2}. Tie groups: |d|=2 (1 occurrence, rank 1), |d|=4 (3
// occurrences, occupying ranks 2-4, average rank 3), |d|=6 (2 occurrences, occupying ranks 5-6,
// average rank 5.5).
// Assigning average ranks back to each signed difference:
//   d= 4 -> rank 3 (positive)   d=-4 -> rank 3 (negative)   d= 6 -> rank 5.5 (positive)
//   d=-6 -> rank 5.5 (negative) d= 4 -> rank 3 (positive)   d= 2 -> rank 1 (positive)
// W+ = 3 + 5.5 + 3 + 1 = 12.5, W- = 3 + 5.5 = 8.5 (check: 12.5 + 8.5 = 21 = 6*7/2).
// W = min(12.5, 8.5) = 8.5.
// Also cross-check the tie-corrected variance directly: mean_W = 6*7/4 = 10.5,
// var_W = 6*7*13/24 - tie_sum/48 where tie_sum = (3^3-3) + (2^3-2) = 24 + 6 = 30, so
// var_W = 22.75 - 30/48 = 22.75 - 0.625 = 22.125. With the 0.5 continuity correction
// (W < mean_W, so cc = +0.5): z = (8.5 - 10.5 + 0.5) / sqrt(22.125) = -1.5 / sqrt(22.125).
TEST(StatsExtTest, wilcoxon_signed_rank_hand_computed_with_ties) {
    const std::vector<double> x{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    const std::vector<double> y{6.0, 24.0, 24.0, 46.0, 46.0, 58.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 6);
    EXPECT_NEAR(result.w_stat, 8.5, 1e-9);
    const double expected_var_w = 22.125;
    const double expected_z = (8.5 - 10.5 + 0.5) / std::sqrt(expected_var_w);
    EXPECT_NEAR(result.z_stat, expected_z, 1e-9);
}

TEST(StatsExtTest, wilcoxon_signed_rank_symmetric_under_swap_of_x_and_y) {
    // Swapping x and y flips the sign of every difference, which swaps W+ and W- -- but since
    // the reported statistic is W = min(W+, W-), the result must be identical either way.
    const std::vector<double> x{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    const std::vector<double> y{7.0, 27.0, 25.0, 42.0, 41.0, 64.0};
    const auto forward = wilcoxon_signed_rank(x, y);
    const auto backward = wilcoxon_signed_rank(y, x);
    EXPECT_EQ(forward.n_eff, backward.n_eff);
    EXPECT_NEAR(forward.w_stat, backward.w_stat, 1e-12);
    EXPECT_NEAR(forward.p_value, backward.p_value, 1e-12);
}

TEST(StatsExtTest, wilcoxon_signed_rank_bounds_sanity_finite_and_in_range) {
    const std::vector<double> x{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    const std::vector<double> y{9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0, 9.0};
    const auto result = wilcoxon_signed_rank(x, y);
    ASSERT_GT(result.n_eff, 0);
    const double n_d = static_cast<double>(result.n_eff);
    EXPECT_TRUE(std::isfinite(result.w_stat));
    EXPECT_TRUE(std::isfinite(result.z_stat));
    EXPECT_TRUE(std::isfinite(result.p_value));
    EXPECT_GE(result.w_stat, 0.0);
    EXPECT_LE(result.w_stat, n_d * (n_d + 1.0) / 2.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

TEST(StatsExtTest, wilcoxon_signed_rank_single_pair_deterministic) {
    // A single non-zero-difference pair: n_eff = 1, rank = 1. With only one pair, either W+ or
    // W- takes the whole rank sum (1) and the other is 0, so W = min(W+, W-) = 0 always.
    // mean_W = 1*2/4 = 0.5, var_W = 1*2*3/24 = 0.25 (no ties possible with only one value),
    // sd_W = 0.5. W (0) is below mean_W, so cc = +0.5:
    // z = (0 - 0.5 + 0.5) / 0.5 = 0 exactly, giving p_value = 2*(1 - norm_cdf(0)) = 1.0 exactly.
    const std::vector<double> x{10.0};
    const std::vector<double> y{7.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 1);
    EXPECT_NEAR(result.w_stat, 0.0, 1e-12);
    EXPECT_NEAR(result.z_stat, 0.0, 1e-9);
    EXPECT_NEAR(result.p_value, 1.0, 1e-9);
}

// All 8 differences share the same magnitude (5) but are evenly split between positive and
// negative signs, so the tied ranks are perfectly balanced between W+ and W-: W+ = W- = 18
// (mean_W = 8*9/4 = 18 exactly), so this is the textbook "no effect" outcome even though every
// single difference is individually large.
TEST(StatsExtTest, wilcoxon_signed_rank_all_tied_balanced_signs_not_significant) {
    const std::vector<double> x{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
    const std::vector<double> y{15.0, 25.0, 35.0, 45.0, 45.0, 55.0, 65.0, 75.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 8);
    EXPECT_NEAR(result.w_stat, 18.0, 1e-9);
    EXPECT_GT(result.p_value, 0.8);
}

TEST(StatsExtTest, wilcoxon_signed_rank_single_zero_pair_among_many) {
    // Exactly one pair out of six is an exact tie (x[2] == y[2] == 30); the other five all have
    // a non-zero, distinct-magnitude difference.
    const std::vector<double> x{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    const std::vector<double> y{12.0, 25.0, 30.0, 33.0, 58.0, 54.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 5);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

TEST(StatsExtTest, wilcoxon_signed_rank_large_sample_strong_effect_with_ties) {
    // 20 pairs, every difference negative (y always larger), with a handful of repeated
    // magnitudes to exercise the tie-correction path alongside a strong, clearly significant
    // effect.
    std::vector<double> x;
    std::vector<double> y;
    for (int i = 0; i < 20; ++i) {
        const double xv = static_cast<double>(10 * (i + 1));
        const double margin = 5.0 + static_cast<double>(i % 5);  // repeats every 5 -> ties
        x.push_back(xv);
        y.push_back(xv + margin);
    }
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 20);
    EXPECT_NEAR(result.w_stat, 0.0, 1e-9);
    EXPECT_LT(result.p_value, 1e-4);
}

TEST(StatsExtTest, wilcoxon_signed_rank_n_eff_tracks_non_zero_pair_count) {
    // Fix 8 pairs total, starting as all-exact-ties (y == x), then progressively turn more of
    // them into non-zero-difference pairs; n_eff must track the number of non-zero pairs
    // exactly (equivalently, 8 - number_of_remaining_zero_pairs) each time.
    const std::vector<double> base_x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    for (int nonzero_count = 0; nonzero_count <= 8; ++nonzero_count) {
        std::vector<double> y = base_x;
        for (int i = 0; i < nonzero_count; ++i) {
            y[static_cast<size_t>(i)] += 3.0;  // introduce a non-zero difference
        }
        const auto result = wilcoxon_signed_rank(base_x, y);
        EXPECT_EQ(result.n_eff, nonzero_count) << "nonzero_count=" << nonzero_count;
    }
}

TEST(StatsExtTest, wilcoxon_signed_rank_small_no_tie_opposite_signs_bounds) {
    const std::vector<double> x{5.0, 12.0, 8.0, 20.0};
    const std::vector<double> y{8.0, 10.0, 15.0, 15.0};
    const auto result = wilcoxon_signed_rank(x, y);
    EXPECT_EQ(result.n_eff, 4);
    EXPECT_GE(result.w_stat, 0.0);
    EXPECT_LE(result.w_stat, 10.0);  // n(n+1)/2 = 4*5/2 = 10
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

// ---------------------------------------------------------------------------
// partial_correlation
// ---------------------------------------------------------------------------

TEST(StatsExtTest, partial_correlation_reduces_to_correlation_when_z_uncorrelated) {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    for (int i = 0; i < 100; ++i) {
        x.push_back(static_cast<double>(i));
        y.push_back(2.0 * static_cast<double>(i) + 1.0);
        z.push_back((i % 2 == 0) ? 1.0 : -1.0);
    }
    EXPECT_NEAR(correlation(x, z), 0.0, 0.05);
    EXPECT_NEAR(correlation(y, z), 0.0, 0.05);
    EXPECT_NEAR(partial_correlation(x, y, z), correlation(x, y), 0.05);
}

TEST(StatsExtTest, partial_correlation_spurious_correlation_drops_after_controlling_z) {
    std::vector<double> z;
    std::vector<double> x;
    std::vector<double> y;
    for (int i = 0; i < 100; ++i) {
        const double zi = static_cast<double>(i);
        z.push_back(zi);
        x.push_back(zi + 0.01 * std::sin(static_cast<double>(i)));
        y.push_back(zi + 0.02 * std::cos(static_cast<double>(i)));
    }
    EXPECT_GT(std::abs(correlation(x, y)), 0.99);
    EXPECT_NEAR(partial_correlation(x, y, z), 0.0, 0.15);
}

TEST(StatsExtTest, partial_correlation_degenerate_inputs) {
    const std::vector<double> empty;
    const std::vector<double> one{1.0};
    const std::vector<double> two{1.0, 2.0};
    EXPECT_DOUBLE_EQ(partial_correlation(empty, empty, empty), 0.0);
    EXPECT_DOUBLE_EQ(partial_correlation(one, one, one), 0.0);
    EXPECT_DOUBLE_EQ(partial_correlation(two, one, one), 0.0);
}

// ---------------------------------------------------------------------------
// variance_inflation_factor / vif
// ---------------------------------------------------------------------------

TEST(StatsExtTest, vif_orthogonal_design_near_one) {
    std::vector<std::vector<double>> X;
    constexpr int n = 64;
    X.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n);
        X.push_back({std::sin(t), std::cos(t), std::sin(2.0 * t)});
    }
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(variance_inflation_factor(X, j), 1.0, 0.15);
        EXPECT_NEAR(vif(X, j), variance_inflation_factor(X, j), 1e-12);
    }
}

TEST(StatsExtTest, vif_near_multicollinearity_is_large) {
    std::vector<std::vector<double>> X;
    constexpr int n = 50;
    X.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double a = static_cast<double>(i);
        const double b = static_cast<double>(i * i) / static_cast<double>(n);
        X.push_back({a, b, a + b + 1e-8 * static_cast<double>(i)});
    }
    EXPECT_GT(variance_inflation_factor(X, 2), 100.0);
}

TEST(StatsExtTest, vif_degenerate_inputs) {
    const std::vector<std::vector<double>> empty;
    const std::vector<std::vector<double>> one_col{{1.0}, {2.0}, {3.0}};
    const std::vector<std::vector<double>> jagged{{1.0, 2.0}, {3.0}};
    EXPECT_DOUBLE_EQ(variance_inflation_factor(empty, 0), 1.0);
    EXPECT_DOUBLE_EQ(variance_inflation_factor(one_col, 0), 1.0);
    EXPECT_DOUBLE_EQ(variance_inflation_factor(jagged, 0), 0.0);
    EXPECT_DOUBLE_EQ(variance_inflation_factor(one_col, 1), 0.0);
}
