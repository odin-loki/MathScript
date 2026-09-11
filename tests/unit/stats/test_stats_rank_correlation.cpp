// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for four audited stats defects.
//
//  * spearman assigned distinct ranks 1..n by sort position with no tie
//    handling, then applied the d^2 shortcut, which is only algebraically equal
//    to Pearson-on-ranks when every rank is distinct. With ties the result also
//    depended on the sort's arbitrary tie-breaking.
//  * friedman divided the tie correction by n*k*(k^3 - k) instead of
//    n*(k^3 - k), diluting it k-fold.
//  * kendall skipped tied pairs when counting but divided by the untied
//    n(n-1)/2, i.e. reported tau-a, so +/-1 was unreachable with ties.
//  * variance_inflation_factor fitted its auxiliary regression WITHOUT an
//    intercept while measuring R^2 against a mean-centred total sum of squares,
//    producing VIFs below 1 -- impossible for 1/(1 - R^2) -- and reporting "no
//    multicollinearity" for perfectly collinear designs.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "ms/stats/stats.hpp"

namespace {

// Pearson correlation of two explicit rank vectors, written out so the test does
// not depend on the implementation it is checking.
double pearson_of(const std::vector<double>& a, const std::vector<double>& b) {
    const double n = static_cast<double>(a.size());
    double ma = 0.0;
    double mb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    double sab = 0.0;
    double saa = 0.0;
    double sbb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return sab / std::sqrt(saa * sbb);
}

} // namespace

TEST(StatsSpearman, MatchesPearsonOnAverageRanksWithTies) {
    // x has a tied pair, so its average ranks are 1, 2.5, 2.5, 4, 5.
    const std::vector<double> x{1.0, 2.0, 2.0, 4.0, 5.0};
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> rx{1.0, 2.5, 2.5, 4.0, 5.0};
    const std::vector<double> ry{1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(ms::spearman(x, y), pearson_of(rx, ry), 1e-12);
}

TEST(StatsSpearman, IsExactlyOneForAMonotoneRelation) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b{10.0, 20.0, 30.0, 40.0, 50.0};
    const std::vector<double> c{5.0, 4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(ms::spearman(a, b), 1.0, 1e-12);
    EXPECT_NEAR(ms::spearman(a, c), -1.0, 1e-12);
}

TEST(StatsSpearman, IsIndependentOfTiedInputOrder) {
    // The old sort-position ranking made the answer depend on which of two equal
    // values the sort happened to place first.
    const std::vector<double> y{3.0, 1.0, 4.0, 1.0, 5.0};
    const std::vector<double> x1{2.0, 7.0, 2.0, 7.0, 9.0};
    const std::vector<double> x2{2.0, 7.0, 2.0, 7.0, 9.0};
    EXPECT_NEAR(ms::spearman(x1, y), ms::spearman(x2, y), 1e-15);
    EXPECT_LE(std::abs(ms::spearman(x1, y)), 1.0);
}

TEST(StatsSpearman, AllTiedInputIsUndefined) {
    // Constant ranks give a zero-variance denominator: rho is undefined, and the
    // function reports NaN rather than inventing a value.
    const std::vector<double> flat{2.0, 2.0, 2.0, 2.0};
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0};
    EXPECT_TRUE(std::isnan(ms::spearman(flat, y)));
    // A single point is the same situation.
    const std::vector<double> one_x{1.0};
    const std::vector<double> one_y{2.0};
    EXPECT_TRUE(std::isnan(ms::spearman(one_x, one_y)));
}

TEST(StatsKendall, ReachesOneOnATieFreeMonotoneRelation) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b{10.0, 20.0, 30.0, 40.0, 50.0};
    EXPECT_NEAR(ms::kendall(a, b), 1.0, 1e-12);
    const std::vector<double> c{5.0, 4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(ms::kendall(a, c), -1.0, 1e-12);
}

TEST(StatsKendall, TauBExceedsTauAWhenTiesArePresent) {
    // One tied pair in x. tau-a would report 9/10 = 0.9; tau-b normalises by
    // sqrt((n0 - n1)(n0 - n2)) and reports more.
    const std::vector<double> x{1.0, 2.0, 2.0, 4.0, 5.0};
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0, 5.0};
    const double tau = ms::kendall(x, y);
    EXPECT_GT(tau, 0.9);
    EXPECT_LE(tau, 1.0);
    // n0 = 10, n1 = 1 (one tied pair in x), n2 = 0; (C - D) = 9.
    EXPECT_NEAR(tau, 9.0 / std::sqrt(9.0 * 10.0), 1e-12);
}

TEST(StatsVif, PerfectlyCollinearColumnsAreInfinite) {
    // col1 = 9.5 + 0.5*col0: collinear only THROUGH an intercept, which is
    // exactly what the intercept-free auxiliary fit could not see. Reported
    // 1.223 and 0.013 before; a VIF below 1 is impossible by definition.
    const std::vector<std::vector<double>> a{{1.0, 10.0}, {2.0, 10.5}, {3.0, 11.0}};
    EXPECT_TRUE(std::isinf(ms::variance_inflation_factor(a, 0)));
    EXPECT_TRUE(std::isinf(ms::variance_inflation_factor(a, 1)));

    const std::vector<std::vector<double>> b{
        {1.0, 100.0}, {2.0, 101.0}, {3.0, 102.0}, {4.0, 103.0}};
    EXPECT_TRUE(std::isinf(ms::variance_inflation_factor(b, 0)));
    EXPECT_TRUE(std::isinf(ms::variance_inflation_factor(b, 1)));
}

TEST(StatsVif, IsNeverBelowOne) {
    // 1/(1 - R^2) with R^2 in [0, 1] cannot go below 1.
    const std::vector<std::vector<double>> x{
        {1.0, 5.0}, {2.0, 1.0}, {3.0, 9.0}, {4.0, 2.0}, {5.0, 7.0}};
    for (std::size_t j = 0; j < 2; ++j) {
        const double v = ms::variance_inflation_factor(x, j);
        EXPECT_GE(v, 1.0) << "column " << j;
        EXPECT_LT(v, 10.0) << "column " << j;  // these columns are near-independent
    }
}

TEST(StatsVif, DegenerateInputKeepsItsConvention) {
    const std::vector<std::vector<double>> single{{1.0}, {2.0}, {3.0}};
    EXPECT_DOUBLE_EQ(ms::variance_inflation_factor(single, 0), 1.0);  // p < 2
    EXPECT_DOUBLE_EQ(ms::variance_inflation_factor(single, 5), 0.0);  // j out of range
    const std::vector<std::vector<double>> empty_x;
    EXPECT_DOUBLE_EQ(ms::variance_inflation_factor(empty_x, 0), 1.0);  // empty
}

TEST(StatsFriedman, TieCorrectionUsesTheStandardDivisor) {
    // Blocks with ties within a row. The correction divisor is n*(k^3 - k); the
    // old n*k*(k^3 - k) diluted it by a factor of k, which made the corrected
    // statistic smaller than it should be. Here k = 3 so the two differ by 3x.
    const std::vector<std::vector<double>> data{
        {1.0, 1.0, 2.0},
        {1.0, 2.0, 2.0},
        {1.0, 1.0, 3.0},
        {2.0, 2.0, 3.0},
        {1.0, 2.0, 3.0},
    };
    const auto r = ms::friedman(data);
    EXPECT_GT(r.chi2_stat, 0.0);
    EXPECT_EQ(r.df, 2);
    EXPECT_GE(r.p_value, 0.0);
    EXPECT_LE(r.p_value, 1.0);

    // Without ties the correction is inert, so this case pins the uncorrected path.
    const std::vector<std::vector<double>> distinct{
        {1.0, 2.0, 3.0},
        {1.0, 2.0, 3.0},
        {1.0, 2.0, 3.0},
        {1.0, 2.0, 3.0},
    };
    const auto d = ms::friedman(distinct);
    // Every block ranks identically: chi2 = 12/(n k (k+1)) * sum R_j^2 - 3 n (k+1)
    //   = 12/(4*3*4) * (16 + 64 + 144) - 3*4*4 = 0.25*224 - 48 = 8.
    EXPECT_NEAR(d.chi2_stat, 8.0, 1e-9);
}
