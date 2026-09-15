// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: three things in `src/stats/stats.cpp` that ran and that nothing asserted.
//
// A sample of 24 mutants at seed 79 against the ten suites that cover the file
// scored 78.3%. Three survivors are addressed here; the reasons they survived
// are each specific:
//
//   shapiro_wilk          Royston's normalising transformation has two branches,
//                         n <= 11 and n > 11, with different polynomial
//                         coefficients. Signs inside the SMALL-sample branch's
//                         `mu` and `sigma` mutate freely -- nothing asserts a
//                         p-value for a sample of eleven or fewer.
//   weighted_correlation  its size guard is three `||` clauses; turning the first
//                         into `&&` lets a w of the wrong length through, which
//                         is then read past its end. Nothing passed it
//                         mismatched sizes at all.
//   friedman              the tie correction fires on
//                         `tie_cubed_sum > 0 && tie_denom > 0`; as `||` it
//                         divides by a zero denominator. `tie_denom` is
//                         n*(k^3 - k), which is zero at k = 1, and no test runs
//                         Friedman on a single treatment.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/stats/stats.hpp"

using namespace ms;

TEST(StatsSmallSample, ShapiroWilkSeparatesNormalFromSkewedBelowTwelve) {
    // n <= 11 takes its own branch of the normalising transformation, with its
    // own four `mu` coefficients and its own three for `sigma`. A sign flip in
    // either moves the z-score by several units, which moves the p-value from
    // one end of its range to the other -- so asserting which END it lands on is
    // enough, and is a far more robust statement than a reference p-value.
    const std::vector<double> near_normal_8{-1.2, -0.7, -0.3, -0.1, 0.15, 0.4, 0.85, 1.3};
    const std::vector<double> near_normal_11{-1.6, -1.1, -0.75, -0.4, -0.15, 0.05,
                                             0.3,  0.6,  0.95,  1.35, 1.8};
    for (const auto* sample : {&near_normal_8, &near_normal_11}) {
        const auto r = shapiro_wilk(*sample);
        EXPECT_GT(r.w_stat, 0.95) << "W for a near-normal sample of " << sample->size();
        EXPECT_LE(r.w_stat, 1.0);
        EXPECT_GT(r.p_value, 0.5)
            << "a near-normal sample of " << sample->size() << " was called non-normal";
        EXPECT_LE(r.p_value, 1.0);
    }

    // Strongly skewed, same sizes: one long tail is what the test exists to find.
    const std::vector<double> skewed_8{0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 9.0};
    const std::vector<double> one_outlier_10{1, 1, 1, 1, 1, 1, 1, 1, 1, 40};
    for (const auto* sample : {&skewed_8, &one_outlier_10}) {
        const auto r = shapiro_wilk(*sample);
        EXPECT_LT(r.w_stat, 0.7) << "W for a skewed sample of " << sample->size();
        EXPECT_GT(r.w_stat, 0.0);
        EXPECT_LT(r.p_value, 0.01)
            << "a strongly skewed sample of " << sample->size() << " was called normal";
        EXPECT_GE(r.p_value, 0.0);
    }

    // And the statistic's defining invariance, which holds whatever the
    // coefficients are: W is unchanged by any positive rescaling and any shift,
    // because it is a ratio of two quadratic forms in the centred order
    // statistics. This is the half a p-value threshold cannot check.
    for (const double scale : {0.001, 0.5, 3.0, 1000.0}) {
        for (const double shift : {-50.0, 0.0, 7.25}) {
            std::vector<double> moved;
            moved.reserve(near_normal_11.size());
            for (const double v : near_normal_11) {
                moved.push_back(scale * v + shift);
            }
            const auto base = shapiro_wilk(near_normal_11);
            const auto got = shapiro_wilk(moved);
            EXPECT_NEAR(got.w_stat, base.w_stat, 1e-9)
                << "W changed under scale " << scale << ", shift " << shift;
            EXPECT_NEAR(got.p_value, base.p_value, 1e-9)
                << "p changed under scale " << scale << ", shift " << shift;
        }
    }

    // The two assertions above bracket the ends of the range, and the ends are
    // where `sigma` stops mattering: it divides the z-score, so shrinking it
    // only pushes an already-extreme p further out. What pins `sigma` is a
    // p-value in the MIDDLE, and these are borderline samples chosen for that.
    //
    // The numbers are Royston's algorithm applied to these points -- determined
    // by its published coefficients, not chosen -- so pinning them pins the
    // transformation, in the same way the compressed-format test pins bytes.
    struct Borderline {
        std::vector<double> sample;
        double w;
        double p;
    };
    const Borderline borderline[] = {
        {{0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.7, 3.0}, 0.874540, 0.166871},
        {{0.2, 0.5, 0.8, 1.0, 1.3, 1.7, 2.4, 4.2}, 0.870873, 0.153720},
        {{1, 2, 3, 4, 5, 6, 7, 15}, 0.827500, 0.055913},
        {{2, 3, 5, 8, 13, 21, 34}, 0.869995, 0.185610},
    };
    for (const auto& b : borderline) {
        const auto r = shapiro_wilk(b.sample);
        EXPECT_NEAR(r.w_stat, b.w, 1e-6) << "W for a borderline sample of " << b.sample.size();
        EXPECT_NEAR(r.p_value, b.p, 1e-6)
            << "p for a borderline sample of " << b.sample.size()
            << " -- the middle of the range is where sigma is visible";
    }

    // A constant sample has no spread and cannot be judged; it must not produce a
    // NaN or divide by its zero denominator.
    const std::vector<double> flat(9, 4.0);
    const auto r = shapiro_wilk(flat);
    EXPECT_TRUE(std::isfinite(r.w_stat));
    EXPECT_TRUE(std::isfinite(r.p_value));
}

TEST(StatsSmallSample, WeightedCorrelationRefusesMismatchedLengths) {
    // Three `||` clauses, and nothing had ever given it inputs that trip any of
    // them -- so the guard could be any combination of the three and answer every
    // test in the tree identically. Each length is wrong on its own here.
    const std::vector<double> three{1.0, 2.0, 3.0};
    const std::vector<double> two{1.0, 2.0};
    const std::vector<double> four{1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(weighted_correlation(three, three, two), 0.0) << "short weights accepted";
    EXPECT_EQ(weighted_correlation(three, three, four), 0.0) << "long weights accepted";
    EXPECT_EQ(weighted_correlation(three, two, three), 0.0) << "short y accepted";
    EXPECT_EQ(weighted_correlation(three, four, three), 0.0) << "long y accepted";
    EXPECT_EQ(weighted_correlation(two, three, four), 0.0) << "all three differing accepted";
    EXPECT_EQ(weighted_correlation({}, {}, {}), 0.0) << "empty accepted";

    // And it still works when they DO match, so the guard is not simply refusing
    // everything: equal weights reduce to the ordinary correlation, and a perfect
    // line has correlation 1.
    const std::vector<double> y{2.0, 4.0, 6.0};
    const std::vector<double> ones{1.0, 1.0, 1.0};
    EXPECT_NEAR(weighted_correlation(three, y, ones), 1.0, 1e-12);
    const std::vector<double> down{6.0, 4.0, 2.0};
    EXPECT_NEAR(weighted_correlation(three, down, ones), -1.0, 1e-12);
}

TEST(StatsSmallSample, FriedmanHandlesASingleTreatmentAndTotalTies) {
    // The tie correction divides by n*(k^3 - k). At k = 1 that denominator is
    // zero, and the guard's two conditions have to BOTH hold for the division to
    // happen -- which is exactly what an `||` would break, and what no test with
    // two or more treatments can see.
    const std::vector<std::vector<double>> one_treatment{{1.0}, {2.0}, {3.0}, {2.5}};
    const auto single = friedman(one_treatment);
    EXPECT_EQ(single.df, 0);
    EXPECT_TRUE(std::isfinite(single.chi2_stat)) << "k = 1 produced " << single.chi2_stat;
    EXPECT_DOUBLE_EQ(single.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(single.p_value, 1.0);

    // Every row entirely tied: the correction's numerator is at its maximum while
    // the statistic itself is zero, which is the other end of the same guard.
    const std::vector<std::vector<double>> all_tied{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
    const auto tied = friedman(all_tied);
    EXPECT_EQ(tied.df, 2);
    EXPECT_TRUE(std::isfinite(tied.chi2_stat));
    EXPECT_DOUBLE_EQ(tied.chi2_stat, 0.0);
    EXPECT_DOUBLE_EQ(tied.p_value, 1.0);

    // A real difference still registers, so the two above are not passing because
    // the function always returns zero.
    const std::vector<std::vector<double>> separated{
        {1, 5, 9}, {2, 6, 10}, {1.5, 5.5, 9.5}, {0.5, 4.5, 8.5}, {2.5, 6.5, 10.5}};
    const auto real = friedman(separated);
    EXPECT_EQ(real.df, 2);
    EXPECT_GT(real.chi2_stat, 5.0) << "a perfectly consistent ranking scored " << real.chi2_stat;
    EXPECT_LT(real.p_value, 0.05);
}
