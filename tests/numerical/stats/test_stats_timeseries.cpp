// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Wave 56: Stats time-series, bootstrap, new descriptive tests
#include "ms/stats/stats.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

// -----------------------------------------------------------------------
// Geometric mean
// -----------------------------------------------------------------------
TEST(StatsGeometricMean, KnownValue) {
    std::vector<double> x = {1.0, 4.0, 16.0};
    // geometric mean = (1 * 4 * 16)^(1/3) = 64^(1/3) = 4
    EXPECT_NEAR(ms::geometric_mean(x), 4.0, 1e-10);
}

TEST(StatsGeometricMean, AllOnes) {
    std::vector<double> x = {1.0, 1.0, 1.0, 1.0};
    EXPECT_NEAR(ms::geometric_mean(x), 1.0, 1e-12);
}

TEST(StatsGeometricMean, SingleValue) {
    std::vector<double> x = {7.5};
    EXPECT_NEAR(ms::geometric_mean(x), 7.5, 1e-12);
}

// -----------------------------------------------------------------------
// Harmonic mean
// -----------------------------------------------------------------------
TEST(StatsHarmonicMean, KnownValue) {
    // HM(1, 2, 4) = 3 / (1 + 1/2 + 1/4) = 3 / 1.75 = 12/7
    std::vector<double> x = {1.0, 2.0, 4.0};
    EXPECT_NEAR(ms::harmonic_mean(x), 12.0 / 7.0, 1e-10);
}

TEST(StatsHarmonicMean, AllEqual) {
    std::vector<double> x = {5.0, 5.0, 5.0};
    EXPECT_NEAR(ms::harmonic_mean(x), 5.0, 1e-12);
}

TEST(StatsHarmonicMean, LeThanArithmeticMean) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    EXPECT_LT(ms::harmonic_mean(x), ms::mean(x));
}

// -----------------------------------------------------------------------
// RMS
// -----------------------------------------------------------------------
TEST(StatsRMS, KnownValue) {
    // rms([3, 4]) = sqrt((9+16)/2) = sqrt(12.5) ~ 3.536
    std::vector<double> x = {3.0, 4.0};
    EXPECT_NEAR(ms::rms(x), std::sqrt(12.5), 1e-10);
}

TEST(StatsRMS, AllOnes) {
    std::vector<double> x = {1.0, 1.0, 1.0};
    EXPECT_NEAR(ms::rms(x), 1.0, 1e-12);
}

// -----------------------------------------------------------------------
// MAD
// -----------------------------------------------------------------------
TEST(StatsMAD, KnownValue) {
    // mad([1, 1, 2, 2, 4, 6, 9]) median=2, devs=[1,1,0,0,2,4,7], median dev=1
    std::vector<double> x = {1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0};
    EXPECT_NEAR(ms::mad(x, false), 1.0, 1e-10);
    EXPECT_NEAR(ms::mad(x), 1.4826, 1e-10);
}

TEST(StatsMAD, Uniform) {
    std::vector<double> x = {5.0, 5.0, 5.0};
    EXPECT_NEAR(ms::mad(x), 0.0, 1e-12);
}

// -----------------------------------------------------------------------
// IQR
// -----------------------------------------------------------------------
TEST(StatsIQR, KnownValue) {
    // For [1,2,3,4,5,6,7,8], Q1=2.5, Q3=6.5 => IQR=4.0
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double iq = ms::iqr(x);
    EXPECT_GT(iq, 0.0);
    EXPECT_TRUE(std::isfinite(iq));
}

TEST(StatsIQR, Positive) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_GT(ms::iqr(x), 0.0);
}

// -----------------------------------------------------------------------
// Trimmed mean
// -----------------------------------------------------------------------
TEST(StatsTrimmedMean, KnownValue) {
    // Trimmed 20% of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double tm = ms::trimmed_mean(x, 0.1);  // trim 10% from each end => drop 1 on each side
    EXPECT_NEAR(tm, ms::mean(std::vector<double>{2, 3, 4, 5, 6, 7, 8, 9}), 0.01);
}

TEST(StatsTrimmedMean, IsFinite) {
    std::vector<double> x = {1.0, 5.0, 100.0, 200.0, 3.0, 2.0};
    EXPECT_TRUE(std::isfinite(ms::trimmed_mean(x, 0.2)));
}

// -----------------------------------------------------------------------
// Spearman correlation
// -----------------------------------------------------------------------
TEST(StatsSpearman, PerfectMonotone) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    EXPECT_NEAR(ms::spearman(x, y), 1.0, 1e-10);
}

TEST(StatsSpearman, PerfectAntiMonotone) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {10.0, 8.0, 6.0, 4.0, 2.0};
    EXPECT_NEAR(ms::spearman(x, y), -1.0, 1e-10);
}

TEST(StatsSpearman, RangeIsMinusOneToOne) {
    std::vector<double> x = {1.0, 3.0, 2.0, 5.0, 4.0};
    std::vector<double> y = {3.0, 1.0, 4.0, 2.0, 5.0};
    double s = ms::spearman(x, y);
    EXPECT_GE(s, -1.0);
    EXPECT_LE(s,  1.0);
}

// -----------------------------------------------------------------------
// Kendall tau
// -----------------------------------------------------------------------
TEST(StatsKendall, PerfectAgreement) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};
    EXPECT_NEAR(ms::kendall(x, y), 1.0, 1e-10);
}

TEST(StatsKendall, PerfectDisagreement) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(ms::kendall(x, y), -1.0, 1e-10);
}

TEST(StatsKendall, RangeCheck) {
    std::vector<double> x = {1.0, 3.0, 2.0, 4.0};
    std::vector<double> y = {2.0, 1.0, 3.0, 4.0};
    double k = ms::kendall(x, y);
    EXPECT_GE(k, -1.0);
    EXPECT_LE(k,  1.0);
}

namespace {

// The definition, written out: every pair against every other, tau-b's denominator.
//
// This is what `ms::kendall` used to BE, kept here as the reference it is now checked
// against. It was replaced because it is quadratic -- 3.5 s at n = 20000, 13.9 s at
// n = 40000, and the REPL's `stats_kendall` did not finish on 100000 observations --
// by Knight's O(n log n) formulation, which reaches the same number through an
// inversion count. The point of this test is that "the same number" is exact and not
// approximate.
double kendall_by_definition(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    if (n != y.size() || n == 0) {
        return 0.0;
    }
    long long concordant = 0;
    long long discordant = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double product = (x[i] - x[j]) * (y[i] - y[j]);
            if (product > 0.0) {
                ++concordant;
            } else if (product < 0.0) {
                --discordant;  // accumulated negative, added below
            }
        }
    }
    const auto tie_pairs = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double total = 0.0;
        std::size_t i = 0;
        while (i < v.size()) {
            std::size_t j = i + 1;
            while (j < v.size() && v[j] == v[i]) {
                ++j;
            }
            const double t = static_cast<double>(j - i);
            total += t * (t - 1.0) / 2.0;
            i = j;
        }
        return total;
    };
    const double nd = static_cast<double>(n);
    const double n0 = nd * (nd - 1.0) / 2.0;
    const double denom = std::sqrt((n0 - tie_pairs(x)) * (n0 - tie_pairs(y)));
    return (denom > 0.0) ? static_cast<double>(concordant + discordant) / denom : 0.0;
}

} // namespace

TEST(StatsKendall, MatchesTheQuadraticDefinitionExactly) {
    // A deterministic linear congruential sequence rather than <random>, so the cases
    // are the same on every platform and a failure is reproducible from the test alone.
    std::uint64_t state = 20260911u;
    const auto next = [&state]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((state >> 33) % 1000u);
    };
    // Four regimes, because ties are where the two formulations could diverge:
    // continuous, ties in x only, ties in y only, and heavy ties in both.
    for (int regime = 0; regime < 4; ++regime) {
        for (const std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{7},
                                    std::size_t{16}, std::size_t{41}, std::size_t{90}}) {
            std::vector<double> x(n);
            std::vector<double> y(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double a = next();
                const double b = next();
                x[i] = (regime == 1 || regime == 3) ? std::floor(a / 400.0) : a;
                y[i] = (regime == 2 || regime == 3) ? std::floor(b / 400.0) : b;
            }
            EXPECT_NEAR(ms::kendall(x, y), kendall_by_definition(x, y), 1e-12)
                << "regime " << regime << ", n = " << n;
        }
    }
}

TEST(StatsKendall, TiesInEveryValueGiveZeroRatherThanNaN) {
    // Every pair is tied, so the tau-b denominator is zero. The identity the fast form
    // uses reaches that case by a different route -- n1 and n2 both equal n0 -- and it
    // has to land on the same answer.
    const std::vector<double> x(20, 7.0);
    const std::vector<double> y(20, 3.0);
    EXPECT_DOUBLE_EQ(ms::kendall(x, y), 0.0);
    EXPECT_DOUBLE_EQ(ms::kendall(x, y), kendall_by_definition(x, y));
}

TEST(StatsKendall, AHundredThousandObservationsIsAnOrdinaryRequest) {
    // The quadratic form was 5e9 comparisons here and did not return. This asserts the
    // ANSWER rather than a duration: a strictly increasing pair is tau = 1 exactly, and
    // an exactly reversed one is -1, which no partial or truncated computation produces.
    const std::size_t n = 100000;
    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> reversed(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = 2.0 * static_cast<double>(i) + 1.0;
        reversed[i] = -static_cast<double>(i);
    }
    EXPECT_NEAR(ms::kendall(x, y), 1.0, 1e-12);
    EXPECT_NEAR(ms::kendall(x, reversed), -1.0, 1e-12);
}

// -----------------------------------------------------------------------
// chi2_gof
// -----------------------------------------------------------------------
TEST(StatsChi2GOF, UniformIsZero) {
    std::vector<double> obs = {10.0, 10.0, 10.0, 10.0};
    std::vector<double> exp = {10.0, 10.0, 10.0, 10.0};
    EXPECT_NEAR(ms::chi2_gof(obs, exp), 0.0, 1e-12);
}

TEST(StatsChi2GOF, NonZero) {
    std::vector<double> obs = {12.0, 8.0, 10.0, 10.0};
    std::vector<double> exp = {10.0, 10.0, 10.0, 10.0};
    double chi2 = ms::chi2_gof(obs, exp);
    EXPECT_GT(chi2, 0.0);
    EXPECT_TRUE(std::isfinite(chi2));
}

// -----------------------------------------------------------------------
// KS test
// -----------------------------------------------------------------------
TEST(StatsKSTest, UniformCDF_SmallStat) {
    // For uniform samples, KS stat should be small
    std::vector<double> x;
    for (int i = 1; i <= 20; ++i) x.push_back(i / 20.0);
    double dn = ms::ks_test(x, [](double t) { return t; });
    EXPECT_LT(dn, 0.1);
    EXPECT_GT(dn, 0.0);
}

// -----------------------------------------------------------------------
// Multiple regression
// -----------------------------------------------------------------------
TEST(StatsMultipleRegression, LinearModel) {
    // y = 1 + 2*x1 + 3*x2
    std::vector<std::vector<double>> X = {
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 2.0, 1.0},
    };
    std::vector<double> y = {1.0, 3.0, 4.0, 6.0, 8.0};
    auto beta = ms::multiple_regression(X, y);
    ASSERT_EQ(beta.size(), 3u);
    EXPECT_NEAR(beta[0], 1.0, 0.01);
    EXPECT_NEAR(beta[1], 2.0, 0.01);
    EXPECT_NEAR(beta[2], 3.0, 0.01);
}

// -----------------------------------------------------------------------
// ACF
// -----------------------------------------------------------------------
TEST(StatsACF, Lag0IsOne) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto a = ms::acf(x, 3);
    ASSERT_GE(a.size(), 1u);
    EXPECT_NEAR(a[0], 1.0, 1e-12);
}

TEST(StatsACF, DecayingForNoise) {
    // White noise has ACF near 0 for lag > 0
    std::vector<double> x = {1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1};
    auto a = ms::acf(x, 4);
    EXPECT_NEAR(a[0], 1.0, 1e-12);
    // Other lags should be finite
    for (size_t i = 1; i < a.size(); ++i)
        EXPECT_TRUE(std::isfinite(a[i]));
}

TEST(StatsACF, SizeCheck) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto a = ms::acf(x, 3);
    EXPECT_EQ(a.size(), 4u);  // lags 0..3
}

TEST(StatsACF, ARProcess_PositiveLag1) {
    // AR(1) with phi=0.8: x_t = 0.8 x_{t-1} + e_t  => acf(1) ≈ 0.8
    std::vector<double> x = {1.0};
    for (int i = 1; i < 200; ++i)
        x.push_back(0.8 * x.back() + 0.1 * std::sin(i * 1.23));
    auto a = ms::acf(x, 5);
    EXPECT_GT(a[1], 0.0);
}

// -----------------------------------------------------------------------
// PACF
// -----------------------------------------------------------------------
TEST(StatsPACF, Lag0IsOne) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    auto pa = ms::pacf(x, 3);
    ASSERT_GE(pa.size(), 1u);
    EXPECT_NEAR(pa[0], 1.0, 1e-12);
}

TEST(StatsPACF, SizeCheck) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto pa = ms::pacf(x, 4);
    EXPECT_EQ(pa.size(), 5u);
}

TEST(StatsPACF, FiniteValues) {
    std::vector<double> x = {1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5};
    auto pa = ms::pacf(x, 3);
    for (auto v : pa) EXPECT_TRUE(std::isfinite(v));
}

// -----------------------------------------------------------------------
// AR fit
// -----------------------------------------------------------------------
TEST(StatsARFit, AR1_RecoverCoefficient) {
    // Generate AR(1) with phi=0.7 using pseudo-random white noise
    std::vector<double> x = {0.0};
    // Simple LCG for deterministic but noise-like sequence
    uint32_t state = 12345;
    for (int i = 1; i < 1000; ++i) {
        state = state * 1664525u + 1013904223u;
        double e = (static_cast<double>(state) / 4294967296.0 - 0.5) * 0.3;
        x.push_back(0.7 * x.back() + e);
    }
    auto phi = ms::arfit(x, 1);
    ASSERT_GE(phi.size(), 1u);
    // Should recover approximately 0.7 (allow wider tolerance)
    EXPECT_NEAR(phi[0], 0.7, 0.2);
}

TEST(StatsARFit, CoefficientsAreFinite) {
    std::vector<double> x = {1.0, 1.5, 2.0, 1.8, 2.1, 2.5, 2.3};
    auto phi = ms::arfit(x, 2);
    for (auto v : phi) EXPECT_TRUE(std::isfinite(v));
}

// -----------------------------------------------------------------------
// Bootstrap
// -----------------------------------------------------------------------
TEST(StatsBootstrapMean, NearTrueMean) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double bm = ms::bootstrap_mean(x, 500, 42);
    double true_mean = ms::mean(x);
    EXPECT_NEAR(bm, true_mean, 0.5);
}

TEST(StatsBootstrapMean, IsFinite) {
    std::vector<double> x = {2.0, 4.0, 6.0, 8.0};
    double bm = ms::bootstrap_mean(x, 100, 1);
    EXPECT_TRUE(std::isfinite(bm));
}

TEST(StatsBootstrapCI, CIContainsTrueMean) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto [lo, hi] = ms::bootstrap_ci(x, 0.95, 1000, 42);
    double m = ms::mean(x);
    EXPECT_LT(lo, m);
    EXPECT_GT(hi, m);
}

TEST(StatsBootstrapCI, LoLtHi) {
    std::vector<double> x = {1.0, 3.0, 5.0, 7.0, 9.0};
    auto [lo, hi] = ms::bootstrap_ci(x, 0.95, 500, 7);
    EXPECT_LT(lo, hi);
}
