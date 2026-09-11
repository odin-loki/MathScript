// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: the six things in `src/ml/ml.cpp` that ran and that nothing asserted.
//
// A sample of 24 mutants at seed 59 against the three suites that cover the file
// scored 65.2%, on 214 tests. Every survivor was a formula rather than a
// control-flow branch, and each one is invisible for its own specific reason --
// which is the interesting part, because the reasons are not "the test is
// missing" but "the test that exists cannot see this":
//
//   r2_score          1 - ss_res/ss_tot became 1 + ss_res/ss_tot. On a perfect
//                     fit ss_res is zero and both give 1, which is the only case
//                     the suite scores.
//   StandardScaler    (x - mean)*(x - mean) became (x + mean)*(x - mean) --
//                     which is sum(x^2) - n*mean^2, the same number, precisely
//                     because the value subtracted is the mean. Equivalent, and
//                     the assertion below measures it: the recovered std_ agrees
//                     with an independently computed one to 1e-9 either way.
//   StandardScaler    the 1e-12 that keeps a zero-variance column out of
//                     sqrt() of a negative changed sign.
//   var_tanh          the derivative 1 - t^2 became 1 + t^2. Nothing
//                     differentiated a tanh and checked the number.
//   TSNE::kl_diverge  the guard "fewer than 2 points" became "fewer than 3" --
//                     which turns out to be equivalent, for a reason worth
//                     writing down rather than guessing at.
//   GaussianMixture   the per-feature variance accumulator started at 1 instead
//                     of 0, and the loop that fills it started at feature 1.
//                     Nothing asserted a fitted variance.

#define _USE_MATH_DEFINES
#include "ms/ml/ml.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ms::ml;

namespace {

double column_mean(const Mat& X, int j) {
    double s = 0.0;
    for (const auto& row : X) s += row[static_cast<std::size_t>(j)];
    return s / static_cast<double>(X.size());
}

double column_sd(const Mat& X, int j) {
    const double m = column_mean(X, j);
    double s = 0.0;
    for (const auto& row : X) {
        const double d = row[static_cast<std::size_t>(j)] - m;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(X.size()));
}

} // namespace

TEST(MlProperties, R2IsOneMinusTheResidualShare) {
    const Vec truth{1.0, 2.0, 3.0, 4.0, 5.0};

    // A perfect fit is 1 -- and it is the ONLY case in which the sign in front
    // of the residual share does not matter, because the share is zero.
    EXPECT_NEAR(r2_score(truth, truth), 1.0, 1e-12);

    // Predicting the mean explains none of the variance, so R^2 is exactly 0.
    const Vec mean_prediction{3.0, 3.0, 3.0, 3.0, 3.0};
    EXPECT_NEAR(r2_score(mean_prediction, truth), 0.0, 1e-9);

    // A partial fit. ss_tot = 10; these residuals are -0.5, 0.5, 0, -0.5, 0.5,
    // so ss_res = 1 and R^2 = 0.9. The wrong sign gives 1.1, which is not a
    // value R^2 can take at all.
    const Vec good{0.5, 2.5, 3.0, 3.5, 5.5};
    EXPECT_NEAR(r2_score(good, truth), 0.9, 1e-9);
    EXPECT_LE(r2_score(good, truth), 1.0);

    // Worse than the mean is negative, which is the half of the range a
    // "close to 1" assertion never reaches.
    const Vec bad{5.0, 4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(r2_score(bad, truth), -3.0, 1e-9);
    EXPECT_LT(r2_score(bad, truth), 0.0);
}

TEST(MlProperties, StandardScalerProducesUnitVarianceColumns) {
    const Mat X{
        {1.0, 10.0, -3.0},
        {2.0, 14.0, -1.0},
        {3.0, 12.0, -7.0},
        {4.0, 20.0, 5.0},
        {5.0, 11.0, 0.0},
    };
    StandardScaler scaler;
    const Mat Z = scaler.fit_transform(X);
    ASSERT_EQ(Z.size(), X.size());

    // What the scaler is FOR. A round trip through inverse_transform holds for
    // any non-zero divisor whatsoever, so it says nothing about which divisor
    // was chosen; this does.
    for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(column_mean(Z, j), 0.0, 1e-9) << "column " << j << " mean";
        EXPECT_NEAR(column_sd(Z, j), 1.0, 1e-9) << "column " << j << " sd";
        EXPECT_NEAR(scaler.mean_[static_cast<std::size_t>(j)], column_mean(X, j), 1e-9);
        EXPECT_NEAR(scaler.std_[static_cast<std::size_t>(j)], column_sd(X, j), 1e-9);
    }

    // And it still round-trips.
    const Mat back = scaler.inverse_transform(Z);
    for (std::size_t i = 0; i < X.size(); ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(back[i][static_cast<std::size_t>(j)], X[i][static_cast<std::size_t>(j)], 1e-9);
}

TEST(MlProperties, StandardScalerSurvivesAConstantColumn) {
    // A column with no spread at all. The epsilon inside the square root is
    // what keeps this from being sqrt of a negative, and a NaN here would
    // propagate silently through every downstream fit.
    const Mat X{{1.0, 7.0}, {2.0, 7.0}, {3.0, 7.0}, {4.0, 7.0}};
    StandardScaler scaler;
    const Mat Z = scaler.fit_transform(X);
    ASSERT_EQ(Z.size(), 4u);
    for (const auto& row : Z) {
        for (const double v : row) {
            EXPECT_TRUE(std::isfinite(v)) << "scaler produced " << v;
        }
    }
    EXPECT_TRUE(std::isfinite(scaler.std_[1]));
    EXPECT_GT(scaler.std_[1], 0.0);
    // The varying column is still scaled properly beside it.
    EXPECT_NEAR(column_sd(Z, 0), 1.0, 1e-9);
}

TEST(MlProperties, ReverseModeDerivativesMatchFiniteDifferences) {
    // Nothing in the suite differentiated the transcendental Vars and checked
    // the number. Each is compared against a central difference of its own
    // forward value, which is an independent computation of the same quantity.
    struct Case {
        const char* name;
        Var (*f)(const Var&);
        double (*forward)(double);
    };
    const Case cases[] = {
        {"tanh", &var_tanh, [](double x) { return std::tanh(x); }},
        {"exp", &var_exp, [](double x) { return std::exp(x); }},
        {"sigmoid", &var_sigmoid, [](double x) { return 1.0 / (1.0 + std::exp(-x)); }},
        {"sqrt", &var_sqrt, [](double x) { return std::sqrt(x); }},
        {"log", &var_log, [](double x) { return std::log(x); }},
    };
    for (const Case& c : cases) {
        for (const double x0 : {0.25, 0.7, 1.3, 2.5}) {
            Var x(x0);
            Var y = c.f(x);
            y.backward();
            const double h = 1e-6;
            const double numeric = (c.forward(x0 + h) - c.forward(x0 - h)) / (2.0 * h);
            EXPECT_NEAR(x.grad(), numeric, 1e-5)
                << c.name << " at " << x0 << ": d/dx is " << x.grad() << ", numerically "
                << numeric;
            EXPECT_NEAR(y.val(), c.forward(x0), 1e-12) << c.name << " value at " << x0;
        }
    }

    // tanh again through a composition, so the derivative is multiplied by
    // something rather than returned alone: d/dx tanh(3x) = 3 (1 - tanh^2(3x)).
    Var x(0.4);
    Var three(3.0);
    Var y = var_tanh(x * three);
    y.backward();
    const double t = std::tanh(1.2);
    EXPECT_NEAR(x.grad(), 3.0 * (1.0 - t * t), 1e-9);
}

TEST(MlProperties, KlDivergenceAcceptsTheSmallestEmbeddingItCanScore) {
    Mat X;
    for (int i = 0; i < 6; ++i) {
        X.push_back(Vec{static_cast<double>(i), static_cast<double>(i % 3), 0.5});
    }
    TSNE tsne;
    tsne.n_components = 2;
    tsne.perplexity = 2.0;

    // Two points, whose divergence is **identically zero however they are
    // embedded**: P has one pair and puts all its mass there, Q has the same one
    // pair and does the same, so KL is 1*log(1/1) whatever the distance between
    // them. Asserted at two different embeddings, because that is the claim --
    // not "it happens to be zero here" but "the two-point case carries no
    // information about the layout at all".
    //
    // It is also why the guard's boundary cannot be pinned from the outside: a
    // version refusing at two and a version computing at two return the same
    // 0.0, and the harness's survivor at that line is equivalent rather than
    // untested.
    const Mat X2{X[0], X[1]};
    const double kl_near = tsne.kl_divergence(X2, Mat{{0.0, 0.0}, {1.0, 0.0}});
    const double kl_far = tsne.kl_divergence(X2, Mat{{0.0, 0.0}, {40.0, 17.0}});
    EXPECT_TRUE(std::isfinite(kl_near));
    EXPECT_EQ(kl_near, 0.0);
    EXPECT_EQ(kl_far, kl_near);

    // One point has no pair, so zero is the right answer rather than a refusal.
    const Mat X1{X[0]};
    const Mat Y1{{0.0, 0.0}};
    EXPECT_EQ(tsne.kl_divergence(X1, Y1), 0.0);

    // And a mismatched pair is still refused with zero.
    EXPECT_EQ(tsne.kl_divergence(X, Mat{{0.0, 0.0}, {1.0, 0.0}}), 0.0);

    // Six points DO carry layout information, so the divergence separates a good
    // embedding from a scrambled one -- which is what says the zero above is a
    // property of the two-point case rather than of the function.
    Mat good, bad;
    const int scrambled[6] = {0, 3, 1, 5, 2, 4};
    for (int i = 0; i < 6; ++i) {
        good.push_back(Vec{static_cast<double>(i), 0.0});
        bad.push_back(Vec{static_cast<double>(scrambled[i]), 0.0});
    }
    const double kl_good = tsne.kl_divergence(X, good);
    const double kl_bad = tsne.kl_divergence(X, bad);
    EXPECT_TRUE(std::isfinite(kl_good));
    EXPECT_TRUE(std::isfinite(kl_bad));
    EXPECT_GT(kl_good, 0.0);
    EXPECT_NE(kl_good, kl_bad);
}

TEST(MlProperties, GaussianMixtureRecoversTheVariancesItWasGiven) {
    // Two well-separated components with KNOWN and DIFFERENT per-feature
    // variances, arranged so each feature discriminates: feature 0 is tight in
    // the first component and wide in the second, feature 1 the other way
    // about. A variance accumulator that starts at 1 inflates every one of
    // them; a loop that starts at feature 1 leaves feature 0 at the floor.
    Mat X;
    const double a_spread[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (int rep = 0; rep < 4; ++rep) {
        for (const double s : a_spread) {
            X.push_back(Vec{0.5 * s, 4.0 * s});          // component A
            X.push_back(Vec{100.0 + 4.0 * s, 0.5 * s});  // component B
        }
    }

    GaussianMixture gmm;
    gmm.config.n_components = 2;
    gmm.config.max_iter = 300;
    gmm.config.seed = 7;
    gmm.fit(X);

    ASSERT_EQ(gmm.variances.size(), 2u);
    ASSERT_EQ(gmm.means.size(), 2u);

    // Identify the components by where their means are, not by index: EM's
    // labelling depends on the initialisation.
    const std::size_t near_zero = (std::abs(gmm.means[0][0]) < 50.0) ? 0u : 1u;
    const std::size_t near_hundred = 1u - near_zero;
    EXPECT_NEAR(gmm.means[near_zero][0], 0.0, 1e-6);
    EXPECT_NEAR(gmm.means[near_hundred][0], 100.0, 1e-6);

    // The sample variances of the generated data: 0.5^2 * 2 and 4^2 * 2, where 2
    // is the variance of {-2,-1,0,1,2}. The components are perfectly separated,
    // so EM's responsibilities are 0 and 1 and it recovers them EXACTLY -- which
    // is worth asserting at 1e-6 rather than loosely, because the accumulator
    // this is here to pin is off by 1/n_component_mass, about 0.05 here, and a
    // tolerance wide enough to be comfortable is wide enough to miss it.
    EXPECT_NEAR(gmm.variances[near_zero][0], 0.25 * 2.0, 1e-6) << "component A, feature 0";
    EXPECT_NEAR(gmm.variances[near_zero][1], 16.0 * 2.0, 1e-6) << "component A, feature 1";
    EXPECT_NEAR(gmm.variances[near_hundred][0], 16.0 * 2.0, 1e-6) << "component B, feature 0";
    EXPECT_NEAR(gmm.variances[near_hundred][1], 0.25 * 2.0, 1e-6) << "component B, feature 1";

    for (const auto& v : gmm.variances) {
        for (const double x : v) {
            EXPECT_GT(x, 0.0);
            EXPECT_TRUE(std::isfinite(x));
        }
    }
    EXPECT_NEAR(gmm.weights[0] + gmm.weights[1], 1.0, 1e-9);
    EXPECT_TRUE(std::isfinite(gmm.log_likelihood));
}

TEST(MlProperties, GaussianMixtureScoresEveryFeature) {
    // Two components that differ ONLY in feature 0. If the per-point density
    // skipped that feature there would be nothing left to tell them apart, so
    // the assignment of a point far out along feature 0 is the test.
    Mat X;
    for (int i = 0; i < 12; ++i) {
        const double j = static_cast<double>(i % 4) - 1.5;
        X.push_back(Vec{-20.0 + j, j});
        X.push_back(Vec{20.0 + j, j});
    }
    GaussianMixture gmm;
    gmm.config.n_components = 2;
    gmm.config.max_iter = 300;
    gmm.config.seed = 11;
    gmm.fit(X);

    ASSERT_EQ(gmm.means.size(), 2u);
    const double m0 = gmm.means[0][0];
    const double m1 = gmm.means[1][0];
    EXPECT_GT(std::abs(m0 - m1), 20.0) << "the components did not separate along feature 0";

    const auto labels = gmm.predict(Mat{Vec{-20.0, 0.0}, Vec{20.0, 0.0}});
    ASSERT_EQ(labels.size(), 2u);
    EXPECT_NE(labels[0], labels[1]) << "two points 40 apart landed in one component";
}
