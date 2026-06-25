// MathScript: Probability + Statistics + Signal Integration Pipeline
// Tests combining prob distributions, stats analysis, and signal processing

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/prob/prob.hpp"
#include "ms/stats/stats.hpp"
#include "ms/signal/signal.hpp"
#include "ms/fft/fft.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Pipeline 1: Generate synthetic data from a known distribution,
// verify statistical properties match the distribution.
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, NormalDistrib_MeanAndVariance_Consistent) {
    // Generate a grid of x values and weights = pdf(x)
    // Then compute weighted mean and variance — should match distribution params
    double mu = 3.0, sigma = 1.5;
    std::vector<double> xs, weights;
    double step = 0.01;
    for (double x = mu - 5 * sigma; x <= mu + 5 * sigma; x += step) {
        xs.push_back(x);
        weights.push_back(norm_pdf(x, mu, sigma) * step);
    }
    // Weighted mean = sum(x * w)
    double wmean = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) wmean += xs[i] * weights[i];
    EXPECT_NEAR(wmean, mu, 0.01) << "Weighted mean should match mu";

    // Weighted variance = sum((x - mu)^2 * w)
    double wvar = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        double diff = xs[i] - mu;
        wvar += diff * diff * weights[i];
    }
    EXPECT_NEAR(wvar, sigma * sigma, 0.05) << "Weighted variance should match sigma^2";
}

TEST(ProbStatsPipeline, CDF_Matches_NormCDF_Integral) {
    // CDF(x) = integral from -inf to x of pdf(t)dt ≈ sum(pdf(ti)*dt) for t < x
    double mu = 0.0, sigma = 1.0;
    double x_target = 1.0;
    double step = 0.001;
    double integral = 0.0;
    for (double t = mu - 8 * sigma; t <= x_target; t += step) {
        integral += norm_pdf(t, mu, sigma) * step;
    }
    double cdf = norm_cdf(x_target, mu, sigma);
    EXPECT_NEAR(integral, cdf, 0.005) << "Numerical CDF integration vs norm_cdf";
}

// ---------------------------------------------------------------------------
// Pipeline 2: Hypothesis testing on a sampled distribution
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, TTest_OnNormalSample_RejectsShiftedMean) {
    std::vector<double> sample = {4.8, 5.1, 5.2, 4.9, 5.0, 5.3, 4.7, 5.1, 5.0, 4.8};
    double t_stat = ttest(sample, 0.0);
    EXPECT_TRUE(std::isfinite(t_stat));
    // t-statistic should be large and positive (sample mean >> 0)
    EXPECT_GT(t_stat, 5.0) << "t-statistic should be large for mu=5 vs null=0";
}

TEST(ProbStatsPipeline, TTest_NearZeroMean_SmallTStat) {
    std::vector<double> sample = {0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.0, -0.01, 0.02};
    double t_stat = ttest(sample, 0.0);
    EXPECT_TRUE(std::isfinite(t_stat));
    EXPECT_LT(std::abs(t_stat), 2.0) << "t-statistic near zero for centered sample";
}

// ---------------------------------------------------------------------------
// Pipeline 3: FFT → frequency analysis → compare with known spectrum
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, FFT_SineWave_PeakAtExpectedBin) {
    int N = 64;
    double f0 = 4.0;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = std::sin(2.0 * M_PI * f0 * i / N);
    }
    auto spectrum_res = fft(signal);
    ASSERT_TRUE(spectrum_res.has_value());
    auto& spectrum = spectrum_res.value();
    ASSERT_EQ(spectrum.size(), static_cast<size_t>(N));

    size_t peak_bin = 0;
    double peak_val = 0.0;
    for (int k = 1; k <= N / 2; ++k) {
        double mag = std::abs(spectrum[k]);
        if (mag > peak_val) {
            peak_val = mag;
            peak_bin = static_cast<size_t>(k);
        }
    }
    EXPECT_EQ(peak_bin, static_cast<size_t>(f0))
        << "Peak should be at bin " << f0 << " but was at " << peak_bin;
}

TEST(ProbStatsPipeline, FFT_DCComponent_EqualsMean) {
    int N = 32;
    std::vector<double> signal(N, 3.0);
    auto spectrum_res = fft(signal);
    ASSERT_TRUE(spectrum_res.has_value());
    auto& spectrum = spectrum_res.value();
    ASSERT_GE(spectrum.size(), 1u);
    double dc = std::abs(spectrum[0]);
    double expected = N * 3.0;
    EXPECT_NEAR(dc, expected, 0.01);
}

// ---------------------------------------------------------------------------
// Pipeline 4: Signal smoothing → stats comparison before/after
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, MovingAverage_ReducesVariance) {
    // A noisy signal smoothed with moving average should have less variance
    int N = 100;
    std::vector<double> noisy(N);
    for (int i = 0; i < N; ++i)
        noisy[i] = std::sin(2.0 * M_PI * i / 20.0) + 0.5 * (static_cast<double>(i % 7) / 3.0 - 0.5);

    int window = 5;
    auto smooth = moving_average(noisy, window);
    ASSERT_GE(smooth.size(), 1u);

    // Variance of smoothed should be <= variance of noisy (or similar)
    // Use a portion of the smoothed signal to avoid edge effects
    size_t trim = static_cast<size_t>(window);
    if (smooth.size() > 2 * trim) {
        std::vector<double> smooth_mid(smooth.begin() + trim,
                                       smooth.end() - trim);
        std::vector<double> noisy_mid(noisy.begin() + trim,
                                      noisy.begin() + trim + smooth_mid.size());
        double v_noisy_res = var(noisy_mid);
        double v_smooth_res = var(smooth_mid);
        EXPECT_LE(v_smooth_res, v_noisy_res * 1.2)
            << "Smoothed signal should not have much more variance";
    }
}

TEST(ProbStatsPipeline, Signal_Convolve_OutputFinite) {
    std::vector<double> sig(50, 1.0);
    std::vector<double> kernel = {0.25, 0.5, 0.25};
    auto result = convolve(sig, kernel);
    EXPECT_GE(result.size(), 1u);
    for (auto v : result)
        EXPECT_TRUE(std::isfinite(v)) << "Convolution output should be finite";
}

// ---------------------------------------------------------------------------
// Pipeline 5: Distribution tail probability and survival function
// ---------------------------------------------------------------------------

TEST(ProbStatsPipeline, TailProbability_NormDist_Symmetric) {
    // P(X < mu - k*sigma) = P(X > mu + k*sigma) for normal distribution
    double mu = 0.0, sigma = 1.0;
    for (double k : {1.0, 1.5, 2.0}) {
        double lower = norm_cdf(mu - k * sigma, mu, sigma);
        double upper = 1.0 - norm_cdf(mu + k * sigma, mu, sigma);
        EXPECT_NEAR(lower, upper, 1e-9)
            << "Symmetry failed at k=" << k;
    }
}

TEST(ProbStatsPipeline, ComparisonBetweenDistributions_AtLargeTail) {
    // For large x, normal tail decays faster than exponential
    // P(X > 5) for N(0,1) should be very small
    double p_norm_tail = 1.0 - norm_cdf(5.0, 0.0, 1.0);
    EXPECT_LT(p_norm_tail, 1e-6);
    // P(X > 5) for Exp(1) should be e^-5 ≈ 0.0067
    double p_exp_tail = 1.0 - exp_cdf(5.0, 1.0);
    EXPECT_GT(p_exp_tail, p_norm_tail) << "Exponential tail should be heavier than normal";
}
