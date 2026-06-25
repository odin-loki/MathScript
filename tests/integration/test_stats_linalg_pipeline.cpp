// MathScript Integration Tests: Stats + Linalg + Signal Pipeline
// End-to-end pipeline: generate data, run stats, fit, solve, filter

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <span>
#include <numeric>
#include <algorithm>

#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"
#include "ms/signal/signal.hpp"
#include "ms/fft/fft.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// Stats → LinearRegression → Verify
// ---------------------------------------------------------------------------

TEST(StatsLinalgPipeline, LinearRegressionOnSineData) {
    // Generate y ≈ 2x + 1 with slight noise
    const int N = 50;
    std::vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<double>(i) / 10.0;
        y[i] = 2.0 * x[i] + 1.0;  // perfect linear
    }
    auto reg = linear_regression(x, y);
    EXPECT_NEAR(reg.slope, 2.0, 1e-6);
    EXPECT_NEAR(reg.intercept, 1.0, 1e-6);
    EXPECT_NEAR(reg.r_squared, 1.0, 1e-6);
}

TEST(StatsLinalgPipeline, PredictiveStats_SampleVsPopulation) {
    // Normal-like data: mean=5, stddev≈1
    std::vector<double> data = {4.0, 4.5, 5.0, 5.0, 5.5, 6.0, 5.2, 4.8, 5.1, 4.9};
    double m = mean(data);
    double s = stddev(data);
    EXPECT_NEAR(m, 5.0, 0.3);
    EXPECT_GT(s, 0.0);

    // t-test against true mean 5.0
    double t = ttest(data, 5.0);
    // t should be small (near 0) since sample mean ≈ 5.0
    EXPECT_LT(std::abs(t), 3.0);
}

TEST(StatsLinalgPipeline, CorrelationPipeline) {
    // Perfectly correlated data
    std::vector<double> x(20), y(20);
    for (int i = 0; i < 20; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = 3.0 * x[i] + 2.0;
    }
    double r = correlation(x, y);
    EXPECT_NEAR(r, 1.0, 1e-8);

    // Perfectly anti-correlated
    std::vector<double> z(20);
    for (int i = 0; i < 20; ++i) z[i] = -3.0 * x[i];
    double r2 = correlation(x, z);
    EXPECT_NEAR(r2, -1.0, 1e-8);
}

TEST(StatsLinalgPipeline, PercentileRanking) {
    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) data[i] = static_cast<double>(i + 1);
    double p25 = percentile(data, 25.0);
    double p50 = percentile(data, 50.0);
    double p75 = percentile(data, 75.0);
    EXPECT_LT(p25, p50);
    EXPECT_LT(p50, p75);
    EXPECT_GT(p25, 20.0);
    EXPECT_LT(p75, 80.0);
}

// ---------------------------------------------------------------------------
// Signal → FFT → Stats pipeline
// ---------------------------------------------------------------------------

TEST(StatsLinalgPipeline, HammingWindow_Then_FFT) {
    const size_t N = 64;
    auto window = hamming(N);
    ASSERT_EQ(window.size(), N);

    // Apply window to a sine wave at bin 8
    std::vector<double> signal(N);
    for (size_t i = 0; i < N; ++i) {
        double t = 2.0 * M_PI * 8.0 * i / N;
        signal[i] = std::sin(t) * window[i];
    }

    auto spectrum = fft(signal);
    ASSERT_TRUE(spectrum.has_value());
    EXPECT_EQ(spectrum.value().size(), N);

    // Find peak frequency bin
    size_t peak_bin = 0;
    double peak_mag = 0.0;
    for (size_t i = 0; i < N/2; ++i) {
        double mag = std::abs(spectrum.value()[i]);
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_bin = i;
        }
    }
    EXPECT_EQ(peak_bin, 8u) << "Peak should be at bin 8";
}

TEST(StatsLinalgPipeline, MovingAverage_Smoothing_Effect) {
    // Moving average should reduce variance
    std::vector<double> noisy(100);
    for (int i = 0; i < 100; ++i) {
        noisy[i] = (double)(i % 5) - 2.0;  // alternating
    }
    auto smoothed = moving_average(noisy, 5);
    ASSERT_EQ(smoothed.size(), noisy.size());

    double var_original = var(noisy);
    double var_smoothed = var(smoothed);
    EXPECT_LT(var_smoothed, var_original) << "Smoothed signal should have lower variance";
}

TEST(StatsLinalgPipeline, LowpassFilter_OutputFinite) {
    const size_t N = 64;
    const double fs = 1000.0;
    const double cutoff = 100.0;

    std::vector<double> signal(N);
    for (size_t i = 0; i < N; ++i) {
        signal[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
    }

    auto filtered = lowpass(signal, cutoff, fs);
    ASSERT_EQ(filtered.size(), N);
    for (double v : filtered) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(StatsLinalgPipeline, Convolve_BoxFilter_OutputSize) {
    // Convolution output size = len(a) + len(b) - 1
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    std::vector<double> kernel = {1.0/3.0, 1.0/3.0, 1.0/3.0};

    auto convolved = convolve(signal, kernel);
    EXPECT_EQ(convolved.size(), signal.size() + kernel.size() - 1);

    // All output values should be finite
    for (double v : convolved) {
        EXPECT_TRUE(std::isfinite(v));
    }

    // Moving average output should also be valid
    auto averaged = moving_average(signal, 3);
    EXPECT_EQ(averaged.size(), signal.size());
    for (double v : averaged) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ---------------------------------------------------------------------------
// Stats → Linalg: PCA-like operations
// ---------------------------------------------------------------------------

TEST(StatsLinalgPipeline, CovarianceMatrix_ThenEigSym) {
    // Build a 5x2 data matrix with known covariance structure
    DMatrix data({{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {4.0, 8.0}, {5.0, 10.0}});

    // Compute sample mean (column-wise)
    double mean_x = 3.0, mean_y = 6.0;

    // Build 2x2 covariance matrix
    DMatrix cov(2, 2, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        double dx = data(i, 0) - mean_x;
        double dy = data(i, 1) - mean_y;
        cov(0, 0) += dx * dx;
        cov(0, 1) += dx * dy;
        cov(1, 0) += dx * dy;
        cov(1, 1) += dy * dy;
    }
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            cov(i, j) /= 4.0;  // sample covariance

    // Eigendecompose covariance matrix
    auto eig_result = eig_sym(cov);
    ASSERT_TRUE(eig_result.has_value());

    // Eigenvalues should be non-negative for semi-definite covariance matrix
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_GE(eig_result.value().values(i, 0), -1e-8);
    }
}

TEST(StatsLinalgPipeline, VarAndStddev_Relationship) {
    // stddev^2 == var
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double v = var(data);
    double sd = stddev(data);
    EXPECT_NEAR(sd * sd, v, 1e-10);
    EXPECT_GT(v, 0.0);
    EXPECT_GT(sd, 0.0);
}

TEST(StatsLinalgPipeline, SkewnessKurtosisFinite) {
    // Symmetric data: skewness ≈ 0
    std::vector<double> symmetric = {-3,-2,-1,-1,0,0,0,0,1,1,1,2,3};
    double sk = skewness(symmetric);
    double ku = kurtosis(symmetric);
    EXPECT_TRUE(std::isfinite(sk));
    EXPECT_TRUE(std::isfinite(ku));
    EXPECT_NEAR(sk, 0.0, 0.5);  // approximately symmetric

    // Right-skewed data should have positive skewness
    std::vector<double> right_skewed = {1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 10.0, 20.0};
    double sk2 = skewness(right_skewed);
    EXPECT_TRUE(std::isfinite(sk2));
    EXPECT_GT(sk2, 0.0) << "right-skewed data should have positive skewness";
}

// ---------------------------------------------------------------------------
// Multi-module: ODE → stats → signal
// ---------------------------------------------------------------------------

TEST(StatsLinalgPipeline, Signal_Correlate_AutocorrIsSymmetric) {
    std::vector<double> sig = {1.0, 2.0, 3.0, 2.0, 1.0};
    auto ac = correlate(sig, sig);

    EXPECT_EQ(ac.size(), 2 * sig.size() - 1);

    // Autocorrelation should be symmetric: ac[i] ≈ ac[N-1-i]
    size_t N = ac.size();
    for (size_t i = 0; i < N/2; ++i) {
        EXPECT_NEAR(ac[i], ac[N - 1 - i], 1e-9);
    }
    // Peak should be at center
    double center = ac[N/2];
    for (double v : ac) {
        EXPECT_LE(v, center + 1e-9);
    }
}

TEST(StatsLinalgPipeline, RFFT_Then_Stats) {
    const size_t N = 128;
    std::vector<double> time_signal(N);
    for (size_t i = 0; i < N; ++i) {
        time_signal[i] = std::sin(2.0 * M_PI * 4.0 * i / N);
    }

    auto spectrum = rfft(time_signal);
    ASSERT_TRUE(spectrum.has_value());

    // Compute magnitude spectrum
    std::vector<double> magnitudes;
    for (const auto& c : spectrum.value()) {
        magnitudes.push_back(std::abs(c));
    }

    double m = mean(magnitudes);
    double max_m = max_value(magnitudes);
    EXPECT_TRUE(std::isfinite(m));
    EXPECT_GT(max_m, 0.0);

    // Peak should be at bin 4
    size_t peak_bin = std::max_element(magnitudes.begin(), magnitudes.end()) - magnitudes.begin();
    EXPECT_EQ(peak_bin, 4u) << "RFFT peak should be at bin 4 for 4 Hz sine";
}
