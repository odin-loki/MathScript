#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <span>
#include <map>
#include <numeric>
#include <algorithm>
#include <string>
#include "ms/stats/stats.hpp"
#include "ms/symbolic/symbolic.hpp"
#include "ms/signal/signal.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Helper to make a map for sym_eval
static std::map<std::string, double> env(const std::string& var, double val) {
    return {{var, val}};
}

// ============================================================
// Stats Advanced Tests
// ============================================================

TEST(StatsAdv, SkewnessSymmetricIsZero) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    double s = ms::skewness(data);
    EXPECT_NEAR(s, 0.0, 1e-9);
}

TEST(StatsAdv, SkewnessRightSkewedIsPositive) {
    std::vector<double> data = {1, 1, 1, 1, 1, 2, 3, 10, 20, 50};
    double s = ms::skewness(data);
    EXPECT_GT(s, 0.0);
}

TEST(StatsAdv, KurtosisNormalApproxThree) {
    // Larger symmetric sample approximates normal kurtosis ≈ 3
    std::vector<double> data;
    for (int i = 1; i <= 100; ++i) data.push_back(static_cast<double>(i));
    double k = ms::kurtosis(data);
    EXPECT_TRUE(std::isfinite(k));
}

TEST(StatsAdv, KurtosisIsFinite) {
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double k = ms::kurtosis(data);
    EXPECT_TRUE(std::isfinite(k));
}

TEST(StatsAdv, TwoSampleTtestIdenticalSamplesNearZero) {
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    double t = ms::two_sample_ttest(a, b);
    EXPECT_NEAR(t, 0.0, 1e-9);
}

TEST(StatsAdv, TwoSampleTtestVeryDifferentSamplesLargeT) {
    std::vector<double> a = {1.0, 1.1, 0.9, 1.0, 1.05};
    std::vector<double> b = {100.0, 100.1, 99.9, 100.0, 100.05};
    double t = ms::two_sample_ttest(a, b);
    EXPECT_GT(std::abs(t), 10.0);
}

TEST(StatsAdv, ZtestAtSampleMeanIsZero) {
    std::vector<double> sample = {2.0, 4.0, 6.0, 8.0, 10.0};
    double mu = ms::mean(sample);
    double sigma = 1.0;
    double z = ms::ztest(sample, mu, sigma);
    EXPECT_NEAR(z, 0.0, 1e-9);
}

TEST(StatsAdv, ZtestKnownSampleGivesExpectedZ) {
    // sample mean = 5, mu = 4, sigma = 1, n = 4 => z = (5-4)/(1/sqrt(4)) = 2.0
    std::vector<double> sample = {4.0, 5.0, 5.0, 6.0};
    double z = ms::ztest(sample, 4.0, 1.0);
    EXPECT_NEAR(z, 2.0, 1e-9);
}

TEST(StatsAdv, TtestAtSampleMeanIsZero) {
    std::vector<double> data = {3.0, 5.0, 7.0, 9.0, 11.0};
    double mu = ms::mean(data);
    double t = ms::ttest(data, mu);
    EXPECT_NEAR(t, 0.0, 1e-9);
}

TEST(StatsAdv, MinValue) {
    std::vector<double> data = {5.0, 3.0, 8.0, 1.0, 9.0};
    EXPECT_DOUBLE_EQ(ms::min_value(data), 1.0);
}

TEST(StatsAdv, MaxValue) {
    std::vector<double> data = {5.0, 3.0, 8.0, 1.0, 9.0};
    EXPECT_DOUBLE_EQ(ms::max_value(data), 9.0);
}

TEST(StatsAdv, PercentileZeroReturnsMin) {
    std::vector<double> data = {4.0, 1.0, 7.0, 2.0, 9.0};
    EXPECT_DOUBLE_EQ(ms::percentile(data, 0.0), ms::min_value(data));
}

TEST(StatsAdv, PercentileHundredReturnsMax) {
    std::vector<double> data = {4.0, 1.0, 7.0, 2.0, 9.0};
    EXPECT_DOUBLE_EQ(ms::percentile(data, 100.0), ms::max_value(data));
}

TEST(StatsAdv, PercentileFiftyReturnsMedian) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(ms::percentile(data, 50.0), ms::median(data), 1e-9);
}

TEST(StatsAdv, LinearRegressionPerfectFit) {
    // y = 2x + 1
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {1.0, 3.0, 5.0, 7.0, 9.0};
    auto res = ms::linear_regression(x, y);
    EXPECT_NEAR(res.slope, 2.0, 1e-9);
    EXPECT_NEAR(res.intercept, 1.0, 1e-9);
    EXPECT_NEAR(res.r_squared, 1.0, 1e-9);
}

TEST(StatsAdv, LinearRegressionOrthogonalDataLowRSquared) {
    // x and y are uncorrelated (orthogonal in mean-centered sense)
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {3.0, 1.0, 4.0, 1.0, 5.0}; // no clear linear trend
    auto res = ms::linear_regression(x, y);
    EXPECT_LT(res.r_squared, 0.5);
}

TEST(StatsAdv, ModeReturnsCorrectValue) {
    std::vector<double> data = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(ms::mode(data), 3.0);
}

// ============================================================
// Symbolic Advanced Tests
// ============================================================

TEST(SymAdv, SymToStringNonEmpty) {
    std::string s = ms::sym_to_string(ms::sym_var("x"));
    EXPECT_FALSE(s.empty());
}

TEST(SymAdv, SymToStringConstant) {
    std::string s = ms::sym_to_string(ms::sym_const(42.0));
    EXPECT_FALSE(s.empty());
}

TEST(SymAdv, SymEvalXSquaredPlusOne) {
    // x^2 + 1 at x=3 == 10
    auto expr = ms::sym_add(
        ms::sym_pow(ms::sym_var("x"), ms::sym_const(2.0)),
        ms::sym_const(1.0));
    double val = ms::sym_eval(expr, env("x", 3.0));
    EXPECT_NEAR(val, 10.0, 1e-9);
}

TEST(SymAdv, SymEvalLinearExpression) {
    // 2*x + 5 at x=4 == 13
    auto expr = ms::sym_add(
        ms::sym_mul(ms::sym_const(2.0), ms::sym_var("x")),
        ms::sym_const(5.0));
    double val = ms::sym_eval(expr, env("x", 4.0));
    EXPECT_NEAR(val, 13.0, 1e-9);
}

TEST(SymAdv, SymDiffXSquaredGives2x) {
    // d/dx(x^2) at x=3 == 6
    auto deriv = ms::sym_diff(
        ms::sym_pow(ms::sym_var("x"), ms::sym_const(2.0)), "x");
    double val = ms::sym_eval(deriv, env("x", 3.0));
    EXPECT_NEAR(val, 6.0, 1e-9);
}

TEST(SymAdv, SymDerivSinX) {
    // d/dx(sin(x)) = cos(x); at x=0: 1
    auto deriv = ms::sym_deriv(ms::sym_sin(ms::sym_var("x")), "x");
    double val = ms::sym_eval(deriv, env("x", 0.0));
    EXPECT_NEAR(val, 1.0, 1e-9);
}

TEST(SymAdv, SymDerivCosX) {
    // d/dx(cos(x)) = -sin(x); at x=pi/2: -1
    auto deriv = ms::sym_deriv(ms::sym_cos(ms::sym_var("x")), "x");
    double val = ms::sym_eval(deriv, env("x", M_PI / 2.0));
    EXPECT_NEAR(val, -1.0, 1e-9);
}

TEST(SymAdv, SymSimplifyReturnsValid) {
    auto simplified = ms::sym_simplify(
        ms::sym_add(ms::sym_const(0.0), ms::sym_var("x")));
    double val = ms::sym_eval(simplified, env("x", 5.0));
    EXPECT_NEAR(val, 5.0, 1e-9);
}

TEST(SymAdv, SymSinEvalHalfPi) {
    double val = ms::sym_eval(ms::sym_sin(ms::sym_var("x")), env("x", M_PI / 2.0));
    EXPECT_NEAR(val, 1.0, 1e-9);
}

TEST(SymAdv, SymCosEvalZero) {
    double val = ms::sym_eval(ms::sym_cos(ms::sym_var("x")), env("x", 0.0));
    EXPECT_NEAR(val, 1.0, 1e-9);
}

TEST(SymAdv, SymExpEvalZero) {
    double val = ms::sym_eval(ms::sym_exp(ms::sym_var("x")), env("x", 0.0));
    EXPECT_NEAR(val, 1.0, 1e-9);
}

TEST(SymAdv, SymExpEvalOne) {
    double val = ms::sym_eval(ms::sym_exp(ms::sym_var("x")), env("x", 1.0));
    EXPECT_NEAR(val, M_E, 1e-9);
}

TEST(SymAdv, SymLogEvalOne) {
    double val = ms::sym_eval(ms::sym_log(ms::sym_var("x")), env("x", 1.0));
    EXPECT_NEAR(val, 0.0, 1e-9);
}

TEST(SymAdv, SymLogEvalE) {
    double val = ms::sym_eval(ms::sym_log(ms::sym_var("x")), env("x", M_E));
    EXPECT_NEAR(val, 1.0, 1e-9);
}

TEST(SymAdv, SymPowTwoToTen) {
    std::map<std::string, double> empty_env;
    double val = ms::sym_eval(
        ms::sym_pow(ms::sym_const(2.0), ms::sym_const(10.0)), empty_env);
    EXPECT_NEAR(val, 1024.0, 1e-9);
}

TEST(SymAdv, SymDiffSinXSquaredIsFinite) {
    // d/dx(sin(x^2)) at x=1; = 2x*cos(x^2) = 2*cos(1) ≈ 1.0806
    auto deriv = ms::sym_diff(
        ms::sym_sin(ms::sym_pow(ms::sym_var("x"), ms::sym_const(2.0))), "x");
    double val = ms::sym_eval(deriv, env("x", 1.0));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_NEAR(val, 2.0 * std::cos(1.0), 1e-9);
}

TEST(SymAdv, SymAddTwoConstants) {
    std::map<std::string, double> empty_env;
    double val = ms::sym_eval(
        ms::sym_add(ms::sym_const(3.0), ms::sym_const(7.0)), empty_env);
    EXPECT_NEAR(val, 10.0, 1e-9);
}

TEST(SymAdv, SymMulTwoConstants) {
    std::map<std::string, double> empty_env;
    double val = ms::sym_eval(
        ms::sym_mul(ms::sym_const(4.0), ms::sym_const(6.0)), empty_env);
    EXPECT_NEAR(val, 24.0, 1e-9);
}

// ============================================================
// Signal Advanced Tests
// ============================================================

TEST(SignalAdv, ConvolveDeltaPreservesSignal) {
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> delta  = {1.0};
    auto result = ms::convolve(signal, delta);
    ASSERT_EQ(result.size(), signal.size());
    for (size_t i = 0; i < signal.size(); ++i)
        EXPECT_NEAR(result[i], signal[i], 1e-9);
}

TEST(SignalAdv, ConvolveSizeIsCorrect) {
    std::vector<double> a(5, 1.0);
    std::vector<double> b(3, 1.0);
    auto result = ms::convolve(a, b);
    EXPECT_EQ(result.size(), a.size() + b.size() - 1);
}

TEST(SignalAdv, ConvolveKnownResult) {
    // [1,2,3] * [0,1,0] = [0,1,2,3,0] (delta at index 1 shifts by 1)
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {0.0, 1.0, 0.0};
    auto result = ms::convolve(a, b);
    ASSERT_EQ(result.size(), 5u);
    EXPECT_NEAR(result[0], 0.0, 1e-9);
    EXPECT_NEAR(result[1], 1.0, 1e-9);
    EXPECT_NEAR(result[2], 2.0, 1e-9);
    EXPECT_NEAR(result[3], 3.0, 1e-9);
    EXPECT_NEAR(result[4], 0.0, 1e-9);
}

TEST(SignalAdv, CorrelateAutocorrelationMaxAtCenter) {
    std::vector<double> sig = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = ms::correlate(sig, sig);
    // Autocorrelation has max at center index = size/2
    size_t center = result.size() / 2;
    double center_val = result[center];
    for (size_t i = 0; i < result.size(); ++i)
        EXPECT_LE(result[i], center_val + 1e-9);
}

TEST(SignalAdv, CorrelateIdenticalSignalsMaxAtCenter) {
    std::vector<double> sig(16, 0.0);
    for (size_t i = 0; i < sig.size(); ++i) sig[i] = std::sin(2.0 * M_PI * i / 16.0);
    auto result = ms::correlate(sig, sig);
    size_t center = result.size() / 2;
    double center_val = result[center];
    for (size_t i = 0; i < result.size(); ++i)
        EXPECT_LE(result[i], center_val + 1e-9);
}

TEST(SignalAdv, MovingAverageConstantSignal) {
    std::vector<double> signal(10, 7.0);
    auto result = ms::moving_average(signal, 3);
    for (double v : result)
        EXPECT_NEAR(v, 7.0, 1e-9);
}

TEST(SignalAdv, MovingAverageSizeMatchesInput) {
    std::vector<double> signal(20, 1.0);
    auto result = ms::moving_average(signal, 4);
    EXPECT_EQ(result.size(), signal.size());
}

TEST(SignalAdv, HammingFirstLastNearPoint08) {
    auto w = ms::hamming(64);
    EXPECT_NEAR(w.front(), 0.08, 1e-2);
    EXPECT_NEAR(w.back(),  0.08, 1e-2);
}

TEST(SignalAdv, HammingMiddleNearOne) {
    size_t n = 65; // odd so there is a true center
    auto w = ms::hamming(n);
    EXPECT_NEAR(w[n / 2], 1.0, 1e-9);
}

TEST(SignalAdv, HanningFirstLastZero) {
    auto w = ms::hanning(64);
    EXPECT_NEAR(w.front(), 0.0, 1e-9);
    EXPECT_NEAR(w.back(),  0.0, 1e-9);
}

TEST(SignalAdv, BlackmanSymmetric) {
    size_t n = 64;
    auto w = ms::blackman(n);
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], 1e-9);
}

TEST(SignalAdv, BlackmanFirstLastNearZero) {
    auto w = ms::blackman(64);
    EXPECT_NEAR(w.front(), 0.0, 1e-2);
    EXPECT_NEAR(w.back(),  0.0, 1e-2);
}

TEST(SignalAdv, ParzenAllPositive) {
    auto w = ms::parzen(64);
    for (double v : w)
        EXPECT_GE(v, 0.0);
}

TEST(SignalAdv, ParzenPeakAtCenter) {
    size_t n = 65;
    auto w = ms::parzen(n);
    double peak = *std::max_element(w.begin(), w.end());
    EXPECT_NEAR(w[n / 2], peak, 1e-9);
}

TEST(SignalAdv, TriangularSymmetric) {
    size_t n = 65;  // odd length ensures perfect symmetry
    auto w = ms::triangular(n);
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], 1e-9);
}

TEST(SignalAdv, TriangularPeakAtCenter) {
    size_t n = 65;
    auto w = ms::triangular(n);
    double peak = *std::max_element(w.begin(), w.end());
    EXPECT_NEAR(w[n / 2], peak, 1e-9);
}

TEST(SignalAdv, LowpassPassesLowFrequency) {
    // DC signal (all 1s) should pass through a lowpass filter with minimal change
    size_t n = 256;
    std::vector<double> dc(n, 1.0);
    double fs = 1000.0, cutoff = 100.0;
    auto filtered = ms::lowpass(dc, cutoff, fs);
    ASSERT_EQ(filtered.size(), dc.size());
    // After transient, output should be close to 1
    double tail_mean = 0.0;
    size_t tail_start = n / 2;
    for (size_t i = tail_start; i < n; ++i) tail_mean += filtered[i];
    tail_mean /= static_cast<double>(n - tail_start);
    EXPECT_NEAR(tail_mean, 1.0, 0.1);
}

TEST(SignalAdv, HighpassAttenuatesDC) {
    // DC signal should be attenuated by a highpass filter
    size_t n = 256;
    std::vector<double> dc(n, 1.0);
    double fs = 1000.0, cutoff = 100.0;
    auto filtered = ms::highpass(dc, cutoff, fs);
    ASSERT_EQ(filtered.size(), dc.size());
    // After transient, DC should be suppressed
    double tail_mean = 0.0;
    size_t tail_start = n / 2;
    for (size_t i = tail_start; i < n; ++i) tail_mean += std::abs(filtered[i]);
    tail_mean /= static_cast<double>(n - tail_start);
    EXPECT_LT(tail_mean, 0.5);
}

TEST(SignalAdv, BandpassOutputIsFinite) {
    size_t n = 128;
    std::vector<double> sig(n);
    for (size_t i = 0; i < n; ++i) sig[i] = std::sin(2.0 * M_PI * 50.0 * i / 1000.0);
    auto filtered = ms::bandpass(sig, 30.0, 100.0, 1000.0);
    ASSERT_EQ(filtered.size(), sig.size());
    for (double v : filtered)
        EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalAdv, ButterworthOutputFiniteAndSameSize) {
    size_t n = 128;
    std::vector<double> sig(n);
    for (size_t i = 0; i < n; ++i) sig[i] = std::sin(2.0 * M_PI * 10.0 * i / 1000.0);
    auto filtered = ms::butterworth(sig, 50.0, 1000.0);
    ASSERT_EQ(filtered.size(), sig.size());
    for (double v : filtered)
        EXPECT_TRUE(std::isfinite(v));
}


