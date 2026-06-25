#pragma once

#include <functional>
#include <span>
#include <tuple>
#include <vector>

namespace ms {

// --- Descriptive ---
double mean(std::span<const double> data);
double var(std::span<const double> data);
double stddev(std::span<const double> data);
double median(std::span<const double> data);
double min_value(std::span<const double> data);
double max_value(std::span<const double> data);
double mode(std::span<const double> data);
double percentile(std::span<const double> data, double p);
double skewness(std::span<const double> data);
double kurtosis(std::span<const double> data);
double geometric_mean(std::span<const double> data);
double harmonic_mean(std::span<const double> data);
double rms(std::span<const double> data);
double mad(std::span<const double> data);     // median absolute deviation
double iqr(std::span<const double> data);     // interquartile range
double trimmed_mean(std::span<const double> data, double frac);

// --- Correlation ---
double correlation(std::span<const double> x, std::span<const double> y);
double spearman(std::span<const double> x, std::span<const double> y);
double kendall(std::span<const double> x, std::span<const double> y);

// --- Hypothesis tests ---
double ttest(std::span<const double> sample, double mu);
double two_sample_ttest(std::span<const double> a, std::span<const double> b);
double ztest(std::span<const double> sample, double mu, double sigma);
double chi2_gof(std::span<const double> observed, std::span<const double> expected);
double ks_test(std::span<const double> x,
               std::function<double(double)> cdf);

// --- Regression ---
struct LinearRegressionResult {
    double slope;
    double intercept;
    double r_squared;
};
LinearRegressionResult linear_regression(std::span<const double> x,
                                          std::span<const double> y);
std::vector<double> multiple_regression(
    const std::vector<std::vector<double>>& X,
    std::span<const double> y);

// --- Time series ---
std::vector<double> acf(std::span<const double> x, int max_lag);
std::vector<double> pacf(std::span<const double> x, int max_lag);
std::vector<double> arfit(std::span<const double> x, int p); // AR(p) coefficients

// --- Bootstrap ---
double bootstrap_mean(std::span<const double> data, int n_boot = 1000,
                      unsigned seed = 42);
std::pair<double, double> bootstrap_ci(std::span<const double> data,
                                        double level = 0.95,
                                        int n_boot = 1000,
                                        unsigned seed = 42);

} // namespace ms
