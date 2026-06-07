#pragma once

#include <span>
#include <tuple>
#include <vector>

namespace ms {

double mean(std::span<const double> data);
double var(std::span<const double> data);
double stddev(std::span<const double> data);
double median(std::span<const double> data);
double min_value(std::span<const double> data);
double max_value(std::span<const double> data);
double mode(std::span<const double> data);
double percentile(std::span<const double> data, double p);
double ttest(std::span<const double> sample, double mu);
double correlation(std::span<const double> x, std::span<const double> y);

double skewness(std::span<const double> data);
double kurtosis(std::span<const double> data);
double two_sample_ttest(std::span<const double> a, std::span<const double> b);
double ztest(std::span<const double> sample, double mu, double sigma);

struct LinearRegressionResult {
    double slope;
    double intercept;
    double r_squared;
};

LinearRegressionResult linear_regression(std::span<const double> x, std::span<const double> y);

} // namespace ms
