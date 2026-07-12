#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <memory>

namespace ms::izaac {
class CSPRNG;
}

namespace ms::cypha {

struct NIGParams {
    double mu = 0.0;
    double alpha = 1.0;
    double beta = 0.0;
    double delta = 1.0;
};

struct DifConfig {
    size_t input_dim = 1;
    size_t output_dim = 1;
    size_t n_experts = 8;
    double learning_rate = 0.01;
    double gh_protection = 0.95;
    bool online = true;
    size_t max_components = 100;
};

struct PredictionInterval {
    Matrix<double> mean;
    Matrix<double> lower;
    Matrix<double> upper;
    double nig_alpha = 0.0;
    double nig_beta = 0.0;
};

class DifModel {
public:
    explicit DifModel(const DifConfig& cfg);

    Result<void> update(const Matrix<double>& x, const Matrix<double>& y);
    Result<Matrix<double>> predict(const Matrix<double>& x) const;
    Result<PredictionInterval> predict_interval(const Matrix<double>& x) const;
    Result<double> ood_score(const Matrix<double>& x) const;
    bool gh_gate(const Matrix<double>& x) const;

private:
    DifConfig cfg_;
    Matrix<double> weights_;
    Matrix<double> bias_;
    NIGParams uncertainty_;
    size_t updates_ = 0;
};

NIGParams nig_fit(const Matrix<double>& data);
double nig_pdf(double x, const NIGParams& params);
double nig_cdf(double x, const NIGParams& params);
Matrix<double> nig_sample(const NIGParams& params, size_t n, izaac::CSPRNG& rng);

/// Closed-form mean of the NIG(alpha, beta, delta, mu) distribution:
/// mean = mu + delta * beta / sqrt(alpha^2 - beta^2).
/// Returns NaN if params are outside the valid domain (|beta| >= alpha or delta <= 0).
double nig_mean(const NIGParams& params);

/// Closed-form variance of the NIG(alpha, beta, delta, mu) distribution:
/// variance = delta * alpha^2 / (alpha^2 - beta^2)^1.5.
/// Returns NaN if params are outside the valid domain (|beta| >= alpha or delta <= 0).
double nig_variance(const NIGParams& params);

} // namespace ms::cypha
