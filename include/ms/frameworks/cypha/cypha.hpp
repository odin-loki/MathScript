#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <memory>

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

class DifModel {
public:
    explicit DifModel(const DifConfig& cfg);

    Result<void> update(const Matrix<double>& x, const Matrix<double>& y);
    Result<Matrix<double>> predict(const Matrix<double>& x) const;
    Result<double> ood_score(const Matrix<double>& x) const;

private:
    DifConfig cfg_;
    Matrix<double> weights_;
    Matrix<double> bias_;
    NIGParams uncertainty_;
    size_t updates_ = 0;
};

NIGParams nig_fit(const Matrix<double>& data);
double nig_pdf(double x, const NIGParams& params);

} // namespace ms::cypha
