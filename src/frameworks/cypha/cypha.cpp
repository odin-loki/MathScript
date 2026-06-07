#include "ms/frameworks/cypha/cypha.hpp"

#include <cmath>

namespace ms::cypha {

DifModel::DifModel(const DifConfig& cfg) : cfg_(cfg) {
    weights_ = Matrix<double>(cfg.output_dim, cfg.input_dim, 0.0);
    bias_ = Matrix<double>(cfg.output_dim, 1, 0.0);
    uncertainty_ = {0.0, 2.0, 0.0, 1.0};
}

Result<void> DifModel::update(const Matrix<double>& x, const Matrix<double>& y) {
    if (x.rows() != cfg_.input_dim || y.rows() != cfg_.output_dim || x.cols() != 1 || y.cols() != 1) {
        return std::unexpected(DimensionMismatch{x.rows(), y.rows()});
    }

    Matrix<double> prediction(cfg_.output_dim, 1, 0.0);
    for (size_t o = 0; o < cfg_.output_dim; ++o) {
        double sum = bias_(o, 0);
        for (size_t i = 0; i < cfg_.input_dim; ++i) {
            sum += weights_(o, i) * x(i, 0);
        }
        prediction(o, 0) = sum;
        const double err = y(o, 0) - sum;
        bias_(o, 0) += cfg_.learning_rate * err;
        for (size_t i = 0; i < cfg_.input_dim; ++i) {
            weights_(o, i) += cfg_.learning_rate * err * x(i, 0);
        }
    }

    uncertainty_.mu = prediction(0, 0);
    uncertainty_.delta = std::max(0.1, uncertainty_.delta * (1.0 - cfg_.learning_rate * 0.1));
    ++updates_;
    return {};
}

Result<Matrix<double>> DifModel::predict(const Matrix<double>& x) const {
    if (x.rows() != cfg_.input_dim || x.cols() != 1) {
        return std::unexpected(DimensionMismatch{x.rows(), cfg_.input_dim});
    }
    Matrix<double> out(cfg_.output_dim, 1, 0.0);
    for (size_t o = 0; o < cfg_.output_dim; ++o) {
        double sum = bias_(o, 0);
        for (size_t i = 0; i < cfg_.input_dim; ++i) {
            sum += weights_(o, i) * x(i, 0);
        }
        out(o, 0) = sum;
    }
    return out;
}

Result<double> DifModel::ood_score(const Matrix<double>& x) const {
    if (x.rows() != cfg_.input_dim || x.cols() != 1) {
        return std::unexpected(DimensionMismatch{x.rows(), cfg_.input_dim});
    }
    double energy = 0.0;
    for (size_t i = 0; i < x.rows(); ++i) {
        energy += x(i, 0) * x(i, 0);
    }
    return std::sqrt(energy) / static_cast<double>(cfg_.input_dim);
}

NIGParams nig_fit(const Matrix<double>& data) {
    NIGParams params;
    if (data.rows() == 0) {
        return params;
    }
    double sum = 0.0;
    for (size_t i = 0; i < data.rows(); ++i) {
        sum += data(i, 0);
    }
    params.mu = sum / static_cast<double>(data.rows());
    double var = 0.0;
    for (size_t i = 0; i < data.rows(); ++i) {
        const double d = data(i, 0) - params.mu;
        var += d * d;
    }
    params.delta = std::sqrt(var / static_cast<double>(data.rows()) + 1e-6);
    params.alpha = 2.0;
    return params;
}

double nig_pdf(double x, const NIGParams& params) {
    const double z =
        params.alpha * params.delta / std::sqrt(params.delta * params.delta + (x - params.mu) * (x - params.mu));
    return std::exp(-z) / (2.0 * params.delta + 1e-12);
}

} // namespace ms::cypha
