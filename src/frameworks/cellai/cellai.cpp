#include "ms/frameworks/cellai/cellai.hpp"

#include <cmath>

namespace ms::cellai {

Matrix<double> hebbian_update(
    const Matrix<double>& w,
    const Matrix<double>& x,
    const Matrix<double>& y,
    double learning_rate) {
    Matrix<double> updated = w;
    for (size_t i = 0; i < w.rows(); ++i) {
        for (size_t j = 0; j < w.cols(); ++j) {
            const double pre = i < x.rows() ? x(i, 0) : 0.0;
            const double post = j < y.rows() ? y(j, 0) : 0.0;
            updated(i, j) += learning_rate * pre * post;
        }
    }
    return updated;
}

CellMemory::CellMemory(size_t input_dim, size_t memory_dim, std::vector<double> time_scales)
    : input_dim_(input_dim),
      memory_dim_(memory_dim),
      time_scales_(std::move(time_scales)),
      state_(memory_dim, 1, 0.0),
      long_term_(memory_dim, 1, 0.0) {}

Result<Matrix<double>> CellMemory::step(const Matrix<double>& input) {
    if (input.rows() != input_dim_ || input.cols() != 1) {
        return std::unexpected(DimensionMismatch{input.rows(), input_dim_});
    }
    for (size_t i = 0; i < memory_dim_; ++i) {
        const size_t src = i % input_dim_;
        state_(i, 0) = 0.9 * state_(i, 0) + 0.1 * input(src, 0);
        long_term_(i, 0) = 0.99 * long_term_(i, 0) + 0.01 * state_(i, 0);
    }
    return state_;
}

Result<Matrix<double>> CellMemory::recall(double time_scale) const {
    Matrix<double> out(memory_dim_, 1, 0.0);
    const double weight = std::exp(-time_scale);
    for (size_t i = 0; i < memory_dim_; ++i) {
        out(i, 0) = weight * state_(i, 0) + (1.0 - weight) * long_term_(i, 0);
    }
    return out;
}

void CellMemory::reset() {
    for (size_t i = 0; i < state_.rows(); ++i) {
        state_(i, 0) = 0.0;
    }
    for (size_t i = 0; i < long_term_.rows(); ++i) {
        long_term_(i, 0) = 0.0;
    }
}

Matrix<double> cell_to_cypha_features(
    const CellMemory& cell,
    const std::vector<double>& time_scales) {
    Matrix<double> features(time_scales.size(), 1, 0.0);
    for (size_t i = 0; i < time_scales.size(); ++i) {
        const auto recalled = cell.recall(time_scales[i]).value();
        double sum = 0.0;
        for (size_t r = 0; r < recalled.rows(); ++r) {
            sum += recalled(r, 0);
        }
        features(i, 0) = sum;
    }
    return features;
}

} // namespace ms::cellai
