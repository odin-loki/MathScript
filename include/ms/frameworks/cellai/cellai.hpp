#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <memory>
#include <vector>

namespace ms::cellai {

Matrix<double> hebbian_update(
    const Matrix<double>& w,
    const Matrix<double>& x,
    const Matrix<double>& y,
    double learning_rate);

class CellMemory {
public:
    explicit CellMemory(size_t input_dim, size_t memory_dim, std::vector<double> time_scales);

    Result<Matrix<double>> step(const Matrix<double>& input);
    Result<Matrix<double>> recall(double time_scale) const;
    Result<void> consolidate();
    void reset();

    // Read-only snapshot for tests and diagnostics (no direct mutation of internal state).
    const Matrix<double>& long_term_state() const { return long_term_; }

private:
    size_t input_dim_;
    size_t memory_dim_;
    std::vector<double> time_scales_;
    Matrix<double> state_;
    Matrix<double> long_term_;
};

double energy(const Matrix<double>& w, const Matrix<double>& v, const Matrix<double>& h);

Matrix<double> cell_to_cypha_features(
    const CellMemory& cell,
    const std::vector<double>& time_scales);

} // namespace ms::cellai
