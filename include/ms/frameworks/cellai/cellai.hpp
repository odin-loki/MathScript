#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <memory>
#include <span>
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

/// Converts a vector of Boltzmann-machine energies into normalized Gibbs/
/// softmax sampling weights: p[i] = exp(-energies[i] / temperature) /
/// sum_j exp(-energies[j] / temperature).
///
/// Numerically stable: the minimum energy is subtracted from every term
/// before exponentiating (log-sum-exp trick), so the largest exponential
/// argument is always 0 and overflow/underflow in exp() is avoided
/// regardless of the absolute magnitude of the input energies.
///
/// Zero-temperature convention: `temperature <= 0` is treated as the
/// zero-temperature limit of the Boltzmann distribution rather than
/// producing a division by zero. In that limit all probability mass
/// concentrates on the minimum-energy state(s), with ties split evenly
/// among them.
///
/// Returns an empty vector for an empty `energies` span.
std::vector<double> boltzmann_weights(std::span<const double> energies, double temperature = 1.0);

Matrix<double> cell_to_cypha_features(
    const CellMemory& cell,
    const std::vector<double>& time_scales);

} // namespace ms::cellai
