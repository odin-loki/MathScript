#pragma once

#include "ms/core/matrix.hpp"
#include <cstdint>
#include <cmath>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ms::gria {

enum class ComputeClass { Reversible, Critical, Irreversible };

double entropy(std::span<const double> data, size_t bins = 16);

template<typename T>
double compute_alpha(
    std::span<const T> input,
    const std::function<std::vector<T>(std::span<const T>)>& transform) {
    if (input.empty()) {
        return 0.0;
    }
    std::vector<double> in(input.begin(), input.end());
    const auto out = transform(input);
    std::vector<double> transformed(out.begin(), out.end());
    const double h_in = entropy(in);
    if (h_in <= 1e-12) {
        return 0.0;
    }
    const double h_out = entropy(transformed);
    const double alpha = 1.0 - (h_out / h_in);
    if (alpha < 0.0) {
        return 0.0;
    }
    if (alpha > 1.0) {
        return 1.0;
    }
    return alpha;
}

double matrix_alpha(const Matrix<double>& x, const Matrix<double>& fx);
bool is_critical(double alpha, double tolerance = 0.05);
ComputeClass classify(double alpha);

namespace gf2n {

uint64_t mul(uint64_t a, uint64_t b, uint64_t poly);
uint64_t pow(uint64_t a, uint64_t exp, uint64_t poly);
uint64_t inv(uint64_t a, uint64_t poly);
std::vector<uint64_t> generate_field(int n);

} // namespace gf2n

namespace ca {

std::vector<uint8_t> step(std::span<const uint8_t> state, uint8_t rule);
double langton_lambda(uint8_t rule);
double alpha_ca(uint8_t rule, size_t steps, size_t width);

/// Hamming distance between two CA configurations: the number of index
/// positions at which `a` and `b` disagree.
///
/// If `a` and `b` have equal length, this is the classic bitwise Hamming
/// distance, in `[0, a.size()]`.
///
/// If `a` and `b` have different lengths, there is no configuration cell to
/// compare against beyond the shorter one, so every position past the
/// shorter length is defensively counted as differing. The result is
/// `(mismatches over the common prefix) + |a.size() - b.size()|`. This keeps
/// the function total (no exceptions/asserts on mismatched input) while
/// still bounding the result by `max(a.size(), b.size())`.
///
/// @return 0 for identical configurations; `a.size()` (== `b.size()`) when
///         every position differs and the lengths match.
size_t hamming_distance(std::span<const uint8_t> a, std::span<const uint8_t> b);

/// Evolves `config_a` and `config_b` forward `n_steps` under the same
/// `rule` (via `step`), recording the Hamming distance between them after
/// each step. This produces a "divergence over time" trace useful for
/// visualising sensitivity to initial conditions -- e.g. a chaotic-regime
/// rule (per `langton_lambda`) applied to two nearly-identical
/// configurations tends to show a growing distance, while an
/// ordered-regime rule tends to keep it small or settle it to zero.
///
/// The returned trajectory has length `n_steps + 1`: index 0 is the Hamming
/// distance between the two *initial* configurations (before any
/// stepping), and index `i` (for `1 <= i <= n_steps`) is the distance after
/// `i` applications of `step`. `n_steps <= 0` yields a single-element
/// trajectory containing just the initial distance.
///
/// `config_a` and `config_b` need not have equal length: `hamming_distance`'s
/// mismatched-length convention applies at every recorded step.
std::vector<size_t> divergence_trajectory(std::vector<uint8_t> config_a,
                                           std::vector<uint8_t> config_b,
                                           uint8_t rule,
                                           int n_steps);

/// Settling time to convergence: evolves `config_a` and `config_b` forward
/// under `rule`, and returns the smallest step count `i` (`0 <= i <=
/// n_steps`) at which their Hamming distance first reaches 0, or
/// `std::nullopt` if they remain distinct through all `n_steps` steps (or
/// if `n_steps < 0`).
///
/// A return value of 0 means the two configurations were already
/// identical before any stepping. Since `step` is a deterministic function
/// of the current configuration, once the distance reaches 0 it stays 0
/// forever after (both configurations follow the same future trajectory).
std::optional<size_t> settling_time(std::vector<uint8_t> config_a,
                                     std::vector<uint8_t> config_b,
                                     uint8_t rule,
                                     int n_steps);

} // namespace ca

namespace lfsr {

uint64_t step(uint64_t state, uint64_t poly);
double alpha_lfsr(uint64_t poly, size_t steps);
bool is_maximal(uint64_t poly, int n);

} // namespace lfsr

void register_dispatch_hint(const std::string& op, double alpha);
double dispatch_hint_alpha(const std::string& op);

} // namespace ms::gria
