#pragma once

#include "ms/core/matrix.hpp"
#include <cstdint>
#include <cmath>
#include <functional>
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

} // namespace ca

namespace lfsr {

uint64_t step(uint64_t state, uint64_t poly);
double alpha_lfsr(uint64_t poly, size_t steps);
bool is_maximal(uint64_t poly, int n);

} // namespace lfsr

void register_dispatch_hint(const std::string& op, double alpha);
double dispatch_hint_alpha(const std::string& op);

} // namespace ms::gria
