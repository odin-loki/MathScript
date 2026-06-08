#include "ms/poly/poly.hpp"

namespace ms {

std::vector<double> poly_eval(const std::vector<double>& coeffs, double x) {
    double value = 0.0;
    for (size_t i = coeffs.size(); i > 0; --i) {
        value = value * x + coeffs[i - 1];
    }
    return {value};
}

std::vector<double> poly_add(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = (a.size() > b.size()) ? a.size() : b.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] += a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        out[i] += b[i];
    }
    return out;
}

std::vector<double> poly_sub(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = (a.size() > b.size()) ? a.size() : b.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] += a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        out[i] -= b[i];
    }
    return out;
}

std::vector<double> poly_mul(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    std::vector<double> out(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            out[i + j] += a[i] * b[j];
        }
    }
    return out;
}

std::vector<double> poly_deriv(const std::vector<double>& coeffs) {
    if (coeffs.size() <= 1) {
        return {0.0};
    }
    std::vector<double> out(coeffs.size() - 1);
    for (size_t i = 1; i < coeffs.size(); ++i) {
        out[i - 1] = coeffs[i] * static_cast<double>(i);
    }
    return out;
}

} // namespace ms
