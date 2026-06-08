#pragma once

#include <vector>

namespace ms {

std::vector<double> poly_eval(const std::vector<double>& coeffs, double x);
std::vector<double> poly_add(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> poly_sub(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> poly_mul(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> poly_deriv(const std::vector<double>& coeffs);

} // namespace ms
