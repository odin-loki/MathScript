#pragma once

#include <span>

namespace ms::cuda {

void add_inplace(std::span<double> a, std::span<const double> b, double alpha = 1.0);
void fill(std::span<double> out, double value);
void mul_inplace(std::span<double> a, std::span<const double> b);
void scale(std::span<double> a, double alpha);

} // namespace ms::cuda
