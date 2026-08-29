#pragma once

#include <cstddef>

namespace ms::cuda {

bool try_device_add_inplace(double* a, const double* b, double alpha, std::size_t n);
bool try_device_fill(double* out, double value, std::size_t n);
bool try_device_mul_inplace(double* a, const double* b, std::size_t n);
bool try_device_scale(double* a, double alpha, std::size_t n);

} // namespace ms::cuda
