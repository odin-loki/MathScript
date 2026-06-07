#pragma once

#include <complex>
#include <vector>
#include "ms/error/error_types.hpp"

namespace ms::cuda {

Result<std::vector<std::complex<double>>> fft(const std::vector<double>& x);
Result<std::vector<double>> ifft(const std::vector<std::complex<double>>& x);

} // namespace ms::cuda
