#pragma once

#include <complex>
#include <span>
#include <vector>
#include "ms/error/error_types.hpp"

namespace ms {

Result<std::vector<std::complex<double>>> fft(const std::vector<double>& x);
Result<std::vector<double>> ifft(const std::vector<std::complex<double>>& x);
Result<std::vector<std::complex<double>>> fft2(const std::vector<std::complex<double>>& data);
Result<std::vector<std::complex<double>>> ifft2(const std::vector<std::complex<double>>& data);
Result<std::vector<std::complex<double>>> dft(std::span<const double> data);

Result<std::vector<std::complex<double>>> rfft(const std::vector<double>& x);
Result<std::vector<double>> irfft(const std::vector<std::complex<double>>& x, size_t n);

std::vector<std::complex<double>> fftshift(const std::vector<std::complex<double>>& x);
std::vector<std::complex<double>> ifftshift(const std::vector<std::complex<double>>& x);

Result<std::vector<double>> dct2(const std::vector<double>& x);
Result<std::vector<double>> idct2(const std::vector<double>& x);
Result<std::vector<double>> dst2(const std::vector<double>& x);
Result<std::vector<double>> idst2(const std::vector<double>& x);

} // namespace ms
