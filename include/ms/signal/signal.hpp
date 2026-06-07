#pragma once

#include <vector>

namespace ms {

std::vector<double> butterworth(const std::vector<double>& x, double cutoff, double fs);
std::vector<double> lowpass(const std::vector<double>& x, double cutoff, double fs);
std::vector<double> highpass(const std::vector<double>& x, double cutoff, double fs);
std::vector<double> bandpass(const std::vector<double>& x, double low, double high, double fs);
std::vector<double> convolve(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> correlate(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> moving_average(const std::vector<double>& x, size_t window);
std::vector<double> hamming(size_t n);
std::vector<double> hanning(size_t n);
std::vector<double> blackman(size_t n);
std::vector<double> parzen(size_t n);
std::vector<double> triangular(size_t n);

} // namespace ms
