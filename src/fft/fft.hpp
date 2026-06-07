// MathScript FFT Library Header

#pragma once

#include <vector>
#include <complex>

namespace ms {

// FFT - Cooley-Tukey radix-2
template<typename T>
std::vector<std::complex<T>>
fft(const std::vector<T>& x) {
    if (x.size() < 2 || x.size() % 2 != 0) return x;
    
    // Recursively divide and conquer
    std::vector<std::complex<T>> even(x.size() / 2), odd(x.size() / 2);
    for (size_t i = 0; i < x.size() / 2; ++i) {
        even[i] = std::complex<T>(x[2 * i], 0);
        odd[i] = std::complex<T>(x[2 * i + 1], 0);
    }
    
    auto even_fft = fft(even);
    auto odd_fft = fft(odd);
    
    // Combine with twiddle factors
    std::vector<std::complex<T>> result(x.size());
    size_t n = x.size();
    for (size_t k = 0; k < n / 2; ++k) {
        std::complex<T> w = std::polar(1.0, -2.0 * M_PI * k / n);
        result[k] = even_fft[k] + w * odd_fft[k];
        result[k + n / 2] = even_fft[k] - w * odd_fft[k];
    }
    
    return result;
}

// Inverse FFT
template<typename T>
std::vector<T>
ifft(const std::vector<std::complex<T>>& x) {
    auto result = fft(x);
    std::vector<T> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = static_cast<T>(result[i].real() / x.size());
    }
    return out;
}

// 2D FFT
template<typename T>
std::vector<std::complex<T>>
fft2(const std::vector<std::complex<T>>& data) {
    // Simplified - would implement 2D properly
    return fft(data);
}

// DFT for arbitrary size (Bluestein)
template<typename T>
std::vector<std::complex<T>>
dft(std::span<const T> data) {
    // Placeholder for arbitrary size DFT
    return std::vector<std::complex<T>>();
}

} // namespace ms