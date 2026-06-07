#include "ms/fft/fft.hpp"
#include "ms/cuda/fft.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

namespace {

std::vector<std::complex<double>> fft_recursive(std::vector<std::complex<double>> x) {
    const size_t n = x.size();
    if (n <= 1) {
        return x;
    }
    if (n % 2 != 0) {
        std::vector<std::complex<double>> out(n);
        for (size_t k = 0; k < n; ++k) {
            std::complex<double> sum(0.0, 0.0);
            for (size_t t = 0; t < n; ++t) {
                const double angle = -2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(n);
                sum += x[t] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            out[k] = sum;
        }
        return out;
    }

    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    even = fft_recursive(std::move(even));
    odd = fft_recursive(std::move(odd));

    std::vector<std::complex<double>> out(n);
    for (size_t k = 0; k < n / 2; ++k) {
        const double angle = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n);
        const std::complex<double> w(std::cos(angle), std::sin(angle));
        const std::complex<double> t = w * odd[k];
        out[k] = even[k] + t;
        out[k + n / 2] = even[k] - t;
    }
    return out;
}

size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

} // namespace

Result<std::vector<std::complex<double>>> fft(const std::vector<double>& x) {
    if (x.empty()) {
        return std::vector<std::complex<double>>{};
    }

    const size_t n = next_power_of_two(x.size());
    const auto decision = decide(n, ExecPolicy::AUTO);
    if (decision.backend == Backend::CUDA) {
        if (auto gpu = cuda::fft(x)) {
            return gpu;
        }
    }

    std::vector<std::complex<double>> input(n);
    for (size_t i = 0; i < x.size(); ++i) {
        input[i] = std::complex<double>(x[i], 0.0);
    }

    return fft_recursive(std::move(input));
}

Result<std::vector<double>> ifft(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return std::vector<double>{};
    }

    std::vector<std::complex<double>> conj(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        conj[i] = std::conj(x[i]);
    }

    auto spectrum = fft_recursive(std::move(conj));
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = spectrum[i].real() / static_cast<double>(x.size());
    }
    return out;
}

Result<std::vector<std::complex<double>>> fft2(const std::vector<std::complex<double>>& data) {
    return data;
}

Result<std::vector<std::complex<double>>> dft(std::span<const double> data) {
    std::vector<double> copy(data.begin(), data.end());
    return fft(copy);
}

Result<std::vector<std::complex<double>>> rfft(const std::vector<double>& x) {
    auto full = fft(x);
    if (!full) {
        return std::unexpected(full.error());
    }
    const size_t n = full->size();
    const size_t out_len = n / 2 + 1;
    std::vector<std::complex<double>> out(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        out[i] = (*full)[i];
    }
    return out;
}

Result<std::vector<double>> irfft(const std::vector<std::complex<double>>& x, size_t n) {
    if (x.empty() || n == 0) {
        return std::vector<double>{};
    }
    const size_t full_len = next_power_of_two(n);
    std::vector<std::complex<double>> full(full_len);
    for (size_t i = 0; i < x.size() && i < full_len; ++i) {
        full[i] = x[i];
    }
    for (size_t i = x.size(); i < full_len; ++i) {
        const size_t k = full_len - i;
        full[i] = std::conj(full[k]);
    }
    return ifft(full);
}

std::vector<std::complex<double>> fftshift(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return x;
    }
    const size_t n = x.size();
    const size_t half = n / 2;
    std::vector<std::complex<double>> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[(i + half) % n];
    }
    return out;
}

std::vector<std::complex<double>> ifftshift(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return x;
    }
    const size_t n = x.size();
    const size_t half = (n + 1) / 2;
    std::vector<std::complex<double>> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[(i + half) % n];
    }
    return out;
}

Result<std::vector<double>> dct2(const std::vector<double>& x) {
    const size_t n = x.size();
    if (n == 0) {
        return std::vector<double>{};
    }
    std::vector<double> out(n);
    const double scale0 = 1.0 / std::sqrt(static_cast<double>(n));
    const double scale = std::sqrt(2.0 / static_cast<double>(n));
    for (size_t k = 0; k < n; ++k) {
        double sum = 0.0;
        for (size_t t = 0; t < n; ++t) {
            sum += x[t] * std::cos(M_PI * static_cast<double>((t + 0.5) * k) / static_cast<double>(n));
        }
        out[k] = (k == 0 ? scale0 : scale) * sum;
    }
    return out;
}

Result<std::vector<double>> idct2(const std::vector<double>& x) {
    const size_t n = x.size();
    if (n == 0) {
        return std::vector<double>{};
    }
    std::vector<double> out(n);
    const double scale0 = 1.0 / std::sqrt(static_cast<double>(n));
    const double scale = std::sqrt(2.0 / static_cast<double>(n));
    for (size_t t = 0; t < n; ++t) {
        double sum = scale0 * x[0];
        for (size_t k = 1; k < n; ++k) {
            sum += scale * x[k] * std::cos(M_PI * static_cast<double>((t + 0.5) * k) / static_cast<double>(n));
        }
        out[t] = sum;
    }
    return out;
}

Result<std::vector<double>> dst2(const std::vector<double>& x) {
    const size_t n = x.size();
    if (n == 0) {
        return std::vector<double>{};
    }
    std::vector<double> out(n);
    const double scale = std::sqrt(2.0 / static_cast<double>(n + 1));
    for (size_t k = 0; k < n; ++k) {
        double sum = 0.0;
        for (size_t t = 0; t < n; ++t) {
            sum += x[t] * std::sin(M_PI * static_cast<double>((t + 1) * (k + 1)) / static_cast<double>(n + 1));
        }
        out[k] = scale * sum;
    }
    return out;
}

} // namespace ms
