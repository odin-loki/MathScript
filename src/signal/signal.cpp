#include "ms/signal/signal.hpp"
#include "ms/fft/fft.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

namespace {

std::vector<double> fft_lowpass(const std::vector<double>& x, double cutoff, double fs) {
    if (x.empty()) {
        return x;
    }
    auto spectrum = fft(x);
    if (!spectrum) {
        return x;
    }
    const size_t n = spectrum->size();
    const double nyquist = fs / 2.0;
    const size_t cutoff_bin = static_cast<size_t>(
        std::min(static_cast<double>(n / 2), cutoff / nyquist * static_cast<double>(n / 2)));

    for (size_t i = cutoff_bin + 1; i < n; ++i) {
        const size_t mirror = n - i;
        if (mirror < n) {
            (*spectrum)[i] = 0.0;
            if (mirror > 0) {
                (*spectrum)[mirror] = 0.0;
            }
        }
    }

    auto restored = ifft(*spectrum);
    if (!restored) {
        return x;
    }
    restored->resize(x.size());
    return *restored;
}

} // namespace

std::vector<double> butterworth(const std::vector<double>& x, double cutoff, double fs) {
    return fft_lowpass(x, cutoff, fs);
}

std::vector<double> lowpass(const std::vector<double>& x, double cutoff, double fs) {
    return fft_lowpass(x, cutoff, fs);
}

std::vector<double> highpass(const std::vector<double>& x, double cutoff, double fs) {
    if (x.empty()) {
        return x;
    }
    auto lp = fft_lowpass(x, cutoff, fs);
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] - lp[i];
    }
    return out;
}

std::vector<double> bandpass(const std::vector<double>& x, double low, double high, double fs) {
    auto hp = highpass(x, low, fs);
    return lowpass(hp, high, fs);
}

std::vector<double> convolve(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i + j] += a[i] * b[j];
        }
    }
    return result;
}

std::vector<double> correlate(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> rev_b(b.rbegin(), b.rend());
    return convolve(a, rev_b);
}

std::vector<double> moving_average(const std::vector<double>& x, size_t window) {
    if (x.empty() || window == 0) {
        return x;
    }
    std::vector<double> out(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        const size_t start = (i >= window - 1) ? i - window + 1 : 0;
        double sum = 0.0;
        for (size_t j = start; j <= i; ++j) {
            sum += x[j];
        }
        out[i] = sum / static_cast<double>(i - start + 1);
    }
    return out;
}

std::vector<double> hamming(size_t n) {
    std::vector<double> h(n);
    if (n == 0) {
        return h;
    }
    if (n == 1) {
        h[0] = 1.0;
        return h;
    }
    for (size_t i = 0; i < n; ++i) {
        h[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n - 1));
    }
    return h;
}

std::vector<double> hanning(size_t n) {
    std::vector<double> w(n);
    if (n == 0) {
        return w;
    }
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }
    for (size_t i = 0; i < n; ++i) {
        w[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n - 1)));
    }
    return w;
}

std::vector<double> blackman(size_t n) {
    std::vector<double> w(n);
    if (n == 0) {
        return w;
    }
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }
    const double a0 = 0.42;
    const double a1 = 0.5;
    const double a2 = 0.08;
    for (size_t i = 0; i < n; ++i) {
        const double t = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n - 1);
        w[i] = a0 - a1 * std::cos(t) + a2 * std::cos(2.0 * t);
    }
    return w;
}

std::vector<double> parzen(size_t n) {
    std::vector<double> w(n);
    if (n == 0) {
        return w;
    }
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }
    const double half = static_cast<double>(n - 1) / 2.0;
    for (size_t i = 0; i < n; ++i) {
        const double u = std::abs(static_cast<double>(i) - half) / half;
        if (u <= 0.5) {
            w[i] = 1.0 - 6.0 * u * u * (1.0 - u);
        } else {
            w[i] = 2.0 * std::pow(1.0 - u, 3.0);
        }
    }
    return w;
}

std::vector<double> triangular(size_t n) {
    std::vector<double> w(n);
    if (n == 0) {
        return w;
    }
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }
    for (size_t i = 0; i < n; ++i) {
        if (i <= n / 2) {
            w[i] = static_cast<double>(i + 1) / static_cast<double>((n / 2) + 1);
        } else {
            w[i] = static_cast<double>(n - i) / static_cast<double>((n / 2) + 1);
        }
    }
    return w;
}

} // namespace ms
