#include "ms/signal/signal.hpp"
#include "ms/fft/fft.hpp"
#include <cmath>
#include <complex>

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

size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

struct SegmentPlan {
    size_t hop;
    size_t n_segments;
    size_t n_fft;
    size_t n_freq_bins;
};

Result<SegmentPlan> plan_segments(size_t x_size, size_t segment_len, double overlap_frac,
                                  const char* fn) {
    if (segment_len == 0) {
        return std::unexpected(DomainError{fn, "segment_len must be > 0"});
    }
    if (x_size == 0) {
        return std::unexpected(DomainError{fn, "input signal must not be empty"});
    }
    if (segment_len > x_size) {
        return std::unexpected(DomainError{fn, "segment_len must be <= signal length"});
    }
    if (overlap_frac < 0.0 || overlap_frac >= 1.0) {
        return std::unexpected(DomainError{fn, "overlap_frac must be in [0, 1)"});
    }

    const double hop_d = static_cast<double>(segment_len) * (1.0 - overlap_frac);
    if (hop_d <= 0.0) {
        return std::unexpected(DomainError{fn, "overlap_frac too large; segment hop would be zero"});
    }
    size_t hop = static_cast<size_t>(std::floor(hop_d));
    if (hop == 0) {
        hop = 1;
    }

    size_t n_segments = 0;
    for (size_t start = 0; start + segment_len <= x_size; start += hop) {
        ++n_segments;
    }
    if (n_segments == 0) {
        return std::unexpected(DomainError{fn, "no complete segments fit in signal"});
    }

    const size_t n_fft = next_power_of_two(segment_len);
    return SegmentPlan{hop, n_segments, n_fft, n_fft / 2 + 1};
}

std::vector<double> segment_frequencies(size_t n_fft, double fs) {
    std::vector<double> frequencies(n_fft / 2 + 1);
    for (size_t k = 0; k < frequencies.size(); ++k) {
        frequencies[k] = static_cast<double>(k) * fs / static_cast<double>(n_fft);
    }
    return frequencies;
}

void apply_one_sided_psd_scaling(std::vector<double>& psd, size_t n_fft) {
    if (psd.size() <= 1) {
        return;
    }
    for (size_t k = 1; k + 1 < psd.size(); ++k) {
        psd[k] *= 2.0;
    }
    if (n_fft % 2 != 0) {
        psd.back() *= 2.0;
    }
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

Result<PSDResult> welch_psd(const std::vector<double>& x, double fs,
                             size_t segment_len, double overlap_frac) {
    if (fs <= 0.0) {
        return std::unexpected(DomainError{"welch_psd", "fs must be > 0"});
    }

    const auto plan = plan_segments(x.size(), segment_len, overlap_frac, "welch_psd");
    if (!plan) {
        return std::unexpected(plan.error());
    }

    const auto window = hanning(segment_len);
    double win_sq_sum = 0.0;
    for (const double w : window) {
        win_sq_sum += w * w;
    }
    const double scale = 1.0 / (fs * win_sq_sum);

    std::vector<double> psd(plan->n_freq_bins, 0.0);
    size_t seg_count = 0;

    for (size_t start = 0; start + segment_len <= x.size(); start += plan->hop) {
        std::vector<double> segment(segment_len);
        for (size_t i = 0; i < segment_len; ++i) {
            segment[i] = x[start + i] * window[i];
        }

        const auto spec = rfft(segment);
        if (!spec) {
            return std::unexpected(spec.error());
        }

        for (size_t k = 0; k < plan->n_freq_bins && k < spec->size(); ++k) {
            psd[k] += std::norm((*spec)[k]);
        }
        ++seg_count;
    }

    for (double& p : psd) {
        p = (p / static_cast<double>(seg_count)) * scale;
    }
    apply_one_sided_psd_scaling(psd, plan->n_fft);

    return PSDResult{segment_frequencies(plan->n_fft, fs), std::move(psd)};
}

Result<SpectrogramResult> spectrogram(const std::vector<double>& x, double fs,
                                       size_t segment_len, double overlap_frac) {
    if (fs <= 0.0) {
        return std::unexpected(DomainError{"spectrogram", "fs must be > 0"});
    }

    const auto plan = plan_segments(x.size(), segment_len, overlap_frac, "spectrogram");
    if (!plan) {
        return std::unexpected(plan.error());
    }

    const auto window = hanning(segment_len);
    const auto frequencies = segment_frequencies(plan->n_fft, fs);

    SpectrogramResult out;
    out.frequencies = frequencies;
    out.times.reserve(plan->n_segments);
    out.magnitude = Matrix<double>(plan->n_segments, plan->n_freq_bins);

    size_t row = 0;
    for (size_t start = 0; start + segment_len <= x.size(); start += plan->hop) {
        std::vector<double> segment(segment_len);
        for (size_t i = 0; i < segment_len; ++i) {
            segment[i] = x[start + i] * window[i];
        }

        const auto spec = rfft(segment);
        if (!spec) {
            return std::unexpected(spec.error());
        }

        out.times.push_back((static_cast<double>(start) + static_cast<double>(segment_len) / 2.0) / fs);

        for (size_t k = 0; k < plan->n_freq_bins && k < spec->size(); ++k) {
            out.magnitude(row, k) = std::abs((*spec)[k]);
        }
        ++row;
    }

    return out;
}

} // namespace ms
