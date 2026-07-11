#include "ms/signal/signal.hpp"
#include "ms/fft/fft.hpp"
#include "ms/poly/poly.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>

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

    for (size_t k = 0; k < n; ++k) {
        const size_t wrapped = std::min(k, n - k);
        if (wrapped > cutoff_bin) {
            (*spectrum)[k] = 0.0;
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

std::vector<std::complex<double>> complex_ifft(const std::vector<std::complex<double>>& x) {
    if (x.empty()) {
        return {};
    }
    std::vector<std::complex<double>> conj(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        conj[i] = std::conj(x[i]);
    }
    auto spectrum = fft_recursive(std::move(conj));
    const double inv_n = 1.0 / static_cast<double>(x.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        spectrum[i] = std::conj(spectrum[i]) * inv_n;
    }
    return spectrum;
}

void apply_hilbert_frequency_mask(std::vector<std::complex<double>>& spectrum) {
    const size_t n = spectrum.size();
    if (n == 0) {
        return;
    }
    for (size_t k = 1; k < n / 2; ++k) {
        spectrum[k] *= 2.0;
    }
    if (n % 2 == 0) {
        for (size_t k = n / 2 + 1; k < n; ++k) {
            spectrum[k] = 0.0;
        }
    } else {
        for (size_t k = (n + 1) / 2; k < n; ++k) {
            spectrum[k] = 0.0;
        }
    }
}

std::vector<double> unwrap_phase(const std::vector<double>& wrapped) {
    if (wrapped.empty()) {
        return wrapped;
    }
    std::vector<double> out(wrapped.size());
    out[0] = wrapped[0];
    for (size_t i = 1; i < wrapped.size(); ++i) {
        double delta = wrapped[i] - wrapped[i - 1];
        while (delta > M_PI) {
            delta -= 2.0 * M_PI;
        }
        while (delta <= -M_PI) {
            delta += 2.0 * M_PI;
        }
        out[i] = out[i - 1] + delta;
    }
    return out;
}

std::vector<double> demean(const std::vector<double>& x) {
    if (x.empty()) {
        return x;
    }
    double mean = 0.0;
    for (double v : x) {
        mean += v;
    }
    mean /= static_cast<double>(x.size());
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] - mean;
    }
    return out;
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

std::vector<double> upsample(const std::vector<double>& x, int n) {
    if (x.empty() || n <= 0) {
        return {};
    }
    const size_t step = static_cast<size_t>(n);
    std::vector<double> out(x.size() * step, 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        out[i * step] = x[i];
    }
    return out;
}

std::vector<double> downsample(const std::vector<double>& x, int n) {
    if (x.empty() || n <= 0) {
        return {};
    }
    const size_t step = static_cast<size_t>(n);
    std::vector<double> out;
    out.reserve(x.size() / step + 1);
    for (size_t i = 0; i < x.size(); i += step) {
        out.push_back(x[i]);
    }
    return out;
}

std::vector<double> decimate(const std::vector<double>& x, int q) {
    if (x.empty() || q <= 0) {
        return {};
    }
    const double new_nyquist = 1.0 / (2.0 * static_cast<double>(q));
    const auto filtered = lowpass(x, 0.8 * new_nyquist, 1.0);
    return downsample(filtered, q);
}

std::vector<double> interpolate(const std::vector<double>& x, int p) {
    if (x.empty() || p <= 0) {
        return {};
    }
    const auto stuffed = upsample(x, p);
    const double new_nyquist = 1.0 / (2.0 * static_cast<double>(p));
    auto filtered = lowpass(stuffed, 0.8 * new_nyquist, 1.0);
    const double gain = static_cast<double>(p);
    for (double& v : filtered) {
        v *= gain;
    }
    return filtered;
}

std::vector<double> resample(const std::vector<double>& x, int p, int q) {
    if (x.empty() || p <= 0 || q <= 0) {
        return {};
    }
    return decimate(interpolate(x, p), q);
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

std::vector<std::complex<double>> hilbert(const std::vector<double>& x) {
    if (x.empty()) {
        return {};
    }
    const auto spectrum = fft(x);
    if (!spectrum) {
        return {};
    }
    auto masked = *spectrum;
    apply_hilbert_frequency_mask(masked);
    auto analytic = complex_ifft(masked);
    analytic.resize(x.size());
    return analytic;
}

std::vector<double> envelope(const std::vector<double>& x) {
    const auto analytic = hilbert(x);
    if (analytic.empty()) {
        return {};
    }
    std::vector<double> out(analytic.size());
    for (size_t i = 0; i < analytic.size(); ++i) {
        out[i] = std::abs(analytic[i]);
    }
    return out;
}

std::vector<double> instantaneous_phase(const std::vector<double>& x) {
    const auto analytic = hilbert(x);
    if (analytic.empty()) {
        return {};
    }
    std::vector<double> out(analytic.size());
    for (size_t i = 0; i < analytic.size(); ++i) {
        out[i] = std::arg(analytic[i]);
    }
    return out;
}

std::vector<double> instantaneous_freq(const std::vector<double>& x, double fs) {
    if (x.empty()) {
        return {};
    }
    const auto phase = unwrap_phase(instantaneous_phase(x));
    std::vector<double> freq(x.size(), 0.0);
    if (phase.size() < 2) {
        return freq;
    }
    const double scale = fs / (2.0 * M_PI);
    for (size_t i = 0; i + 1 < phase.size(); ++i) {
        freq[i] = (phase[i + 1] - phase[i]) * scale;
    }
    freq.back() = freq[freq.size() - 2];
    return freq;
}

std::vector<double> xcorr(const std::vector<double>& a, const std::vector<double>& b, int max_lag) {
    if (max_lag < 0 || a.empty() || b.empty()) {
        return {};
    }
    const auto full = correlate(a, b);
    const ptrdiff_t zero_idx = static_cast<ptrdiff_t>(b.size() - 1);
    const size_t out_len = static_cast<size_t>(2 * max_lag + 1);
    std::vector<double> out(out_len, 0.0);
    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        const ptrdiff_t idx = zero_idx - lag;
        if (idx >= 0 && idx < static_cast<ptrdiff_t>(full.size())) {
            out[static_cast<size_t>(lag + max_lag)] = full[static_cast<size_t>(idx)];
        }
    }
    return out;
}

std::vector<double> xcov(const std::vector<double>& a, const std::vector<double>& b, int max_lag) {
    return xcorr(demean(a), demean(b), max_lag);
}

std::vector<double> autocorr(const std::vector<double>& x, int max_lag) {
    return xcorr(x, x, max_lag);
}

Matrix<double> conv2(const Matrix<double>& A, const Matrix<double>& B) {
    if (A.empty() || B.empty()) {
        return Matrix<double>{};
    }
    const size_t out_rows = A.rows() + B.rows() - 1;
    const size_t out_cols = A.cols() + B.cols() - 1;
    Matrix<double> out(out_rows, out_cols, 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            for (size_t k = 0; k < B.rows(); ++k) {
                for (size_t l = 0; l < B.cols(); ++l) {
                    out(i + k, j + l) += A(i, j) * B(k, l);
                }
            }
        }
    }
    return out;
}

std::vector<double> deconv(const std::vector<double>& y, const std::vector<double>& b) {
    return poly::poly_div_quot(y, b);
}

std::vector<double> filter(const std::vector<double>& b, const std::vector<double>& a,
                            const std::vector<double>& x) {
    if (x.empty() || b.empty()) {
        return {};
    }

    const double a0 = a.empty() ? 1.0 : a[0];
    std::vector<double> bn = b;
    std::vector<double> an = a.empty() ? std::vector<double>{1.0} : a;
    if (a0 != 1.0 && a0 != 0.0) {
        for (double& v : bn) {
            v /= a0;
        }
        for (double& v : an) {
            v /= a0;
        }
    }

    std::vector<double> y(x.size(), 0.0);
    for (size_t n = 0; n < x.size(); ++n) {
        double acc = 0.0;
        for (size_t j = 0; j < bn.size(); ++j) {
            if (n >= j) {
                acc += bn[j] * x[n - j];
            }
        }
        for (size_t k = 1; k < an.size(); ++k) {
            if (n >= k) {
                acc -= an[k] * y[n - k];
            }
        }
        y[n] = acc;
    }
    return y;
}

std::vector<double> filtfilt(const std::vector<double>& b, const std::vector<double>& a,
                              const std::vector<double>& x) {
    if (x.empty() || b.empty()) {
        return {};
    }

    const size_t n = x.size();
    size_t pad_len = 3 * std::max(a.size(), b.size());
    if (n <= 1) {
        pad_len = 0;
    } else if (pad_len >= n) {
        pad_len = n - 1;
    }

    std::vector<double> padded;
    padded.reserve(n + 2 * pad_len);
    for (size_t j = 0; j < pad_len; ++j) {
        const size_t k = pad_len - j; // k = pad_len .. 1
        padded.push_back(2.0 * x[0] - x[k]);
    }
    padded.insert(padded.end(), x.begin(), x.end());
    for (size_t j = 0; j < pad_len; ++j) {
        const size_t k = j + 1; // k = 1 .. pad_len
        padded.push_back(2.0 * x[n - 1] - x[n - 1 - k]);
    }

    auto forward = filter(b, a, padded);
    std::reverse(forward.begin(), forward.end());
    auto backward = filter(b, a, forward);
    std::reverse(backward.begin(), backward.end());

    return std::vector<double>(backward.begin() + static_cast<ptrdiff_t>(pad_len),
                                backward.end() - static_cast<ptrdiff_t>(pad_len));
}

namespace {

std::vector<double> fir_window(int n_taps, FirWindow window) {
    const size_t n = static_cast<size_t>(n_taps);
    switch (window) {
        case FirWindow::Hamming:
            return hamming(n);
        case FirWindow::Hann:
            return hanning(n);
        case FirWindow::Blackman:
            return blackman(n);
        case FirWindow::Rectangular:
        default:
            return std::vector<double>(n, 1.0);
    }
}

} // namespace

std::vector<double> firwin(int n_taps, double cutoff, FirWindow window) {
    if (n_taps <= 0) {
        return {};
    }

    const size_t n = static_cast<size_t>(n_taps);
    const double center = static_cast<double>(n_taps - 1) / 2.0;
    const double omega_c = M_PI * cutoff;

    std::vector<double> taps(n);
    for (size_t i = 0; i < n; ++i) {
        const double m = static_cast<double>(i) - center;
        if (std::abs(m) < 1e-12) {
            taps[i] = cutoff;
        } else {
            taps[i] = std::sin(omega_c * m) / (M_PI * m);
        }
    }

    const auto win = fir_window(n_taps, window);
    for (size_t i = 0; i < n; ++i) {
        taps[i] *= win[i];
    }

    double sum = 0.0;
    for (double v : taps) {
        sum += v;
    }
    if (std::abs(sum) > 1e-12) {
        for (double& v : taps) {
            v /= sum;
        }
    }
    return taps;
}

std::vector<double> firwin_highpass(int n_taps, double cutoff, FirWindow window) {
    if (n_taps <= 0 || n_taps % 2 == 0) {
        return {};
    }

    auto taps = firwin(n_taps, cutoff, window);
    for (double& v : taps) {
        v = -v;
    }
    const size_t center = static_cast<size_t>(n_taps - 1) / 2;
    taps[center] += 1.0;
    return taps;
}

} // namespace ms
