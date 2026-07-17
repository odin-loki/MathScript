#include "ms/signal/signal.hpp"
#include "ms/core/operations.hpp"
#include "ms/fft/fft.hpp"
#include "ms/poly/poly.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <set>
#include <span>

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

struct WelchSetup {
    SegmentPlan plan;
    std::vector<double> window;
    double scale;
    std::vector<double> frequencies;
};

Result<WelchSetup> prepare_welch(size_t signal_len, double fs, size_t segment_len,
                                  double overlap_frac, const char* fn) {
    if (fs <= 0.0) {
        return std::unexpected(DomainError{fn, "fs must be > 0"});
    }

    const auto plan = plan_segments(signal_len, segment_len, overlap_frac, fn);
    if (!plan) {
        return std::unexpected(plan.error());
    }

    const auto window = hanning(segment_len);
    double win_sq_sum = 0.0;
    for (const double w : window) {
        win_sq_sum += w * w;
    }
    const double scale = 1.0 / (fs * win_sq_sum);

    return WelchSetup{*plan, window, scale, segment_frequencies(plan->n_fft, fs)};
}

Result<void> rfft_windowed_segment(const std::vector<double>& signal, size_t start,
                                   size_t segment_len, const std::vector<double>& window,
                                   std::vector<double>& segment_buf,
                                   std::vector<std::complex<double>>& spec_buf) {
    if (segment_buf.size() != segment_len) {
        segment_buf.resize(segment_len);
    }
    for (size_t i = 0; i < segment_len; ++i) {
        segment_buf[i] = signal[start + i] * window[i];
    }
    const auto spec = rfft(segment_buf);
    if (!spec) {
        return std::unexpected(spec.error());
    }
    spec_buf = std::move(*spec);
    return {};
}

Result<std::vector<std::complex<double>>> rfft_windowed_segment(
    const std::vector<double>& signal, size_t start, size_t segment_len,
    const std::vector<double>& window) {
    std::vector<double> segment;
    std::vector<std::complex<double>> spec;
    const auto status = rfft_windowed_segment(signal, start, segment_len, window, segment, spec);
    if (!status) {
        return std::unexpected(status.error());
    }
    return spec;
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

namespace {

constexpr size_t kConvolveFftCrossover = 64;

std::vector<double> convolve_direct(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i + j] += a[i] * b[j];
        }
    }
    return result;
}

bool convolve_use_fft(size_t a_len, size_t b_len) {
    if (a_len == 0 || b_len == 0) {
        return false;
    }
    const size_t out_len = a_len + b_len - 1;
    if (out_len < kConvolveFftCrossover) {
        return false;
    }
    return a_len >= kConvolveFftCrossover || b_len >= kConvolveFftCrossover;
}

std::vector<double> convolve_fft_linear(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t out_len = a.size() + b.size() - 1;
    const size_t n_fft = next_power_of_two(out_len);

    std::vector<double> a_pad(n_fft, 0.0);
    std::vector<double> b_pad(n_fft, 0.0);
    std::copy(a.begin(), a.end(), a_pad.begin());
    std::copy(b.begin(), b.end(), b_pad.begin());

    const auto spec_a = fft(a_pad);
    const auto spec_b = fft(b_pad);
    if (!spec_a || !spec_b) {
        return {};
    }

    std::vector<std::complex<double>> product(n_fft);
    for (size_t k = 0; k < n_fft; ++k) {
        product[k] = (*spec_a)[k] * (*spec_b)[k];
    }

    const auto restored = ifft(product);
    if (!restored) {
        return {};
    }

    std::vector<double> result = *restored;
    result.resize(out_len);
    return result;
}

std::vector<double> convolve_fft(const std::vector<double>& a, const std::vector<double>& b) {
    auto result = convolve_fft_linear(a, b);
    if (result.empty() && !a.empty() && !b.empty()) {
        return convolve_direct(a, b);
    }
    return result;
}

bool xcorr_use_fft_partial(size_t a_len, size_t b_len, int max_lag) {
    if (max_lag < 0 || a_len == 0 || b_len == 0) {
        return false;
    }
    const size_t full_len = a_len + b_len - 1;
    const size_t out_len = static_cast<size_t>(2 * max_lag + 1);
    if (out_len >= full_len) {
        return false;
    }
    if (!convolve_use_fft(a_len, b_len)) {
        return false;
    }
    const size_t n = std::max(a_len, b_len);
    return static_cast<size_t>(max_lag) <= n / 8 || out_len * 4 < full_len;
}

std::vector<double> xcorr_direct_lags(const std::vector<double>& a, const std::vector<double>& b,
                                      int max_lag) {
    const size_t out_len = static_cast<size_t>(2 * max_lag + 1);
    std::vector<double> out(out_len, 0.0);
    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const ptrdiff_t bi = static_cast<ptrdiff_t>(lag) + static_cast<ptrdiff_t>(i);
            if (bi >= 0 && bi < static_cast<ptrdiff_t>(b.size())) {
                sum += a[i] * b[static_cast<size_t>(bi)];
            }
        }
        out[static_cast<size_t>(lag + max_lag)] = sum;
    }
    return out;
}

std::vector<double> xcorr_slice_from_correlate(const std::vector<double>& full, size_t b_len,
                                               int max_lag) {
    const ptrdiff_t zero_idx = static_cast<ptrdiff_t>(b_len - 1);
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

std::vector<double> xcorr_fft_lags(const std::vector<double>& a, const std::vector<double>& b,
                                   int max_lag) {
    std::vector<double> rev_b(b.rbegin(), b.rend());
    auto linear = convolve_fft_linear(a, rev_b);
    if (linear.empty()) {
        return xcorr_direct_lags(a, b, max_lag);
    }
    return xcorr_slice_from_correlate(linear, b.size(), max_lag);
}

} // namespace

std::vector<double> convolve(const std::vector<double>& a, const std::vector<double>& b) {
    if (convolve_use_fft(a.size(), b.size())) {
        return convolve_fft(a, b);
    }
    return convolve_direct(a, b);
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
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i];
        if (i >= window) {
            sum -= x[i - window];
        }
        const size_t count = std::min(i + 1, window);
        out[i] = sum / static_cast<double>(count);
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
    const auto setup = prepare_welch(x.size(), fs, segment_len, overlap_frac, "welch_psd");
    if (!setup) {
        return std::unexpected(setup.error());
    }

    std::vector<double> psd(setup->plan.n_freq_bins, 0.0);
    std::vector<double> segment_buf;
    std::vector<std::complex<double>> spec_buf;
    segment_buf.reserve(segment_len);
    spec_buf.reserve(setup->plan.n_freq_bins);
    size_t seg_count = 0;

    for (size_t start = 0; start + segment_len <= x.size(); start += setup->plan.hop) {
        const auto status =
            rfft_windowed_segment(x, start, segment_len, setup->window, segment_buf, spec_buf);
        if (!status) {
            return std::unexpected(status.error());
        }

        for (size_t k = 0; k < setup->plan.n_freq_bins && k < spec_buf.size(); ++k) {
            psd[k] += std::norm(spec_buf[k]);
        }
        ++seg_count;
    }

    for (double& p : psd) {
        p = (p / static_cast<double>(seg_count)) * setup->scale;
    }
    apply_one_sided_psd_scaling(psd, setup->plan.n_fft);

    return PSDResult{setup->frequencies, std::move(psd)};
}

Result<CoherenceResult> coherence(const std::vector<double>& x, const std::vector<double>& y,
                                   double fs, size_t nperseg, double overlap_frac) {
    if (x.size() != y.size()) {
        return std::unexpected(DomainError{"coherence", "x and y must have the same length"});
    }

    const auto setup = prepare_welch(x.size(), fs, nperseg, overlap_frac, "coherence");
    if (!setup) {
        return std::unexpected(setup.error());
    }

    std::vector<double> pxx(setup->plan.n_freq_bins, 0.0);
    std::vector<double> pyy(setup->plan.n_freq_bins, 0.0);
    std::vector<std::complex<double>> pxy(setup->plan.n_freq_bins, 0.0);
    size_t seg_count = 0;

    for (size_t start = 0; start + nperseg <= x.size(); start += setup->plan.hop) {
        const auto spec_x = rfft_windowed_segment(x, start, nperseg, setup->window);
        if (!spec_x) {
            return std::unexpected(spec_x.error());
        }
        const auto spec_y = rfft_windowed_segment(y, start, nperseg, setup->window);
        if (!spec_y) {
            return std::unexpected(spec_y.error());
        }

        for (size_t k = 0; k < setup->plan.n_freq_bins && k < spec_x->size() && k < spec_y->size();
             ++k) {
            pxx[k] += std::norm((*spec_x)[k]);
            pyy[k] += std::norm((*spec_y)[k]);
            pxy[k] += (*spec_x)[k] * std::conj((*spec_y)[k]);
        }
        ++seg_count;
    }

    const double inv_seg = 1.0 / static_cast<double>(seg_count);
    std::vector<double> coh(setup->plan.n_freq_bins, 0.0);
    for (size_t k = 0; k < coh.size(); ++k) {
        const double auto_x = pxx[k] * inv_seg * setup->scale;
        const double auto_y = pyy[k] * inv_seg * setup->scale;
        const auto cross = pxy[k] * inv_seg * setup->scale;
        const double denom = auto_x * auto_y;
        if (denom > 0.0) {
            coh[k] = std::norm(cross) / denom;
            if (coh[k] > 1.0) {
                coh[k] = 1.0;
            }
        }
    }

    return CoherenceResult{setup->frequencies, std::move(coh)};
}

Result<SpectrogramResult> spectrogram(const std::vector<double>& x, double fs,
                                       size_t segment_len, double overlap_frac) {
    const auto setup = prepare_welch(x.size(), fs, segment_len, overlap_frac, "spectrogram");
    if (!setup) {
        return std::unexpected(setup.error());
    }

    SpectrogramResult out;
    out.frequencies = setup->frequencies;
    out.times.reserve(setup->plan.n_segments);
    out.magnitude = Matrix<double>(setup->plan.n_segments, setup->plan.n_freq_bins);

    std::vector<double> segment_buf;
    std::vector<std::complex<double>> spec_buf;
    segment_buf.reserve(segment_len);
    spec_buf.reserve(setup->plan.n_freq_bins);

    size_t row = 0;
    for (size_t start = 0; start + segment_len <= x.size(); start += setup->plan.hop) {
        const auto status =
            rfft_windowed_segment(x, start, segment_len, setup->window, segment_buf, spec_buf);
        if (!status) {
            return std::unexpected(status.error());
        }

        out.times.push_back((static_cast<double>(start) + static_cast<double>(segment_len) / 2.0) / fs);

        for (size_t k = 0; k < setup->plan.n_freq_bins && k < spec_buf.size(); ++k) {
            out.magnitude(row, k) = std::abs(spec_buf[k]);
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

    const size_t full_len = a.size() + b.size() - 1;
    const size_t out_len = static_cast<size_t>(2 * max_lag + 1);

    if (out_len >= full_len) {
        return xcorr_slice_from_correlate(correlate(a, b), b.size(), max_lag);
    }
    if (xcorr_use_fft_partial(a.size(), b.size(), max_lag)) {
        return xcorr_fft_lags(a, b, max_lag);
    }
    return xcorr_direct_lags(a, b, max_lag);
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

namespace {

struct Zpk {
    std::vector<std::complex<double>> z;
    std::vector<std::complex<double>> p;
    double k = 1.0;
};

std::vector<double> poly_mul_asc(const std::vector<double>& a, const std::vector<double>& b) {
    return poly::poly_mul(a, b);
}

std::vector<double> poly_from_poles_zinv(const std::vector<std::complex<double>>& poles) {
    std::vector<double> poly{1.0};
    std::vector<bool> used(poles.size(), false);
    for (size_t i = 0; i < poles.size(); ++i) {
        if (used[i]) {
            continue;
        }
        const auto& pole = poles[i];
        if (std::abs(pole.imag()) < 1e-12) {
            const std::vector<double> factor{1.0, -pole.real()};
            poly = poly_mul_asc(poly, factor);
            used[i] = true;
            continue;
        }
        for (size_t j = i + 1; j < poles.size(); ++j) {
            if (!used[j] && std::abs(poles[j] - std::conj(pole)) < 1e-10) {
                const std::vector<double> factor{
                    1.0,
                    -2.0 * pole.real(),
                    std::norm(pole),
                };
                poly = poly_mul_asc(poly, factor);
                used[i] = used[j] = true;
                break;
            }
        }
    }
    return poly;
}

std::vector<double> poly_from_zeros_zinv(const std::vector<std::complex<double>>& zeros) {
    return poly_from_poles_zinv(zeros);
}

Zpk cheb1ap(int order, double rp_db) {
    const double epsilon = std::sqrt(std::pow(10.0, 0.1 * rp_db) - 1.0);
    const double mu = std::asinh(1.0 / epsilon) / static_cast<double>(order);

    Zpk sys;
    sys.p.reserve(static_cast<size_t>(order));
    for (int m = -order + 1; m < order; m += 2) {
        const double theta = M_PI * static_cast<double>(m) / (2.0 * static_cast<double>(order));
        sys.p.emplace_back(-std::sinh(mu) * std::cos(theta), -std::cosh(mu) * std::sin(theta));
    }

    std::complex<double> neg_p_prod{1.0, 0.0};
    for (const auto& pole : sys.p) {
        neg_p_prod *= -pole;
    }
    sys.k = neg_p_prod.real();
    if (order % 2 == 0) {
        sys.k /= std::sqrt(1.0 + epsilon * epsilon);
    }
    return sys;
}

Zpk cheb2ap(int order, double rs_db) {
    const double de = 1.0 / std::sqrt(std::pow(10.0, 0.1 * rs_db) - 1.0);
    const double mu = std::asinh(1.0 / de) / static_cast<double>(order);

    Zpk sys;
    sys.z.reserve(static_cast<size_t>(order));
    sys.p.reserve(static_cast<size_t>(order));

    if (order % 2 == 0) {
        for (int m = -order + 1; m < order; m += 2) {
            const double zm = M_PI * static_cast<double>(m) / (2.0 * static_cast<double>(order));
            sys.z.emplace_back(0.0, 1.0 / std::sin(zm));
        }
    } else {
        for (int m = -order + 1; m < 0; m += 2) {
            const double zm = M_PI * static_cast<double>(m) / (2.0 * static_cast<double>(order));
            sys.z.emplace_back(0.0, 1.0 / std::sin(zm));
        }
        for (int m = 2; m < order; m += 2) {
            const double zm = M_PI * static_cast<double>(m) / (2.0 * static_cast<double>(order));
            sys.z.emplace_back(0.0, 1.0 / std::sin(zm));
        }
    }

    for (int m = -order + 1; m < order; m += 2) {
        const double theta = M_PI * static_cast<double>(m) / (2.0 * static_cast<double>(order));
        const std::complex<double> sinh_term{
            std::sinh(mu) * std::cos(theta),
            std::cosh(mu) * std::sin(theta),
        };
        sys.p.push_back(-1.0 / sinh_term);
    }

    std::complex<double> neg_p_prod{1.0, 0.0};
    std::complex<double> neg_z_prod{1.0, 0.0};
    for (const auto& pole : sys.p) {
        neg_p_prod *= -pole;
    }
    for (const auto& zero : sys.z) {
        neg_z_prod *= -zero;
    }
    sys.k = (neg_p_prod / neg_z_prod).real();
    return sys;
}

Zpk lp2lp_zpk(const Zpk& sys, double wo) {
    Zpk out;
    out.z.reserve(sys.z.size());
    out.p.reserve(sys.p.size());
    for (const auto& z : sys.z) {
        out.z.push_back(wo * z);
    }
    for (const auto& p : sys.p) {
        out.p.push_back(wo * p);
    }
    const int degree = static_cast<int>(sys.p.size()) - static_cast<int>(sys.z.size());
    out.k = sys.k * std::pow(wo, static_cast<double>(degree));
    return out;
}

Zpk lp2hp_zpk(const Zpk& sys, double wo) {
    Zpk out;
    out.z.reserve(sys.z.size() + sys.p.size());
    out.p.reserve(sys.p.size());

    std::complex<double> neg_z_prod{1.0, 0.0};
    std::complex<double> neg_p_prod{1.0, 0.0};
    for (const auto& z : sys.z) {
        neg_z_prod *= -z;
    }
    for (const auto& p : sys.p) {
        neg_p_prod *= -p;
    }
    out.k = sys.k * (neg_z_prod / neg_p_prod).real();

    for (const auto& z : sys.z) {
        if (std::abs(z) < 1e-15) {
            out.z.emplace_back(std::numeric_limits<double>::infinity(), 0.0);
        } else {
            out.z.push_back(wo / z);
        }
    }
    for (const auto& p : sys.p) {
        out.p.push_back(wo / p);
    }
    const int degree = static_cast<int>(sys.p.size()) - static_cast<int>(sys.z.size());
    for (int i = 0; i < degree; ++i) {
        out.z.emplace_back(0.0, 0.0);
    }
    return out;
}

Zpk bilinear_zpk(const Zpk& sys, double fs) {
    const double fs2 = 2.0 * fs;
    Zpk out;
    out.z.reserve(sys.z.size() + sys.p.size());
    out.p.reserve(sys.p.size());

    for (const auto& z : sys.z) {
        out.z.push_back((fs2 + z) / (fs2 - z));
    }
    for (const auto& p : sys.p) {
        out.p.push_back((fs2 + p) / (fs2 - p));
    }

    const int degree = static_cast<int>(sys.p.size()) - static_cast<int>(sys.z.size());
    for (int i = 0; i < degree; ++i) {
        out.z.emplace_back(-1.0, 0.0);
    }

    std::complex<double> num{1.0, 0.0};
    std::complex<double> den{1.0, 0.0};
    for (const auto& z : sys.z) {
        num *= (fs2 - z);
    }
    for (const auto& p : sys.p) {
        den *= (fs2 - p);
    }
    out.k = sys.k * (num / den).real();
    return out;
}

bool valid_cutoff(double cutoff, double fs, FilterType type) {
    if (!(cutoff > 0.0) || !(fs > 0.0) || !(cutoff < fs / 2.0)) {
        return false;
    }
    (void)type;
    return true;
}

} // namespace

IirCoeffs cheby1(int order, double rp_db, double cutoff, double fs, FilterType type) {
    if (order < 1 || rp_db < 0.0 || !valid_cutoff(cutoff, fs, type)) {
        return {};
    }

    const double wn = cutoff / (fs / 2.0);
    // Match scipy.signal.iirfilter: prewarp normalized Wn, then bilinear with fs=2.
    const double warped = 4.0 * std::tan(M_PI * wn / 2.0);

    Zpk sys = cheb1ap(order, rp_db);
    if (type == FilterType::Highpass) {
        sys = lp2hp_zpk(sys, warped);
    } else {
        sys = lp2lp_zpk(sys, warped);
    }
    sys = bilinear_zpk(sys, 2.0);

    IirCoeffs coeffs;
    coeffs.a = poly_from_poles_zinv(sys.p);
    coeffs.b = poly_from_zeros_zinv(sys.z);
    if (std::abs(sys.k - 1.0) > 1e-15) {
        for (double& v : coeffs.b) {
            v *= sys.k;
        }
    }
    if (!coeffs.a.empty() && std::abs(coeffs.a[0]) > 1e-15) {
        const double scale = coeffs.a[0];
        for (double& v : coeffs.b) {
            v /= scale;
        }
        for (double& v : coeffs.a) {
            v /= scale;
        }
    }
    return coeffs;
}

IirCoeffs cheby2(int order, double rs_db, double cutoff, double fs, FilterType type) {
    if (order < 1 || rs_db < 0.0 || !valid_cutoff(cutoff, fs, type)) {
        return {};
    }

    const double wn = cutoff / (fs / 2.0);
    const double warped = 4.0 * std::tan(M_PI * wn / 2.0);

    Zpk sys = cheb2ap(order, rs_db);
    if (type == FilterType::Highpass) {
        sys = lp2hp_zpk(sys, warped);
    } else {
        sys = lp2lp_zpk(sys, warped);
    }
    sys = bilinear_zpk(sys, 2.0);

    IirCoeffs coeffs;
    coeffs.a = poly_from_poles_zinv(sys.p);
    coeffs.b = poly_from_zeros_zinv(sys.z);
    if (std::abs(sys.k - 1.0) > 1e-15) {
        for (double& v : coeffs.b) {
            v *= sys.k;
        }
    }
    if (!coeffs.a.empty() && std::abs(coeffs.a[0]) > 1e-15) {
        const double scale = coeffs.a[0];
        for (double& v : coeffs.b) {
            v /= scale;
        }
        for (double& v : coeffs.a) {
            v /= scale;
        }
    }
    return coeffs;
}

namespace {

void filter_df2t(std::span<const double> b, std::span<const double> a, std::span<const double> x,
                 std::span<double> y) {
    if (x.empty() || b.empty() || y.size() < x.size()) {
        return;
    }

    const double a0 = a.empty() ? 1.0 : a[0];
    const double inv_a0 = (a0 != 1.0 && a0 != 0.0) ? (1.0 / a0) : 1.0;
    const size_t b_len = b.size();
    const size_t a_len = a.empty() ? 1 : a.size();
    const size_t order = std::max(b_len, a_len) - 1;
    std::vector<double> z(order, 0.0);

    for (size_t n = 0; n < x.size(); ++n) {
        const double xi = x[n];
        double yi = b[0] * inv_a0 * xi;
        if (order > 0) {
            yi += z[0];
        }
        for (size_t i = 0; i + 1 < order; ++i) {
            const double bi = (i + 1 < b_len) ? b[i + 1] * inv_a0 : 0.0;
            const double ai = (i + 1 < a_len) ? a[i + 1] * inv_a0 : 0.0;
            z[i] = z[i + 1] + bi * xi - ai * yi;
        }
        if (order > 0) {
            const double bi = (order < b_len) ? b[order] * inv_a0 : 0.0;
            const double ai = (order < a_len) ? a[order] * inv_a0 : 0.0;
            z[order - 1] = bi * xi - ai * yi;
        }
        y[n] = yi;
    }
}

void filter_df2t_in_place(std::span<const double> b, std::span<const double> a,
                          std::span<double> x_y) {
    if (x_y.empty() || b.empty()) {
        return;
    }

    const double a0 = a.empty() ? 1.0 : a[0];
    const double inv_a0 = (a0 != 1.0 && a0 != 0.0) ? (1.0 / a0) : 1.0;
    const size_t b_len = b.size();
    const size_t a_len = a.empty() ? 1 : a.size();
    const size_t order = std::max(b_len, a_len) - 1;
    std::vector<double> z(order, 0.0);

    for (size_t n = 0; n < x_y.size(); ++n) {
        const double xi = x_y[n];
        double yi = b[0] * inv_a0 * xi;
        if (order > 0) {
            yi += z[0];
        }
        for (size_t i = 0; i + 1 < order; ++i) {
            const double bi = (i + 1 < b_len) ? b[i + 1] * inv_a0 : 0.0;
            const double ai = (i + 1 < a_len) ? a[i + 1] * inv_a0 : 0.0;
            z[i] = z[i + 1] + bi * xi - ai * yi;
        }
        if (order > 0) {
            const double bi = (order < b_len) ? b[order] * inv_a0 : 0.0;
            const double ai = (order < a_len) ? a[order] * inv_a0 : 0.0;
            z[order - 1] = bi * xi - ai * yi;
        }
        x_y[n] = yi;
    }
}

void filter_sos_section(std::span<const double> x, std::span<double> y,
                        const std::array<double, 6>& section) {
    if (x.empty() || y.size() < x.size()) {
        return;
    }

    const double inv_a0 =
        (section[3] != 1.0 && section[3] != 0.0) ? (1.0 / section[3]) : 1.0;
    const double b0 = section[0] * inv_a0;
    const double b1 = section[1] * inv_a0;
    const double b2 = section[2] * inv_a0;
    const double a1 = section[4] * inv_a0;
    const double a2 = section[5] * inv_a0;
    double z1 = 0.0;
    double z2 = 0.0;

    for (size_t n = 0; n < x.size(); ++n) {
        const double xi = x[n];
        const double yi = b0 * xi + z1;
        z1 = z2 + b1 * xi - a1 * yi;
        z2 = b2 * xi - a2 * yi;
        y[n] = yi;
    }
}

void filter_sos_section_in_place(std::span<double> y, const std::array<double, 6>& section) {
    if (y.empty()) {
        return;
    }

    const double inv_a0 =
        (section[3] != 1.0 && section[3] != 0.0) ? (1.0 / section[3]) : 1.0;
    const double b0 = section[0] * inv_a0;
    const double b1 = section[1] * inv_a0;
    const double b2 = section[2] * inv_a0;
    const double a1 = section[4] * inv_a0;
    const double a2 = section[5] * inv_a0;
    double z1 = 0.0;
    double z2 = 0.0;

    for (size_t n = 0; n < y.size(); ++n) {
        const double xi = y[n];
        const double yi = b0 * xi + z1;
        z1 = z2 + b1 * xi - a1 * yi;
        z2 = b2 * xi - a2 * yi;
        y[n] = yi;
    }
}

} // namespace

std::vector<double> filter(const std::vector<double>& b, const std::vector<double>& a,
                            const std::vector<double>& x) {
    if (x.empty() || b.empty()) {
        return {};
    }

    std::vector<double> y(x.size());
    filter(b, a, std::span<const double>(x), std::span<double>(y));
    return y;
}

void filter(const std::vector<double>& b, const std::vector<double>& a,
            std::span<const double> x, std::span<double> y) {
    filter_df2t(std::span<const double>(b), std::span<const double>(a), x, y);
}

void filter_in_place(const std::vector<double>& b, const std::vector<double>& a,
                     std::span<double> x_y) {
    filter_df2t_in_place(std::span<const double>(b), std::span<const double>(a), x_y);
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

std::vector<double> sosfilt(const std::vector<std::array<double, 6>>& sos,
                             const std::vector<double>& x) {
    if (x.empty()) {
        return sos.empty() ? x : std::vector<double>{};
    }
    if (sos.empty()) {
        return x;
    }

    std::vector<double> y(x.size());
    filter_sos_section(std::span<const double>(x), std::span<double>(y), sos.front());
    for (size_t i = 1; i < sos.size(); ++i) {
        filter_sos_section_in_place(std::span<double>(y), sos[i]);
    }
    return y;
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

namespace {

// Builds the fixed FIR taps (length window_length) of the centered Savitzky-Golay smoothing
// filter: A is the Vandermonde design matrix of integer window offsets -half..+half raised to
// powers 0..polyorder; v solves the (polyorder+1)x(polyorder+1) normal-equations system
// (A^T A) v = e_0 (e_0 selects the polynomial's value at offset 0, the window center); the taps
// are h = A*v, the first row of the least-squares pseudoinverse (A^T A)^{-1} A^T. Returns an
// empty vector if the normal-equations solve fails (e.g. a degenerate/singular design, which
// should not occur for the valid window_length/polyorder ranges validated by the caller).
std::vector<double> savgol_coefficients(int window_length, int polyorder) {
    const int m = window_length;
    const int half = (m - 1) / 2;
    const size_t n_coef = static_cast<size_t>(polyorder + 1);

    Matrix<double> A(static_cast<size_t>(m), n_coef);
    for (int i = 0; i < m; ++i) {
        const double offset = static_cast<double>(i - half);
        double power = 1.0;
        for (size_t j = 0; j < n_coef; ++j) {
            A(static_cast<size_t>(i), j) = power;
            power *= offset;
        }
    }

    Matrix<double> ata(n_coef, n_coef, 0.0);
    for (size_t j = 0; j < n_coef; ++j) {
        for (size_t k = 0; k < n_coef; ++k) {
            double sum = 0.0;
            for (int i = 0; i < m; ++i) {
                sum += A(static_cast<size_t>(i), j) * A(static_cast<size_t>(i), k);
            }
            ata(j, k) = sum;
        }
    }

    Matrix<double> e0(n_coef, 1, 0.0);
    e0(0, 0) = 1.0;

    const auto v = solve(ata, e0);
    if (!v) {
        return {};
    }

    std::vector<double> h(static_cast<size_t>(m), 0.0);
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n_coef; ++j) {
            sum += A(static_cast<size_t>(i), j) * (*v)(j, 0);
        }
        h[static_cast<size_t>(i)] = sum;
    }
    return h;
}

} // namespace

std::vector<double> savgol(const std::vector<double>& x, int window_length, int polyorder) {
    if (window_length <= 0 || window_length % 2 == 0) {
        return {};
    }
    if (polyorder < 0 || polyorder >= window_length) {
        return {};
    }
    if (x.size() < static_cast<size_t>(window_length)) {
        return {};
    }

    const auto h = savgol_coefficients(window_length, polyorder);
    if (h.size() != static_cast<size_t>(window_length)) {
        return {};
    }

    const size_t half = static_cast<size_t>((window_length - 1) / 2);
    const size_t n = x.size();

    // Boundary points without a full centered window are left unfiltered (copied from x); see
    // the @note on boundary handling in signal.hpp.
    std::vector<double> y = x;
    for (size_t center = half; center + half < n; ++center) {
        double acc = 0.0;
        for (size_t j = 0; j < h.size(); ++j) {
            acc += h[j] * x[center - half + j];
        }
        y[center] = acc;
    }
    return y;
}

namespace {

inline void median_compare_swap(double& a, double& b) {
    if (a > b) {
        std::swap(a, b);
    }
}

// Fixed 3-comparator sorting network: median of three values in O(1).
inline double median_of_three(double a, double b, double c) {
    median_compare_swap(a, b);
    median_compare_swap(b, c);
    median_compare_swap(a, b);
    return b;
}

// Fixed 9-comparator full-sort network for five values: median at index 2 in O(1).
inline double median_of_five(double a, double b, double c, double d, double e) {
    median_compare_swap(a, b);
    median_compare_swap(d, e);
    median_compare_swap(a, c);
    median_compare_swap(b, e);
    median_compare_swap(b, c);
    median_compare_swap(a, b);
    median_compare_swap(c, e);
    median_compare_swap(c, d);
    median_compare_swap(b, c);
    return c;
}

std::vector<double> median_filter_sliding_multiset(const std::vector<double>& x, size_t half,
                                                    size_t window_length) {
    const size_t n = x.size();
    std::vector<double> y = x;

    std::multiset<double> window;
    for (size_t j = 0; j < window_length; ++j) {
        window.insert(x[j]);
    }

    auto median_it = std::next(window.begin(), static_cast<std::ptrdiff_t>(half));
    y[half] = *median_it;

    for (size_t center = half + 1; center + half < n; ++center) {
        const double outgoing = x[center - half - 1];
        const double incoming = x[center + half];
        const auto outgoing_it = window.find(outgoing);
        window.erase(outgoing_it);
        window.insert(incoming);
        median_it = std::next(window.begin(), static_cast<std::ptrdiff_t>(half));
        y[center] = *median_it;
    }
    return y;
}

} // namespace

std::vector<double> median_filter(const std::vector<double>& x, int window_length) {
    if (window_length <= 0 || window_length % 2 == 0) {
        return {};
    }
    if (x.size() < static_cast<size_t>(window_length)) {
        return {};
    }

    const size_t half = static_cast<size_t>((window_length - 1) / 2);
    const size_t n = x.size();

    // Boundary points without a full centered window are left unfiltered (copied from x),
    // mirroring savgol's boundary convention exactly; see the @note in signal.hpp.
    std::vector<double> y = x;

    if (window_length == 1) {
        return y;
    }
    if (window_length == 3) {
        for (size_t center = half; center + half < n; ++center) {
            y[center] = median_of_three(x[center - 1], x[center], x[center + 1]);
        }
        return y;
    }
    if (window_length == 5) {
        for (size_t center = half; center + half < n; ++center) {
            y[center] = median_of_five(x[center - 2], x[center - 1], x[center], x[center + 1],
                                       x[center + 2]);
        }
        return y;
    }

    return median_filter_sliding_multiset(x, half, static_cast<size_t>(window_length));
}

LMSResult lms_adaptive_filter(const std::vector<double>& x, const std::vector<double>& d,
                               int filter_length, double mu) {
    if (filter_length <= 0 || x.size() != d.size() ||
        x.size() < static_cast<size_t>(filter_length)) {
        return LMSResult{};
    }

    const size_t n_taps = static_cast<size_t>(filter_length);
    const size_t n = x.size();

    LMSResult result;
    result.output.assign(n, 0.0);
    result.error.assign(n, 0.0);
    result.weights.assign(n_taps, 0.0);

    for (size_t i = 0; i < n; ++i) {
        double y_n = 0.0;
        for (size_t k = 0; k < n_taps; ++k) {
            if (i >= k) {
                y_n += result.weights[k] * x[i - k];
            }
        }
        const double e_n = d[i] - y_n;
        result.output[i] = y_n;
        result.error[i] = e_n;

        for (size_t k = 0; k < n_taps; ++k) {
            if (i >= k) {
                result.weights[k] += mu * e_n * x[i - k];
            }
        }
    }

    return result;
}

std::vector<std::complex<double>> czt(const std::vector<double>& x, int m,
                                       std::complex<double> w, std::complex<double> a) {
    if (x.empty() || m <= 0) {
        return {};
    }

    const size_t n = x.size();
    const size_t m_size = static_cast<size_t>(m);
    const size_t max_nm = std::max(m_size, n);

    std::vector<std::complex<double>> wk2(max_nm);
    for (size_t k = 0; k < max_nm; ++k) {
        const double k_d = static_cast<double>(k);
        wk2[k] = std::pow(w, k_d * k_d / 2.0);
    }

    std::vector<std::complex<double>> ak(n);
    for (size_t idx = 0; idx < n; ++idx) {
        const double idx_d = static_cast<double>(idx);
        ak[idx] = x[idx] * std::pow(a, -idx_d) * wk2[idx];
    }

    const size_t ww_len = n + m_size - 1;
    std::vector<std::complex<double>> ww(ww_len);
    for (size_t i = 0; i + 1 < n; ++i) {
        ww[i] = wk2[n - 1 - i];
    }
    for (size_t k = 0; k < m_size; ++k) {
        ww[n - 1 + k] = wk2[k];
    }

    const size_t conv_len = n + m_size - 1;
    const size_t nfft = next_power_of_two(conv_len);

    std::vector<std::complex<double>> Ak(nfft, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < n; ++i) {
        Ak[i] = ak[i];
    }

    std::vector<std::complex<double>> Ww(nfft, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < ww_len; ++i) {
        Ww[i] = 1.0 / ww[i];
    }

    Ak = fft_recursive(std::move(Ak));
    Ww = fft_recursive(std::move(Ww));
    for (size_t i = 0; i < nfft; ++i) {
        Ak[i] *= Ww[i];
    }

    auto fk = complex_ifft(std::move(Ak));
    if (fk.size() < n - 1 + m_size) {
        return {};
    }

    std::vector<std::complex<double>> result(m_size);
    for (size_t k = 0; k < m_size; ++k) {
        result[k] = fk[n - 1 + k] * wk2[k];
    }
    return result;
}

std::vector<std::complex<double>> czt_zoom_fft(const std::vector<double>& x, double f_start,
                                                double f_stop, int m, double fs) {
    if (x.empty() || m <= 0 || fs <= 0.0 || f_start < 0.0 || f_stop < f_start ||
        f_stop > fs / 2.0) {
        return {};
    }

    const double delta_f = (m > 1) ? (f_stop - f_start) / static_cast<double>(m - 1) : 0.0;
    const double arg_w = -2.0 * M_PI * delta_f / fs;
    const std::complex<double> w(std::cos(arg_w), std::sin(arg_w));
    const double arg_a = 2.0 * M_PI * f_start / fs;
    const std::complex<double> a(std::cos(arg_a), std::sin(arg_a));

    return czt(x, m, w, a);
}

} // namespace ms
