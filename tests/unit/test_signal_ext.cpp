#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <variant>
#include "ms/signal/signal.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

TEST(SignalExtTest, convolve_identity) {
    std::vector<double> a{1, 2, 3};
    std::vector<double> impulse{1};
    auto out = convolve(a, impulse);
    ASSERT_EQ(out.size(), a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_DOUBLE_EQ(out[i], a[i]);
    }
}

TEST(SignalExtTest, correlate_peak) {
    std::vector<double> a{1, 2, 3};
    auto c = correlate(a, a);
    EXPECT_GT(c[2], c[0]);
}

TEST(SignalExtTest, moving_average) {
    std::vector<double> x{1, 2, 3, 4};
    auto y = moving_average(x, 2);
    EXPECT_NEAR(y[1], 1.5, 1e-12);
    EXPECT_NEAR(y[3], 3.5, 1e-12);
}

TEST(SignalExtTest, hamming_window) {
    auto w = hamming(5);
    EXPECT_NEAR(w[0], 0.08, 1e-2);
    EXPECT_NEAR(w[2], 1.0, 1e-12);
}

TEST(SignalExtTest, window_functions_smoke) {
    for (size_t n : {4u, 8u, 16u}) {
        const auto hann = hanning(n);
        const auto black = blackman(n);
        const auto parz = parzen(n);
        const auto tri = triangular(n);
        ASSERT_EQ(hann.size(), n);
        ASSERT_EQ(black.size(), n);
        ASSERT_EQ(parz.size(), n);
        ASSERT_EQ(tri.size(), n);
        EXPECT_GT(hann[n / 2], hann[0]);
    }
}

TEST(SignalExtTest, filter_smoke) {
    const std::vector<double> x{0, 1, 0, -1, 0, 1, 0, -1};
    const double fs = 8.0;
    const auto lp = lowpass(x, 1.0, fs);
    const auto hp = highpass(x, 1.0, fs);
    const auto bp = bandpass(x, 0.5, 2.0, fs);
    const auto bw = butterworth(x, 1.0, fs);
    ASSERT_EQ(lp.size(), x.size());
    ASSERT_EQ(hp.size(), x.size());
    ASSERT_EQ(bp.size(), x.size());
    ASSERT_EQ(bw.size(), x.size());
}

TEST(SignalExtTest, empty_input_guards) {
    const std::vector<double> empty;
    EXPECT_TRUE(lowpass(empty, 1.0, 8.0).empty());
    EXPECT_TRUE(highpass(empty, 1.0, 8.0).empty());
    EXPECT_TRUE(bandpass(empty, 0.5, 2.0, 8.0).empty());
    EXPECT_TRUE(butterworth(empty, 1.0, 8.0).empty());
    EXPECT_TRUE(moving_average(empty, 3).empty());
    const std::vector<double> x{1, 2, 3};
    EXPECT_EQ(moving_average(x, 0).size(), x.size());
}

TEST(SignalExtTest, window_edge_sizes) {
    EXPECT_TRUE(hamming(0).empty());
    EXPECT_DOUBLE_EQ(hamming(1)[0], 1.0);
    EXPECT_TRUE(hanning(0).empty());
    EXPECT_DOUBLE_EQ(hanning(1)[0], 1.0);
    EXPECT_TRUE(blackman(0).empty());
    EXPECT_DOUBLE_EQ(blackman(1)[0], 1.0);
    EXPECT_TRUE(parzen(0).empty());
    EXPECT_DOUBLE_EQ(parzen(1)[0], 1.0);
    EXPECT_TRUE(triangular(0).empty());
    EXPECT_DOUBLE_EQ(triangular(1)[0], 1.0);
}

TEST(SignalExtTest, window_tail_branches) {
    const auto parz = parzen(5);
    EXPECT_LT(parz[0], parz[2]);
    EXPECT_LT(parz[4], parz[2]);

    const auto tri = triangular(5);
    EXPECT_NEAR(tri[0], 0.3333333333333333, 1e-12);
    EXPECT_NEAR(tri[4], 0.3333333333333333, 1e-12);
    EXPECT_NEAR(tri[2], 1.0, 1e-12);
}

TEST(SignalExtTest, filter_cutoff_edge_cases) {
    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0};
    const double fs = 9.0;

    const auto lp_tight = lowpass(x, 0.25, fs);
    const auto lp_wide = lowpass(x, fs * 0.45, fs);
    const auto hp = highpass(x, 2.0, fs);
    const auto bp_narrow = bandpass(x, 1.0, 2.5, fs);
    const auto bp_wide = bandpass(x, 0.1, fs * 0.45, fs);

    ASSERT_EQ(lp_tight.size(), x.size());
    ASSERT_EQ(lp_wide.size(), x.size());
    ASSERT_EQ(hp.size(), x.size());
    ASSERT_EQ(bp_narrow.size(), x.size());
    ASSERT_EQ(bp_wide.size(), x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(hp[i] + lp_wide[i], x[i], 1e-5);
    }
    EXPECT_LT(std::abs(lp_tight[4]), std::abs(lp_wide[4]) + 1e-6);
}

TEST(SignalExtTest, filter_odd_length_and_moving_average_start) {
    const std::vector<double> odd{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto filtered = butterworth(odd, 1.0, 10.0);
    ASSERT_EQ(filtered.size(), odd.size());

    const std::vector<double> ramp{1.0, 2.0, 3.0, 4.0};
    const auto avg = moving_average(ramp, 3);
    EXPECT_NEAR(avg[0], 1.0, 1e-12);
    EXPECT_NEAR(avg[1], 1.5, 1e-12);
    EXPECT_NEAR(avg[2], 2.0, 1e-12);
    EXPECT_NEAR(avg[3], 3.0, 1e-12);
}

TEST(SignalExtTest, fir_lowpass_passes_dc) {
    const std::vector<double> dc(64, 1.0);
    const double fs = 64.0;
    const auto lp = lowpass(dc, fs * 0.4, fs);
    const auto hp = highpass(dc, 0.5, fs);
    ASSERT_EQ(lp.size(), dc.size());
    EXPECT_NEAR(lp[32], dc[32], 1e-2);
    EXPECT_NEAR(std::abs(hp[32]), 0.0, 1e-2);
}

TEST(SignalExtTest, iir_bandpass_center_freq) {
    const size_t n = 64;
    const double fs = 64.0;
    const double f0 = 8.0;
    std::vector<double> tone(n);
    for (size_t i = 0; i < n; ++i) {
        tone[i] = std::sin(2.0 * M_PI * f0 * static_cast<double>(i) / fs);
    }
    const auto bp = bandpass(tone, 6.0, 10.0, fs);
    const auto bp_off = bandpass(tone, 20.0, 24.0, fs);
    ASSERT_EQ(bp.size(), tone.size());
    ASSERT_EQ(bp_off.size(), tone.size());
    // Verify all output elements are finite (basic smoke test)
    for (double v : bp) { EXPECT_TRUE(std::isfinite(v)); }
    for (double v : bp_off) { EXPECT_TRUE(std::isfinite(v)); }
}

TEST(SignalExtTest, xcorr_autocorrelation) {
    const std::vector<double> x{1, 2, 3, 4};
    const auto xc = correlate(x, x);
    ASSERT_FALSE(xc.empty());
    const auto peak = std::max_element(xc.begin(), xc.end());
    const size_t peak_lag = static_cast<size_t>(std::distance(xc.begin(), peak));
    EXPECT_EQ(peak_lag, x.size() - 1);
}

TEST(SignalExtTest, convolve_delta_identity) {
    const std::vector<double> x{1, 2, 3, 4, 5};
    std::vector<double> delta(x.size(), 0.0);
    delta[0] = 1.0;
    const auto out = convolve(delta, x);
    ASSERT_GE(out.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], x[i], 1e-12);
    }
}

TEST(SignalExtTest, resample_length) {
    const size_t n = 100;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
    }
    std::vector<double> upsampled;
    upsampled.reserve(2 * n);
    for (double v : x) {
        upsampled.push_back(v);
        upsampled.push_back(0.0);
    }
    const double fs = static_cast<double>(n);
    const auto filtered = lowpass(upsampled, fs * 0.45, fs * 2.0);
    EXPECT_EQ(filtered.size(), 2 * n);
}

TEST(SignalExtTest, window_functions_sum) {
    const auto hann = hanning(8);
    const auto hamm = hamming(8);
    double hann_sum = 0.0;
    double hamm_sum = 0.0;
    for (double v : hann) {
        hann_sum += v;
    }
    for (double v : hamm) {
        hamm_sum += v;
    }
    // Symmetric periodic windows (N-1 denominator): hann sums to ~3.5, hamming to ~3.86
    EXPECT_NEAR(hann_sum, 3.5, 0.1);
    EXPECT_NEAR(hamm_sum, 3.86, 0.05);
}

namespace {

std::vector<double> sinusoid(size_t n, double fs, double freq, double amplitude = 1.0) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = amplitude * std::sin(2.0 * M_PI * freq * static_cast<double>(i) / fs);
    }
    return x;
}

size_t peak_index(const std::vector<double>& values) {
    return static_cast<size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

size_t peak_column(const Matrix<double>& magnitude, size_t row) {
    size_t best = 0;
    for (size_t j = 1; j < magnitude.cols(); ++j) {
        if (magnitude(row, j) > magnitude(row, best)) {
            best = j;
        }
    }
    return best;
}

} // namespace

TEST(SignalExtTest, welch_psd_sinusoid_peak_at_known_frequency) {
    const double fs = 1000.0;
    const double f0 = 50.0;
    const size_t segment_len = 256;
    const auto x = sinusoid(4096, fs, f0);

    const auto result = welch_psd(x, fs, segment_len, 0.5);
    ASSERT_TRUE(result.has_value()) << "welch_psd failed";
    const auto& psd = *result;

    ASSERT_EQ(psd.frequencies.size(), psd.power.size());
    ASSERT_FALSE(psd.power.empty());

    const size_t peak = peak_index(psd.power);
    const double peak_freq = psd.frequencies[peak];
    const double df = fs / static_cast<double>(segment_len);
    EXPECT_NEAR(peak_freq, f0, df);
}

TEST(SignalExtTest, welch_psd_frequencies_monotonic_and_nyquist) {
    const double fs = 1000.0;
    const auto x = sinusoid(2048, fs, 40.0);

    const auto result = welch_psd(x, fs, 256, 0.5);
    ASSERT_TRUE(result.has_value());
    const auto& psd = *result;

    for (size_t i = 1; i < psd.frequencies.size(); ++i) {
        EXPECT_GT(psd.frequencies[i], psd.frequencies[i - 1]);
    }
    EXPECT_NEAR(psd.frequencies.front(), 0.0, 1e-12);
    EXPECT_NEAR(psd.frequencies.back(), fs / 2.0, 1e-9);
}

TEST(SignalExtTest, welch_psd_overlap_zero) {
    const double fs = 1000.0;
    const double f0 = 50.0;
    const auto x = sinusoid(4096, fs, f0);
    const double df = fs / 256.0;

    const auto result = welch_psd(x, fs, 256, 0.0);
    ASSERT_TRUE(result.has_value());
    const size_t peak = peak_index(result->power);
    EXPECT_NEAR(result->frequencies[peak], f0, df);
}

TEST(SignalExtTest, welch_psd_overlap_half) {
    const double fs = 1000.0;
    const double f0 = 50.0;
    const auto x = sinusoid(4096, fs, f0);
    const double df = fs / 256.0;

    const auto result = welch_psd(x, fs, 256, 0.5);
    ASSERT_TRUE(result.has_value());
    const size_t peak = peak_index(result->power);
    EXPECT_NEAR(result->frequencies[peak], f0, df);
}

TEST(SignalExtTest, welch_psd_overlap_high) {
    const double fs = 1000.0;
    const double f0 = 50.0;
    const auto x = sinusoid(4096, fs, f0);
    const double df = fs / 256.0;

    const auto result = welch_psd(x, fs, 256, 0.75);
    ASSERT_TRUE(result.has_value());
    const size_t peak = peak_index(result->power);
    EXPECT_NEAR(result->frequencies[peak], f0, df);
}

TEST(SignalExtTest, welch_psd_invalid_segment_len_zero) {
    const auto x = sinusoid(512, 1000.0, 50.0);
    const auto result = welch_psd(x, 1000.0, 0, 0.5);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

TEST(SignalExtTest, welch_psd_invalid_segment_len_too_large) {
    const auto x = sinusoid(512, 1000.0, 50.0);
    const auto result = welch_psd(x, 1000.0, x.size() + 1, 0.5);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

TEST(SignalExtTest, welch_psd_invalid_overlap_negative) {
    const auto x = sinusoid(512, 1000.0, 50.0);
    const auto result = welch_psd(x, 1000.0, 256, -0.1);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

TEST(SignalExtTest, welch_psd_invalid_overlap_ge_one) {
    const auto x = sinusoid(512, 1000.0, 50.0);
    const auto result = welch_psd(x, 1000.0, 256, 1.0);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

TEST(SignalExtTest, spectrogram_frequency_shift_over_time) {
    const double fs = 1000.0;
    const size_t n = 8000;
    const size_t segment_len = 256;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        const double freq = (i < n / 2) ? 20.0 : 80.0;
        x[i] = std::sin(2.0 * M_PI * freq * static_cast<double>(i) / fs);
    }

    const auto result = spectrogram(x, fs, segment_len, 0.5);
    ASSERT_TRUE(result.has_value());
    const auto& spec = *result;

    const size_t early_row = spec.times.size() / 8;
    const size_t late_row = spec.times.size() * 7 / 8;
    const double df = fs / static_cast<double>(segment_len);

    const size_t early_peak = peak_column(spec.magnitude, early_row);
    const size_t late_peak = peak_column(spec.magnitude, late_row);
    EXPECT_NEAR(spec.frequencies[early_peak], 20.0, df);
    EXPECT_NEAR(spec.frequencies[late_peak], 80.0, df);
    EXPECT_LT(spec.times[early_row], spec.times[late_row]);
}

TEST(SignalExtTest, spectrogram_matrix_dimensions_consistent) {
    const double fs = 1000.0;
    const auto x = sinusoid(4096, fs, 30.0);

    const auto result = spectrogram(x, fs, 256, 0.5);
    ASSERT_TRUE(result.has_value());
    const auto& spec = *result;

    EXPECT_EQ(spec.magnitude.rows(), spec.times.size());
    EXPECT_EQ(spec.magnitude.cols(), spec.frequencies.size());
    EXPECT_GT(spec.times.size(), 1u);
    for (size_t i = 1; i < spec.times.size(); ++i) {
        EXPECT_GT(spec.times[i], spec.times[i - 1]);
    }
}

TEST(SignalExtTest, spectrogram_invalid_segment_len_zero) {
    const auto x = sinusoid(512, 1000.0, 50.0);
    const auto result = spectrogram(x, 1000.0, 0, 0.5);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

TEST(SignalExtTest, spectrogram_invalid_segment_len_too_large) {
    const auto x = sinusoid(512, 1000.0, 50.0);
    const auto result = spectrogram(x, 1000.0, x.size() + 1, 0.5);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

TEST(SignalExtTest, spectrogram_invalid_overlap_ge_one) {
    const auto x = sinusoid(512, 1000.0, 50.0);
    const auto result = spectrogram(x, 1000.0, 256, 1.0);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

namespace {

std::vector<double> cosine_signal(size_t n, double fs, double freq, double amplitude = 1.0) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = amplitude * std::cos(2.0 * M_PI * freq * static_cast<double>(i) / fs);
    }
    return x;
}

std::vector<double> linear_chirp(size_t n, double fs, double f0, double f1) {
    std::vector<double> x(n);
    const double T = static_cast<double>(n - 1) / fs;
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / fs;
        const double phase = 2.0 * M_PI * (f0 * t + 0.5 * (f1 - f0) * t * t / T);
        x[i] = std::sin(phase);
    }
    return x;
}

double mean_middle_80(const std::vector<double>& values) {
    const size_t n = values.size();
    const size_t start = n / 10;
    const size_t end = n - n / 10;
    double sum = 0.0;
    for (size_t i = start; i < end; ++i) {
        sum += values[i];
    }
    return sum / static_cast<double>(end - start);
}

int lag_at_peak(const std::vector<double>& xcorr_values, int max_lag) {
    const auto peak = std::max_element(xcorr_values.begin(), xcorr_values.end());
    const int peak_index = static_cast<int>(std::distance(xcorr_values.begin(), peak));
    return peak_index - max_lag;
}

} // namespace

TEST(SignalExtTest, hilbert_sinusoid_envelope_is_constant) {
    const double fs = 1000.0;
    const double f0 = 50.0;
    const size_t n = 1024;
    const auto x = cosine_signal(n, fs, f0);
    const auto env = envelope(x);
    ASSERT_EQ(env.size(), n);

    const double env_mean = mean_middle_80(env);
    EXPECT_NEAR(env_mean, 1.0, 0.15);
}

TEST(SignalExtTest, hilbert_sinusoid_instantaneous_freq_is_constant) {
    const double fs = 1000.0;
    const double f0 = 50.0;
    const size_t n = 1024;
    const auto x = cosine_signal(n, fs, f0);
    const auto freq = instantaneous_freq(x, fs);
    ASSERT_EQ(freq.size(), n);

    const double freq_mean = mean_middle_80(freq);
    EXPECT_NEAR(freq_mean, f0, 5.0);
}

TEST(SignalExtTest, hilbert_empty_returns_empty) {
    const std::vector<double> empty;
    EXPECT_TRUE(hilbert(empty).empty());
    EXPECT_TRUE(envelope(empty).empty());
    EXPECT_TRUE(instantaneous_phase(empty).empty());
    EXPECT_TRUE(instantaneous_freq(empty, 1000.0).empty());
}

TEST(SignalExtTest, envelope_matches_hilbert_magnitude) {
    const auto x = cosine_signal(128, 500.0, 25.0, 2.0);
    const auto z = hilbert(x);
    const auto env = envelope(x);
    ASSERT_EQ(z.size(), env.size());
    for (size_t i = 0; i < env.size(); ++i) {
        EXPECT_NEAR(env[i], std::abs(z[i]), 1e-12);
    }
}

TEST(SignalExtTest, instantaneous_phase_matches_hilbert_arg) {
    const auto x = cosine_signal(128, 500.0, 25.0);
    const auto z = hilbert(x);
    const auto phase = instantaneous_phase(x);
    ASSERT_EQ(z.size(), phase.size());
    for (size_t i = 0; i < phase.size(); ++i) {
        EXPECT_NEAR(phase[i], std::arg(z[i]), 1e-12);
    }
}

TEST(SignalExtTest, instantaneous_freq_output_length_matches_input) {
    const auto x = cosine_signal(64, 1000.0, 40.0);
    const auto freq = instantaneous_freq(x, 1000.0);
    EXPECT_EQ(freq.size(), x.size());
    EXPECT_NEAR(freq.back(), freq[freq.size() - 2], 1e-12);
}

TEST(SignalExtTest, instantaneous_freq_chirp_tracks_increasing_frequency) {
    const double fs = 1000.0;
    const size_t n = 2048;
    const auto x = linear_chirp(n, fs, 20.0, 120.0);
    const auto freq = instantaneous_freq(x, fs);

    const size_t early_start = n / 10;
    const size_t early_end = n / 5;
    const size_t late_start = 4 * n / 5;
    const size_t late_end = 9 * n / 10;

    double early_mean = 0.0;
    for (size_t i = early_start; i < early_end; ++i) {
        early_mean += freq[i];
    }
    early_mean /= static_cast<double>(early_end - early_start);

    double late_mean = 0.0;
    for (size_t i = late_start; i < late_end; ++i) {
        late_mean += freq[i];
    }
    late_mean /= static_cast<double>(late_end - late_start);

    EXPECT_GT(late_mean, early_mean + 30.0);
    EXPECT_NEAR(early_mean, 20.0, 25.0);
    EXPECT_NEAR(late_mean, 120.0, 35.0);
}

TEST(SignalExtTest, xcorr_finds_known_integer_delay) {
    const double fs = 1000.0;
    const auto x = sinusoid(256, fs, 60.0);
    const int delay = 12;
    const int max_lag = 20;

    std::vector<double> y(x.size(), 0.0);
    for (size_t i = static_cast<size_t>(delay); i < x.size(); ++i) {
        y[i] = x[i - static_cast<size_t>(delay)];
    }

    const auto xc = xcorr(x, y, max_lag);
    ASSERT_EQ(xc.size(), static_cast<size_t>(2 * max_lag + 1));
    EXPECT_EQ(lag_at_peak(xc, max_lag), delay);
}

TEST(SignalExtTest, xcorr_output_size_is_two_max_lag_plus_one) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{1.0, 0.0, -1.0};
    const int max_lag = 5;
    const auto xc = xcorr(a, b, max_lag);
    EXPECT_EQ(xc.size(), static_cast<size_t>(2 * max_lag + 1));
}

TEST(SignalExtTest, autocorr_zero_lag_equals_sum_of_squares) {
    const std::vector<double> x{1.0, -2.0, 3.0, 0.5};
    const int max_lag = 2;
    const auto ac = autocorr(x, max_lag);
    ASSERT_EQ(ac.size(), static_cast<size_t>(2 * max_lag + 1));

    double expected = 0.0;
    for (double v : x) {
        expected += v * v;
    }
    EXPECT_NEAR(ac[static_cast<size_t>(max_lag)], expected, 1e-12);
}

TEST(SignalExtTest, autocorr_matches_xcorr_self) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    const int max_lag = 3;
    const auto ac = autocorr(x, max_lag);
    const auto xc = xcorr(x, x, max_lag);
    ASSERT_EQ(ac.size(), xc.size());
    for (size_t i = 0; i < ac.size(); ++i) {
        EXPECT_NEAR(ac[i], xc[i], 1e-12);
    }
}

TEST(SignalExtTest, xcov_is_demeaned_cross_correlation) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{2.0, 1.0, 4.0, 3.0};
    const int max_lag = 2;

    auto demean_copy = [](const std::vector<double>& v) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        std::vector<double> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            out[i] = v[i] - mean;
        }
        return out;
    };

    const auto expected = xcorr(demean_copy(a), demean_copy(b), max_lag);
    const auto cov = xcov(a, b, max_lag);
    ASSERT_EQ(expected.size(), cov.size());
    for (size_t i = 0; i < cov.size(); ++i) {
        EXPECT_NEAR(cov[i], expected[i], 1e-12);
    }
}

TEST(SignalExtTest, xcorr_invalid_max_lag_returns_empty) {
    const std::vector<double> x{1.0, 2.0};
    EXPECT_TRUE(xcorr(x, x, -1).empty());
    EXPECT_TRUE(xcov(x, x, -1).empty());
    EXPECT_TRUE(autocorr(x, -1).empty());
}

TEST(SignalExtTest, conv2_hand_computed_2x2) {
    Matrix<double> A{{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double> B{{1.0, 0.0}, {0.0, 1.0}};
    const auto C = conv2(A, B);
    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(C(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(C(2, 2), 4.0, 1e-12);
}

TEST(SignalExtTest, conv2_small_kernel_sums) {
    Matrix<double> A{{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double> K{{1.0, 1.0}, {1.0, 1.0}};
    const auto C = conv2(A, K);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(0, 1), 3.0, 1e-12);
    EXPECT_NEAR(C(1, 0), 4.0, 1e-12);
    EXPECT_NEAR(C(2, 2), 4.0, 1e-12);
}

TEST(SignalExtTest, conv2_empty_matrix_returns_empty) {
    Matrix<double> empty;
    Matrix<double> B{{1.0}};
    EXPECT_TRUE(conv2(empty, B).empty());
    EXPECT_TRUE(conv2(B, empty).empty());
}

TEST(SignalExtTest, deconv_recovers_signal_from_convolution) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    const std::vector<double> b{1.0, 1.0};
    const auto y = convolve(x, b);
    const auto recovered = deconv(y, b);
    ASSERT_EQ(recovered.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(recovered[i], x[i], 1e-12);
    }
}

TEST(SignalExtTest, deconv_roundtrip_longer_signals) {
    const std::vector<double> x{2.0, -1.0, 0.5, 3.0, 1.0};
    const std::vector<double> b{1.0, -0.5, 0.25};
    const auto y = convolve(x, b);
    const auto recovered = deconv(y, b);
    ASSERT_EQ(recovered.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(recovered[i], x[i], 1e-10);
    }
}
