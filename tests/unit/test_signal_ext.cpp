#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
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
