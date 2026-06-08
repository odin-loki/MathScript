#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include "ms/signal/signal.hpp"

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
