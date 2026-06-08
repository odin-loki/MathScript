#include <gtest/gtest.h>

#include "ms/signal/signal.hpp"
#include <algorithm>

using namespace ms;

TEST(SignalFilterTest, butterworth_and_lowpass) {
    const std::vector<double> x{0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0};
    const auto bw = butterworth(x, 0.25, 1.0);
    ASSERT_EQ(bw.size(), x.size());
    const auto lp = lowpass(x, 0.25, 1.0);
    ASSERT_EQ(lp.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(lp[i], bw[i], 1e-12);
    }
}

TEST(SignalFilterTest, highpass_and_bandpass) {
    const std::vector<double> x{0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0};
    const auto hp = highpass(x, 0.1, 1.0);
    ASSERT_EQ(hp.size(), x.size());
    const auto bp = bandpass(x, 0.05, 0.4, 1.0);
    ASSERT_EQ(bp.size(), x.size());
}

TEST(SignalFilterTest, empty_input) {
    const std::vector<double> empty;
    EXPECT_TRUE(butterworth(empty, 0.2, 1.0).empty());
    EXPECT_TRUE(highpass(empty, 0.2, 1.0).empty());
}

TEST(SignalFilterTest, window_functions) {
    const auto hann = hanning(8);
    const auto black = blackman(8);
    const auto parz = parzen(8);
    const auto tri = triangular(8);
    ASSERT_EQ(hann.size(), 8u);
    ASSERT_EQ(black.size(), 8u);
    ASSERT_EQ(parz.size(), 8u);
    ASSERT_EQ(tri.size(), 8u);
    EXPECT_NEAR(hann[0], 0.0, 1e-12);
    EXPECT_NEAR(hann[7], 0.0, 1e-12);
    EXPECT_GT(*std::max_element(black.begin(), black.end()), 0.9);
    EXPECT_NEAR(tri[0], 1.0 / 5.0, 1e-12);
    EXPECT_EQ(hanning(0).size(), 0u);
    EXPECT_DOUBLE_EQ(hanning(1)[0], 1.0);
}
