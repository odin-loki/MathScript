#include <gtest/gtest.h>
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
