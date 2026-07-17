// MathScript: moving_average O(n) sliding-window tests (Wave 218)

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/signal/signal.hpp"

using namespace ms;

namespace {

std::vector<double> naive_moving_average(const std::vector<double>& x, size_t window) {
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

void expect_matches_naive(const std::vector<double>& x, size_t window) {
    const auto fast = moving_average(x, window);
    const auto ref = naive_moving_average(x, window);
    ASSERT_EQ(fast.size(), ref.size());
    for (size_t i = 0; i < fast.size(); ++i) {
        EXPECT_NEAR(fast[i], ref[i], 1e-12) << "i=" << i << " window=" << window;
    }
}

} // namespace

TEST(MovingAverageSlidingWindow, EmptyInput_ReturnsEmpty) {
    const std::vector<double> empty;
    EXPECT_TRUE(moving_average(empty, 5).empty());
}

TEST(MovingAverageSlidingWindow, WindowZero_ReturnsUnchanged) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    const auto out = moving_average(x, 0);
    ASSERT_EQ(out.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(out[i], x[i]);
    }
}

TEST(MovingAverageSlidingWindow, WindowOne_IsIdentity) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto out = moving_average(x, 1);
    ASSERT_EQ(out.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], x[i], 1e-12);
    }
}

TEST(MovingAverageSlidingWindow, WindowGreaterThanLength_GrowingAverage) {
    const std::vector<double> ramp{1.0, 2.0, 3.0, 4.0};
    const auto out = moving_average(ramp, 10);
    EXPECT_NEAR(out[0], 1.0, 1e-12);
    EXPECT_NEAR(out[1], 1.5, 1e-12);
    EXPECT_NEAR(out[2], 2.0, 1e-12);
    EXPECT_NEAR(out[3], 2.5, 1e-12);
}

TEST(MovingAverageSlidingWindow, PartialWindowAtStart_HandComputed) {
    const std::vector<double> ramp{1.0, 2.0, 3.0, 4.0};
    const auto out = moving_average(ramp, 3);
    EXPECT_NEAR(out[0], 1.0, 1e-12);
    EXPECT_NEAR(out[1], 1.5, 1e-12);
    EXPECT_NEAR(out[2], 2.0, 1e-12);
    EXPECT_NEAR(out[3], 3.0, 1e-12);
}

TEST(MovingAverageSlidingWindow, LargeN_MatchesNaiveReference) {
    constexpr size_t n = 65536;
    constexpr size_t window = 128;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(0.013 * static_cast<double>(i)) + 0.25 * static_cast<double>(i % 17);
    }
    expect_matches_naive(x, window);
}

TEST(MovingAverageSlidingWindow, RandomishData_MultipleWindowsMatchNaive) {
    const std::vector<double> x{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0};
    for (size_t window : {2u, 3u, 4u, 7u, 11u, 100u}) {
        expect_matches_naive(x, window);
    }
}
