#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "ms/signal/signal.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Parameterized tests for all window functions
// ---------------------------------------------------------------------------

struct WindowParam {
    std::string name;
    std::function<std::vector<double>(size_t)> fn;
};

class WindowFunctionTest : public ::testing::TestWithParam<WindowParam> {};

INSTANTIATE_TEST_SUITE_P(
    AllWindows, WindowFunctionTest,
    ::testing::Values(
        WindowParam{"hamming",   [](size_t n) { return hamming(n);   }},
        WindowParam{"hanning",   [](size_t n) { return hanning(n);   }},
        WindowParam{"blackman",  [](size_t n) { return blackman(n);  }},
        WindowParam{"parzen",    [](size_t n) { return parzen(n);    }},
        WindowParam{"triangular",[](size_t n) { return triangular(n);}})
);

TEST_P(WindowFunctionTest, length_matches_n) {
    const auto& p = GetParam();
    for (size_t n : {4u, 8u, 16u, 64u}) {
        EXPECT_EQ(p.fn(n).size(), n);
    }
}

TEST_P(WindowFunctionTest, all_values_in_0_1) {
    const auto& p = GetParam();
    const auto w = p.fn(16);
    for (double v : w) {
        EXPECT_GE(v, -1e-12);  // >= 0
        EXPECT_LE(v, 1.0 + 1e-12);  // <= 1
    }
}

TEST_P(WindowFunctionTest, finite_values) {
    const auto& p = GetParam();
    const auto w = p.fn(32);
    for (double v : w) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST_P(WindowFunctionTest, sum_is_positive) {
    const auto& p = GetParam();
    const auto w = p.fn(16);
    double sum = 0.0;
    for (double v : w) sum += v;
    EXPECT_GT(sum, 0.0);
}

// ---------------------------------------------------------------------------
// Direct smoke tests for the 3 new window types
// ---------------------------------------------------------------------------

TEST(BlackmanTest, N8_values_symmetric) {
    const auto w = blackman(8);
    ASSERT_EQ(w.size(), 8u);
    // Blackman window: symmetric
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(w[i], w[7 - i], 1e-10);
    }
}

TEST(BlackmanTest, first_and_last_near_zero) {
    const auto w = blackman(8);
    EXPECT_NEAR(w[0], 0.0, 0.01);
    EXPECT_NEAR(w[7], 0.0, 0.01);
}

TEST(ParzenTest, N8_values_finite) {
    const auto w = parzen(8);
    ASSERT_EQ(w.size(), 8u);
    for (double v : w) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GE(v, 0.0);
    }
}

TEST(TriangularTest, N8_center_max) {
    const auto w = triangular(8);
    ASSERT_EQ(w.size(), 8u);
    // Triangular window: peak at center
    double max_val = *std::max_element(w.begin(), w.end());
    EXPECT_NEAR(max_val, 1.0, 0.25);  // center should be near 1
}
