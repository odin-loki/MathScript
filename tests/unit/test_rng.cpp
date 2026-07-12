#include <gtest/gtest.h>
#include "ms/core/rng.hpp"
#include "ms/linalg/linalg.hpp"

#include <cmath>
#include <random>

using namespace ms;

TEST(RngTest, inactive_returns_zero) {
    clear_session_rng();
    EXPECT_DOUBLE_EQ(session_uniform(), 0.0);
}

TEST(RngTest, set_and_use_uniform) {
    static int call_count = 0;
    call_count = 0;
    auto custom_uniform = []() -> double {
        ++call_count;
        return 0.75;
    };
    auto dummy_normal = []() -> double { return 0.0; };

    set_session_rng(custom_uniform, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    const double val = session_uniform();
    EXPECT_DOUBLE_EQ(val, 0.75);
    EXPECT_GE(call_count, 1);

    clear_session_rng();
}

TEST(RngTest, set_and_use_normal) {
    static int normal_calls = 0;
    normal_calls = 0;
    auto dummy_uniform = []() -> double { return 0.0; };
    auto custom_normal  = []() -> double {
        ++normal_calls;
        return -1.5;
    };

    set_session_rng(dummy_uniform, custom_normal);
    ASSERT_TRUE(session_rng_active());

    const double val = session_normal();
    EXPECT_DOUBLE_EQ(val, -1.5);
    EXPECT_GE(normal_calls, 1);

    clear_session_rng();
}

TEST(RngTest, clear_deactivates) {
    auto fn = []() -> double { return 0.5; };
    set_session_rng(fn, fn);
    EXPECT_TRUE(session_rng_active());

    clear_session_rng();
    EXPECT_FALSE(session_rng_active());
}

TEST(RngTest, exponential_exact_value_for_fixed_uniform_half) {
    auto fixed_uniform = []() -> double { return 0.5; };
    auto dummy_normal  = []() -> double { return 0.0; };

    set_session_rng(fixed_uniform, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    // For u = 0.5: -log(1 - 0.5) / rate = -log(0.5) / rate.
    EXPECT_DOUBLE_EQ(session_exponential(1.0), -std::log(0.5));
    EXPECT_DOUBLE_EQ(session_exponential(2.0), -std::log(0.5) / 2.0);
    EXPECT_DOUBLE_EQ(session_exponential(0.5), -std::log(0.5) / 0.5);

    clear_session_rng();
}

TEST(RngTest, exponential_exact_value_for_fixed_uniform_other) {
    auto fixed_uniform = []() -> double { return 0.75; };
    auto dummy_normal  = []() -> double { return 0.0; };

    set_session_rng(fixed_uniform, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    // For u = 0.75: -log(1 - 0.75) / rate = -log(0.25) / rate.
    EXPECT_DOUBLE_EQ(session_exponential(1.0), -std::log(0.25));
    EXPECT_DOUBLE_EQ(session_exponential(4.0), -std::log(0.25) / 4.0);

    clear_session_rng();
}

TEST(RngTest, exponential_statistical_mean_matches_theory) {
    auto real_uniform = []() -> double {
        static std::mt19937 engine(12345);
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    };
    auto dummy_normal = []() -> double { return 0.0; };

    set_session_rng(real_uniform, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    constexpr double rate = 1.0;
    constexpr int n = 20000;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += session_exponential(rate);
    }
    const double mean = sum / n;
    // Theoretical mean = 1/rate = 1.0, theoretical std dev = 1/rate, so the
    // standard error at n=20000 is ~0.0071; a 20% relative tolerance is
    // generous enough to avoid flakiness while still catching real bugs.
    EXPECT_NEAR(mean, 1.0 / rate, 0.20 * (1.0 / rate));

    clear_session_rng();
}

TEST(RngTest, exponential_samples_are_always_non_negative) {
    auto real_uniform = []() -> double {
        static std::mt19937 engine(777);
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    };
    auto dummy_normal = []() -> double { return 0.0; };

    set_session_rng(real_uniform, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    for (int i = 0; i < 10000; ++i) {
        EXPECT_GE(session_exponential(3.0), 0.0);
    }

    clear_session_rng();
}

TEST(RngTest, exponential_rate_scaling_halves_mean) {
    auto real_uniform_a = []() -> double {
        static std::mt19937 engine(2024);
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    };
    auto dummy_normal = []() -> double { return 0.0; };

    set_session_rng(real_uniform_a, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    constexpr int n = 20000;
    double sum_rate1 = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_rate1 += session_exponential(1.0);
    }
    const double mean_rate1 = sum_rate1 / n;
    clear_session_rng();

    auto real_uniform_b = []() -> double {
        static std::mt19937 engine(2024);
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    };
    set_session_rng(real_uniform_b, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    double sum_rate2 = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_rate2 += session_exponential(2.0);
    }
    const double mean_rate2 = sum_rate2 / n;
    clear_session_rng();

    // rate=2.0 has theoretical mean 0.5, half that of rate=1.0's mean of 1.0.
    EXPECT_NEAR(mean_rate2, mean_rate1 / 2.0, 0.15 * mean_rate1);
}

TEST(RngTest, exponential_invalid_rate_returns_nan) {
    auto fixed_uniform = []() -> double { return 0.5; };
    auto dummy_normal  = []() -> double { return 0.0; };

    set_session_rng(fixed_uniform, dummy_normal);
    ASSERT_TRUE(session_rng_active());

    EXPECT_TRUE(std::isnan(session_exponential(0.0)));
    EXPECT_TRUE(std::isnan(session_exponential(-1.0)));

    clear_session_rng();
}

TEST(RngTest, exponential_no_active_rng_matches_uniform_fallback) {
    clear_session_rng();
    ASSERT_FALSE(session_rng_active());

    // session_uniform() falls back to 0.0 when inactive, so this must equal
    // -log(1.0 - session_uniform()) / rate == -log(1.0) / rate == 0.0.
    EXPECT_DOUBLE_EQ(session_exponential(1.0), -std::log(1.0 - session_uniform()) / 1.0);
    EXPECT_DOUBLE_EQ(session_exponential(1.0), 0.0);
    EXPECT_DOUBLE_EQ(session_exponential(5.0), 0.0);
}

TEST(RngTest, construction_uses_session_rng) {
    auto fixed_uniform = []() -> double { return 0.5; };
    auto fixed_normal  = []() -> double { return 0.5; };

    set_session_rng(fixed_uniform, fixed_normal);
    ASSERT_TRUE(session_rng_active());

    auto M = rand<double>(2, 2, 42);
    for (size_t i = 0; i < M.rows(); ++i) {
        for (size_t j = 0; j < M.cols(); ++j) {
            EXPECT_DOUBLE_EQ(M(i, j), 0.5);
        }
    }

    clear_session_rng();
}
