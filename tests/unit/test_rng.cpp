#include <gtest/gtest.h>
#include "ms/core/rng.hpp"
#include "ms/linalg/linalg.hpp"

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
