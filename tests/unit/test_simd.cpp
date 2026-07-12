#include <gtest/gtest.h>
#include "ms/simd/simd.hpp"
#include <cmath>
#include <numeric>
#include <vector>

using namespace ms::simd;

TEST(SimdTest, dispatch_detects_isa) {
    const auto info = dispatch_info();
    EXPECT_TRUE(info.isa.sse2);
}

TEST(SimdTest, add_matches_scalar) {
    std::vector<double> a(16);
    std::vector<double> b(16);
    std::vector<double> out(16);
    std::vector<double> ref(16);
    std::iota(a.begin(), a.end(), 0.0);
    std::iota(b.begin(), b.end(), 1.0);
    for (size_t i = 0; i < a.size(); ++i) {
        ref[i] = a[i] + b[i];
    }
    add(a, b, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-12);
    }
}

TEST(SimdTest, dot_and_gemv) {
    std::vector<double> a{1, 2, 3, 4};
    std::vector<double> b{4, 3, 2, 1};
    EXPECT_NEAR(dot(a, b), 20.0, 1e-12);

    std::vector<double> A{1, 0, 0, 1};
    auto y = gemv(A, 2, 2, b);
    ASSERT_EQ(y.size(), 2);
    EXPECT_NEAR(y[0], 4.0, 1e-12);
    EXPECT_NEAR(y[1], 3.0, 1e-12);
}

TEST(SimdTest, axpy_matches_scalar) {
    std::vector<double> x{1, 2, 3, 4};
    std::vector<double> y{4, 3, 2, 1};
    std::vector<double> ref = y;
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] += 0.5 * x[i];
    }
    axpy(0.5, x, y);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], ref[i], 1e-12);
    }
}

TEST(SimdTest, exp_map) {
    std::vector<double> x{0.0, 1.0, 2.0};
    std::vector<double> out(3);
    exp_map(x, out);
    EXPECT_NEAR(out[0], 1.0, 1e-12);
    EXPECT_NEAR(out[1], std::exp(1.0), 1e-9);
}

TEST(SimdTest, sum_small_hand_computable) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(sum(x), 10.0);
}

TEST(SimdTest, sum_empty_span_is_zero) {
    std::vector<double> empty;
    EXPECT_EQ(sum(empty), 0.0);
}

TEST(SimdTest, sum_single_element) {
    std::vector<double> x{5.0};
    EXPECT_DOUBLE_EQ(sum(x), 5.0);
}

TEST(SimdTest, sum_non_multiple_of_vector_width_tail_7) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    // Manually computed: 1+2+3+4+5+6+7 = 28
    EXPECT_DOUBLE_EQ(sum(x), 28.0);
}

TEST(SimdTest, sum_non_multiple_of_vector_width_tail_13) {
    std::vector<double> x(13);
    std::iota(x.begin(), x.end(), 1.0);
    // Manually computed: 1+2+...+13 = 13*14/2 = 91
    EXPECT_DOUBLE_EQ(sum(x), 91.0);
}

TEST(SimdTest, sum_large_array_closed_form) {
    constexpr size_t n = 1000;
    std::vector<double> x(n);
    std::iota(x.begin(), x.end(), 1.0);
    const double expected = static_cast<double>(n) * (static_cast<double>(n) + 1.0) / 2.0;
    EXPECT_NEAR(sum(x), expected, 1e-6);
}

TEST(SimdTest, sum_large_array_all_ones) {
    constexpr size_t n = 1024;
    std::vector<double> x(n, 1.0);
    EXPECT_DOUBLE_EQ(sum(x), static_cast<double>(n));
}

TEST(SimdTest, sum_exact_sign_cancellation) {
    std::vector<double> x{5.0, -5.0, 3.0};
    EXPECT_DOUBLE_EQ(sum(x), 3.0);
}

TEST(SimdTest, sum_mixed_sign_large_magnitude_cancellation) {
    std::vector<double> x{1e10, 1.0, -1e10};
    EXPECT_NEAR(sum(x), 1.0, 1e-3);
}

TEST(SimdTest, sum_cross_check_against_naive_loop_various_sizes) {
    for (const size_t n : {0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 200}) {
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) * 0.5 - 3.0;
        }
        double naive = 0.0;
        for (size_t i = 0; i < n; ++i) {
            naive += x[i];
        }
        EXPECT_NEAR(sum(x), naive, 1e-9) << "mismatch at n=" << n;
    }
}

TEST(SimdTest, sum_squares_pythagorean_triple) {
    std::vector<double> x{3.0, 4.0};
    EXPECT_DOUBLE_EQ(sum_squares(x), 25.0);
}

TEST(SimdTest, sum_squares_empty_span_is_zero) {
    std::vector<double> empty;
    EXPECT_EQ(sum_squares(empty), 0.0);
}

TEST(SimdTest, sum_squares_matches_manual_computation_with_tail) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    double expected = 0.0;
    for (double v : x) {
        expected += v * v;
    }
    EXPECT_NEAR(sum_squares(x), expected, 1e-12);
}
