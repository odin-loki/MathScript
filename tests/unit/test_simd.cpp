#include <gtest/gtest.h>
#include "ms/simd/simd.hpp"
#include <cmath>
#include <cstdlib>
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

TEST(SimdTest, dot_cross_check_against_naive_loop_various_sizes) {
    for (const size_t n : {0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 200}) {
        std::vector<double> a(n);
        std::vector<double> b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<double>(i) * 0.25 + 1.0;
            b[i] = static_cast<double>(i) * 0.5 - 2.0;
        }
        double naive = 0.0;
        for (size_t i = 0; i < n; ++i) {
            naive += a[i] * b[i];
        }
        EXPECT_NEAR(dot(a, b), naive, 1e-9) << "mismatch at n=" << n;
    }
}

TEST(SimdTest, dot_cross_check_large_array) {
    constexpr size_t n = 1024;
    std::vector<double> a(n);
    std::vector<double> b(n);
    std::iota(a.begin(), a.end(), 1.0);
    std::iota(b.begin(), b.end(), 0.5);
    const double expected = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    EXPECT_NEAR(dot(a, b), expected, 1e-6);
}

TEST(SimdTest, dot_cross_check_odd_length_tail) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    const std::vector<double> b{2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const double expected = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    EXPECT_NEAR(dot(a, b), expected, 1e-12);
}

TEST(SimdTest, dot_cross_check_mixed_sign) {
    const std::vector<double> a{1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0};
    const std::vector<double> b{-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0};
    const double expected = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    EXPECT_NEAR(dot(a, b), expected, 1e-12);
}

TEST(SimdTest, dot_cross_check_simd_matches_scalar_dispatch) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    const std::vector<double> b{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
#if defined(_WIN32)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "1");
#else
    setenv("MS_SIMD_FORCE_SCALAR", "1", 1);
#endif
    const double scalar_ref = dot(a, b);
#if defined(_WIN32)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "0");
#else
    unsetenv("MS_SIMD_FORCE_SCALAR");
#endif
    const double simd_result = dot(a, b);
    EXPECT_NEAR(simd_result, scalar_ref, 1e-12);
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

TEST(SimdTest, norm_l2_pythagorean_triple) {
    std::vector<double> x{3.0, 4.0};
    EXPECT_DOUBLE_EQ(norm_l2(x), 5.0);
}

TEST(SimdTest, norm_l2_empty_span_is_zero) {
    std::vector<double> empty;
    EXPECT_EQ(norm_l2(empty), 0.0);
}

TEST(SimdTest, norm_l2_single_element) {
    std::vector<double> x{7.0};
    EXPECT_DOUBLE_EQ(norm_l2(x), 7.0);
}

TEST(SimdTest, norm_l2_unit_vector) {
    std::vector<double> x{1.0, 0.0, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(norm_l2(x), 1.0);
}

TEST(SimdTest, norm_l2_negative_values_match_positive) {
    std::vector<double> pos{1.0, -2.0, 3.0, -4.0};
    std::vector<double> neg{-1.0, 2.0, -3.0, 4.0};
    EXPECT_NEAR(norm_l2(pos), norm_l2(neg), 1e-12);
    EXPECT_NEAR(norm_l2(pos), std::sqrt(30.0), 1e-12);
}

TEST(SimdTest, norm_l2_non_multiple_of_vector_width_tail_7) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double expected = 0.0;
    for (double v : x) {
        expected += v * v;
    }
    EXPECT_NEAR(norm_l2(x), std::sqrt(expected), 1e-12);
}

TEST(SimdTest, norm_l2_large_array_closed_form) {
    constexpr size_t n = 1000;
    std::vector<double> x(n);
    std::iota(x.begin(), x.end(), 1.0);
    const double sum_sq =
        static_cast<double>(n) * (static_cast<double>(n) + 1.0) *
        (2.0 * static_cast<double>(n) + 1.0) / 6.0;
    EXPECT_NEAR(norm_l2(x), std::sqrt(sum_sq), 1e-6);
}

TEST(SimdTest, norm_l2_zero_vector) {
    std::vector<double> x(16, 0.0);
    EXPECT_DOUBLE_EQ(norm_l2(x), 0.0);
}

TEST(SimdTest, norm_l2_cross_check_against_naive_loop_various_sizes) {
    for (const size_t n : {0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 200}) {
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) * 0.25 - 1.0;
        }
        double sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum_sq += x[i] * x[i];
        }
        EXPECT_NEAR(norm_l2(x), std::sqrt(sum_sq), 1e-9) << "mismatch at n=" << n;
    }
}

TEST(SimdTest, norm_l2_cross_check_simd_matches_scalar_dispatch) {
    const std::vector<double> x{1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0};
#if defined(_WIN32)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "1");
#else
    setenv("MS_SIMD_FORCE_SCALAR", "1", 1);
#endif
    const double scalar_ref = norm_l2(x);
#if defined(_WIN32)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "0");
#else
    unsetenv("MS_SIMD_FORCE_SCALAR");
#endif
    const double simd_result = norm_l2(x);
    EXPECT_NEAR(simd_result, scalar_ref, 1e-12);
}

TEST(SimdTest, sub_matches_scalar) {
    std::vector<double> a(16);
    std::vector<double> b(16);
    std::vector<double> out(16);
    std::iota(a.begin(), a.end(), 0.0);
    std::iota(b.begin(), b.end(), 1.0);
    sub(a, b, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], a[i] - b[i], 1e-12);
    }
}

TEST(SimdTest, sub_single_element) {
    const std::vector<double> a{7.5};
    const std::vector<double> b{2.5};
    std::vector<double> out(1);
    sub(a, b, out);
    EXPECT_NEAR(out[0], 5.0, 1e-12);
}

TEST(SimdTest, sub_negative_values) {
    const std::vector<double> a{-1.0, -2.0, -3.0, -4.0};
    const std::vector<double> b{1.0, 2.0, 3.0, 4.0};
    std::vector<double> out(4);
    sub(a, b, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], -2.0 * (static_cast<double>(i) + 1.0), 1e-12);
    }
}

TEST(SimdTest, sub_non_multiple_of_vector_width_tail_7) {
    const std::vector<double> a{10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0};
    const std::vector<double> b{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    std::vector<double> out(7);
    sub(a, b, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], a[i] - b[i], 1e-12);
    }
}

TEST(SimdTest, sub_non_multiple_of_vector_width_tail_13) {
    std::vector<double> a(13);
    std::vector<double> b(13);
    std::vector<double> out(13);
    std::iota(a.begin(), a.end(), 20.0);
    std::iota(b.begin(), b.end(), 1.0);
    sub(a, b, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], a[i] - b[i], 1e-12);
    }
}

TEST(SimdTest, sub_cross_check_against_naive_loop_various_sizes) {
    for (const size_t n : {0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 200}) {
        std::vector<double> a(n);
        std::vector<double> b(n);
        std::vector<double> out(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<double>(i) * 1.25;
            b[i] = static_cast<double>(i) * 0.75 + 2.0;
        }
        sub(a, b, out);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(out[i], a[i] - b[i], 1e-12) << "mismatch at n=" << n << " i=" << i;
        }
    }
}

TEST(SimdTest, abs_matches_scalar) {
    const std::vector<double> x{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    std::vector<double> out(x.size());
    abs(x, out);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], std::abs(x[i]), 1e-12);
    }
}

TEST(SimdTest, abs_single_element) {
    const std::vector<double> x{-3.5};
    std::vector<double> out(1);
    abs(x, out);
    EXPECT_NEAR(out[0], 3.5, 1e-12);
}

TEST(SimdTest, abs_zero) {
    const std::vector<double> x{0.0, -0.0, 0.0, 0.0};
    std::vector<double> out(4);
    abs(x, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_DOUBLE_EQ(out[i], 0.0);
    }
}

TEST(SimdTest, abs_non_multiple_of_vector_width_tail_7) {
    const std::vector<double> x{-7.0, 6.0, -5.0, 4.0, -3.0, 2.0, -1.0};
    std::vector<double> out(7);
    abs(x, out);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], std::abs(x[i]), 1e-12);
    }
}

TEST(SimdTest, abs_non_multiple_of_vector_width_tail_13) {
    std::vector<double> x(13);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i) - 6.0;
    }
    std::vector<double> out(13);
    abs(x, out);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], std::abs(x[i]), 1e-12);
    }
}

TEST(SimdTest, abs_cross_check_against_naive_loop_various_sizes) {
    for (const size_t n : {0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 200}) {
        std::vector<double> x(n);
        std::vector<double> out(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i) * 0.5 - 3.0;
        }
        abs(x, out);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(out[i], std::abs(x[i]), 1e-12) << "mismatch at n=" << n << " i=" << i;
        }
    }
}
