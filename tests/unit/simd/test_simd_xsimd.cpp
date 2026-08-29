#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "ms/simd/simd.hpp"

using namespace ms::simd;

TEST(SimdXsimdTest, batch_width_force_scalar_then_restore) {
#if defined(_WIN32)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "1");
#else
    setenv("MS_SIMD_FORCE_SCALAR", "1", 1);
#endif
    EXPECT_EQ(batch_width(), 1u);

#if defined(_WIN32)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "0");
#else
    unsetenv("MS_SIMD_FORCE_SCALAR");
#endif
    const std::size_t w = batch_width();
    EXPECT_TRUE(w == 1u || w >= 2u);
}

TEST(SimdXsimdTest, fill_lengths_with_value) {
    constexpr double kVal = 3.25;
    for (const size_t n : {0, 1, 7, 16, 64}) {
        std::vector<double> x(n, 0.0);
        fill(x, kVal);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_DOUBLE_EQ(x[i], kVal) << "n=" << n << " i=" << i;
        }
    }
}

TEST(SimdXsimdTest, add_mul_dot_sum_abs_exp_map_match_naive) {
    for (const size_t n : {0, 1, 3, 4, 5, 7, 8, 15, 17, 64}) {
        std::vector<double> a(n);
        std::vector<double> b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<double>(i) * 0.25 - 2.0;
            b[i] = static_cast<double>(i) * 0.5 + 1.0;
        }

        std::vector<double> sum_out(n);
        std::vector<double> mul_out(n);
        std::vector<double> abs_out(n);
        std::vector<double> exp_out(n);
        add(a, b, sum_out);
        mul(a, b, mul_out);
        abs(a, abs_out);
        exp_map(a, exp_out);

        double naive_dot = 0.0;
        double naive_sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(sum_out[i], a[i] + b[i], 1e-12) << "add n=" << n << " i=" << i;
            EXPECT_NEAR(mul_out[i], a[i] * b[i], 1e-12) << "mul n=" << n << " i=" << i;
            EXPECT_NEAR(abs_out[i], std::abs(a[i]), 1e-12) << "abs n=" << n << " i=" << i;
            EXPECT_NEAR(exp_out[i], std::exp(a[i]), 1e-9) << "exp_map n=" << n << " i=" << i;
            naive_dot += a[i] * b[i];
            naive_sum += a[i];
        }
        EXPECT_NEAR(dot(a, b), naive_dot, 1e-9) << "dot n=" << n;
        EXPECT_NEAR(sum(a), naive_sum, 1e-9) << "sum n=" << n;
    }
}

TEST(SimdXsimdTest, dispatch_info_batch_width_matches_batch_width) {
    EXPECT_EQ(dispatch_info().batch_width, batch_width());
}

TEST(SimdXsimdTest, kernel_is_scalar_or_avx2) {
    const auto info = dispatch_info();
    EXPECT_TRUE(info.active == Kernel::Scalar || info.active == Kernel::Avx2);
}
