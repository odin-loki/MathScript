#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "ms/simd/simd.hpp"

using namespace ms::simd;

TEST(SimdExtTest, mul_and_scale) {
    const std::vector<double> a{1, 2, 3, 4};
    const std::vector<double> b{2, 3, 4, 5};
    std::vector<double> prod(4);
    std::vector<double> scaled(4);
    mul(a, b, prod);
    scale(2.0, a, scaled);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(prod[i], a[i] * b[i], 1e-12);
        EXPECT_NEAR(scaled[i], 2.0 * a[i], 1e-12);
    }
}

TEST(SimdExtTest, gemv_rectangular) {
    const std::vector<double> A{1, 2, 3, 4, 5, 6};
    const std::vector<double> x{1, 2};
    const auto y = gemv(A, 3, 2, x);
    ASSERT_EQ(y.size(), 3u);
    EXPECT_NEAR(y[0], 5.0, 1e-12);
    EXPECT_NEAR(y[2], 17.0, 1e-12);
}

TEST(SimdExtTest, add_axpy_dot_exp_map) {
    const std::vector<double> a{1, 2, 3, 4, 5, 6};
    const std::vector<double> b{2, 3, 4, 5, 6, 7};
    std::vector<double> sum(6);
    std::vector<double> y{1, 1, 1, 1, 1, 1};
    std::vector<double> out(6);

    add(a, b, sum);
    axpy(2.0, a, y);
    exp_map(a, out);

    EXPECT_NEAR(dot(a, b), std::inner_product(a.begin(), a.end(), b.begin(), 0.0), 1e-12);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(sum[i], a[i] + b[i], 1e-12);
        EXPECT_NEAR(y[i], 1.0 + 2.0 * a[i], 1e-12);
        EXPECT_NEAR(out[i], std::exp(a[i]), 1e-12);
    }
}

TEST(SimdExtTest, dispatch_info_reports_kernel) {
    const auto info = dispatch_info();
    EXPECT_TRUE(info.active == Kernel::Scalar || info.active == Kernel::Avx2);
}

TEST(SimdExtTest, odd_length_vectors_hit_avx2_tails) {
    const std::vector<double> a{1, 2, 3, 4, 5};
    const std::vector<double> b{2, 3, 4, 5, 6};
    std::vector<double> prod(5);
    std::vector<double> scaled(5);
    std::vector<double> sum(5);
    std::vector<double> y{1, 1, 1, 1, 1};
    std::vector<double> out(5);

    mul(a, b, prod);
    scale(1.5, a, scaled);
    add(a, b, sum);
    axpy(0.5, a, y);
    exp_map(a, out);

    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(prod[i], a[i] * b[i], 1e-12);
        EXPECT_NEAR(scaled[i], 1.5 * a[i], 1e-12);
        EXPECT_NEAR(sum[i], a[i] + b[i], 1e-12);
        EXPECT_NEAR(y[i], 1.0 + 0.5 * a[i], 1e-12);
        EXPECT_NEAR(out[i], std::exp(a[i]), 1e-12);
    }
    EXPECT_NEAR(dot(a, b), std::inner_product(a.begin(), a.end(), b.begin(), 0.0), 1e-12);
}

TEST(SimdExtTest, scalar_dispatch_via_env_override) {
#if defined(_MSC_VER)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "1");
#else
    setenv("MS_SIMD_FORCE_SCALAR", "1", 1);
#endif
    const std::vector<double> a{1, 2, 3, 4, 5, 6, 7};
    const std::vector<double> b{2, 3, 4, 5, 6, 7, 8};
    std::vector<double> prod(7);
    std::vector<double> scaled(7);
    std::vector<double> sum(7);
    std::vector<double> y(7, 1.0);
    std::vector<double> out(7);

    EXPECT_EQ(dispatch_info().active, Kernel::Scalar);
    mul(a, b, prod);
    scale(2.0, a, scaled);
    add(a, b, sum);
    axpy(2.0, a, y);
    exp_map(a, out);
    EXPECT_NEAR(dot(a, b), std::inner_product(a.begin(), a.end(), b.begin(), 0.0), 1e-12);

    const auto gemv_y = gemv(std::vector<double>{1, 2, 3, 4, 5, 6}, 2, 3, std::vector<double>{1, 2, 3});
    ASSERT_EQ(gemv_y.size(), 2u);
    EXPECT_NEAR(gemv_y[0], 14.0, 1e-12);
    EXPECT_NEAR(gemv_y[1], 32.0, 1e-12);

#if defined(_MSC_VER)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "0");
#else
    unsetenv("MS_SIMD_FORCE_SCALAR");
#endif
}
