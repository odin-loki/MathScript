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
#if defined(_WIN32)
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

#if defined(_WIN32)
    _putenv_s("MS_SIMD_FORCE_SCALAR", "0");
#else
    unsetenv("MS_SIMD_FORCE_SCALAR");
#endif
}

TEST(SimdExtTest, horizontal_sum) {
    const std::vector<double> a{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> ones(8, 1.0);
    EXPECT_NEAR(dot(a, ones), 36.0, 1e-12);
}

TEST(SimdExtTest, fma_accuracy) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{5.0, 6.0, 7.0, 8.0};
    const std::vector<double> c{0.5, -1.0, 2.0, 3.0};
    std::vector<double> prod(a.size());
    std::vector<double> y(a.size());
    mul(a, b, prod);
    add(prod, c, y);
    for (size_t i = 0; i < a.size(); ++i) {
        const double expected = std::fma(a[i], b[i], c[i]);
        EXPECT_NEAR(y[i], expected, 1e-12);
    }
}

TEST(SimdExtTest, vectorized_saxpy) {
    const std::vector<double> x{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> y{8, 7, 6, 5, 4, 3, 2, 1};
    const std::vector<double> ref = y;
    const double alpha = 0.25;
    axpy(alpha, x, y);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(y[i], ref[i] + alpha * x[i], 1e-12);
    }
}

TEST(SimdExtTest, masked_load_store) {
    const std::vector<double> a{1, 2, 3, 4, 5, 6, 7};
    const std::vector<double> b{2, 3, 4, 5, 6, 7, 8};
    std::vector<double> out(7);
    add(a, b, out);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(out[i], a[i] + b[i], 1e-12);
    }
}

TEST(SimdExtTest, gather_scatter) {
    const std::vector<double> data{10, 20, 30, 40, 50, 60, 70, 80};
    const std::vector<size_t> idx{0, 2, 4, 6};
    std::vector<double> gathered(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        gathered[i] = data[idx[i]];
    }
    const std::vector<double> expected{10, 30, 50, 70};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gathered[i], expected[i], 1e-12);
    }
    const auto y = gemv(std::vector<double>{1, 0, 1, 0, 1, 0, 1, 0,
                                            0, 1, 0, 1, 0, 1, 0, 1},
                        2, 8, data);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_NEAR(y[0], 10 + 30 + 50 + 70, 1e-12);
    EXPECT_NEAR(y[1], 20 + 40 + 60 + 80, 1e-12);
}
