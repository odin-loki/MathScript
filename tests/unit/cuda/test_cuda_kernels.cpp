#include <gtest/gtest.h>
#include <vector>

#include "ms/cuda/elementwise.hpp"

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include "ms/cuda/buffer.hpp"
#endif

using namespace ms;

TEST(CudaKernelsTest, add_inplace_alpha_two_lengths) {
    for (const size_t n : {3, 7, 64}) {
        std::vector<double> a(n);
        std::vector<double> b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<double>(i) + 1.0;
            b[i] = static_cast<double>(i) * 0.5;
        }
        const std::vector<double> a0 = a;
        cuda::add_inplace(a, b, 2.0);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(a[i], a0[i] + 2.0 * b[i], 1e-12) << "n=" << n << " i=" << i;
        }
    }
}

TEST(CudaKernelsTest, fill_length_7_and_64) {
    for (const size_t n : {7, 64}) {
        std::vector<double> x(n, 1.0);
        cuda::fill(x, 3.25);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_DOUBLE_EQ(x[i], 3.25) << "n=" << n << " i=" << i;
        }
    }
}

TEST(CudaKernelsTest, mul_inplace_length_7) {
    std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    const std::vector<double> b{2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    cuda::mul_inplace(a, b);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], (static_cast<double>(i) + 1.0) * (static_cast<double>(i) + 2.0), 1e-12);
    }
}

TEST(CudaKernelsTest, scale_length_64) {
    std::vector<double> a(64);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<double>(i);
    }
    cuda::scale(a, 1.5);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], 1.5 * static_cast<double>(i), 1e-12);
    }
}

TEST(CudaKernelsTest, empty_spans_do_nothing_harmful) {
    std::vector<double> empty_a;
    std::vector<double> empty_b;
    cuda::add_inplace(empty_a, empty_b, 2.0);
    cuda::fill(empty_a, 1.0);
    cuda::mul_inplace(empty_a, empty_b);
    cuda::scale(empty_a, 3.0);
    EXPECT_TRUE(empty_a.empty());
    EXPECT_TRUE(empty_b.empty());
}

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
TEST(CudaKernelsTest, device_apis_match_cpu_when_available) {
    if (!cuda::available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    const std::vector<double> b{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
    std::vector<double> cpu_add = a;
    for (size_t i = 0; i < cpu_add.size(); ++i) {
        cpu_add[i] += 2.0 * b[i];
    }
    cuda::add_inplace(a, b, 2.0);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], cpu_add[i], 1e-12);
    }

    std::vector<double> filled(7, 0.0);
    cuda::fill(filled, 9.0);
    for (double v : filled) {
        EXPECT_DOUBLE_EQ(v, 9.0);
    }

    std::vector<double> prod{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    cuda::mul_inplace(prod, b);
    for (size_t i = 0; i < prod.size(); ++i) {
        EXPECT_NEAR(prod[i], (static_cast<double>(i) + 1.0) * b[i], 1e-12);
    }

    std::vector<double> scaled(64);
    for (size_t i = 0; i < scaled.size(); ++i) {
        scaled[i] = static_cast<double>(i);
    }
    cuda::scale(scaled, 0.25);
    for (size_t i = 0; i < scaled.size(); ++i) {
        EXPECT_NEAR(scaled[i], 0.25 * static_cast<double>(i), 1e-12);
    }
}
#endif
