#include <gtest/gtest.h>
#include "ms/cuda/buffer.hpp"
#include "ms/cuda/elementwise.hpp"
#include "ms/cuda/fft.hpp"
#include <cmath>

using namespace ms;

TEST(CudaFftTest, returns_padded_spectrum) {
    std::vector<double> signal{1, 2, 3, 4};
    auto spectrum = cuda::fft(signal).value();
    EXPECT_EQ(spectrum.size(), 4);
    EXPECT_GT(std::abs(spectrum[0]), 0.0);
}

TEST(CudaFftTest, dc_component) {
    std::vector<double> signal{2, 2, 2, 2};
    auto spectrum = cuda::fft(signal).value();
    EXPECT_NEAR(spectrum[0].real(), 8.0, 1e-3);
}

TEST(CudaElementwiseTest, add_inplace) {
    std::vector<double> a{1, 2, 3};
    std::vector<double> b{4, 5, 6};
    cuda::add_inplace(a, b);
    EXPECT_DOUBLE_EQ(a[0], 5.0);
    EXPECT_DOUBLE_EQ(a[2], 9.0);
    cuda::fill(a, 0.0);
    EXPECT_DOUBLE_EQ(a[1], 0.0);
}

TEST(CudaFftTest, ifft_roundtrip) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!cuda::available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
#else
    GTEST_SKIP() << "CUDA disabled at build time";
#endif

    const std::vector<double> signal{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto spectrum = cuda::fft(signal);
    ASSERT_TRUE(spectrum.has_value());

    const size_t n = spectrum->size();
    std::vector<std::complex<double>> trimmed(n);
    for (size_t i = 0; i < n; ++i) {
        trimmed[i] = (*spectrum)[i];
    }

    const auto restored = cuda::ifft(trimmed);
    ASSERT_TRUE(restored.has_value());
    ASSERT_GE(restored->size(), signal.size());

    for (size_t i = 0; i < signal.size(); ++i) {
        EXPECT_NEAR((*restored)[i], signal[i], 1e-9);
    }
}

TEST(CudaStreamPoolTest, acquire_returns_null_when_unavailable) {
#if !(defined(MS_HAS_CUDA) && MS_HAS_CUDA)
    cuda::StreamPool pool;
    EXPECT_EQ(pool.acquire(), nullptr);
#else
    if (!cuda::available()) {
        cuda::StreamPool pool;
        EXPECT_EQ(pool.acquire(), nullptr);
    }
#endif
}

TEST(CudaStreamPoolTest, acquire_release_reuses_streams) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!cuda::available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    cuda::StreamPool pool;
    void* first = pool.acquire();
    ASSERT_NE(first, nullptr);
    void* second = pool.acquire();
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first, second);

    pool.release(first);
    pool.release(second);

    void* reused = pool.acquire();
    ASSERT_NE(reused, nullptr);
    pool.release(reused);
#endif
}
