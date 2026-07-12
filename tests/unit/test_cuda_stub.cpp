#include <gtest/gtest.h>
#include "ms/core/operations.hpp"
#include "ms/cuda/buffer.hpp"
#include "ms/cuda/blas.hpp"
#include "ms/cuda/fft.hpp"
#include "ms/cuda/nvml.hpp"
#include "ms/cuda/solver.hpp"
#include "ms/runtime/dispatch.hpp"
#include <complex>

using namespace ms;

TEST(CudaStubTest, not_available_by_default) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (cuda::device_count() == 0) {
        EXPECT_FALSE(cuda::available());
    }
#else
    EXPECT_FALSE(cuda::available());
    EXPECT_EQ(cuda::device_count(), 0);
#endif
}

TEST(CudaStubTest, host_buffer_roundtrip) {
    auto buffer = cuda::make_device_buffer(4 * sizeof(double));
    double src[4] = {1, 2, 3, 4};
    double dst[4] = {0, 0, 0, 0};
    cuda::copy_host_to_device(src, buffer, sizeof(src));
    cuda::copy_device_to_host(buffer, dst, sizeof(dst));
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(dst[i], src[i]);
    }
}

TEST(CudaStubTest, matmul_matches_cpu) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    ColMatrix<double> B{{5, 6}, {7, 8}};
    auto gpu = cuda::matmul(A, B);
    ASSERT_TRUE(gpu.has_value());
    auto cpu = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(cpu.has_value());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            EXPECT_NEAR((*gpu)(i, j), (*cpu)(i, j), 1e-9);
        }
    }
}

TEST(CudaStubTest, ifft_stub) {
    const std::vector<std::complex<double>> spectrum{{1.0, 0.5}, {2.0, -0.5}, {3.0, 0.0}, {0.0, 1.0}};
    const auto result = cuda::ifft(spectrum);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        EXPECT_DOUBLE_EQ((*result)[i], spectrum[i].real());
    }
}

TEST(CudaStubTest, lu_stub) {
    ColMatrix<double> A{{4, 1}, {1, 3}};
    const auto result = cuda::lu(A);
    EXPECT_FALSE(result.has_value());
}

TEST(CudaStubTest, solve_stub) {
    ColMatrix<double> A{{4, 1}, {1, 3}};
    ColMatrix<double> b{{1}, {2}};
    const auto result = cuda::solve(A, b);
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (cuda::available()) {
        ASSERT_TRUE(result.has_value());
        return;
    }
#endif
    EXPECT_FALSE(result.has_value());
}

TEST(CudaNvmlTest, device_memory_free_matches_device_stats_composition) {
    const auto stats = cuda::device_stats(0);
    const size_t expected =
        stats.memory_total_bytes >= stats.memory_used_bytes
            ? stats.memory_total_bytes - stats.memory_used_bytes
            : 0;
    EXPECT_EQ(cuda::device_memory_free(0), expected);
}

TEST(CudaNvmlTest, device_memory_free_stub_is_zero) {
#if !(defined(MS_HAS_CUDA) && MS_HAS_CUDA)
    EXPECT_EQ(cuda::device_memory_free(0), 0u);
#else
    if (!cuda::available()) {
        EXPECT_EQ(cuda::device_memory_free(0), 0u);
    }
#endif
}

TEST(CudaNvmlTest, default_argument_matches_explicit_device_zero) {
    EXPECT_EQ(cuda::device_memory_free(), cuda::device_memory_free(0));
}

TEST(CudaNvmlTest, multiple_device_indices_do_not_crash) {
    for (const int device : {0, 1, 7}) {
        const auto stats = cuda::device_stats(device);
        const size_t expected =
            stats.memory_total_bytes >= stats.memory_used_bytes
                ? stats.memory_total_bytes - stats.memory_used_bytes
                : 0;
        EXPECT_EQ(cuda::device_memory_free(device), expected);
    }
}

TEST(CudaNvmlTest, negative_device_index_does_not_crash) {
    EXPECT_NO_THROW({
        const size_t free_bytes = cuda::device_memory_free(-1);
        EXPECT_GE(free_bytes, 0u);
    });
}

TEST(CudaNvmlTest, out_of_range_device_index_does_not_crash) {
    EXPECT_NO_THROW({
        const size_t free_bytes = cuda::device_memory_free(9999);
        EXPECT_GE(free_bytes, 0u);
    });
}

TEST(CudaNvmlTest, result_never_underflows) {
    for (const int device : {-1, 0, 1, 7, 9999}) {
        const size_t free_bytes = cuda::device_memory_free(device);
        // size_t is unsigned; an underflowed subtraction would produce a
        // value near SIZE_MAX, so a sane upper bound catches that case.
        EXPECT_LE(free_bytes, 64ULL * 1024 * 1024 * 1024 * 1024);
    }
}
