#include <gtest/gtest.h>
#include "ms/core/operations.hpp"
#include "ms/cuda/buffer.hpp"
#include "ms/cuda/blas.hpp"
#include "ms/runtime/dispatch.hpp"

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
