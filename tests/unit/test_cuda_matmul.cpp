#include <gtest/gtest.h>
#include "ms/core/operations.hpp"
#include "ms/cuda/buffer.hpp"
#include "ms/cuda/blas.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;

namespace {

ColMatrix<double> sample_a() {
    return ColMatrix<double>{{1, 2, 3}, {4, 5, 6}};
}

ColMatrix<double> sample_b() {
    return ColMatrix<double>{{7, 8}, {9, 10}, {11, 12}};
}

void expect_matrices_near(
    const ColMatrix<double>& expected,
    const ColMatrix<double>& actual,
    double tol = 1e-9) {
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    for (size_t i = 0; i < expected.rows(); ++i) {
        for (size_t j = 0; j < expected.cols(); ++j) {
            EXPECT_NEAR(expected(i, j), actual(i, j), tol);
        }
    }
}

} // namespace

TEST(CudaMatmulStubTest, cuda_unavailable_when_disabled) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (cuda::device_count() == 0) {
        EXPECT_FALSE(cuda::available());
    }
#else
    EXPECT_FALSE(cuda::available());
    EXPECT_FALSE(has_cuda());
    EXPECT_EQ(cuda::device_count(), 0);
#endif
}

TEST(CudaMatmulStubTest, direct_cuda_matmul_matches_cpu) {
    const auto A = sample_a();
    const auto B = sample_b();
    auto gpu = cuda::matmul(A, B);
    ASSERT_TRUE(gpu.has_value());
    auto cpu = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(cpu.has_value());
    expect_matrices_near(*cpu, *gpu);
}

TEST(CudaMatmulStubTest, gpu_policy_falls_back_to_cpu_without_cuda) {
    if (has_cuda()) {
        GTEST_SKIP() << "CUDA hardware/build available; stub fallback not under test";
    }

    const auto A = sample_a();
    const auto B = sample_b();
    const auto decision = decide(A.rows(), ExecPolicy::GPU);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_EQ(decision.policy, ExecPolicy::GPU);

    auto gpu_policy = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    auto cpu_ref = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(gpu_policy.has_value());
    ASSERT_TRUE(cpu_ref.has_value());
    expect_matrices_near(*cpu_ref, *gpu_policy);
}

TEST(CudaMatmulStubTest, auto_policy_falls_back_to_cpu_without_cuda) {
    if (has_cuda()) {
        GTEST_SKIP() << "CUDA hardware/build available; stub fallback not under test";
    }

    const auto A = sample_a();
    const auto B = sample_b();
    const auto n = (std::max)({A.rows(), A.cols(), B.cols()});

    const auto small = decide(n, ExecPolicy::AUTO);
    EXPECT_EQ(small.backend, Backend::CPU);
    EXPECT_EQ(small.policy, ExecPolicy::CPU);

    const auto large = decide(4096, ExecPolicy::AUTO);
    EXPECT_EQ(large.backend, Backend::CPU);
    EXPECT_EQ(large.policy, ExecPolicy::CPU);

    auto auto_result = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    auto cpu_ref = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(auto_result.has_value());
    ASSERT_TRUE(cpu_ref.has_value());
    expect_matrices_near(*cpu_ref, *auto_result);
}

TEST(CudaMatmulStubTest, auto_large_workload_stays_cpu_without_cuda) {
    if (has_cuda()) {
        GTEST_SKIP() << "CUDA hardware/build available; stub fallback not under test";
    }

    ColMatrix<double> A = zeros<double>(300, 300);
    ColMatrix<double> B = zeros<double>(300, 300);
    for (size_t i = 0; i < 300; ++i) {
        A(i, i) = 1.0;
        B(i, i) = static_cast<double>(i + 1);
    }

    const auto decision = decide(300, ExecPolicy::AUTO);
    EXPECT_EQ(decision.backend, Backend::CPU);

    auto auto_result = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    auto cpu_ref = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(auto_result.has_value());
    ASSERT_TRUE(cpu_ref.has_value());
    expect_matrices_near(*cpu_ref, *auto_result);
}
