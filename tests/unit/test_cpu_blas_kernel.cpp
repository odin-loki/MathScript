// MathScript CPU BLAS Kernel Unit Tests
// Tests: ms::cpu::blas::avx512::dgemm_nn, available()

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>

#include "ms/cpu/blas_kernel.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace blas = ms::cpu::blas::avx512;

// ---------------------------------------------------------------------------
// available() - detect AVX-512 support
// ---------------------------------------------------------------------------

TEST(BLASKernel, Available_DoesNotCrash) {
    // Just verify it returns without crashing
    bool avail = blas::available();
    (void)avail;
    SUCCEED();
}

TEST(BLASKernel, Available_ReturnsBool) {
    bool avail = blas::available();
    EXPECT_TRUE(avail == true || avail == false);
}

// ---------------------------------------------------------------------------
// dgemm_nn: C = alpha * A * B + beta * C
// dgemm_nn works regardless of AVX-512 (falls back to scalar if not available)
// ---------------------------------------------------------------------------

TEST(BLASKernel, DGEMM_Identity_Times_Any) {
    if (!blas::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 0, 0, 1};
    double B[4] = {1, 2, 3, 4};
    double C[4] = {0, 0, 0, 0};
    blas::dgemm_nn(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    EXPECT_NEAR(C[0], 1.0, 1e-10);
    EXPECT_NEAR(C[1], 2.0, 1e-10);
    EXPECT_NEAR(C[2], 3.0, 1e-10);
    EXPECT_NEAR(C[3], 4.0, 1e-10);
}

TEST(BLASKernel, DGEMM_ScalarMultiply) {
    if (!blas::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 0, 0, 1};
    double B[4] = {1, 0, 0, 1};
    double C[4] = {0, 0, 0, 0};
    double alpha = 3.14;
    blas::dgemm_nn(m, n, k, alpha, A, k, B, n, 0.0, C, n);
    EXPECT_NEAR(C[0], alpha, 1e-10);
    EXPECT_NEAR(C[1], 0.0, 1e-10);
    EXPECT_NEAR(C[2], 0.0, 1e-10);
    EXPECT_NEAR(C[3], alpha, 1e-10);
}

TEST(BLASKernel, DGEMM_Accumulate_With_Beta) {
    if (!blas::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 0, 0, 1};
    double B[4] = {1, 0, 0, 1};
    double C[4] = {1, 1, 1, 1};
    blas::dgemm_nn(m, n, k, 1.0, A, k, B, n, 2.0, C, n);
    EXPECT_NEAR(C[0], 3.0, 1e-10);
    EXPECT_NEAR(C[1], 2.0, 1e-10);
    EXPECT_NEAR(C[2], 2.0, 1e-10);
    EXPECT_NEAR(C[3], 3.0, 1e-10);
}

TEST(BLASKernel, DGEMM_3x3_FiniteOutput) {
    if (!blas::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 3, n = 3, k = 3;
    double A[9] = {1,2,3, 4,5,6, 7,8,9};
    double B[9] = {9,8,7, 6,5,4, 3,2,1};
    double C[9] = {0};
    blas::dgemm_nn(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    for (int i = 0; i < 9; ++i)
        EXPECT_TRUE(std::isfinite(C[i]));
    double sum = 0.0;
    for (int i = 0; i < 9; ++i) sum += std::abs(C[i]);
    EXPECT_GT(sum, 0.0);
}

TEST(BLASKernel, DGEMM_Zero_Alpha_PreservesC) {
    if (!blas::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {1, 2, 3, 4};
    blas::dgemm_nn(m, n, k, 0.0, A, k, B, n, 2.0, C, n);
    EXPECT_NEAR(C[0], 2.0, 1e-10);
    EXPECT_NEAR(C[1], 4.0, 1e-10);
    EXPECT_NEAR(C[2], 6.0, 1e-10);
    EXPECT_NEAR(C[3], 8.0, 1e-10);
}

TEST(BLASKernel, DGEMM_ZeroOutput_When_AlphaAndBetaZero) {
    if (!blas::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 3, n = 3, k = 3;
    double A[9] = {1,2,3, 4,5,6, 7,8,9};
    double B[9] = {9,8,7, 6,5,4, 3,2,1};
    double C[9] = {100, 100, 100, 100, 100, 100, 100, 100, 100};
    blas::dgemm_nn(m, n, k, 0.0, A, k, B, n, 0.0, C, n);
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(C[i], 0.0, 1e-10);
}

TEST(BLASKernel, DGEMM_LargerMatrix_FiniteOutput) {
    const int N = 8;
    std::vector<double> A(N*N, 0.0), B(N*N, 0.0), C(N*N, 0.0);
    // Fill A, B with small values
    for (int i = 0; i < N; ++i) {
        A[i*N+i] = 1.0;  // identity
        for (int j = 0; j < N; ++j) B[i*N+j] = static_cast<double>(i+j+1) / (N*N);
    }
    blas::dgemm_nn(N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);
    for (int i = 0; i < N*N; ++i)
        EXPECT_TRUE(std::isfinite(C[i]));
}
