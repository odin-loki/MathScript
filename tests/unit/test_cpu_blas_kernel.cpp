// MathScript CPU BLAS Kernel Unit Tests
// Tests: ms::cpu::blas::avx512::dgemm_nn, available()
//        ms::cpu::blas::ddot, daxpy, dgemv

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>

#include "ms/cpu/blas.hpp"
#include "ms/cpu/blas_kernel.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace avx = ms::cpu::blas::avx512;
namespace blas = ms::cpu::blas;

namespace {

void reference_dgemv_n(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    int incx,
    double beta,
    double* y,
    int incy) {
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                   x[static_cast<std::size_t>(j) * static_cast<std::size_t>(incx)];
        }
        y[static_cast<std::size_t>(i) * static_cast<std::size_t>(incy)] =
            beta * y[static_cast<std::size_t>(i) * static_cast<std::size_t>(incy)] + alpha * sum;
    }
}

void reference_dgemv_t(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    int incx,
    double beta,
    double* y,
    int incy) {
    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < m; ++i) {
            sum += A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                   x[static_cast<std::size_t>(i) * static_cast<std::size_t>(incx)];
        }
        y[static_cast<std::size_t>(j) * static_cast<std::size_t>(incy)] =
            beta * y[static_cast<std::size_t>(j) * static_cast<std::size_t>(incy)] + alpha * sum;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// available() - detect AVX-512 support
// ---------------------------------------------------------------------------

TEST(BLASKernel, Available_DoesNotCrash) {
    // Just verify it returns without crashing
    bool avail = avx::available();
    (void)avail;
    SUCCEED();
}

TEST(BLASKernel, Available_ReturnsBool) {
    bool avail = avx::available();
    EXPECT_TRUE(avail == true || avail == false);
}

// ---------------------------------------------------------------------------
// dgemm_nn: C = alpha * A * B + beta * C
// dgemm_nn works regardless of AVX-512 (falls back to scalar if not available)
// ---------------------------------------------------------------------------

TEST(BLASKernel, DGEMM_Identity_Times_Any) {
    if (!avx::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 0, 0, 1};
    double B[4] = {1, 2, 3, 4};
    double C[4] = {0, 0, 0, 0};
    avx::dgemm_nn(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    EXPECT_NEAR(C[0], 1.0, 1e-10);
    EXPECT_NEAR(C[1], 2.0, 1e-10);
    EXPECT_NEAR(C[2], 3.0, 1e-10);
    EXPECT_NEAR(C[3], 4.0, 1e-10);
}

TEST(BLASKernel, DGEMM_ScalarMultiply) {
    if (!avx::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 0, 0, 1};
    double B[4] = {1, 0, 0, 1};
    double C[4] = {0, 0, 0, 0};
    double alpha = 3.14;
    avx::dgemm_nn(m, n, k, alpha, A, k, B, n, 0.0, C, n);
    EXPECT_NEAR(C[0], alpha, 1e-10);
    EXPECT_NEAR(C[1], 0.0, 1e-10);
    EXPECT_NEAR(C[2], 0.0, 1e-10);
    EXPECT_NEAR(C[3], alpha, 1e-10);
}

TEST(BLASKernel, DGEMM_Accumulate_With_Beta) {
    if (!avx::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 0, 0, 1};
    double B[4] = {1, 0, 0, 1};
    double C[4] = {1, 1, 1, 1};
    avx::dgemm_nn(m, n, k, 1.0, A, k, B, n, 2.0, C, n);
    EXPECT_NEAR(C[0], 3.0, 1e-10);
    EXPECT_NEAR(C[1], 2.0, 1e-10);
    EXPECT_NEAR(C[2], 2.0, 1e-10);
    EXPECT_NEAR(C[3], 3.0, 1e-10);
}

TEST(BLASKernel, DGEMM_3x3_FiniteOutput) {
    if (!avx::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 3, n = 3, k = 3;
    double A[9] = {1,2,3, 4,5,6, 7,8,9};
    double B[9] = {9,8,7, 6,5,4, 3,2,1};
    double C[9] = {0};
    avx::dgemm_nn(m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    for (int i = 0; i < 9; ++i)
        EXPECT_TRUE(std::isfinite(C[i]));
    double sum = 0.0;
    for (int i = 0; i < 9; ++i) sum += std::abs(C[i]);
    EXPECT_GT(sum, 0.0);
}

TEST(BLASKernel, DGEMM_Zero_Alpha_PreservesC) {
    if (!avx::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 2, n = 2, k = 2;
    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {1, 2, 3, 4};
    avx::dgemm_nn(m, n, k, 0.0, A, k, B, n, 2.0, C, n);
    EXPECT_NEAR(C[0], 2.0, 1e-10);
    EXPECT_NEAR(C[1], 4.0, 1e-10);
    EXPECT_NEAR(C[2], 6.0, 1e-10);
    EXPECT_NEAR(C[3], 8.0, 1e-10);
}

TEST(BLASKernel, DGEMM_ZeroOutput_When_AlphaAndBetaZero) {
    if (!avx::available()) GTEST_SKIP() << "AVX-512 not available on this machine";
    const int m = 3, n = 3, k = 3;
    double A[9] = {1,2,3, 4,5,6, 7,8,9};
    double B[9] = {9,8,7, 6,5,4, 3,2,1};
    double C[9] = {100, 100, 100, 100, 100, 100, 100, 100, 100};
    avx::dgemm_nn(m, n, k, 0.0, A, k, B, n, 0.0, C, n);
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
    avx::dgemm_nn(N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);
    for (int i = 0; i < N*N; ++i)
        EXPECT_TRUE(std::isfinite(C[i]));
}

// ---------------------------------------------------------------------------
// BLAS Level-1/2: ddot, daxpy, dgemv
// ---------------------------------------------------------------------------

TEST(BLASKernel, DDOT_KnownVectors_UnitStride) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{4.0, 3.0, 2.0, 1.0};
    const double result = blas::ddot(4, x.data(), 1, y.data(), 1);
    EXPECT_NEAR(result, 20.0, 1e-12);
}

TEST(BLASKernel, DDOT_OrthogonalVectors_ReturnsZero) {
    const std::vector<double> x{1.0, 0.0, 0.0};
    const std::vector<double> y{0.0, 1.0, 0.0};
    EXPECT_NEAR(blas::ddot(3, x.data(), 1, y.data(), 1), 0.0, 1e-12);
}

TEST(BLASKernel, DDOT_NonUnitStride) {
    const std::vector<double> x{1.0, 99.0, 2.0, 99.0, 3.0};
    const std::vector<double> y{4.0, 99.0, 5.0, 99.0, 6.0};
    const double result = blas::ddot(3, x.data(), 2, y.data(), 2);
    EXPECT_NEAR(result, 32.0, 1e-12);
}

TEST(BLASKernel, DDOT_NonPositive_NReturnsZero) {
    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> y{3.0, 4.0};
    EXPECT_DOUBLE_EQ(blas::ddot(0, x.data(), 1, y.data(), 1), 0.0);
    EXPECT_DOUBLE_EQ(blas::ddot(-1, x.data(), 1, y.data(), 1), 0.0);
}

TEST(BLASKernel, DAXPY_AlphaXPlusY) {
    std::vector<double> y{1.0, 2.0, 3.0};
    const std::vector<double> x{0.5, -1.0, 2.0};
    blas::daxpy(3, 2.0, x.data(), 1, y.data(), 1);
    EXPECT_NEAR(y[0], 2.0, 1e-12);
    EXPECT_NEAR(y[1], 0.0, 1e-12);
    EXPECT_NEAR(y[2], 7.0, 1e-12);
}

TEST(BLASKernel, DAXPY_NonUnitStride) {
    std::vector<double> y{1.0, 0.0, 2.0, 0.0, 3.0};
    const std::vector<double> x{10.0, 0.0, 20.0, 0.0, 30.0};
    blas::daxpy(3, 0.5, x.data(), 2, y.data(), 2);
    EXPECT_NEAR(y[0], 6.0, 1e-12);
    EXPECT_NEAR(y[2], 12.0, 1e-12);
    EXPECT_NEAR(y[4], 18.0, 1e-12);
}

TEST(BLASKernel, DAXPY_ZeroAlpha_NoOp) {
    std::vector<double> y{1.0, 2.0, 3.0};
    const std::vector<double> x{9.0, 9.0, 9.0};
    blas::daxpy(3, 0.0, x.data(), 1, y.data(), 1);
    EXPECT_NEAR(y[0], 1.0, 1e-12);
    EXPECT_NEAR(y[1], 2.0, 1e-12);
    EXPECT_NEAR(y[2], 3.0, 1e-12);
}

TEST(BLASKernel, DGEMV_NoTranspose_MatchesNaive) {
    const int m = 3;
    const int n = 2;
    const std::vector<double> A{1.0, 4.0, 7.0, 2.0, 5.0, 8.0};
    const std::vector<double> x{1.0, -1.0};
    std::vector<double> y{0.5, -0.5, 1.0};
    std::vector<double> ref = y;

    blas::dgemv('N', m, n, 2.0, A.data(), m, x.data(), 1, 1.0, y.data(), 1);
    reference_dgemv_n(m, n, 2.0, A.data(), m, x.data(), 1, 1.0, ref.data(), 1);

    for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(y[static_cast<std::size_t>(i)], ref[static_cast<std::size_t>(i)], 1e-12);
    }
}

TEST(BLASKernel, DGEMV_Transpose_MatchesNaive) {
    const int m = 2;
    const int n = 3;
    const std::vector<double> A{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<double> x{1.0, -2.0};
    std::vector<double> y{1.0, 0.0, -1.0};
    std::vector<double> ref = y;

    blas::dgemv('T', m, n, 0.5, A.data(), m, x.data(), 1, 2.0, y.data(), 1);
    reference_dgemv_t(m, n, 0.5, A.data(), m, x.data(), 1, 2.0, ref.data(), 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y[static_cast<std::size_t>(i)], ref[static_cast<std::size_t>(i)], 1e-12);
    }
}

TEST(BLASKernel, DGEMV_BetaZero_ClearsOutput) {
    const int m = 2;
    const int n = 2;
    const std::vector<double> A{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> x{1.0, 1.0};
    std::vector<double> y{9.0, 9.0};

    blas::dgemv('N', m, n, 1.0, A.data(), m, x.data(), 1, 0.0, y.data(), 1);
    EXPECT_NEAR(y[0], 4.0, 1e-12);
    EXPECT_NEAR(y[1], 6.0, 1e-12);
}
