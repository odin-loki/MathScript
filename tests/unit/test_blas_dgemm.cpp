#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ms/cpu/blas.hpp"
#include "ms/cpu/blas_kernel.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;

namespace {

void reference_dgemm_nn(
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                       B[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(p)];
            }
            const std::size_t idx =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i);
            C[idx] = alpha * sum + beta * C[idx];
        }
    }
}

} // namespace

TEST(BlasDgemmTest, matches_reference_32x32) {
    constexpr int m = 32;
    constexpr int k = 24;
    constexpr int n = 16;
    ColMatrix<double> A(m, k);
    ColMatrix<double> B(k, n);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < m; ++i) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.01 * static_cast<double>(i + 1) + 0.001 * static_cast<double>(j);
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            B(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.02 * static_cast<double>(i) - 0.003 * static_cast<double>(j);
        }
    }

    std::vector<double> C(m * n, 0.5);
    std::vector<double> ref(m * n, 0.5);
    cpu::blas::dgemm('N', 'N', m, n, k, 1.0, A.data(), m, B.data(), k, 1.0, C.data(), m);
    reference_dgemm_nn(m, n, k, 1.0, A.data(), m, B.data(), k, 1.0, ref.data(), m);

    for (std::size_t idx = 0; idx < C.size(); ++idx) {
        EXPECT_NEAR(C[idx], ref[idx], 1e-10);
    }
}

TEST(BlasDgemmTest, matmul_uses_own_blas) {
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}};
    ColMatrix<double> B{{7, 8}, {9, 10}, {11, 12}};
    auto C = matmul(A, B).value();
    EXPECT_NEAR(C(0, 0), 58.0, 1e-12);
    EXPECT_NEAR(C(1, 0), 139.0, 1e-12);
    EXPECT_NEAR(C(0, 1), 64.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 154.0, 1e-12);
}

TEST(BlasDgemmTest, avx512_kernel_matches_rank1_when_available) {
    if (!cpu::blas::avx512::available()) {
        GTEST_SKIP() << "AVX-512F not available on this CPU";
    }

    constexpr int m = 16;
    constexpr int k = 12;
    constexpr int n = 16;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(k));
    ColMatrix<double> B(static_cast<std::size_t>(k), static_cast<std::size_t>(n));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < m; ++i) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.03 * static_cast<double>(i + 1) - 0.01 * static_cast<double>(j);
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            B(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.02 * static_cast<double>(i) + 0.015 * static_cast<double>(j);
        }
    }

    std::vector<double> blocked(m * n, 0.0);
    std::vector<double> reference(m * n, 0.0);
    cpu::blas::avx512::dgemm_nn(m, n, k, 1.0, A.data(), m, B.data(), k, 0.0, blocked.data(), m);
    reference_dgemm_nn(m, n, k, 1.0, A.data(), m, B.data(), k, 0.0, reference.data(), m);

    for (std::size_t idx = 0; idx < blocked.size(); ++idx) {
        EXPECT_NEAR(blocked[idx], reference[idx], 1e-10);
    }
}
