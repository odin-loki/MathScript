#include <gtest/gtest.h>
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/stats/stats.hpp"
#include "ms/simd/simd.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(IntegrationTest, solve_then_residual) {
    DMatrix A{{4, 1}, {1, 3}};
    DMatrix b{{1}, {2}};
    auto x = solve(A, b).value();
    auto Ax = A * x;
    EXPECT_NEAR(Ax(0, 0), b(0, 0), 1e-9);
    EXPECT_NEAR(Ax(1, 0), b(1, 0), 1e-9);
}

TEST(IntegrationTest, svd_rank_pipeline) {
    DMatrix A{{1, 2, 3}, {2, 4, 6}};
    EXPECT_EQ(rank(A).value(), 1);
    EXPECT_GT(cond(DMatrix{{3, 1}, {1, 2}}).value(), 1.0);
}

TEST(IntegrationTest, fft_stats_chain) {
    std::vector<double> signal{1, 2, 3, 4};
    auto spectrum = fft(signal).value();
    std::vector<double> mags(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        mags[i] = std::abs(spectrum[i]);
    }
    EXPECT_GT(mean(mags), 0.0);
}

TEST(IntegrationTest, simd_matvec_matches_dense) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix x{{5}, {6}};
    std::vector<double> flat(A.rows() * A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            flat[i * A.cols() + j] = A(i, j);
        }
    }
    std::vector<double> xv(x.rows());
    for (size_t i = 0; i < x.rows(); ++i) {
        xv[i] = x(i, 0);
    }
    auto y = simd::gemv(flat, A.rows(), A.cols(), xv);
    auto ref = A * x;
    ASSERT_EQ(y.size(), ref.rows());
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], ref(i, 0), 1e-12);
    }
}
