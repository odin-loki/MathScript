#include <gtest/gtest.h>
#include <cmath>
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/stats/stats.hpp"
#include "ms/simd/simd.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;
using RMatrix = RowMatrix<double>;

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

TEST(IntegrationTest, dispatch_execute_policies) {
    execute(decide(32, ExecPolicy::AUTO));
    execute(decide(65536, ExecPolicy::AUTO));
    execute(decide(64, ExecPolicy::GPU));
    execute(decide(128, ExecPolicy::CPU));
    EXPECT_EQ(get_policy_from_error(), ExecPolicy::CPU);
}

namespace {

DMatrix as_col(const RMatrix& A) {
    DMatrix B(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            B(i, j) = A(i, j);
        }
    }
    return B;
}

void expect_svd_reconstruction(const DMatrix& A, const SvdResult& s, double tol) {
    DMatrix Sigma = zeros<double>(s.S.rows(), s.S.rows());
    for (size_t i = 0; i < s.S.rows(); ++i) {
        Sigma(i, i) = s.S(i, 0);
    }
    const DMatrix reconstructed = s.U * Sigma * transpose(s.V);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), tol);
        }
    }
}

} // namespace

TEST(IntegrationTest, svd_row_major_fallback_tall) {
    RMatrix A{{1, 0, 2}, {0, 3, 0}, {4, 0, 5}, {0, 6, 0}};
    const auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->S.rows(), 3u);
    for (size_t i = 0; i < result->S.rows(); ++i) {
        EXPECT_GE(result->S(i, 0), 0.0);
        EXPECT_TRUE(std::isfinite(result->S(i, 0)));
    }
}

TEST(IntegrationTest, svd_row_major_fallback_wide) {
    RMatrix A{{1, 2, 3, 4}, {5, 6, 7, 8}};
    const auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    expect_svd_reconstruction(as_col(A), *result, 1e-5);
}

TEST(IntegrationTest, svd_row_major_rank_deficient) {
    RMatrix A{{1, 2, 3}, {2, 4, 6}};
    const auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->S(0, 0), 0.0);
    EXPECT_NEAR(result->S(1, 0), 0.0, 1e-8);
}
