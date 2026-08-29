#include <gtest/gtest.h>
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/matmul.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using namespace ms::distributed;

namespace {

ColMatrix<double> make_matrix(size_t rows, size_t cols) {
    ColMatrix<double> m(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = static_cast<double>(i * 10 + j + 1);
        }
    }
    return m;
}

} // namespace

TEST(DistMatmulTest, stub_single_rank_matches_local) {
    auto ctx = init(0, nullptr);
    const auto A = make_matrix(4, 3);
    const auto B = make_matrix(3, 2);
    const auto expected = ms::matmul(A, B).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();
    const auto result = matmul(dA, dB, ctx).value();

    for (size_t i = 0; i < expected.rows(); ++i) {
        for (size_t j = 0; j < expected.cols(); ++j) {
            EXPECT_NEAR(result(i, j), expected(i, j), 1e-9);
        }
    }
    finalize(ctx);
}

TEST(DistMatmulTest, stub_block_cyclic_matches_local) {
    auto ctx = init(0, nullptr);
    const auto A = make_matrix(7, 5);
    const auto B = make_matrix(5, 4);
    const auto expected = ms::matmul(A, B).value();

    const auto dA = scatter(A, ctx, Distribution::BlockCyclic).value();
    const auto dB = scatter(B, ctx, Distribution::BlockCyclic).value();
    const auto result = matmul(dA, dB, ctx).value();

    for (size_t i = 0; i < expected.rows(); ++i) {
        for (size_t j = 0; j < expected.cols(); ++j) {
            EXPECT_NEAR(result(i, j), expected(i, j), 1e-9);
        }
    }
    finalize(ctx);
}

TEST(DistMatmulTest, inner_dimension_mismatch_errors) {
    auto ctx = init(0, nullptr);
    const auto A = make_matrix(2, 3);
    const auto B = make_matrix(2, 2);
    const auto dA = scatter(A, ctx).value();
    const auto dB = scatter(B, ctx).value();
    const auto result = matmul(dA, dB, ctx);
    EXPECT_FALSE(result.has_value());
    finalize(ctx);
}

#if defined(MS_HAS_MPI) && MS_HAS_MPI
TEST(DistMatmulMpiTest, live_block_matmul_when_launched_with_two_ranks) {
    auto ctx = init(0, nullptr);
    if (size(ctx) != 2) {
        finalize(ctx);
        GTEST_SKIP() << "Run with mpiexec -n 2 to enable this test";
    }

    const auto A = make_matrix(6, 3);
    const auto B = make_matrix(3, 2);
    const auto expected = ms::matmul(A, B).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();
    const auto result = matmul(dA, dB, ctx);
    ASSERT_TRUE(result.has_value());

    if (rank(ctx) == 0) {
        for (size_t i = 0; i < expected.rows(); ++i) {
            for (size_t j = 0; j < expected.cols(); ++j) {
                EXPECT_NEAR((*result)(i, j), expected(i, j), 1e-9);
            }
        }
    }
    finalize(ctx);
}
#endif
