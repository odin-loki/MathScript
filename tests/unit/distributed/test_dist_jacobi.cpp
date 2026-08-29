#include <gtest/gtest.h>
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using namespace ms::distributed;

namespace {

ColMatrix<double> make_diag_dominant_system(size_t n) {
    ColMatrix<double> A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = static_cast<double>(n + 2);
        if (i > 0) {
            A(i, i - 1) = -0.5;
            A(i - 1, i) = -0.5;
        }
    }
    return A;
}

} // namespace

TEST(JacobiTest, local_small_system_converges) {
    const ColMatrix<double> A{{4.0, 1.0}, {1.0, 3.0}};
    const ColMatrix<double> b{{1.0}, {2.0}};
    const auto result = ms::jacobi(A, b).value();
    const auto Ax = ms::matmul(A, result).value();
    EXPECT_NEAR(Ax(0, 0), b(0, 0), 1e-6);
    EXPECT_NEAR(Ax(1, 0), b(1, 0), 1e-6);
}

TEST(DistJacobiTest, stub_single_rank_matches_local_jacobi) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{4.0, 1.0}, {1.0, 3.0}};
    const ColMatrix<double> b{{1.0}, {2.0}};
    const auto expected = ms::jacobi(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto result = dist_jacobi(dA, db, ctx).value();

    EXPECT_NEAR(result(0, 0), expected(0, 0), 1e-6);
    EXPECT_NEAR(result(1, 0), expected(1, 0), 1e-6);
    finalize(ctx);
}

TEST(DistJacobiTest, stub_identity_system) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{1.0, 0.0}, {0.0, 1.0}};
    const ColMatrix<double> b{{3.0}, {5.0}};

    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    const auto result = dist_jacobi(dA, db, ctx).value();

    EXPECT_NEAR(result(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(result(1, 0), 5.0, 1e-8);
    finalize(ctx);
}

TEST(DistJacobiTest, stub_larger_system_matches_local_jacobi) {
    auto ctx = init(0, nullptr);
    const auto A = make_diag_dominant_system(6);
    ColMatrix<double> b(6, 1);
    for (size_t i = 0; i < 6; ++i) {
        b(i, 0) = static_cast<double>(i + 1);
    }
    const auto expected = ms::jacobi(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto result = dist_jacobi(dA, db, ctx).value();

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(result(i, 0), expected(i, 0), 1e-6);
    }
    finalize(ctx);
}

TEST(DistJacobiTest, dimension_mismatch_errors) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{2.0, 1.0}, {1.0, 2.0}};
    const ColMatrix<double> b{{1.0}, {0.0}, {0.0}};
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    EXPECT_FALSE(dist_jacobi(dA, db, ctx).has_value());
    finalize(ctx);
}

#if defined(MS_HAS_MPI) && MS_HAS_MPI
TEST(DistJacobiMpiTest, live_block_jacobi_when_launched_with_two_ranks) {
    auto ctx = init(0, nullptr);
    if (size(ctx) != 2) {
        finalize(ctx);
        GTEST_SKIP() << "Run with mpiexec -n 2 to enable this test";
    }

    const auto A = make_diag_dominant_system(8);
    ColMatrix<double> b(8, 1);
    for (size_t i = 0; i < 8; ++i) {
        b(i, 0) = static_cast<double>((i + 1) * 2);
    }
    const auto expected = ms::jacobi(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto result = dist_jacobi(dA, db, ctx);
    ASSERT_TRUE(result.has_value());

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR((*result)(i, 0), expected(i, 0), 1e-5);
    }
    finalize(ctx);
}
#endif
