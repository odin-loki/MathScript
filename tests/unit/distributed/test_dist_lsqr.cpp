#include <gtest/gtest.h>
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using namespace ms::distributed;

TEST(LSQRTest, local_identity_system) {
    const ColMatrix<double> A{{1.0, 0.0}, {0.0, 1.0}};
    const ColMatrix<double> b{{3.0}, {5.0}};
    const auto result = ms::lsqr(A, b).value();
    EXPECT_NEAR(result(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(result(1, 0), 5.0, 1e-8);
}

TEST(DistLSQRTest, stub_single_rank_matches_local_lsqr) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{3.0, 1.0}, {1.0, 2.0}};
    const ColMatrix<double> b{{1.0}, {1.0}};
    const auto expected = ms::lsqr(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto result = dist_lsqr(dA, db, ctx).value();

    EXPECT_NEAR(result(0, 0), expected(0, 0), 1e-5);
    EXPECT_NEAR(result(1, 0), expected(1, 0), 1e-5);
    finalize(ctx);
}

TEST(DistLSQRTest, stub_identity_system) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{1.0, 0.0}, {0.0, 1.0}};
    const ColMatrix<double> b{{3.0}, {5.0}};

    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    const auto result = dist_lsqr(dA, db, ctx).value();

    EXPECT_NEAR(result(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(result(1, 0), 5.0, 1e-8);
    finalize(ctx);
}

TEST(DistLSQRTest, dimension_mismatch_errors) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{2.0, 1.0}, {1.0, 2.0}};
    const ColMatrix<double> b{{1.0}, {0.0}, {0.0}};
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    EXPECT_FALSE(dist_lsqr(dA, db, ctx).has_value());
    finalize(ctx);
}
