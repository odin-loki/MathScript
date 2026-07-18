#include <gtest/gtest.h>
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using namespace ms::distributed;

namespace {

ColMatrix<double> make_nonsymmetric_system(size_t n) {
    ColMatrix<double> A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = static_cast<double>(n + 2);
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i + 1 < n) {
            A(i, i + 1) = 0.5;
        }
    }
    return A;
}

} // namespace

TEST(DistGmresTest, stub_single_rank_matches_local_gmres) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{3.0, 1.0}, {1.0, 2.0}};
    const ColMatrix<double> b{{5.0}, {5.0}};
    const auto expected = ms::gmres(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto result = dist_gmres(dA, db, ctx).value();

    EXPECT_NEAR(result(0, 0), expected(0, 0), 1e-6);
    EXPECT_NEAR(result(1, 0), expected(1, 0), 1e-6);
    finalize(ctx);
}

TEST(DistGmresTest, stub_identity_system) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{1.0, 0.0}, {0.0, 1.0}};
    const ColMatrix<double> b{{3.0}, {5.0}};

    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    const auto result = dist_gmres(dA, db, ctx).value();

    EXPECT_NEAR(result(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(result(1, 0), 5.0, 1e-8);
    finalize(ctx);
}

TEST(DistGmresTest, stub_larger_system_matches_local_gmres) {
    auto ctx = init(0, nullptr);
    const auto A = make_nonsymmetric_system(6);
    ColMatrix<double> b(6, 1);
    for (size_t i = 0; i < 6; ++i) {
        b(i, 0) = static_cast<double>(i + 1);
    }
    const auto expected = ms::gmres(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto result = dist_gmres(dA, db, ctx).value();

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(result(i, 0), expected(i, 0), 1e-6);
    }
    finalize(ctx);
}

TEST(DistGmresTest, dimension_mismatch_errors) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{2.0, 1.0}, {1.0, 2.0}};
    const ColMatrix<double> b{{1.0}, {0.0}, {0.0}};
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    EXPECT_FALSE(dist_gmres(dA, db, ctx).has_value());
    finalize(ctx);
}
