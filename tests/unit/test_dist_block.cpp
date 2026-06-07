#include <gtest/gtest.h>
#include "ms/distributed/block.hpp"
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/distributed/solve.hpp"
#include "ms/distributed/linalg.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using namespace ms::distributed;

TEST(DistBlockTest, row_extents_partition) {
    const auto r0 = block_row_extent(10, 0, 4);
    const auto r1 = block_row_extent(10, 1, 4);
    const auto r2 = block_row_extent(10, 2, 4);
    const auto r3 = block_row_extent(10, 3, 4);
    EXPECT_EQ(r0.count + r1.count + r2.count + r3.count, 10);
    EXPECT_EQ(r0.start, 0);
    EXPECT_EQ(r3.start + r3.count, 10);
}

TEST(DistBlockTest, scatter_local_block) {
    auto ctx = init(0, nullptr);
    ctx.size = 4;
    ctx.rank = 1;

    ColMatrix<double> global(8, 2);
    for (size_t i = 0; i < 8; ++i) {
        global(i, 0) = static_cast<double>(i);
        global(i, 1) = static_cast<double>(i) * 2.0;
    }

    auto dist = scatter(global, ctx, Distribution::Block).value();
    EXPECT_EQ(dist.local.rows(), 2);
    EXPECT_EQ(dist.local.cols(), 2);
    EXPECT_DOUBLE_EQ(dist.local(0, 0), 2.0);
    finalize(ctx);
}

TEST(DistBlockTest, gather_reassembles_stub) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global{{1, 2}, {3, 4}, {5, 6}};
    auto dist = scatter(global, ctx).value();
    auto rebuilt = gather(dist, ctx).value();
    for (size_t i = 0; i < global.rows(); ++i) {
        for (size_t j = 0; j < global.cols(); ++j) {
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
        }
    }
    finalize(ctx);
}

TEST(DistBlockTest, distributed_solve_single_rank) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A{{4, 1}, {1, 3}};
    ColMatrix<double> b{{1}, {2}};
    auto dA = scatter(A, ctx).value();
    auto db = scatter(b, ctx).value();
    auto x = distributed::solve(dA, db, ctx).value();
    auto check = ms::solve(A, b).value();
    EXPECT_NEAR(x(0, 0), check(0, 0), 1e-6);
    EXPECT_NEAR(x(1, 0), check(1, 0), 1e-6);
    finalize(ctx);
}

TEST(DistBlockTest, block_cyclic_indices) {
    const auto rows = block_cyclic_row_indices(10, 1, 4);
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0], 1u);
    EXPECT_EQ(rows[1], 5u);
    EXPECT_EQ(rows[2], 9u);
}

TEST(DistBlockTest, block_cyclic_roundtrip) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global(6, 2);
    for (size_t i = 0; i < 6; ++i) {
        global(i, 0) = static_cast<double>(i);
        global(i, 1) = static_cast<double>(i) * 10.0;
    }
    auto dist = scatter(global, ctx, Distribution::BlockCyclic).value();
    EXPECT_EQ(dist.local.rows(), 6u);
    auto rebuilt = gather(dist, ctx).value();
    for (size_t i = 0; i < global.rows(); ++i) {
        for (size_t j = 0; j < global.cols(); ++j) {
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
        }
    }
    finalize(ctx);
}

TEST(DistBlockTest, distributed_eig_matches_local) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A{{4, 1}, {1, 3}};
    auto dA = scatter(A, ctx, Distribution::BlockCyclic).value();
    auto local = ms::eig_sym(A).value();
    auto remote = distributed::eig_sym(dA, ctx).value();
    ASSERT_EQ(local.values.rows(), remote.values.rows());
    for (size_t i = 0; i < local.values.rows(); ++i) {
        EXPECT_NEAR(local.values(i, 0), remote.values(i, 0), 1e-6);
    }
    finalize(ctx);
}
