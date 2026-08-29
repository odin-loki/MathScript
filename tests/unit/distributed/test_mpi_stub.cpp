#include <gtest/gtest.h>
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/mpi_context.hpp"

using namespace ms;
using namespace ms::distributed;

TEST(MpiStubTest, single_rank_context) {
    auto ctx = init(0, nullptr);
    EXPECT_TRUE(ctx.active);
    EXPECT_EQ(rank(ctx), 0);
    EXPECT_EQ(size(ctx), 1);
    EXPECT_EQ(backend_name(ctx), "stub");
    finalize(ctx);
    EXPECT_FALSE(ctx.active);
}

TEST(MpiStubTest, scatter_gather_roundtrip) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global{{1, 2, 3}, {4, 5, 6}};
    auto dist = scatter(global, ctx).value();
    EXPECT_EQ(dist.local.rows(), 2);
    auto rebuilt = gather(dist, ctx).value();
    for (size_t i = 0; i < global.rows(); ++i) {
        for (size_t j = 0; j < global.cols(); ++j) {
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
        }
    }
    finalize(ctx);
}

TEST(MpiStubTest, allreduce_identity) {
    auto ctx = init(0, nullptr);
    EXPECT_DOUBLE_EQ(allreduce_sum(ctx, 3.5), 3.5);
    finalize(ctx);
}

TEST(MpiStubTest, bcast_identity) {
    auto ctx = init(0, nullptr);
    EXPECT_EQ(size(ctx), 1);
    EXPECT_DOUBLE_EQ(bcast(ctx, 3.5), 3.5);
    EXPECT_DOUBLE_EQ(bcast(ctx, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(bcast(ctx, -2.25), -2.25);
    finalize(ctx);
}
