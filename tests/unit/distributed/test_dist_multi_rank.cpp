#include <gtest/gtest.h>
#include "ms/distributed/block.hpp"
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/mpi_context.hpp"

using namespace ms;
using namespace ms::distributed;

namespace {

ColMatrix<double> make_test_matrix(size_t rows, size_t cols) {
    ColMatrix<double> m(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = static_cast<double>(i * 10 + j);
        }
    }
    return m;
}

std::vector<DistMatrix<double>> simulate_shards(
    const ColMatrix<double>& global,
    int nprocs,
    Distribution distribution) {
    std::vector<DistMatrix<double>> shards;
    shards.reserve(static_cast<size_t>(nprocs));
    for (int r = 0; r < nprocs; ++r) {
        auto ctx = init(0, nullptr);
        ctx.rank = r;
        ctx.size = nprocs;
        shards.push_back(scatter(global, ctx, distribution).value());
        finalize(ctx);
    }
    return shards;
}

} // namespace

TEST(DistMultiRankTest, block_partition_sums_to_global_rows) {
    constexpr int nprocs = 4;
    size_t total = 0;
    for (int r = 0; r < nprocs; ++r) {
        const auto ext = block_row_extent(17, r, nprocs);
        total += ext.count;
    }
    EXPECT_EQ(total, 17u);
}

TEST(DistMultiRankTest, block_combine_gather_four_ranks) {
    const auto global = make_test_matrix(9, 2);
    const auto shards = simulate_shards(global, 4, Distribution::Block);
    ASSERT_EQ(shards.size(), 4u);
    const auto rebuilt = combine_gather(shards).value();
    for (size_t i = 0; i < global.rows(); ++i) {
        for (size_t j = 0; j < global.cols(); ++j) {
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
        }
    }
}

TEST(DistMultiRankTest, block_cyclic_combine_gather_four_ranks) {
    const auto global = make_test_matrix(10, 3);
    const auto shards = simulate_shards(global, 4, Distribution::BlockCyclic);
    const auto rebuilt = combine_gather(shards).value();
    for (size_t i = 0; i < global.rows(); ++i) {
        for (size_t j = 0; j < global.cols(); ++j) {
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
        }
    }
}

#if defined(MS_HAS_MPI) && MS_HAS_MPI
TEST(MpiFourRankTest, live_partition_when_launched_with_four_ranks) {
    auto ctx = init(0, nullptr);
    if (size(ctx) != 4) {
        finalize(ctx);
        GTEST_SKIP() << "Run with mpiexec -n 4 to enable this test";
    }

    ColMatrix<double> global = make_test_matrix(12, 2);
    auto local = scatter(global, ctx, Distribution::Block).value();
    EXPECT_GT(local.local.rows(), 0u);
    EXPECT_EQ(local.local.cols(), 2u);

    auto gathered = gather(local, ctx).value();
    if (rank(ctx) == 0) {
        for (size_t i = 0; i < global.rows(); ++i) {
            for (size_t j = 0; j < global.cols(); ++j) {
                EXPECT_DOUBLE_EQ(gathered(i, j), global(i, j));
            }
        }
    }
    finalize(ctx);
}
#endif
