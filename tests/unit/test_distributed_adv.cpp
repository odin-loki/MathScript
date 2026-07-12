// MathScript Distributed Module Advanced Tests
// Tests: MPIContext, block_cyclic_row_indices, allreduce_sum, barrier

#include <gtest/gtest.h>
#include <vector>
#include <cstddef>
#include <string>

#include "ms/distributed/mpi_context.hpp"
#include "ms/distributed/block.hpp"

using namespace ms::distributed;

// ---------------------------------------------------------------------------
// MPIContext: default construction and single-process semantics
// ---------------------------------------------------------------------------

TEST(DistributedAdv, MPIContext_Default) {
    MPIContext ctx;
    EXPECT_EQ(ctx.rank, 0);
    EXPECT_EQ(ctx.size, 1);
    EXPECT_FALSE(ctx.active);
}

TEST(DistributedAdv, Rank_DefaultContext) {
    MPIContext ctx;
    EXPECT_EQ(rank(ctx), 0);
}

TEST(DistributedAdv, Size_DefaultContext) {
    MPIContext ctx;
    EXPECT_EQ(size(ctx), 1);
}

TEST(DistributedAdv, BackendName_ReturnsString) {
    MPIContext ctx;
    std::string name = backend_name(ctx);
    EXPECT_FALSE(name.empty());
}

TEST(DistributedAdv, AllreduceSum_SingleProcess) {
    // In a 1-process context, allreduce_sum(ctx, x) = x
    MPIContext ctx;
    EXPECT_NEAR(allreduce_sum(ctx, 3.14), 3.14, 1e-10);
    EXPECT_NEAR(allreduce_sum(ctx, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(allreduce_sum(ctx, -1.5), -1.5, 1e-10);
}

TEST(DistributedAdv, AllreduceMax_SingleProcess) {
    // In a 1-process context, allreduce_max(ctx, x) = x
    MPIContext ctx;
    EXPECT_NEAR(allreduce_max(ctx, 3.14), 3.14, 1e-10);
    EXPECT_NEAR(allreduce_max(ctx, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(allreduce_max(ctx, -1.5), -1.5, 1e-10);
}

TEST(DistributedAdv, AllreduceMin_SingleProcess) {
    // In a 1-process context, allreduce_min(ctx, x) = x
    MPIContext ctx;
    EXPECT_NEAR(allreduce_min(ctx, 3.14), 3.14, 1e-10);
    EXPECT_NEAR(allreduce_min(ctx, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(allreduce_min(ctx, -1.5), -1.5, 1e-10);
}

TEST(DistributedAdv, AllreduceMax_Idempotent) {
    // Applying allreduce_max twice should give the same result in the stub path
    MPIContext ctx;
    double once = allreduce_max(ctx, 42.0);
    double twice = allreduce_max(ctx, once);
    EXPECT_NEAR(once, twice, 1e-10);
}

TEST(DistributedAdv, AllreduceMin_Idempotent) {
    // Applying allreduce_min twice should give the same result in the stub path
    MPIContext ctx;
    double once = allreduce_min(ctx, -7.0);
    double twice = allreduce_min(ctx, once);
    EXPECT_NEAR(once, twice, 1e-10);
}

TEST(DistributedAdv, AllreduceMinMax_Ordering) {
    // With a single rank, min <= value <= max should hold (in fact, equality)
    MPIContext ctx;
    for (double v : {-100.0, -1.5, 0.0, 1.5, 100.0}) {
        EXPECT_LE(allreduce_min(ctx, v), v);
        EXPECT_LE(v, allreduce_max(ctx, v));
        EXPECT_NEAR(allreduce_min(ctx, v), allreduce_max(ctx, v), 1e-10);
    }
}

TEST(DistributedAdv, AllreduceMaxMin_EdgeCaseMagnitudes) {
    // Very large/small magnitude values should pass through unchanged, no overflow
    MPIContext ctx;
    EXPECT_NEAR(allreduce_max(ctx, 1e300), 1e300, 1e290);
    EXPECT_NEAR(allreduce_min(ctx, -1e300), -1e300, 1e290);
    EXPECT_NEAR(allreduce_max(ctx, 1e-300), 1e-300, 1e-310);
    EXPECT_NEAR(allreduce_min(ctx, 1e-300), 1e-300, 1e-310);
}

TEST(DistributedAdv, AllreduceMaxMin_MultipleContexts) {
    // Mirrors allreduce_sum's usage with a freshly constructed MPIContext each time
    for (double v : {-3.0, 0.0, 2.0}) {
        MPIContext ctx;
        EXPECT_NEAR(allreduce_max(ctx, v), v, 1e-10);
        EXPECT_NEAR(allreduce_min(ctx, v), v, 1e-10);
    }
}

TEST(DistributedAdv, Barrier_DoesNotCrash) {
    MPIContext ctx;
    EXPECT_NO_THROW(barrier(ctx));
}

TEST(DistributedAdv, Finalize_DoesNotCrash) {
    MPIContext ctx;
    EXPECT_NO_THROW(finalize(ctx));
}

// ---------------------------------------------------------------------------
// block_cyclic_row_indices(global_rows, rank, nprocs): round-robin distribution
// block_row_extent(global_rows, rank, nprocs): contiguous block distribution
// ---------------------------------------------------------------------------

TEST(DistributedAdv, BlockCyclicRowIndices_SingleProc) {
    // With 1 process (rank=0, nprocs=1), all rows belong to process 0
    auto indices = block_cyclic_row_indices(10, 0, 1);
    EXPECT_EQ(indices.size(), 10u);
    for (size_t i = 0; i < indices.size(); ++i)
        EXPECT_EQ(indices[i], i);
}

TEST(DistributedAdv, BlockCyclicRowIndices_TwoProcs_Rank0) {
    // Round-robin: rank 0 gets rows 0, 2, 4, 6, 8
    auto idx0 = block_cyclic_row_indices(10, 0, 2);
    EXPECT_EQ(idx0.size(), 5u);
    for (size_t r : idx0) EXPECT_LT(r, 10u);
}

TEST(DistributedAdv, BlockCyclicRowIndices_TwoProcs_Rank1) {
    // Rank 1 gets rows 1, 3, 5, 7, 9
    auto idx1 = block_cyclic_row_indices(10, 1, 2);
    EXPECT_EQ(idx1.size(), 5u);
    for (size_t r : idx1) EXPECT_LT(r, 10u);
}

TEST(DistributedAdv, BlockCyclicRowIndices_Partition_Complete) {
    // Union of all ranks' indices = all rows
    auto idx0 = block_cyclic_row_indices(8, 0, 2);
    auto idx1 = block_cyclic_row_indices(8, 1, 2);
    EXPECT_EQ(idx0.size() + idx1.size(), 8u);
    std::vector<bool> seen(8, false);
    for (size_t r : idx0) { EXPECT_LT(r, 8u); seen[r] = true; }
    for (size_t r : idx1) { EXPECT_LT(r, 8u); seen[r] = true; }
    for (bool s : seen) EXPECT_TRUE(s);
}

TEST(DistributedAdv, BlockCyclicRowIndices_EmptyForHighRank) {
    // Rank >= global_rows: should get 0 rows
    auto idx = block_cyclic_row_indices(5, 10, 11);
    EXPECT_EQ(idx.size(), 0u);
}

TEST(DistributedAdv, BlockRowExtent_SingleProc) {
    auto ext = block_row_extent(10, 0, 1);
    EXPECT_EQ(ext.start, 0u);
    EXPECT_EQ(ext.count, 10u);
}

TEST(DistributedAdv, BlockRowExtent_TwoProcs) {
    auto e0 = block_row_extent(10, 0, 2);
    auto e1 = block_row_extent(10, 1, 2);
    // Together cover all 10 rows
    EXPECT_EQ(e0.count + e1.count, 10u);
    EXPECT_EQ(e1.start, e0.start + e0.count);
}

TEST(DistributedAdv, BlockRowExtent_Uneven) {
    // 7 rows, 3 procs: 7/3=2 rem 1, so rank 0 gets 3, ranks 1-2 get 2 each
    auto e0 = block_row_extent(7, 0, 3);
    EXPECT_EQ(e0.count, 3u);
    auto e1 = block_row_extent(7, 1, 3);
    EXPECT_EQ(e1.count, 2u);
    auto e2 = block_row_extent(7, 2, 3);
    EXPECT_EQ(e2.count, 2u);
}
