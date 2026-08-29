// MathScript DistMatrix Advanced Tests
// Tests: DistMatrix struct, scatter/gather/combine_gather, distributed::solve,
//        MPIContext helpers, allreduce_sum, barrier

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <cstddef>

#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/distributed/solve.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using namespace ms::distributed;

using DMatrix = ms::Matrix<double>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

ColMatrix<double> make_matrix(size_t rows, size_t cols, double base = 1.0) {
    ColMatrix<double> m(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            m(i, j) = base + static_cast<double>(i * cols + j);
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
// DistMatrix: default construction and field access
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, DefaultConstruction_FieldDefaults) {
    DistMatrix<double> dm;
    EXPECT_EQ(dm.distribution, Distribution::Block);
    EXPECT_EQ(dm.owner_rank, 0);
    EXPECT_EQ(dm.global_rows, 0u);
    EXPECT_EQ(dm.global_cols, 0u);
    EXPECT_TRUE(dm.row_map.empty());
}

TEST(DistMatrixAdv, DistributionEnum_BlockValue) {
    DistMatrix<double> dm;
    dm.distribution = Distribution::Block;
    EXPECT_EQ(dm.distribution, Distribution::Block);
    EXPECT_NE(dm.distribution, Distribution::BlockCyclic);
}

TEST(DistMatrixAdv, DistributionEnum_BlockCyclicValue) {
    DistMatrix<double> dm;
    dm.distribution = Distribution::BlockCyclic;
    EXPECT_EQ(dm.distribution, Distribution::BlockCyclic);
    EXPECT_NE(dm.distribution, Distribution::Block);
}

TEST(DistMatrixAdv, DistributionEnum_DistinctValues) {
    EXPECT_NE(Distribution::Block, Distribution::BlockCyclic);
}

TEST(DistMatrixAdv, FieldAssignment_OwnerRank) {
    DistMatrix<double> dm;
    dm.owner_rank = 3;
    EXPECT_EQ(dm.owner_rank, 3);
}

TEST(DistMatrixAdv, FieldAssignment_GlobalDimensions) {
    DistMatrix<double> dm;
    dm.global_rows = 100;
    dm.global_cols = 50;
    EXPECT_EQ(dm.global_rows, 100u);
    EXPECT_EQ(dm.global_cols, 50u);
}

TEST(DistMatrixAdv, FieldAssignment_RowMap) {
    DistMatrix<double> dm;
    dm.row_map = {0, 2, 4, 6};
    ASSERT_EQ(dm.row_map.size(), 4u);
    EXPECT_EQ(dm.row_map[0], 0u);
    EXPECT_EQ(dm.row_map[3], 6u);
}

// ---------------------------------------------------------------------------
// MPIContext: struct defaults and free functions
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, MPIContext_StructDefaults) {
    MPIContext ctx{};
    EXPECT_EQ(ctx.rank, 0);
    EXPECT_EQ(ctx.size, 1);
    EXPECT_FALSE(ctx.active);
}

TEST(DistMatrixAdv, MPIContext_InitActivates) {
    auto ctx = init(0, nullptr);
    EXPECT_TRUE(ctx.active);
    finalize(ctx);
}

TEST(DistMatrixAdv, MPIContext_FinalizeDeactivates) {
    auto ctx = init(0, nullptr);
    finalize(ctx);
    EXPECT_FALSE(ctx.active);
}

TEST(DistMatrixAdv, RankFn_SingleProcess) {
    auto ctx = init(0, nullptr);
    EXPECT_EQ(rank(ctx), 0);
    finalize(ctx);
}

TEST(DistMatrixAdv, SizeFn_SingleProcess) {
    auto ctx = init(0, nullptr);
    EXPECT_EQ(size(ctx), 1);
    finalize(ctx);
}

TEST(DistMatrixAdv, BackendName_NotEmpty) {
    auto ctx = init(0, nullptr);
    std::string name = backend_name(ctx);
    EXPECT_FALSE(name.empty());
    finalize(ctx);
}

TEST(DistMatrixAdv, BackendName_IsStub) {
    auto ctx = init(0, nullptr);
    EXPECT_EQ(backend_name(ctx), "stub");
    finalize(ctx);
}

TEST(DistMatrixAdv, AllreduceSum_SingleProcess_Identity) {
    auto ctx = init(0, nullptr);
    EXPECT_DOUBLE_EQ(allreduce_sum(ctx, 7.0), 7.0);
    EXPECT_DOUBLE_EQ(allreduce_sum(ctx, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(allreduce_sum(ctx, -2.5), -2.5);
    finalize(ctx);
}

TEST(DistMatrixAdv, AllreduceSum_LargeValue) {
    auto ctx = init(0, nullptr);
    const double big = 1e15;
    EXPECT_DOUBLE_EQ(allreduce_sum(ctx, big), big);
    finalize(ctx);
}

TEST(DistMatrixAdv, Barrier_DoesNotThrow) {
    auto ctx = init(0, nullptr);
    EXPECT_NO_THROW(barrier(ctx));
    finalize(ctx);
}

// ---------------------------------------------------------------------------
// scatter: single-rank context
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, Scatter_SingleRank_ReturnsAllRows) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(5, 3);
    auto result = scatter(global, ctx, Distribution::Block);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->local.rows(), 5u);
    EXPECT_EQ(result->local.cols(), 3u);
    finalize(ctx);
}

TEST(DistMatrixAdv, Scatter_Block_GlobalDimensionsPreserved) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(4, 2);
    auto dist = scatter(global, ctx, Distribution::Block).value();
    EXPECT_EQ(dist.global_rows, 4u);
    EXPECT_EQ(dist.global_cols, 2u);
    finalize(ctx);
}

TEST(DistMatrixAdv, Scatter_BlockCyclic_SingleRank_ReturnsAllRows) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(6, 2);
    auto result = scatter(global, ctx, Distribution::BlockCyclic);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->local.rows(), 6u);
    finalize(ctx);
}

TEST(DistMatrixAdv, Scatter_BlockCyclic_DistributionFieldSet) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(4, 2);
    auto dist = scatter(global, ctx, Distribution::BlockCyclic).value();
    EXPECT_EQ(dist.distribution, Distribution::BlockCyclic);
    finalize(ctx);
}

TEST(DistMatrixAdv, Scatter_SingleRank_ValuesCorrect) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global{{10, 20}, {30, 40}, {50, 60}};
    auto dist = scatter(global, ctx).value();
    EXPECT_DOUBLE_EQ(dist.local(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(dist.local(1, 1), 40.0);
    EXPECT_DOUBLE_EQ(dist.local(2, 0), 50.0);
    finalize(ctx);
}

TEST(DistMatrixAdv, Scatter_SimulatedRank1_Of4_CorrectSlice) {
    auto ctx = init(0, nullptr);
    ctx.size = 4;
    ctx.rank = 1;

    ColMatrix<double> global(8, 1);
    for (size_t i = 0; i < 8; ++i)
        global(i, 0) = static_cast<double>(i);

    auto dist = scatter(global, ctx, Distribution::Block).value();
    EXPECT_EQ(dist.local.rows(), 2u);  // 8/4 = 2 rows per rank
    EXPECT_DOUBLE_EQ(dist.local(0, 0), 2.0);  // rank 1 starts at row 2
    EXPECT_DOUBLE_EQ(dist.local(1, 0), 3.0);
    finalize(ctx);
}

// ---------------------------------------------------------------------------
// gather: roundtrip after scatter
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, Gather_Roundtrip_Block_1x1) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global{{42.0}};
    auto dist = scatter(global, ctx).value();
    auto rebuilt = gather(dist, ctx).value();
    EXPECT_DOUBLE_EQ(rebuilt(0, 0), 42.0);
    finalize(ctx);
}

TEST(DistMatrixAdv, Gather_Roundtrip_Block_RowVector) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global(1, 5);
    for (size_t j = 0; j < 5; ++j)
        global(0, j) = static_cast<double>(j + 1);
    auto dist = scatter(global, ctx).value();
    auto rebuilt = gather(dist, ctx).value();
    EXPECT_EQ(rebuilt.rows(), 1u);
    EXPECT_EQ(rebuilt.cols(), 5u);
    for (size_t j = 0; j < 5; ++j)
        EXPECT_DOUBLE_EQ(rebuilt(0, j), global(0, j));
    finalize(ctx);
}

TEST(DistMatrixAdv, Gather_Roundtrip_Block_MultiRow) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(6, 4);
    auto dist = scatter(global, ctx, Distribution::Block).value();
    auto rebuilt = gather(dist, ctx).value();
    ASSERT_EQ(rebuilt.rows(), global.rows());
    ASSERT_EQ(rebuilt.cols(), global.cols());
    for (size_t i = 0; i < global.rows(); ++i)
        for (size_t j = 0; j < global.cols(); ++j)
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
    finalize(ctx);
}

TEST(DistMatrixAdv, Gather_Roundtrip_BlockCyclic) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(8, 3);
    auto dist = scatter(global, ctx, Distribution::BlockCyclic).value();
    auto rebuilt = gather(dist, ctx).value();
    ASSERT_EQ(rebuilt.rows(), global.rows());
    ASSERT_EQ(rebuilt.cols(), global.cols());
    for (size_t i = 0; i < global.rows(); ++i)
        for (size_t j = 0; j < global.cols(); ++j)
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
    finalize(ctx);
}

TEST(DistMatrixAdv, Gather_Roundtrip_NegativeValues) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global{{-1.0, -2.0}, {-3.0, -4.0}};
    auto dist = scatter(global, ctx).value();
    auto rebuilt = gather(dist, ctx).value();
    EXPECT_DOUBLE_EQ(rebuilt(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(rebuilt(1, 1), -4.0);
    finalize(ctx);
}

// ---------------------------------------------------------------------------
// combine_gather: reassemble from shards
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, CombineGather_SingleShard_Roundtrip) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(5, 2);
    auto shard = scatter(global, ctx, Distribution::Block).value();
    std::vector<DistMatrix<double>> shards{shard};
    auto rebuilt = combine_gather(shards).value();
    ASSERT_EQ(rebuilt.rows(), global.rows());
    ASSERT_EQ(rebuilt.cols(), global.cols());
    for (size_t i = 0; i < global.rows(); ++i)
        for (size_t j = 0; j < global.cols(); ++j)
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
    finalize(ctx);
}

TEST(DistMatrixAdv, CombineGather_TwoShards_Block) {
    auto global = make_matrix(6, 2);
    std::vector<DistMatrix<double>> shards;
    for (int r = 0; r < 2; ++r) {
        auto ctx = init(0, nullptr);
        ctx.rank = r;
        ctx.size = 2;
        shards.push_back(scatter(global, ctx, Distribution::Block).value());
        finalize(ctx);
    }
    auto rebuilt = combine_gather(shards).value();
    ASSERT_EQ(rebuilt.rows(), 6u);
    ASSERT_EQ(rebuilt.cols(), 2u);
    for (size_t i = 0; i < global.rows(); ++i)
        for (size_t j = 0; j < global.cols(); ++j)
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
}

TEST(DistMatrixAdv, CombineGather_ThreeShards_Block) {
    auto global = make_matrix(9, 3);
    std::vector<DistMatrix<double>> shards;
    for (int r = 0; r < 3; ++r) {
        auto ctx = init(0, nullptr);
        ctx.rank = r;
        ctx.size = 3;
        shards.push_back(scatter(global, ctx, Distribution::Block).value());
        finalize(ctx);
    }
    auto rebuilt = combine_gather(shards).value();
    ASSERT_EQ(rebuilt.rows(), 9u);
    for (size_t i = 0; i < global.rows(); ++i)
        for (size_t j = 0; j < global.cols(); ++j)
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
}

TEST(DistMatrixAdv, CombineGather_TwoShards_BlockCyclic) {
    auto global = make_matrix(8, 2);
    std::vector<DistMatrix<double>> shards;
    for (int r = 0; r < 2; ++r) {
        auto ctx = init(0, nullptr);
        ctx.rank = r;
        ctx.size = 2;
        shards.push_back(scatter(global, ctx, Distribution::BlockCyclic).value());
        finalize(ctx);
    }
    auto rebuilt = combine_gather(shards).value();
    ASSERT_EQ(rebuilt.rows(), 8u);
    for (size_t i = 0; i < global.rows(); ++i)
        for (size_t j = 0; j < global.cols(); ++j)
            EXPECT_DOUBLE_EQ(rebuilt(i, j), global(i, j));
}

// ---------------------------------------------------------------------------
// distributed::solve
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, DistributedSolve_2x2_SingleRank) {
    auto ctx = init(0, nullptr);
    // 4x + y = 1
    // x + 3y = 2  => x=1/11, y=7/11
    ColMatrix<double> A{{4.0, 1.0}, {1.0, 3.0}};
    ColMatrix<double> b{{1.0}, {2.0}};
    auto dA = scatter(A, ctx).value();
    auto db = scatter(b, ctx).value();
    auto result = distributed::solve(dA, db, ctx);
    ASSERT_TRUE(result.has_value());
    const double x0 = result.value()(0, 0);
    const double x1 = result.value()(1, 0);
    EXPECT_NEAR(x0, 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(x1, 7.0 / 11.0, 1e-6);
    finalize(ctx);
}

TEST(DistMatrixAdv, DistributedSolve_Identity_2x2) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A{{1.0, 0.0}, {0.0, 1.0}};
    ColMatrix<double> b{{3.0}, {5.0}};
    auto dA = scatter(A, ctx).value();
    auto db = scatter(b, ctx).value();
    auto result = distributed::solve(dA, db, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value()(0, 0), 3.0, 1e-10);
    EXPECT_NEAR(result.value()(1, 0), 5.0, 1e-10);
    finalize(ctx);
}

TEST(DistMatrixAdv, DistributedSolve_Diagonal_3x3) {
    auto ctx = init(0, nullptr);
    // 2x=4, 3y=6, 4z=8 => x=2,y=2,z=2
    ColMatrix<double> A{{2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 4.0}};
    ColMatrix<double> b{{4.0}, {6.0}, {8.0}};
    auto dA = scatter(A, ctx).value();
    auto db = scatter(b, ctx).value();
    auto result = distributed::solve(dA, db, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value()(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(result.value()(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(result.value()(2, 0), 2.0, 1e-9);
    finalize(ctx);
}

TEST(DistMatrixAdv, DistributedSolve_ReturnsExpectedType) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A{{2.0, 1.0}, {1.0, 2.0}};
    ColMatrix<double> b{{1.0}, {0.0}};
    auto dA = scatter(A, ctx).value();
    auto db = scatter(b, ctx).value();
    auto result = distributed::solve(dA, db, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().rows(), 2u);
    EXPECT_EQ(result.value().cols(), 1u);
    finalize(ctx);
}

// ---------------------------------------------------------------------------
// Local field of DistMatrix after scatter
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, Scatter_LocalMatrix_CorrectShape) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(7, 5);
    auto dist = scatter(global, ctx, Distribution::Block).value();
    EXPECT_GT(dist.local.rows(), 0u);
    EXPECT_EQ(dist.local.cols(), 5u);
    finalize(ctx);
}

TEST(DistMatrixAdv, Scatter_LocalMatrix_SumMatchesGlobal) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(4, 3);
    double global_sum = 0.0;
    for (size_t i = 0; i < global.rows(); ++i)
        for (size_t j = 0; j < global.cols(); ++j)
            global_sum += global(i, j);

    auto dist = scatter(global, ctx, Distribution::Block).value();
    // Single-rank: local == global, sums must match
    double local_sum = 0.0;
    for (size_t i = 0; i < dist.local.rows(); ++i)
        for (size_t j = 0; j < dist.local.cols(); ++j)
            local_sum += dist.local(i, j);
    EXPECT_DOUBLE_EQ(local_sum, global_sum);
    finalize(ctx);
}

// ---------------------------------------------------------------------------
// Scatter / gather dimension consistency
// ---------------------------------------------------------------------------

TEST(DistMatrixAdv, ScatterGather_LargeMatrix_DimensionsPreserved) {
    auto ctx = init(0, nullptr);
    auto global = make_matrix(20, 10);
    auto dist = scatter(global, ctx).value();
    auto rebuilt = gather(dist, ctx).value();
    EXPECT_EQ(rebuilt.rows(), 20u);
    EXPECT_EQ(rebuilt.cols(), 10u);
    finalize(ctx);
}

TEST(DistMatrixAdv, ScatterGather_FloatPrecision_Values) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> global(3, 2);
    global(0, 0) = 1.23456789012345;
    global(1, 1) = -9.87654321098765;
    global(2, 0) = 3.14159265358979;
    auto dist = scatter(global, ctx).value();
    auto rebuilt = gather(dist, ctx).value();
    EXPECT_DOUBLE_EQ(rebuilt(0, 0), global(0, 0));
    EXPECT_DOUBLE_EQ(rebuilt(1, 1), global(1, 1));
    EXPECT_DOUBLE_EQ(rebuilt(2, 0), global(2, 0));
    finalize(ctx);
}


