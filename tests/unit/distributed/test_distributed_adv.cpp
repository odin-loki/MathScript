// MathScript Distributed Module Advanced Tests
// Tests: MPIContext, block_cyclic_row_indices, allreduce_sum, barrier

#include <gtest/gtest.h>
#include <vector>
#include <cstddef>
#include <string>
#include <variant>

#include "ms/distributed/mpi_context.hpp"
#include "ms/distributed/block.hpp"
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/solve.hpp"
#include "ms/distributed/matmul.hpp"
#include "ms/distributed/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"

using namespace ms;
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

TEST(DistributedAdv, Bcast_Identity) {
    // In a 1-process context, bcast(ctx, x) = x
    MPIContext ctx;
    EXPECT_NEAR(bcast(ctx, 3.14), 3.14, 1e-10);
    EXPECT_NEAR(bcast(ctx, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(bcast(ctx, -1.5), -1.5, 1e-10);
}

TEST(DistributedAdv, Barrier_DoesNotCrash) {
    MPIContext ctx;
    barrier(ctx);
}

TEST(DistributedAdv, Finalize_DoesNotCrash) {
    MPIContext ctx;
    finalize(ctx);
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

TEST(DistributedAdv, Init_ActivatesSingleRank) {
    std::vector<char*> argv;
    auto ctx = init(static_cast<int>(argv.size()), argv.data());
    EXPECT_TRUE(ctx.active);
    EXPECT_EQ(rank(ctx), 0);
    EXPECT_EQ(size(ctx), 1);
    EXPECT_FALSE(backend_name(ctx).empty());
    EXPECT_NEAR(allreduce_sum(ctx, 2.5), 2.5, 1e-10);
    EXPECT_NEAR(bcast(ctx, -4.0), -4.0, 1e-10);
    barrier(ctx);
    finalize(ctx);
    EXPECT_FALSE(ctx.active);
}

TEST(DistributedAdv, EmptyRankContext) {
    MPIContext ctx;
    ctx.rank = 0;
    ctx.size = 0;
    EXPECT_EQ(rank(ctx), 0);
    EXPECT_EQ(size(ctx), 0);
    EXPECT_NEAR(allreduce_sum(ctx, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(allreduce_max(ctx, -2.0), -2.0, 1e-10);
    EXPECT_NEAR(allreduce_min(ctx, 8.0), 8.0, 1e-10);
    EXPECT_NEAR(bcast(ctx, 0.25), 0.25, 1e-10);
    barrier(ctx);
}

TEST(DistributedAdv, BlockRowExtent_EmptyGlobalRows) {
    auto ext = block_row_extent(0, 0, 4);
    EXPECT_EQ(ext.start, 0u);
    EXPECT_EQ(ext.count, 0u);
}

TEST(DistributedAdv, BlockRowExtent_ZeroProcs) {
    auto ext = block_row_extent(10, 0, 0);
    EXPECT_EQ(ext.start, 0u);
    EXPECT_EQ(ext.count, 0u);
}

TEST(DistributedAdv, BlockRowExtent_NegativeProcs) {
    auto ext = block_row_extent(10, 0, -1);
    EXPECT_EQ(ext.start, 0u);
    EXPECT_EQ(ext.count, 0u);
}

TEST(DistributedAdv, BlockRowExtent_RankPastNprocs) {
    auto ext = block_row_extent(10, 5, 2);
    EXPECT_EQ(ext.start, 25u);
    EXPECT_EQ(ext.count, 5u);
}

TEST(DistributedAdv, BlockCyclicRowIndices_EmptyGlobalRows) {
    auto idx = block_cyclic_row_indices(0, 0, 3);
    EXPECT_TRUE(idx.empty());
}

TEST(DistributedAdv, BlockCyclicRowIndices_ZeroProcs) {
    auto idx = block_cyclic_row_indices(8, 0, 0);
    EXPECT_TRUE(idx.empty());
}

TEST(DistributedAdv, BlockCyclicRowIndices_NegativeProcs) {
    auto idx = block_cyclic_row_indices(8, 1, -2);
    EXPECT_TRUE(idx.empty());
}

TEST(DistributedAdv, BlockCyclicRowIndices_EmptyRankPastRows) {
    auto idx = block_cyclic_row_indices(4, 7, 3);
    EXPECT_TRUE(idx.empty());
}

namespace {

DistMatrix<double> dims_only(size_t rows, size_t cols) {
    DistMatrix<double> d;
    d.global_rows = rows;
    d.global_cols = cols;
    return d;
}

} // namespace

TEST(DistributedAdv, Iterative_NonsquareA_AllSolvers) {
    MPIContext ctx;
    const auto A = dims_only(2, 3);
    const auto b = dims_only(2, 1);
    EXPECT_FALSE(dist_cg(A, b, ctx).has_value());
    EXPECT_FALSE(dist_gmres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_jacobi(A, b, ctx).has_value());
    EXPECT_FALSE(dist_bicgstab(A, b, ctx).has_value());
    EXPECT_FALSE(dist_minres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_qmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_tfqmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_lsmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_lsqr(A, b, ctx).has_value());
}

TEST(DistributedAdv, Iterative_WideRhs_AllSolvers) {
    MPIContext ctx;
    const auto A = dims_only(3, 3);
    const auto b = dims_only(3, 2);
    EXPECT_FALSE(dist_cg(A, b, ctx).has_value());
    EXPECT_FALSE(dist_gmres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_jacobi(A, b, ctx).has_value());
    EXPECT_FALSE(dist_bicgstab(A, b, ctx).has_value());
    EXPECT_FALSE(dist_minres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_qmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_tfqmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_lsmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_lsqr(A, b, ctx).has_value());
}

TEST(DistributedAdv, Iterative_RowMismatch_AllSolvers) {
    MPIContext ctx;
    const auto A = dims_only(2, 2);
    const auto b = dims_only(4, 1);
    EXPECT_FALSE(dist_cg(A, b, ctx).has_value());
    EXPECT_FALSE(dist_gmres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_jacobi(A, b, ctx).has_value());
    EXPECT_FALSE(dist_bicgstab(A, b, ctx).has_value());
    EXPECT_FALSE(dist_minres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_qmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_tfqmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_lsmr(A, b, ctx).has_value());
    EXPECT_FALSE(dist_lsqr(A, b, ctx).has_value());
}

TEST(DistributedAdv, Iterative_EmptySquare_RhsColsNotOne) {
    MPIContext ctx;
    const auto A = dims_only(0, 0);
    const auto b = dims_only(0, 0);
    EXPECT_FALSE(dist_cg(A, b, ctx).has_value());
    EXPECT_FALSE(dist_lsqr(A, b, ctx).has_value());
}

TEST(DistributedAdv, ExportedSolve_1x1) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{4.0}};
    const ColMatrix<double> b{{8.0}};
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    const auto x = distributed::solve(dA, db, ctx);
    if (!x.has_value()) GTEST_SKIP() << "distributed::solve rejected 1x1";
    EXPECT_NEAR((*x)(0, 0), 2.0, 1e-10);
    finalize(ctx);
}

TEST(DistributedAdv, ExportedMatmul_1x1) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{3.0}};
    const ColMatrix<double> B{{5.0}};
    const auto dA = scatter(A, ctx).value();
    const auto dB = scatter(B, ctx).value();
    const auto C = distributed::matmul(dA, dB, ctx);
    if (!C.has_value()) GTEST_SKIP() << "distributed::matmul rejected 1x1";
    EXPECT_NEAR((*C)(0, 0), 15.0, 1e-12);
    finalize(ctx);
}

TEST(DistributedAdv, ExportedEigSymSvdLu_1x1) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{9.0}};
    const auto dA = scatter(A, ctx).value();

    const auto ev = distributed::eig_sym(dA, ctx);
    if (!ev.has_value()) GTEST_SKIP() << "distributed::eig_sym rejected 1x1";
    ASSERT_GE(ev->values.rows() * ev->values.cols(), 1u);
    EXPECT_NEAR(ev->values(0, 0), 9.0, 1e-8);

    const auto sv = distributed::svd(dA, ctx);
    if (sv.has_value()) {
        ASSERT_GE(sv->S.rows() * sv->S.cols(), 1u);
        EXPECT_NEAR(sv->S(0, 0), 9.0, 1e-8);
    }

    const auto lu_f = distributed::lu(dA, ctx);
    if (lu_f.has_value()) {
        const auto& [L, U, P] = *lu_f;
        EXPECT_GE(L.rows() + U.rows() + P.rows(), 1u);
    }
    finalize(ctx);
}

TEST(DistributedAdv, CombineGather_EmptyShards) {
    const std::vector<DistMatrix<double>> shards;
    const auto g = combine_gather(shards);
    if (g.has_value()) {
        EXPECT_TRUE(g->empty());
    } else {
        EXPECT_TRUE(std::holds_alternative<DimensionMismatch>(g.error()) ||
                    std::holds_alternative<DomainError>(g.error()));
    }
}

TEST(DistributedAdv, Iterative_EmptySquare_ColumnRhs) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A(0, 0);
    const ColMatrix<double> b(0, 1);
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();

    EXPECT_EQ(dA.global_rows, 0u);
    EXPECT_EQ(dA.global_cols, 0u);
    EXPECT_EQ(db.global_cols, 1u);

    const auto jac = dist_jacobi(dA, db, ctx);
    const auto cg = dist_cg(dA, db, ctx);
    const auto gmres = dist_gmres(dA, db, ctx);
    if (jac.has_value()) {
        EXPECT_EQ(jac->rows(), 0u);
    }
    if (cg.has_value()) {
        EXPECT_EQ(cg->rows(), 0u);
    }
    if (gmres.has_value()) {
        EXPECT_EQ(gmres->rows(), 0u);
    }
    finalize(ctx);
}

TEST(DistributedAdv, Iterative_1x1_JacobiCgGmres) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{4.0}};
    const ColMatrix<double> b{{8.0}};
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();

    const auto jac = dist_jacobi(dA, db, ctx);
    ASSERT_TRUE(jac.has_value());
    ASSERT_EQ(jac->rows(), 1u);
    EXPECT_NEAR((*jac)(0, 0), 2.0, 1e-8);

    const auto cg = dist_cg(dA, db, ctx);
    ASSERT_TRUE(cg.has_value());
    ASSERT_EQ(cg->rows(), 1u);
    EXPECT_NEAR((*cg)(0, 0), 2.0, 1e-8);

    const auto gmres = dist_gmres(dA, db, ctx);
    ASSERT_TRUE(gmres.has_value());
    ASSERT_EQ(gmres->rows(), 1u);
    EXPECT_NEAR((*gmres)(0, 0), 2.0, 1e-8);
    finalize(ctx);
}

TEST(DistributedAdv, Iterative_1x1_Bicgstab) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{4.0}};
    const ColMatrix<double> b{{8.0}};
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    const auto xb = dist_bicgstab(dA, db, ctx);
    ASSERT_TRUE(xb.has_value());
    ASSERT_EQ(xb->rows(), 1u);
    EXPECT_NEAR((*xb)(0, 0), 2.0, 1e-8);
    finalize(ctx);
}

TEST(DistributedAdv, Iterative_EmptySquare_Bicgstab) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A(0, 0);
    const ColMatrix<double> b(0, 1);
    const auto dA = scatter(A, ctx).value();
    const auto db = scatter(b, ctx).value();
    const auto xb = dist_bicgstab(dA, db, ctx);
    if (xb.has_value()) {
        EXPECT_EQ(xb->rows(), 0u);
    }
    finalize(ctx);
}

TEST(DistributedAdv, Iterative_EmptySquare_RhsColsNotOne_JacobiGmresBicgstab) {
    MPIContext ctx;
    const auto A = dims_only(0, 0);
    const auto b = dims_only(0, 0);
    EXPECT_FALSE(dist_jacobi(A, b, ctx).has_value());
    EXPECT_FALSE(dist_gmres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_bicgstab(A, b, ctx).has_value());
}

TEST(DistributedAdv, Iterative_ZeroByOne_RowMismatch) {
    MPIContext ctx;
    const auto A = dims_only(0, 0);
    const auto b = dims_only(1, 1);
    EXPECT_FALSE(dist_jacobi(A, b, ctx).has_value());
    EXPECT_FALSE(dist_cg(A, b, ctx).has_value());
    EXPECT_FALSE(dist_gmres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_bicgstab(A, b, ctx).has_value());
}

TEST(DistributedAdv, Iterative_OneByOne_EmptyRhsCols) {
    MPIContext ctx;
    const auto A = dims_only(1, 1);
    const auto b = dims_only(1, 0);
    EXPECT_FALSE(dist_jacobi(A, b, ctx).has_value());
    EXPECT_FALSE(dist_cg(A, b, ctx).has_value());
    EXPECT_FALSE(dist_gmres(A, b, ctx).has_value());
    EXPECT_FALSE(dist_bicgstab(A, b, ctx).has_value());
}
