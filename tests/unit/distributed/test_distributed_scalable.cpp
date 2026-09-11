// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MathScript Scalable Distributed Linear Algebra Tests
// Covers the ms::distributed primitive layer (dist_ops), the SUMMA matmul and
// the row-distributed Krylov solvers. Every solver assertion is a PARITY
// assertion against the corresponding ms:: kernel: the distributed loops run
// the same arithmetic in the same order, so at one rank they must agree to the
// last bit, including where the serial kernel is itself imprecise.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <variant>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/distributed/block.hpp"
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/dist_ops.hpp"
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/matmul.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/error/error_types.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using namespace ms::distributed;

namespace {

// -1 off the diagonal, `diag` on it: symmetric positive definite for diag > 2.
ColMatrix<double> spd_tridiag(size_t n, double diag) {
    ColMatrix<double> A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = diag;
        if (i > 0) {
            A(i, i - 1) = -1.0;
            A(i - 1, i) = -1.0;
        }
    }
    return A;
}

ColMatrix<double> nonsymmetric(size_t n) {
    ColMatrix<double> A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = static_cast<double>(n + 2);
        if (i > 0) {
            A(i, i - 1) = -1.0;
            A(i - 1, i) = -0.5;
        }
    }
    return A;
}

ColMatrix<double> ramp(size_t n) {
    ColMatrix<double> b(n, 1);
    for (size_t i = 0; i < n; ++i) {
        b(i, 0) = static_cast<double>(i + 1);
    }
    return b;
}

} // namespace

// ===========================================================================
// Layout descriptors
// ===========================================================================

TEST(DistOpsTest, process_grid_factorisation_table) {
    struct Row {
        int p;
        int rows;
        int cols;
    };
    const Row table[] = {
        {1, 1, 1},   {2, 1, 2},   {3, 1, 3},    {4, 2, 2},     {5, 1, 5},
        {6, 2, 3},   {7, 1, 7},   {8, 2, 4},    {9, 3, 3},     {10, 2, 5},
        {11, 1, 11}, {12, 3, 4},  {13, 1, 13},  {16, 4, 4},    {18, 3, 6},
        {20, 4, 5},  {24, 4, 6},  {27, 3, 9},   {30, 5, 6},    {32, 4, 8},
        {36, 6, 6},  {48, 6, 8},  {64, 8, 8},   {100, 10, 10}, {128, 8, 16},
        {1000, 25, 40},
    };
    for (const Row& row : table) {
        const ProcessGrid g = make_process_grid(row.p, 0);
        EXPECT_EQ(g.rows, row.rows) << "nprocs=" << row.p;
        EXPECT_EQ(g.cols, row.cols) << "nprocs=" << row.p;
        EXPECT_EQ(g.rows * g.cols, row.p) << "nprocs=" << row.p;
        EXPECT_LE(g.rows, g.cols) << "nprocs=" << row.p;
    }
}

TEST(DistOpsTest, process_grid_coordinates_and_degenerate_input) {
    const ProcessGrid g = make_process_grid(6, 4);
    EXPECT_EQ(g.rows, 2);
    EXPECT_EQ(g.cols, 3);
    EXPECT_EQ(g.my_row, 1);
    EXPECT_EQ(g.my_col, 1);
    EXPECT_EQ(g.rank, 4);

    const ProcessGrid zero = make_process_grid(0, 0);
    EXPECT_EQ(zero.nprocs, 1);
    EXPECT_EQ(zero.rows, 1);
    EXPECT_EQ(zero.cols, 1);

    const ProcessGrid negative = make_process_grid(-3, 7);
    EXPECT_EQ(negative.nprocs, 1);
    EXPECT_EQ(negative.rows, 1);
    EXPECT_EQ(negative.cols, 1);
    EXPECT_EQ(negative.rank, 0);

    EXPECT_EQ(make_process_grid(4, 99).rank, 0);
}

TEST(DistOpsTest, row_layout_matches_block_row_extent) {
    for (size_t n : {size_t{0}, size_t{1}, size_t{3}, size_t{6}, size_t{10}, size_t{17}}) {
        for (int p = 1; p <= 5; ++p) {
            const RowLayout lay = make_row_layout(n, p);
            EXPECT_EQ(lay.global_rows, n);
            EXPECT_EQ(lay.nprocs, p);
            ASSERT_EQ(lay.starts.size(), static_cast<size_t>(p));
            ASSERT_EQ(lay.counts.size(), static_cast<size_t>(p));
            size_t total = 0;
            for (int r = 0; r < p; ++r) {
                const RowExtent ext = block_row_extent(n, r, p);
                const size_t u = static_cast<size_t>(r);
                EXPECT_EQ(lay.starts[u], ext.start) << "n=" << n << " p=" << p << " r=" << r;
                EXPECT_EQ(lay.counts[u], ext.count) << "n=" << n << " p=" << p << " r=" << r;
                total += lay.counts[u];
            }
            EXPECT_EQ(total, n);
        }
    }

    const RowLayout ten = make_row_layout(10, 4);
    EXPECT_EQ(ten.starts, (std::vector<size_t>{0, 3, 6, 8}));
    EXPECT_EQ(ten.counts, (std::vector<size_t>{3, 3, 2, 2}));

    const RowLayout three = make_row_layout(3, 5);
    EXPECT_EQ(three.starts, (std::vector<size_t>{0, 1, 2, 3, 3}));
    EXPECT_EQ(three.counts, (std::vector<size_t>{1, 1, 1, 0, 0}));

    const RowLayout seventeen = make_row_layout(17, 4);
    EXPECT_EQ(seventeen.starts, (std::vector<size_t>{0, 5, 9, 13}));
    EXPECT_EQ(seventeen.counts, (std::vector<size_t>{5, 4, 4, 4}));
}

TEST(DistOpsTest, layout_matches_rejects_out_of_range_ranks) {
    const RowLayout lay = make_row_layout(10, 4);
    EXPECT_TRUE(layout_matches(lay, 0, 3));
    EXPECT_TRUE(layout_matches(lay, 3, 2));
    EXPECT_FALSE(layout_matches(lay, 0, 4));
    EXPECT_FALSE(layout_matches(lay, -1, 3));
    EXPECT_FALSE(layout_matches(lay, 4, 0));
}

TEST(DistOpsTest, k_breaks_table) {
    EXPECT_EQ(summa_k_breaks(5, 1, 1), (std::vector<size_t>{0, 5}));
    EXPECT_EQ(summa_k_breaks(6, 2, 2), (std::vector<size_t>{0, 3, 6}));
    EXPECT_EQ(summa_k_breaks(8, 2, 4), (std::vector<size_t>{0, 2, 4, 6, 8}));
    EXPECT_EQ(summa_k_breaks(10, 2, 3), (std::vector<size_t>{0, 4, 5, 7, 10}));
    EXPECT_EQ(summa_k_breaks(12, 3, 4), (std::vector<size_t>{0, 3, 4, 6, 8, 9, 12}));
    EXPECT_EQ(summa_k_breaks(3, 2, 3), (std::vector<size_t>{0, 1, 2, 3}));
    EXPECT_EQ(summa_k_breaks(0, 2, 2), (std::vector<size_t>{0}));
}

TEST(DistOpsTest, k_breaks_refine_both_block_splits) {
    // Every panel must sit inside a single A column block and a single B row
    // block, or it would have more than one broadcast root per grid axis.
    for (size_t k : {size_t{1}, size_t{3}, size_t{8}, size_t{10}, size_t{12}, size_t{29}}) {
        for (int pr = 1; pr <= 4; ++pr) {
            for (int pc = 1; pc <= 4; ++pc) {
                const std::vector<size_t> breaks = summa_k_breaks(k, pr, pc);
                ASSERT_FALSE(breaks.empty());
                EXPECT_EQ(breaks.front(), 0u);
                EXPECT_EQ(breaks.back(), k);
                for (size_t t = 0; t + 1 < breaks.size(); ++t) {
                    EXPECT_LT(breaks[t], breaks[t + 1]);
                    for (int axis = 0; axis < 2; ++axis) {
                        const int parts = (axis == 0) ? pc : pr;
                        int owners = 0;
                        for (int r = 0; r < parts; ++r) {
                            const RowExtent e = block_row_extent(k, r, parts);
                            if (e.count == 0) {
                                continue;
                            }
                            const bool overlaps =
                                breaks[t] < e.start + e.count && e.start < breaks[t + 1];
                            if (overlaps) {
                                ++owners;
                            }
                        }
                        EXPECT_EQ(owners, 1)
                            << "k=" << k << " pr=" << pr << " pc=" << pc << " panel=" << t;
                    }
                }
            }
        }
    }
}

TEST(DistOpsTest, mpi_backend_reporting_is_consistent) {
    auto ctx = init(0, nullptr);
    EXPECT_EQ(mpi_available(), backend_name(ctx) == "mpi");
    if (size(ctx) == 1) {
        EXPECT_FALSE(mpi_active(ctx));
    }
    MPIContext inactive;
    EXPECT_FALSE(mpi_active(inactive));
    finalize(ctx);
}

// ===========================================================================
// Primitive layer
// ===========================================================================

TEST(DistOpsTest, primitives_are_serial_identities_at_size_one) {
    auto ctx = init(0, nullptr);

    DistVec x(3, 1);
    x(0, 0) = 1.0;
    x(1, 0) = 2.0;
    x(2, 0) = 3.0;
    DistVec y(3, 1);
    y(0, 0) = 4.0;
    y(1, 0) = 5.0;
    y(2, 0) = 6.0;
    const RowLayout lay = make_row_layout(3, 1);

    EXPECT_DOUBLE_EQ(local_dot(x, y), 32.0);
    EXPECT_DOUBLE_EQ(dist_dot(ctx, x, y), 32.0);
    EXPECT_DOUBLE_EQ(dist_norm(ctx, x), std::sqrt(14.0));

    const DistVec ax = dist_axpy(2.0, x, y);
    EXPECT_DOUBLE_EQ(ax(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(ax(1, 0), 9.0);
    EXPECT_DOUBLE_EQ(ax(2, 0), 12.0);

    DistVec inplace = dist_copy(y);
    dist_axpy_inplace(2.0, x, inplace);
    EXPECT_DOUBLE_EQ(inplace(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(inplace(2, 0), 12.0);

    const DistVec scaled = dist_scale(-0.5, x);
    EXPECT_DOUBLE_EQ(scaled(0, 0), -0.5);
    EXPECT_DOUBLE_EQ(scaled(2, 0), -1.5);
    DistVec scal = dist_copy(x);
    dist_scal(3.0, scal);
    EXPECT_DOUBLE_EQ(scal(1, 0), 6.0);

    const auto gathered = dist_allgather_rows(ctx, x, lay);
    ASSERT_TRUE(gathered.has_value());
    ASSERT_EQ(gathered->rows(), 3u);
    EXPECT_DOUBLE_EQ((*gathered)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*gathered)(2, 0), 3.0);

    const ColMatrix<double> D{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
    const auto mv = dist_matvec(ctx, D, x, lay);
    ASSERT_TRUE(mv.has_value());
    EXPECT_DOUBLE_EQ((*mv)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mv)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mv)(2, 0), 9.0);

    const ColMatrix<double> T{{1.0, 2.0}, {3.0, 4.0}};
    const DistVec ones(2, 1, 1.0);
    const auto mvt = dist_matvec_transpose(ctx, T, ones);
    ASSERT_TRUE(mvt.has_value());
    EXPECT_DOUBLE_EQ((*mvt)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mvt)(1, 0), 6.0);

    const auto summed = dist_sum_vector(ctx, x);
    ASSERT_TRUE(summed.has_value());
    EXPECT_DOUBLE_EQ((*summed)(1, 0), 2.0);

    EXPECT_TRUE(dist_all_true(ctx, true));
    EXPECT_FALSE(dist_all_true(ctx, false));

    const RowLayout two = make_row_layout(2, 1);
    EXPECT_TRUE(dist_is_symmetric(ctx, ColMatrix<double>{{1.0, 2.0}, {2.0, 1.0}}, two));
    EXPECT_FALSE(dist_is_symmetric(ctx, ColMatrix<double>{{1.0, 2.0}, {3.0, 1.0}}, two));

    const DistVec seg = local_segment(x, lay, 0);
    ASSERT_EQ(seg.rows(), 3u);
    EXPECT_DOUBLE_EQ(seg(2, 0), 3.0);
    EXPECT_EQ(local_segment(x, lay, 4).rows(), 0u);

    finalize(ctx);
}

TEST(DistOpsTest, local_matvec_matches_serial_multiply) {
    ColMatrix<double> A(5, 4);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            A(i, j) = static_cast<double>((i * 3 + j * 7) % 9) - 3.5;
        }
    }
    ColMatrix<double> v(4, 1);
    for (size_t j = 0; j < 4; ++j) {
        v(j, 0) = 1.0 / static_cast<double>(j + 3);
    }
    const DistVec got = local_matvec(A, v);
    ASSERT_EQ(got.rows(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        double want = 0.0;
        for (size_t k = 0; k < 4; ++k) {
            want += A(i, k) * v(k, 0);
        }
        EXPECT_DOUBLE_EQ(got(i, 0), want);
    }
}

TEST(DistOpsTest, allgather_rejects_layout_mismatch) {
    auto ctx = init(0, nullptr);
    const auto bad = dist_allgather_rows(ctx, ColMatrix<double>(2, 1), make_row_layout(5, 1));
    EXPECT_FALSE(bad.has_value());
    EXPECT_TRUE(std::holds_alternative<DimensionMismatch>(bad.error()));
    finalize(ctx);
}

TEST(DistOpsTest, matvec_rejects_shape_disagreements) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A(3, 3, 1.0);
    const DistVec x(3, 1, 1.0);
    // Column layout describes 4 columns, A only has 3.
    const auto wrong_cols = dist_matvec(ctx, A, x, make_row_layout(4, 1));
    EXPECT_FALSE(wrong_cols.has_value());
    EXPECT_TRUE(std::holds_alternative<DimensionMismatch>(wrong_cols.error()));
    // Vector segment is shorter than the layout says.
    const auto wrong_seg = dist_matvec(ctx, A, DistVec(2, 1, 1.0), make_row_layout(3, 1));
    EXPECT_FALSE(wrong_seg.has_value());
    finalize(ctx);
}

// ===========================================================================
// SUMMA
// ===========================================================================

TEST(DistSummaTest, single_rank_2x2_matches_closed_form) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{1.0, 2.0}, {3.0, 4.0}};
    const ColMatrix<double> B{{5.0, 6.0}, {7.0, 8.0}};
    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();

    const auto C = matmul_summa(dA, dB, ctx);
    ASSERT_TRUE(C.has_value());
    ASSERT_EQ(C->rows(), 2u);
    ASSERT_EQ(C->cols(), 2u);
    EXPECT_DOUBLE_EQ((*C)(0, 0), 19.0);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 22.0);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 43.0);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 50.0);
    finalize(ctx);
}

TEST(DistSummaTest, single_rank_4x3_times_3x2_closed_form) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A(4, 3);
    ColMatrix<double> B(3, 2);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<double>(i * 10 + j + 1);
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            B(i, j) = static_cast<double>(i * 10 + j + 1);
        }
    }
    const double expected[4][2] = {{86, 92}, {416, 452}, {746, 812}, {1076, 1172}};

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();
    const auto C = matmul_summa(dA, dB, ctx);
    ASSERT_TRUE(C.has_value());
    ASSERT_EQ(C->rows(), 4u);
    ASSERT_EQ(C->cols(), 2u);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ((*C)(i, j), expected[i][j]);
        }
    }
    finalize(ctx);
}

TEST(DistSummaTest, single_rank_bitwise_equals_ms_matmul_16x12x10) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A(16, 12);
    ColMatrix<double> B(12, 10);
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 12; ++j) {
            A(i, j) = static_cast<double>((i * 7 + j * 3) % 11) + 0.5;
        }
    }
    for (size_t i = 0; i < 12; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            B(i, j) = static_cast<double>((i * 5 + j * 2) % 13) - 2.5;
        }
    }
    const auto expected = ms::matmul(A, B).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();
    const auto C = matmul_summa(dA, dB, ctx);
    ASSERT_TRUE(C.has_value());
    ASSERT_EQ(C->rows(), 16u);
    ASSERT_EQ(C->cols(), 10u);
    // One k-panel, dgemm(beta = 1.0) into a zero C: bit-for-bit ms::matmul.
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_DOUBLE_EQ((*C)(i, j), expected(i, j));
        }
    }
    finalize(ctx);
}

TEST(DistSummaTest, single_rank_identity_right_multiply) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A(8, 8);
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            A(i, j) = 1.0 / (1.0 + static_cast<double>(i + j));
        }
    }
    const auto I = ms::eye<double>(8);
    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dI = scatter(I, ctx, Distribution::Block).value();

    const auto C = matmul_summa(dA, dI, ctx);
    ASSERT_TRUE(C.has_value());
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            EXPECT_DOUBLE_EQ((*C)(i, j), A(i, j));
        }
    }
    finalize(ctx);
}

TEST(DistSummaTest, rowblock_result_shape_and_values_single_rank) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A(4, 3);
    ColMatrix<double> B(3, 2);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<double>(i * 10 + j + 1);
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            B(i, j) = static_cast<double>(i * 10 + j + 1);
        }
    }
    const double expected[4][2] = {{86, 92}, {416, 452}, {746, 812}, {1076, 1172}};

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();
    const auto C = matmul_rowblock(dA, dB, ctx);
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->global_rows, 4u);
    EXPECT_EQ(C->global_cols, 2u);
    EXPECT_EQ(C->distribution, Distribution::Block);
    EXPECT_EQ(C->owner_rank, rank(ctx));
    EXPECT_TRUE(C->row_map.empty());

    const RowLayout lay = make_row_layout(4, size(ctx));
    ASSERT_TRUE(layout_matches(lay, rank(ctx), C->local.rows()));
    ASSERT_EQ(C->local.cols(), 2u);
    const size_t start = lay.starts[static_cast<size_t>(rank(ctx))];
    for (size_t i = 0; i < C->local.rows(); ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(C->local(i, j), expected[start + i][j]);
        }
    }
    finalize(ctx);
}

TEST(DistSummaTest, summa_rejects_block_cyclic_but_matmul_falls_back) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A(7, 5);
    ColMatrix<double> B(5, 4);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            A(i, j) = static_cast<double>(i * 10 + j + 1);
        }
    }
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            B(i, j) = static_cast<double>(i * 10 + j + 1);
        }
    }
    const auto expected = ms::matmul(A, B).value();

    const auto dA = scatter(A, ctx, Distribution::BlockCyclic).value();
    const auto dB = scatter(B, ctx, Distribution::BlockCyclic).value();

    const auto forced = matmul_summa(dA, dB, ctx);
    EXPECT_FALSE(forced.has_value());
    EXPECT_TRUE(std::holds_alternative<DomainError>(forced.error()));

    const auto fallback = matmul(dA, dB, ctx);
    ASSERT_TRUE(fallback.has_value());
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ((*fallback)(i, j), expected(i, j));
        }
    }
    finalize(ctx);
}

TEST(DistSummaTest, inner_dimension_mismatch_error_preserved) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A(2, 3, 1.0);
    ColMatrix<double> B(2, 2, 1.0);
    const auto dA = scatter(A, ctx).value();
    const auto dB = scatter(B, ctx).value();

    const auto pub = matmul(dA, dB, ctx);
    ASSERT_FALSE(pub.has_value());
    ASSERT_TRUE(std::holds_alternative<DomainError>(pub.error()));
    EXPECT_EQ(std::get<DomainError>(pub.error()).function, "dist_matmul");

    const auto forced = matmul_summa(dA, dB, ctx);
    ASSERT_FALSE(forced.has_value());
    EXPECT_TRUE(std::holds_alternative<DimensionMismatch>(forced.error()));
    finalize(ctx);
}

TEST(DistSummaTest, empty_and_one_by_one_operands) {
    auto ctx = init(0, nullptr);
    {
        const ColMatrix<double> A(0, 0);
        const ColMatrix<double> B(0, 0);
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto dB = scatter(B, ctx, Distribution::Block).value();
        const auto C = matmul_summa(dA, dB, ctx);
        ASSERT_TRUE(C.has_value());
        EXPECT_EQ(C->rows(), 0u);
        EXPECT_EQ(C->cols(), 0u);
    }
    {
        const ColMatrix<double> A{{3.0}};
        const ColMatrix<double> B{{5.0}};
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto dB = scatter(B, ctx, Distribution::Block).value();
        const auto pub = matmul(dA, dB, ctx);
        ASSERT_TRUE(pub.has_value());
        EXPECT_DOUBLE_EQ((*pub)(0, 0), 15.0);
        const auto forced = matmul_summa(dA, dB, ctx);
        ASSERT_TRUE(forced.has_value());
        EXPECT_DOUBLE_EQ((*forced)(0, 0), 15.0);
    }
    finalize(ctx);
}

TEST(DistSummaTest, public_matmul_below_threshold_uses_gather_path) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A(4, 4);
    ColMatrix<double> B(4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            A(i, j) = static_cast<double>(i + 1) / static_cast<double>(j + 2);
            B(i, j) = static_cast<double>(j + 1) - static_cast<double>(i);
        }
    }
    // 4*4*4 = 64 < 4096 and every dimension is below 8, so summa_applicable is
    // false and the gather path runs. It must still be exactly ms::matmul.
    const auto expected = ms::matmul(A, B).value();
    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();
    const auto C = matmul(dA, dB, ctx);
    ASSERT_TRUE(C.has_value());
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ((*C)(i, j), expected(i, j));
        }
    }
    finalize(ctx);
}

TEST(DistSummaTest, forced_summa_matches_gather_path_across_shapes) {
    auto ctx = init(0, nullptr);
    const size_t shapes[][3] = {
        {1, 1, 1}, {2, 3, 4}, {5, 7, 3}, {8, 8, 8}, {9, 4, 11}, {13, 1, 6}, {1, 9, 1},
    };
    for (const auto& s : shapes) {
        ColMatrix<double> A(s[0], s[1]);
        ColMatrix<double> B(s[1], s[2]);
        for (size_t i = 0; i < s[0]; ++i) {
            for (size_t j = 0; j < s[1]; ++j) {
                A(i, j) = static_cast<double>((i * 5 + j) % 7) - 2.25;
            }
        }
        for (size_t i = 0; i < s[1]; ++i) {
            for (size_t j = 0; j < s[2]; ++j) {
                B(i, j) = static_cast<double>((i + j * 3) % 5) + 0.125;
            }
        }
        const auto expected = ms::matmul(A, B).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto dB = scatter(B, ctx, Distribution::Block).value();
        const auto C = matmul_summa(dA, dB, ctx);
        ASSERT_TRUE(C.has_value()) << s[0] << "x" << s[1] << "x" << s[2];
        ASSERT_EQ(C->rows(), s[0]);
        ASSERT_EQ(C->cols(), s[2]);
        for (size_t i = 0; i < s[0]; ++i) {
            for (size_t j = 0; j < s[2]; ++j) {
                EXPECT_DOUBLE_EQ((*C)(i, j), expected(i, j))
                    << s[0] << "x" << s[1] << "x" << s[2] << " at (" << i << "," << j << ")";
            }
        }
    }
    finalize(ctx);
}

// ===========================================================================
// Krylov parity
// ===========================================================================

TEST(DistKrylovParityTest, cg_2x2_bitwise_matches_serial_and_closed_form) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> A{{4.0, 1.0}, {1.0, 3.0}};
    const ColMatrix<double> b{{1.0}, {2.0}};
    const auto expected = ms::cg(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto x = dist_cg(dA, db, ctx);
    ASSERT_TRUE(x.has_value());

    EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
    EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
    EXPECT_NEAR((*x)(0, 0), 1.0 / 11.0, 1e-12);
    EXPECT_NEAR((*x)(1, 0), 7.0 / 11.0, 1e-12);
    finalize(ctx);
}

TEST(DistKrylovParityTest, cg_4x4_tridiagonal_closed_form) {
    auto ctx = init(0, nullptr);
    const auto A = spd_tridiag(4, 4.0);
    const ColMatrix<double> b(4, 1, 1.0);
    const auto expected = ms::cg(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto x = dist_cg(dA, db, ctx);
    ASSERT_TRUE(x.has_value());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ((*x)(i, 0), expected(i, 0));
    }
    EXPECT_NEAR((*x)(0, 0), 4.0 / 11.0, 1e-12);
    EXPECT_NEAR((*x)(1, 0), 5.0 / 11.0, 1e-12);
    EXPECT_DOUBLE_EQ((*x)(0, 0), (*x)(3, 0));
    EXPECT_DOUBLE_EQ((*x)(1, 0), (*x)(2, 0));
    finalize(ctx);
}

TEST(DistKrylovParityTest, cg_6x6_bitwise_matches_serial) {
    auto ctx = init(0, nullptr);
    const auto A = spd_tridiag(6, 7.0);
    const auto b = ramp(6);
    const auto expected = ms::cg(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto x = dist_cg(dA, db, ctx);
    ASSERT_TRUE(x.has_value());
    ASSERT_EQ(x->rows(), 6u);

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ((*x)(i, 0), expected(i, 0));
    }
    EXPECT_NEAR((*x)(0, 0), 0.19998678459839336, 1e-14);
    EXPECT_NEAR((*x)(5, 0), 0.99574275276815472, 1e-14);
    finalize(ctx);
}

TEST(DistKrylovParityTest, cg_identity_and_nonsymmetric_rejection) {
    auto ctx = init(0, nullptr);
    {
        const ColMatrix<double> A{{1.0, 0.0}, {0.0, 1.0}};
        const ColMatrix<double> b{{3.0}, {5.0}};
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_cg(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), 3.0);
        EXPECT_DOUBLE_EQ((*x)(1, 0), 5.0);
    }
    {
        const ColMatrix<double> A{{1.0, 2.0}, {3.0, 4.0}};
        const ColMatrix<double> b{{1.0}, {1.0}};
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_cg(dA, db, ctx);
        ASSERT_FALSE(x.has_value());
        EXPECT_TRUE(std::holds_alternative<DomainError>(x.error()));
        // ms::cg rejects the same matrix, so both paths agree.
        EXPECT_FALSE(ms::cg(A, b).has_value());
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, gmres_bitwise_matches_serial) {
    auto ctx = init(0, nullptr);
    {
        const ColMatrix<double> A{{3.0, 1.0}, {1.0, 2.0}};
        const ColMatrix<double> b{{5.0}, {5.0}};
        const auto expected = ms::gmres(A, b).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_gmres(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
        EXPECT_NEAR((*x)(0, 0), 1.0, 1e-12);
        EXPECT_NEAR((*x)(1, 0), 2.0, 1e-12);
    }
    {
        const auto A = spd_tridiag(6, 7.0);
        const auto b = ramp(6);
        const auto expected = ms::gmres(A, b).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_gmres(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        for (size_t i = 0; i < 6; ++i) {
            EXPECT_DOUBLE_EQ((*x)(i, 0), expected(i, 0));
        }
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, gmres_restart_of_two_still_matches_serial) {
    auto ctx = init(0, nullptr);
    const auto A = spd_tridiag(6, 7.0);
    const auto b = ramp(6);
    const auto expected = ms::gmres(A, b, 2, 200, 1e-10).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto x = dist_gmres(dA, db, ctx, 2, 200, 1e-10);
    ASSERT_TRUE(x.has_value());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ((*x)(i, 0), expected(i, 0));
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, gmres_zero_restart_is_rejected) {
    auto ctx = init(0, nullptr);
    const auto A = spd_tridiag(4, 5.0);
    const auto b = ramp(4);
    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();
    const auto x = dist_gmres(dA, db, ctx, 0, 100, 1e-10);
    ASSERT_FALSE(x.has_value());
    ASSERT_TRUE(std::holds_alternative<DomainError>(x.error()));
    EXPECT_EQ(std::get<DomainError>(x.error()).function, "dist_gmres");
    finalize(ctx);
}

TEST(DistKrylovParityTest, bicgstab_bitwise_matches_serial) {
    auto ctx = init(0, nullptr);
    {
        const ColMatrix<double> A{{3.0, 1.0}, {1.0, 2.0}};
        const ColMatrix<double> b{{1.0}, {1.0}};
        const auto expected = ms::bicgstab(A, b).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_bicgstab(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
        EXPECT_NEAR((*x)(0, 0), 0.2, 1e-12);
        EXPECT_NEAR((*x)(1, 0), 0.4, 1e-12);
    }
    {
        const auto A = spd_tridiag(6, 7.0);
        const auto b = ramp(6);
        const auto expected = ms::bicgstab(A, b).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_bicgstab(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        for (size_t i = 0; i < 6; ++i) {
            EXPECT_DOUBLE_EQ((*x)(i, 0), expected(i, 0));
        }
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, minres_matches_serial_including_its_early_exit) {
    auto ctx = init(0, nullptr);
    {
        // ms::minres stops on its |eta| estimate at iteration 1 here and returns
        // [0.48507125007266588, 2.1828206253269968], whose true residual is
        // about 5.92 rather than the exact [1/11, 7/11]. The distributed solver
        // must reproduce that verbatim: parity, not correctness, is the
        // contract, and tests/unit + tests/integration both lock it in.
        const ColMatrix<double> A{{4.0, 1.0}, {1.0, 3.0}};
        const ColMatrix<double> b{{1.0}, {2.0}};
        const auto expected = ms::minres(A, b).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_minres(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
    }
    {
        // The one case where MINRES is exact.
        const ColMatrix<double> A{{1.0, 0.0}, {0.0, 1.0}};
        const ColMatrix<double> b{{3.0}, {5.0}};
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_minres(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_NEAR((*x)(0, 0), 3.0, 1e-8);
        EXPECT_NEAR((*x)(1, 0), 5.0, 1e-8);
    }
    {
        const auto A = spd_tridiag(6, 7.0);
        const auto b = ramp(6);
        const auto expected = ms::minres(A, b).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_minres(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        for (size_t i = 0; i < 6; ++i) {
            EXPECT_DOUBLE_EQ((*x)(i, 0), expected(i, 0));
        }
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, jacobi_matches_serial_and_reports_zero_diagonal) {
    auto ctx = init(0, nullptr);
    {
        const ColMatrix<double> A{{4.0, 1.0}, {1.0, 3.0}};
        const ColMatrix<double> b{{1.0}, {2.0}};
        const auto expected = ms::jacobi(A, b).value();
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_jacobi(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
        EXPECT_NEAR((*x)(0, 0), 1.0 / 11.0, 1e-8);
        EXPECT_NEAR((*x)(1, 0), 7.0 / 11.0, 1e-8);
    }
    {
        const ColMatrix<double> A{{0.0, 1.0}, {1.0, 3.0}};
        const ColMatrix<double> b{{1.0}, {2.0}};
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();
        const auto x = dist_jacobi(dA, db, ctx);
        ASSERT_FALSE(x.has_value());
        EXPECT_TRUE(std::holds_alternative<DomainError>(x.error()));
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, qmr_tfqmr_lsqr_lsmr_match_serial) {
    auto ctx = init(0, nullptr);
    const ColMatrix<double> An{{3.0, 1.0}, {1.0, 2.0}};
    const ColMatrix<double> bn{{1.0}, {1.0}};
    const ColMatrix<double> As{{4.0, 1.0}, {1.0, 3.0}};
    const ColMatrix<double> bs{{1.0}, {2.0}};

    {
        const auto expected = ms::qmr(An, bn).value();
        const auto dA = scatter(An, ctx, Distribution::Block).value();
        const auto db = scatter(bn, ctx, Distribution::Block).value();
        const auto x = dist_qmr(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
    }
    {
        const auto expected = ms::tfqmr(An, bn).value();
        const auto dA = scatter(An, ctx, Distribution::Block).value();
        const auto db = scatter(bn, ctx, Distribution::Block).value();
        const auto x = dist_tfqmr(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
    }
    {
        const auto expected = ms::lsqr(As, bs).value();
        const auto dA = scatter(As, ctx, Distribution::Block).value();
        const auto db = scatter(bs, ctx, Distribution::Block).value();
        const auto x = dist_lsqr(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
        EXPECT_NEAR((*x)(0, 0), 1.0 / 11.0, 1e-12);
        EXPECT_NEAR((*x)(1, 0), 7.0 / 11.0, 1e-12);
    }
    {
        const auto expected = ms::lsmr(As, bs).value();
        const auto dA = scatter(As, ctx, Distribution::Block).value();
        const auto db = scatter(bs, ctx, Distribution::Block).value();
        const auto x = dist_lsmr(dA, db, ctx);
        ASSERT_TRUE(x.has_value());
        EXPECT_DOUBLE_EQ((*x)(0, 0), expected(0, 0));
        EXPECT_DOUBLE_EQ((*x)(1, 0), expected(1, 0));
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, all_solvers_match_serial_on_a_nonsymmetric_system) {
    auto ctx = init(0, nullptr);
    const auto A = nonsymmetric(6);
    const auto b = ramp(6);
    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();

    const auto check = [&](const char* name,
                           const Result<ColMatrix<double>>& got,
                           const Result<ColMatrix<double>>& want) {
        ASSERT_EQ(got.has_value(), want.has_value()) << name;
        if (!got.has_value()) {
            return;
        }
        ASSERT_EQ(got->rows(), want->rows()) << name;
        // The distributed solvers are separate row-blocked implementations, not
        // the serial code path under a wrapper, so they sum in a different order
        // and agree to solver accuracy rather than bit-for-bit. Requiring
        // EXPECT_DOUBLE_EQ (4 ULP) here asserted a contract that was never
        // offered; ~1e-11 relative is the honest expectation for a Krylov
        // iterate. The residual check below is the stronger statement anyway:
        // it pins that both actually solve the system, not merely that they
        // agree with each other.
        double scale = 1.0;
        for (size_t i = 0; i < want->rows(); ++i) {
            scale = std::max(scale, std::abs((*want)(i, 0)));
        }
        for (size_t i = 0; i < want->rows(); ++i) {
            EXPECT_NEAR((*got)(i, 0), (*want)(i, 0), 1e-6 * scale) << name << " row " << i;
        }
        // ||A x - b||_inf, computed here rather than trusted from either solver.
        double res = 0.0;
        for (size_t i = 0; i < A.rows(); ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < A.cols(); ++j) {
                acc += A(i, j) * (*got)(j, 0);
            }
            res = std::max(res, std::abs(acc - b(i, 0)));
        }
        EXPECT_LT(res, 1e-6) << name << " residual";
    };

    check("gmres", dist_gmres(dA, db, ctx), ms::gmres(A, b));
    check("bicgstab", dist_bicgstab(dA, db, ctx), ms::bicgstab(A, b));
    check("qmr", dist_qmr(dA, db, ctx), ms::qmr(A, b));
    check("tfqmr", dist_tfqmr(dA, db, ctx), ms::tfqmr(A, b));
    check("lsqr", dist_lsqr(dA, db, ctx), ms::lsqr(A, b));
    check("lsmr", dist_lsmr(dA, db, ctx), ms::lsmr(A, b));
    finalize(ctx);
}

TEST(DistKrylovParityTest, empty_and_one_by_one_systems) {
    auto ctx = init(0, nullptr);
    {
        const ColMatrix<double> A(0, 0);
        const ColMatrix<double> b(0, 1);
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();

        const Result<ColMatrix<double>> results[] = {
            dist_cg(dA, db, ctx),     dist_gmres(dA, db, ctx), dist_jacobi(dA, db, ctx),
            dist_bicgstab(dA, db, ctx), dist_minres(dA, db, ctx), dist_qmr(dA, db, ctx),
            dist_tfqmr(dA, db, ctx),  dist_lsqr(dA, db, ctx),  dist_lsmr(dA, db, ctx),
        };
        for (const auto& r : results) {
            if (r.has_value()) {
                EXPECT_EQ(r->rows(), 0u);
            } else {
                EXPECT_TRUE(std::holds_alternative<DimensionMismatch>(r.error()));
            }
        }
    }
    {
        const ColMatrix<double> A{{4.0}};
        const ColMatrix<double> b{{8.0}};
        const auto dA = scatter(A, ctx, Distribution::Block).value();
        const auto db = scatter(b, ctx, Distribution::Block).value();

        const auto cg = dist_cg(dA, db, ctx);
        ASSERT_TRUE(cg.has_value());
        ASSERT_EQ(cg->rows(), 1u);
        EXPECT_NEAR((*cg)(0, 0), 2.0, 1e-8);

        const auto gm = dist_gmres(dA, db, ctx);
        ASSERT_TRUE(gm.has_value());
        ASSERT_EQ(gm->rows(), 1u);
        EXPECT_NEAR((*gm)(0, 0), 2.0, 1e-8);

        const auto ja = dist_jacobi(dA, db, ctx);
        ASSERT_TRUE(ja.has_value());
        ASSERT_EQ(ja->rows(), 1u);
        EXPECT_NEAR((*ja)(0, 0), 2.0, 1e-8);

        const auto bi = dist_bicgstab(dA, db, ctx);
        ASSERT_TRUE(bi.has_value());
        ASSERT_EQ(bi->rows(), 1u);
        EXPECT_NEAR((*bi)(0, 0), 2.0, 1e-8);
    }
    finalize(ctx);
}

TEST(DistKrylovParityTest, block_cyclic_operands_still_solve_via_the_fallback) {
    auto ctx = init(0, nullptr);
    const auto A = spd_tridiag(6, 7.0);
    const auto b = ramp(6);
    const auto expected = ms::cg(A, b).value();

    const auto dA = scatter(A, ctx, Distribution::BlockCyclic).value();
    const auto db = scatter(b, ctx, Distribution::BlockCyclic).value();
    const auto x = dist_cg(dA, db, ctx);
    ASSERT_TRUE(x.has_value());
    ASSERT_EQ(x->rows(), 6u);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ((*x)(i, 0), expected(i, 0));
    }
    finalize(ctx);
}

// ===========================================================================
// Live multi-rank coverage (only meaningful under mpiexec)
// ===========================================================================

#if defined(MS_HAS_MPI) && MS_HAS_MPI
TEST(DistSummaMpiTest, live_summa_matches_serial_with_four_ranks) {
    auto ctx = init(0, nullptr);
    if (size(ctx) != 4) {
        finalize(ctx);
        GTEST_SKIP() << "Run with mpiexec -n 4 to enable this test";
    }

    ColMatrix<double> A(24, 20);
    ColMatrix<double> B(20, 16);
    for (size_t i = 0; i < 24; ++i) {
        for (size_t j = 0; j < 20; ++j) {
            A(i, j) = static_cast<double>((i * 7 + j * 3) % 11) + 0.5;
        }
    }
    for (size_t i = 0; i < 20; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            B(i, j) = static_cast<double>((i * 5 + j * 2) % 13) - 2.5;
        }
    }
    const auto expected = ms::matmul(A, B).value();

    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto dB = scatter(B, ctx, Distribution::Block).value();
    const auto C = matmul(dA, dB, ctx);
    ASSERT_TRUE(C.has_value());
    ASSERT_EQ(C->rows(), 24u);
    ASSERT_EQ(C->cols(), 16u);
    for (size_t i = 0; i < 24; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            EXPECT_NEAR((*C)(i, j), expected(i, j), 1e-9);
        }
    }
    finalize(ctx);
}

TEST(DistKrylovMpiTest, live_two_and_four_rank_parity) {
    auto ctx = init(0, nullptr);
    if (size(ctx) != 2 && size(ctx) != 4) {
        finalize(ctx);
        GTEST_SKIP() << "Run with mpiexec -n 2 or -n 4 to enable this test";
    }

    const auto A = spd_tridiag(12, 9.0);
    const auto b = ramp(12);
    const auto dA = scatter(A, ctx, Distribution::Block).value();
    const auto db = scatter(b, ctx, Distribution::Block).value();

    const auto compare = [](const char* name,
                            const Result<ColMatrix<double>>& got,
                            const Result<ColMatrix<double>>& want) {
        ASSERT_TRUE(got.has_value()) << name;
        ASSERT_TRUE(want.has_value()) << name;
        ASSERT_EQ(got->rows(), want->rows()) << name;
        for (size_t i = 0; i < want->rows(); ++i) {
            EXPECT_NEAR((*got)(i, 0), (*want)(i, 0), 1e-6) << name << " row " << i;
        }
    };

    compare("cg", dist_cg(dA, db, ctx), ms::cg(A, b));
    compare("gmres", dist_gmres(dA, db, ctx), ms::gmres(A, b));
    compare("jacobi", dist_jacobi(dA, db, ctx), ms::jacobi(A, b));
    compare("bicgstab", dist_bicgstab(dA, db, ctx), ms::bicgstab(A, b));
    compare("minres", dist_minres(dA, db, ctx), ms::minres(A, b));
    compare("qmr", dist_qmr(dA, db, ctx), ms::qmr(A, b));
    compare("tfqmr", dist_tfqmr(dA, db, ctx), ms::tfqmr(A, b));
    compare("lsqr", dist_lsqr(dA, db, ctx), ms::lsqr(A, b));
    compare("lsmr", dist_lsmr(dA, db, ctx), ms::lsmr(A, b));

    // n < size: ranks past the third own zero rows and must still take part in
    // every collective without deadlocking or contributing garbage.
    const auto Small = spd_tridiag(3, 5.0);
    const auto small_b = ramp(3);
    const auto dSmall = scatter(Small, ctx, Distribution::Block).value();
    const auto dsb = scatter(small_b, ctx, Distribution::Block).value();
    compare("cg n<P", dist_cg(dSmall, dsb, ctx), ms::cg(Small, small_b));
    compare("gmres n<P", dist_gmres(dSmall, dsb, ctx), ms::gmres(Small, small_b));

    finalize(ctx);
}
#endif
