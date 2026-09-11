// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include "ms/core/matrix.hpp"
#include "ms/distributed/block.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/error/error_types.hpp"

#include <cstddef>
#include <vector>

namespace ms::distributed {

/// Column-vector segment owned by one rank. Always `local_rows x 1`,
/// column-major, so `.data()` is the contiguous segment MPI sends.
using DistVec = Matrix<double, StorageOrder::ColMajor>;

// ---------------------------------------------------------------------------
// Layout descriptors
// ---------------------------------------------------------------------------

/// @brief Contiguous block-row partition of `global_rows` over `nprocs` ranks.
///
/// Rank r owns the global rows [starts[r], starts[r] + counts[r]). The split is
/// exactly the one `block_row_extent()` produces: the first
/// `global_rows % nprocs` ranks take one extra row each. `starts` and `counts`
/// always hold exactly `nprocs` entries, including the zero counts that appear
/// when `global_rows < nprocs`.
struct RowLayout {
    std::size_t global_rows = 0;
    int nprocs = 1;
    std::vector<std::size_t> starts;
    std::vector<std::size_t> counts;
};

/// @brief Build the block-row partition of `global_rows` over `nprocs` ranks.
/// @note Degenerate input: `nprocs <= 0` is clamped to a single rank, and
///       `global_rows == 0` yields all-zero counts. Never fails.
/// @note O(nprocs) time and space.
RowLayout make_row_layout(std::size_t global_rows, int nprocs);

/// @brief True when `local_rows` is the row count this layout assigns to
///        `my_rank`; a rank outside [0, nprocs) always reports false.
/// @note O(1).
bool layout_matches(const RowLayout& layout, int my_rank, std::size_t local_rows);

/// @brief 2D process grid for SUMMA, in row-major rank order.
///
/// `rows * cols == nprocs` always holds and the coordinates of a rank are
/// `my_row = rank / cols`, `my_col = rank % cols` — the mapping
/// `MPI_Cart_create` produces for a two-dimensional grid with `reorder = 0`.
struct ProcessGrid {
    int nprocs = 1;
    int rank = 0;
    int rows = 1;
    int cols = 1;
    int my_row = 0;
    int my_col = 0;
};

/// @brief Factor `nprocs` into the most square 2D grid: `rows` is the largest
///        divisor d of `nprocs` with d*d <= nprocs, and `cols = nprocs / rows`.
/// @note Pure integer arithmetic (no sqrt), so the factorisation is exact for
///       perfect squares: 36 -> 6x6, 1000 -> 25x40, 128 -> 8x16.
/// @note Degenerate input: `nprocs <= 0` yields the 1x1 grid, and a `rank`
///       outside [0, nprocs) is clamped to 0. O(sqrt(nprocs)). Never fails.
ProcessGrid make_process_grid(int nprocs, int rank);

/// @brief k-panel breakpoints for SUMMA: the sorted, de-duplicated union of
///        A's column-block starts (K split over `grid_cols`), B's row-block
///        starts (K split over `grid_rows`), 0, and K.
///
/// Each resulting panel [b[t], b[t+1]) lies entirely inside one A column block
/// and one B row block, so it has exactly one broadcast root per grid axis.
/// @return `{0}` — that is, no panels at all — when `k_global == 0`.
/// @note O(P log P) for P = grid_rows + grid_cols breakpoints.
std::vector<std::size_t> summa_k_breaks(std::size_t k_global, int grid_rows, int grid_cols);

/// @brief True when this library was built with MPI support AND `ctx` is an
///        initialised, multi-rank context — i.e. when a collective in this
///        module will actually send a message.
bool mpi_active(const MPIContext& ctx);

/// @brief True when the library was compiled with `MS_HAS_MPI == 1`.
/// @note Reported by the library, not by the caller's translation unit, so a
///       test binary that never sees the macro still gets the right answer.
bool mpi_available();

// ---------------------------------------------------------------------------
// Level 1 — purely local, no communication
// ---------------------------------------------------------------------------

/// @brief Sequential dot product of two local segments, k ascending, into one
///        `double` accumulator. Bit-identical to the `dotvec` helper the serial
///        Krylov kernels in src/linalg/iterative.cpp use.
/// @note Degenerate input: reads `min(x.rows(), y.rows())` entries, so a length
///       mismatch truncates instead of reading out of bounds. O(m).
double local_dot(const DistVec& x, const DistVec& y);

/// @brief Out-of-place `alpha * x + y`, element-wise, matching the serial
///        `axpy` in src/linalg/iterative.cpp exactly (including its loop order
///        and its result shape, which is taken from `x`).
/// @note Degenerate input: entries of `y` past its last row are treated as 0,
///       so a short `y` widens rather than reading out of bounds. O(m*n).
DistVec dist_axpy(double alpha, const DistVec& x, const DistVec& y);

/// @brief In-place `y := alpha * x + y`, same rounding as dist_axpy. Entries of
///        `x` past `y`'s last row are ignored. O(m).
void dist_axpy_inplace(double alpha, const DistVec& x, DistVec& y);

/// @brief Out-of-place `alpha * x`, matching the serial `scale_vec` helper.
///        O(m*n).
DistVec dist_scale(double alpha, const DistVec& x);

/// @brief In-place `x := alpha * x`. O(m*n).
void dist_scal(double alpha, DistVec& x);

/// @brief Deep element-wise copy, matching `ms::linalg_detail::copy`. O(m*n).
Matrix<double> dist_copy(const Matrix<double>& x);

/// @brief `y(i) = sum_k A(i,k) * x_full(k)`, k ascending, one accumulator per
///        row. Bit-identical to `ms::linalg_detail::multiply(A, x)` for a
///        single-column `x`, which is what the serial Krylov kernels call.
/// @note `x_full` must be the FULL global vector (`A.cols()` rows); entries
///       past its last row are treated as 0. O(m*n).
DistVec local_matvec(const Matrix<double>& A, const DistVec& x_full);

/// @brief Cut this rank's block-row segment out of a replicated global vector.
/// @return `layout.counts[my_rank] x 1`; an empty vector when `my_rank` is out
///         of range or the segment would run past the end of `full`. O(m).
DistVec local_segment(const DistVec& full, const RowLayout& layout, int my_rank);

// ---------------------------------------------------------------------------
// Level 1 — collective
// ---------------------------------------------------------------------------

/// @brief Global dot product: an MPI_Allreduce(MPI_SUM) over the per-rank
///        partial dots of the two block-row segments.
/// @note With one rank, or in a non-MPI build, this is exactly `local_dot`, so
///       single-rank results are bit-identical to the serial kernels.
///       One allreduce of 1 double; O(m) flops.
double dist_dot(const MPIContext& ctx, const DistVec& x, const DistVec& y);

/// @brief Global Euclidean norm, `sqrt(dist_dot(ctx, x, x))`.
/// @note One allreduce of 1 double; O(m) flops.
double dist_norm(const MPIContext& ctx, const DistVec& x);

/// @brief Global logical AND of a per-rank predicate, carried by an
///        allreduce-min over 1.0/0.0, so every rank reaches the same verdict
///        and no rank returns an error the others do not.
/// @note One allreduce of 1 double.
bool dist_all_true(const MPIContext& ctx, bool local_value);

// ---------------------------------------------------------------------------
// Level 2 — collective
// ---------------------------------------------------------------------------

/// @brief Concatenate every rank's row block into the full global matrix,
///        replicated on ALL ranks (MPI_Allgatherv over a column-major buffer).
/// @return A `layout.global_rows x local.cols()` matrix, identical on every
///         rank. With one rank, or in a non-MPI build, it is a plain copy of
///         `local`.
/// @return DimensionMismatch when `local.rows()` disagrees with the row count
///         the layout assigns to this rank; DomainError when the element count
///         would overflow MPI's `int` counts.
/// @note One allgather of `global_rows * cols` doubles.
Result<Matrix<double>> dist_allgather_rows(
    const MPIContext& ctx,
    const Matrix<double>& local,
    const RowLayout& layout);

/// @brief Distributed sparse-free matvec `y_local = A_local * x_global`.
///
/// `A_local` is this rank's `m_local x n` dense row block (all n columns) and
/// `x_local` is the matching segment of the length-n vector under
/// `col_layout`. The segment is all-gathered into the full length-n vector and
/// `local_matvec` is then applied to this rank's rows.
/// @return `m_local x 1`; DimensionMismatch when `A_local.cols()` differs from
///         `col_layout.global_rows` or `x_local` does not match the layout.
/// @note One allgather of n doubles; O(m_local * n) flops.
Result<DistVec> dist_matvec(
    const MPIContext& ctx,
    const Matrix<double>& A_local,
    const DistVec& x_local,
    const RowLayout& col_layout);

/// @brief Element-wise MPI_Allreduce(MPI_SUM) of a REPLICATED vector, so every
///        rank ends up holding the same sum.
/// @note Identity with one rank or in a non-MPI build. One allreduce of
///       `partial.rows()` doubles.
Result<DistVec> dist_sum_vector(const MPIContext& ctx, const DistVec& partial);

/// @brief Distributed transpose product `v = A^T u`, returned REPLICATED
///        (length `A_local.cols()`) on every rank.
///
/// Each rank forms `v_part(j) = sum_{i in its rows} A_local(i,j) * u_local(i)`
/// in the loop order `for j { for i { ... } }` — the order the serial LSQR and
/// QMR kernels use — and the partials are summed with one MPI_Allreduce.
/// @return `A_local.cols() x 1`; DimensionMismatch when `u_local` is shorter
///         than `A_local`'s row block.
/// @note One allreduce of n doubles; O(m_local * n) flops.
Result<DistVec> dist_matvec_transpose(
    const MPIContext& ctx,
    const Matrix<double>& A_local,
    const DistVec& u_local);

/// @brief Global symmetry test for a block-row-distributed square matrix.
///
/// Rank r holds rows R_r but needs A(j,i) for j outside R_r, so the
/// off-diagonal blocks are swapped with one MPI_Alltoallv: rank r sends peer p
/// the sub-block A[R_r][cols R_p] and receives A[R_p][cols R_r]. Each rank then
/// compares |A(i,j) - A(j,i)| against `tol` over its own rows and the verdicts
/// are combined with `dist_all_true`, so every rank agrees.
/// @return false when the matrix is not square or a local block does not match
///         the layout — the same answer `ms::linalg_detail::is_symmetric` gives
///         for a non-square matrix.
/// @note With one rank, or in a non-MPI build, this is exactly
///       `ms::linalg_detail::is_symmetric(A, tol)`, so `dist_cg` rejects
///       precisely the matrices `ms::cg` rejects. Setup only: never called
///       inside an iteration loop. One alltoall of n^2/P doubles and
///       O(n^2/P) comparisons.
bool dist_is_symmetric(
    const MPIContext& ctx,
    const Matrix<double>& A_local,
    const RowLayout& layout,
    double tol = 1e-10);

// ---------------------------------------------------------------------------
// Level 3 — SUMMA
// ---------------------------------------------------------------------------

/// @brief SUMMA (Scalable Universal Matrix Multiplication) on a 2D process
///        grid, taking and returning block-ROW slices.
///
/// `a_local` and `b_local` are this rank's block-row slices of the global
/// `a_rows x a_cols` A and `b_rows x b_cols` B. The routine
///   1. builds the most square grid (`make_process_grid`) and its Cartesian row
///      and column sub-communicators,
///   2. redistributes A and B from 1D block-row to 2D block form, one
///      MPI_Alltoallv each,
///   3. runs the broadcast-multiply-roll loop over `summa_k_breaks()`: for each
///      k-panel the owning process column broadcasts A's panel along the grid
///      row, the owning process row broadcasts B's panel along the grid column,
///      and every rank accumulates `C_2d += A_panel * B_panel` with one
///      `cpu::blas::dgemm(beta = 1.0)`,
///   4. redistributes C from 2D block back to 1D block-row, one MPI_Alltoallv.
///
/// @return this rank's block-ROW slice of C, `counts[my_rank] x b_cols`.
/// @return DimensionMismatch when `a_cols != b_rows`, or when a local block
///         does not match its block-row layout; DomainError when a message
///         would overflow MPI's `int` counts; DistributedError carrying the MPI
///         status when a collective reports failure.
/// @note With one rank (or a non-MPI build) the grid is 1x1, both
///       redistributions are identity copies, there is exactly one k-panel, no
///       message is sent, and the single `dgemm(beta = 1.0)` into a
///       zero-initialised C is BIT-IDENTICAL to `ms::matmul(A, B)` — `dgemm`
///       skips its beta scaling when beta == 1.0.
/// @note O(M*K*N / P) flops and O(K * (M/Pr + N/Pc)) communicated doubles.
Result<Matrix<double>> summa_matmul_rowblock(
    const MPIContext& ctx,
    const Matrix<double>& a_local, std::size_t a_rows, std::size_t a_cols,
    const Matrix<double>& b_local, std::size_t b_rows, std::size_t b_cols);

} // namespace ms::distributed
