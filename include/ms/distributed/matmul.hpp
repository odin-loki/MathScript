#pragma once

#include "ms/distributed/dist_matrix.hpp"

namespace ms::distributed {

/// @brief Distributed C = A * B for block-row-distributed operands.
///
/// Dispatch, in order: the inner dimensions must agree, else DomainError. The
/// SUMMA path is then taken only when ALL of the following hold, and the
/// gather-and-multiply fallback is used otherwise:
///   * the library was built with MPI support (`mpi_available()`),
///   * `size(ctx) > 1` — a 1x1 process grid has nothing to distribute,
///   * both operands use `Distribution::Block` and their local blocks match the
///     block-row layout for this rank,
///   * `min(M, K, N) >= 8` and `M*K*N >= 4096` — below that the collectives
///     cost more than simply gathering and multiplying.
/// @return the FULL M x N product, identical on every rank.
/// @return DomainError{"dist_matmul", ...} when A's columns and B's rows
///         disagree; otherwise whatever gather or the chosen kernel reports.
/// @note With one rank the fallback runs `ms::matmul` on the gathered operands,
///       so the single-rank result is bit-identical to the serial kernel.
///       SUMMA costs O(M*K*N/P) flops and O(K*(M/Pr + N/Pc)) communicated
///       doubles; the fallback costs O(M*K*N) flops on every rank.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> matmul(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx);

/// @brief Force the SUMMA path regardless of the size thresholds, returning the
///        full M x N product replicated on every rank.
///
/// Exists so that the broadcast-multiply-roll loop is exercised in the default
/// single-rank, no-MPI configuration, where the grid is 1x1, there is exactly
/// one k-panel, no message is sent, and the result is bit-identical to
/// `ms::matmul` on the gathered operands.
/// @return the FULL M x N product on every rank.
/// @return DimensionMismatch when `A.global_cols != B.global_rows`; DomainError
///         when either operand is not `Distribution::Block` or a local block
///         disagrees with its block-row layout.
/// @note O(M*K*N/P) flops; two MPI_Alltoallv redistributions plus one
///       broadcast pair per k-panel, then one MPI_Allgatherv of the result.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> matmul_summa(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx);

/// @brief SUMMA product left distributed: this rank's block-ROW slice of C,
///        skipping the final all-gather.
///
/// @return a DistMatrix whose `local` holds rows [start_r, start_r + count_r)
///         of C under the block-row layout for `size(ctx)` ranks, with
///         `distribution = Distribution::Block`, `owner_rank = rank(ctx)`,
///         `global_rows = A.global_rows`, `global_cols = B.global_cols` and an
///         empty `row_map`. Same error set as `matmul_summa`.
/// @note Use this instead of `matmul_summa` when the product feeds another
///       distributed operation: it saves the O(M*N) all-gather and the O(M*N)
///       replicated storage on every rank.
template<typename S, template<typename> class Alloc = std::allocator>
Result<DistMatrix<S, Alloc>> matmul_rowblock(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx);

} // namespace ms::distributed
