#pragma once

#include "ms/distributed/dist_matrix.hpp"

namespace ms::distributed {

// ---------------------------------------------------------------------------
// Row-distributed Krylov solvers.
//
// Every solver below takes A and b already scattered over the ranks and returns
// the FULL global solution, identical on every rank. When A and b are
// contiguous block rows that match this rank's share of the layout, the
// iteration runs distributed: A stays sharded, only the length-n vectors that
// genuinely need it are replicated, and the per-iteration traffic is one
// MPI_Allgatherv of n doubles per sparse-matrix-vector product plus one
// MPI_Allreduce of a single double per inner product. Anything else — a
// Distribution::BlockCyclic operand, a hand-built DistMatrix whose `local` does
// not match the layout, or an MPI-less build with more than one nominal rank —
// falls back to gathering both operands and running the serial kernel.
//
// Both paths reproduce the corresponding ms:: kernel exactly. With one rank the
// collectives are identities and the arithmetic is performed in the same order
// as src/linalg/iterative.cpp, so the results are BIT-IDENTICAL to the serial
// solver — including the places where the serial solver is imprecise. In
// particular dist_minres reproduces ms::minres's optimistic |eta| stopping
// estimate rather than correcting it.
//
// Shared error contract: DimensionMismatch when A is not square, when b's rows
// do not match A's, or when b is not a single column; otherwise whatever the
// underlying kernel reports.
// ---------------------------------------------------------------------------

/// @brief Distributed conjugate gradient for a symmetric positive definite A.
/// @return x on convergence (||b - A x|| < tol); DomainError{"dist_cg", ...}
///         when A is not symmetric to 1e-10 — the symmetry test is global, one
///         MPI_Alltoallv at setup, so every rank rejects the same matrices
///         ms::cg rejects; ConvergenceFail with the final residual otherwise.
/// @note Per iteration: one distributed SpMV and two global dot products.
///       O(n^2/P) flops and O(n) communicated doubles per iteration.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_cg(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed restarted GMRES(restart) for a general square A.
///
/// Only the Arnoldi basis vectors are distributed; the Hessenberg matrix, the
/// Givens rotations and the back substitution are computed redundantly on every
/// rank from allreduce results, which the MPI standard requires to be identical
/// everywhere. That is the memory win: a rank stores restart+1 segments of
/// length n/P instead of restart+1 full n-vectors.
/// @return x on convergence; DomainError{"dist_gmres", ...} when `restart` is
///         zero (the outer loop would never advance); ConvergenceFail with the
///         final residual when max_iter is exhausted.
/// @note Per Arnoldi step: one distributed SpMV and step+2 global dot products.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_gmres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t restart = 20,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed Jacobi iteration for a square A with a nonzero diagonal.
///
/// The diagonal entry of local row i is A_local(i, start_r + i), which is always
/// present because a block row carries every column, so the sweep itself needs
/// no communication beyond the two SpMVs.
/// @return x_new once ||b - A x_new|| < tol; DomainError{"dist_jacobi", ...}
///         when any rank finds a diagonal entry below 1e-30 — the verdict is
///         combined across ranks so they all return the same error;
///         ConvergenceFail with the final residual otherwise.
/// @note Per iteration: two distributed SpMVs and one global norm.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_jacobi(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed BiCGSTAB for a general square A.
/// @return the current iterate, always. Like ms::bicgstab this solver never
///         reports ConvergenceFail: it breaks out on rho breakdown or when the
///         residual falls below tol and hands back whatever x it has reached.
/// @note Per iteration: two distributed SpMVs and four global dot products.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_bicgstab(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed MINRES (Lanczos plus Givens) for a symmetric A.
/// @return x when the |eta| residual estimate drops below tol, or when the true
///         residual is below 10*tol after max_iter; ConvergenceFail otherwise.
/// @note This mirrors ms::minres operation for operation, INCLUDING its
///       optimistic |eta| stopping test, which declares convergence early on
///       most systems and returns an x whose true residual is much larger than
///       tol. Parity with the serial kernel is the contract here; correcting
///       the estimate belongs in src/linalg/iterative.cpp.
/// @note Per iteration: one distributed SpMV and two global reductions.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_minres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed QMR (Freund & Nachtigal, no look-ahead) for a general
///        square A.
///
/// The two-sided Lanczos recurrence needs one product with A and one with A^T
/// per iteration. A is never transposed into storage: the transpose product is
/// formed as a column-wise partial sum on each rank and combined with one
/// MPI_Allreduce of n doubles, and this rank's block row of the result is what
/// the recurrence consumes.
/// @return x once the recursively updated residual AND the recomputed true
///         residual are within tol*||b||; ConvergenceFail with the final true
///         residual on Lanczos breakdown or once max_iter is spent.
/// @note Per iteration: one SpMV (allgather of n) and one transpose product
///       (allreduce of n), plus a handful of scalar allreduces.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_qmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed transpose-free QMR (Freund 1993) for a general square A.
///
/// The CGS recurrence smoothed by QMR's rotations, so only products with A are
/// needed — no transpose product and therefore no length-n allreduce. Each
/// outer iteration is two SpMVs and two "half steps"; x is updated BEFORE the
/// quasi-residual test and the true residual is what declares convergence.
/// @return x on convergence, or once the true residual is within 10*tol*||b||
///         after the loop; ConvergenceFail with the half-step count otherwise.
/// @note Per iteration: three distributed SpMVs and a few scalar allreduces.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_tfqmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed LSMR (Fong & Saunders 2011) on the Golub-Kahan
///        bidiagonalisation, minimising ||A^T r|| monotonically. Undamped.
///
/// The two vector spaces are split: u (length m) stays block-row distributed,
/// while v, h, hbar and x (length n) are REPLICATED, because the transpose
/// product already produces them replicated via an allreduce. So A*v costs no
/// communication at all and only ||u|| and A^T u cross the network.
/// @return the current iterate, always — LSMR never reports ConvergenceFail.
/// @return DimensionMismatch when A is not square: the distributed entry point
///         keeps that restriction even though ms::lsmr accepts rectangular A.
/// @note Because x is already replicated, no final all-gather is needed.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_lsmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Distributed LSQR (Paige & Saunders) on the Golub-Kahan
///        bidiagonalisation, minimising ||b - A x||.
///
/// Same split of the vector spaces as dist_lsmr: u distributed, v/w/x
/// replicated, one allgather-free A*v and one length-n allreduce for A^T u per
/// iteration.
/// @return the current iterate, always — LSQR never reports ConvergenceFail
///         once past its initial breakdown guards.
/// @return DimensionMismatch when A is not square: the distributed entry point
///         keeps that restriction even though ms::lsqr accepts rectangular A.
template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_lsqr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

} // namespace ms::distributed
