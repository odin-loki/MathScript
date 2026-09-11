// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include "ms/core/operations.hpp"
#include <functional>
#include <vector>

namespace ms {

/// @brief Eigenvalues and right eigenvectors of a real square matrix.
///
/// @note  A real matrix can have complex-conjugate eigenvalue pairs, so the
///        spectrum is carried in two real columns: @c values holds the real
///        parts and @c values_imag the imaginary parts (all zero for a
///        symmetric matrix, and for any matrix with a purely real spectrum).
///        The eigenvectors use LAPACK dgeev's packing: for a conjugate pair
///        occupying rows j and j+1 (values_imag(j,0) > 0 and
///        values_imag(j+1,0) = -values_imag(j,0)), the eigenvector for
///        lambda_j is vectors(:,j) + i*vectors(:,j+1) and the eigenvector for
///        lambda_{j+1} is its complex conjugate. Every purely real eigenvalue
///        owns one real column, normalised to unit 2-norm with its dominant
///        entry positive; each packed complex pair is normalised so that
///        ||Re v||^2 + ||Im v||^2 = 1. Entries are ordered by descending real
///        part, with the two members of a pair kept adjacent.
struct EigResult {
    Matrix<double> values;       ///< n x 1, real parts of the eigenvalues
    Matrix<double> vectors;      ///< n x n, right eigenvectors (packed, see above)
    Matrix<double> values_imag;  ///< n x 1, imaginary parts of the eigenvalues
};

struct SvdResult {
    Matrix<double> U;
    Matrix<double> S;
    Matrix<double> V;
};

struct LdlResult {
    Matrix<double> L;
    Matrix<double> D;
    Matrix<double> P;
};

struct SchurResult {
    Matrix<double> T;
    Matrix<double> Q;
};

struct BidiagResult {
    Matrix<double> U;
    Matrix<double> B;
    Matrix<double> V;
};

// Construction
template<typename S, template<typename> class Alloc = std::allocator>
Matrix<S, StorageOrder::ColMajor, Alloc> rand(size_t m, size_t n, unsigned seed = 42);

template<typename S, template<typename> class Alloc = std::allocator>
Matrix<S, StorageOrder::ColMajor, Alloc> randn(size_t m, size_t n, unsigned seed = 42);

template<typename S, template<typename> class Alloc = std::allocator>
Matrix<S, StorageOrder::ColMajor, Alloc> diag(const std::vector<S>& v);

template<typename S, StorageOrder OA, template<typename> class Alloc>
std::vector<S> diag(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> tril(const Matrix<S, OA, Alloc>& A, int k = 0);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> triu(const Matrix<S, OA, Alloc>& A, int k = 0);

// Basic operations
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> rank(const Matrix<S, OA, Alloc>& A, S tol = S(0));

/// @brief Matrix rank via SVD: count singular values strictly above @p tol.
template<typename S, StorageOrder OA, template<typename> class Alloc>
int matrix_rank(const Matrix<S, OA, Alloc>& A, double tol = 1e-10);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> cond(const Matrix<S, OA, Alloc>& A, int p = 2);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsq(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b);

// Decompositions
/// @brief Symmetric LDL^T factorisation with threshold diagonal pivoting:
///        P^T * A * P == L * D * L^T.
///
/// @note  L is unit lower triangular and D is returned as an n x 1 column of
///        the 1x1 pivots (not as an n x n matrix). P is the symmetric
///        interchange actually used: the natural pivot order is kept unless a
///        diagonal pivot has lost roughly half of double precision relative to
///        the largest remaining candidate, in which case rows and columns are
///        swapped and the swap is recorded in P. P is the identity for every
///        well-conditioned input, so the common case still satisfies
///        A == L*D*L^T.
/// @note  Only 1x1 pivots are produced. A symmetric matrix whose trailing
///        block has no usable diagonal entry needs a 2x2 Bunch-Kaufman block
///        pivot (the canonical example is [[0,1],[1,0]]); that case is
///        reported as a DomainError rather than being mislabelled singular.
/// @return DimensionMismatch if A is not square, DomainError if A is not
///         symmetric or needs a 2x2 pivot, SingularMatrix if a trailing block
///         is genuinely zero.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<LdlResult> ldl(const Matrix<S, OA, Alloc>& A);

/// @brief Upper Hessenberg reduction by Householder similarity:
///        H == Q^T * A * Q for an orthogonal Q, so H has exactly A's trace,
///        determinant, characteristic polynomial and spectrum.
///
/// @note  Each reflector is applied from BOTH sides. A symmetric input comes
///        back symmetric tridiagonal. Q is not part of this signature — use
///        schur(), which performs the same reduction and returns its
///        accumulated orthogonal factor.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> hess(const Matrix<S, OA, Alloc>& A);

/// @brief Golub-Kahan bidiagonalisation: A == U * B * V^T with U (m x m) and
///        V (n x n) orthogonal and B (m x n) upper bidiagonal — its only
///        nonzeros are B(i,i) and B(i,i+1).
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<BidiagResult> bidiag(const Matrix<S, OA, Alloc>& A);

/// @brief Eigenvalues and right eigenvectors of a general real square matrix.
///
/// @note  Computes the real Schur form (Householder Hessenberg reduction plus
///        implicit Francis double-shift QR with deflation) and recovers the
///        eigenvectors by back substitution on the quasi-triangular factor,
///        then transforms them back with the Schur basis. Complex-conjugate
///        pairs are reported exactly — see EigResult for the packing — rather
///        than being flattened onto the real axis. A symmetric input is
///        delegated to eig_sym().
/// @note  For a defective matrix the eigenvectors belonging to a repeated
///        eigenvalue cannot all be independent; the back substitution
///        perturbs an exactly singular pivot by eps*||T|| (as LAPACK dtrevc
///        does), so the returned columns stay finite but the duplicated ones
///        carry no extra information.
/// @return DimensionMismatch if A is not square; ConvergenceFail if the QR
///         iteration exhausts its sweep budget.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<EigResult> eig(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<EigResult> eig_sym(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<SvdResult> svd(const Matrix<S, OA, Alloc>& A);

/// @brief Real Schur decomposition: A == Q * T * Q^T with Q orthogonal and T
///        quasi-upper-triangular.
///
/// @note  1x1 diagonal blocks of T carry real eigenvalues; a 2x2 block with a
///        nonzero subdiagonal carries a complex-conjugate pair (never two
///        adjacent nonzero subdiagonal entries). The reduction is a genuine
///        Householder similarity and its orthogonal factor seeds Q, which is
///        then updated by every implicit Francis double-shift sweep, so the
///        identity A == Q*T*Q^T holds to round-off for any input.
/// @return DimensionMismatch if A is not square; ConvergenceFail{sweeps,
///         largest remaining subdiagonal} if the shifted QR iteration
///         exhausts its budget, instead of a half-reduced T reported as
///         success.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<SchurResult> schur(const Matrix<S, OA, Alloc>& A);

/// @brief Solve the Sylvester equation A*X + X*B = C for X, given square
///        A (n x n), square B (m x m), and C (n x m).
///
/// @note Algorithm: Kronecker-sum vectorization, i.e. it forms the (n*m) x
///       (n*m) matrix K = I_m ⊗ A + B^T ⊗ I_n and solves K*vec(X) = vec(C)
///       (vec stacks columns) via the existing dense solve(). This is the
///       same style of approach ms::control::lyap() uses for the Lyapunov
///       special case (B = A^T). It is O((n*m)^3) — much worse than a true
///       Bartels-Stewart Schur-based reduction (O(n^3 + m^3)) — so it is only
///       intended for small/moderate n, m; it was chosen over Bartels-Stewart
///       here because schur() returns a REAL Schur form, whose 2x2 blocks for
///       complex-conjugate eigenvalues a Bartels-Stewart solver would have to
///       handle as coupled 2x2 Sylvester subproblems, whereas the Kronecker
///       approach is correct for the fully general case (real or
///       complex-conjugate eigenvalues) and much simpler to get right.
/// @return X on success. Returns DimensionMismatch if A/B aren't square or C's
///         shape doesn't match (n x m), or the SingularMatrix/DomainError
///         propagated from solve() if the Sylvester operator K is singular
///         (which happens exactly when A and -B share an eigenvalue, i.e. the
///         equation has no unique solution).
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> solve_sylvester(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& B,
    const Matrix<S, OA, Alloc>& C);

// ---------------------------------------------------------------------------
// Matrix functions (logm, sqrtm, sinm, cosm, funm)
//
// DOMAIN OF VALIDITY -- read this before calling any of the five.
//
// A SYMMETRIC A is diagonalised by an orthogonal V (eig_sym) and f(A) is
// evaluated exactly as V*diag(f(lambda))*V^T. No restriction beyond f being
// finite on the spectrum.
//
// A NON-SYMMETRIC A goes through the real Schur form A = Q*T*Q^T:
//   * sqrtm uses the Bjorck-Hammarling triangular square-root recurrence,
//     which stays well defined for repeated eigenvalues;
//   * logm, sinm, cosm, funm and funm_taylor use the Parlett recurrence, whose
//     divisor is T(j,j) - T(i,i), and two diagonal entries within
//     1e-8*||T||_F make that divisor vanish. There f(A) depends on the
//     DERIVATIVES of f -- the logarithm of the defective [[2,1],[0,2]] is
//     [[ln 2, 1/2],[0, ln 2]], and that 1/2 is (d/dx) ln x at 2 -- so anything
//     sampling f on the spectrum alone is missing information, not merely
//     dividing by zero. logm, sinm, cosm and funm_taylor therefore switch to
//     the BLOCKED Schur-Parlett (Davies-Higham): eigenvalues within
//     0.1*||T||_F are grouped, T is reordered by Givens swaps so the groups sit
//     contiguously, each diagonal block is evaluated by its Taylor series about
//     the group mean, and the off-diagonal blocks come from triangular
//     Sylvester solves. Those four accept repeated and clustered eigenvalues.
//     Plain funm, handed f alone, does not have the derivatives and says so.
// Consequently these inputs are REPORTED AS DomainError rather than served
// with a plausible but wrong matrix:
//   * a complex-conjugate eigenvalue pair (a 2x2 block of the real Schur
//     form) -- a real-arithmetic evaluation cannot represent f on it;
//   * for funm (the overload taking f alone), two diagonal entries of T closer
//     than 1e-8*||T||_F -- use funm_taylor and supply the coefficients;
//   * for logm/sinm/cosm/funm_taylor, a cluster whose Taylor series does not
//     converge, or two clusters too close to separate;
//   * for sqrtm, an eigenvalue below -16*n*eps*||T||_F (no real square root)
//     -- the tolerance is loose enough that a positive semidefinite matrix
//     whose smallest eigenvalue is negative only by round-off still works;
//   * for logm, a non-positive eigenvalue (no real logarithm).
// None of the six clamps, truncates or substitutes a value silently.
// ---------------------------------------------------------------------------

/// @brief Real matrix logarithm: exp(logm(A)) == A. Repeated and clustered
///        eigenvalues go through the blocked Schur-Parlett, so a defective A
///        (e.g. [[2,1],[0,2]]) is served rather than rejected.
/// @return DimensionMismatch if A is not square; DomainError on any input
///         outside the domain described above (in particular a symmetric A
///         that is not positive definite).
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> logm(const Matrix<S, OA, Alloc>& A);

/// @brief Principal real matrix square root: sqrtm(A)*sqrtm(A) == A.
/// @return DimensionMismatch if A is not square; DomainError on any input
///         outside the domain described above (negative eigenvalue, complex
///         pair, or a defective zero eigenvalue that admits no square root).
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sqrtm(const Matrix<S, OA, Alloc>& A);

// ---------------------------------------------------------------------------
// Iterative solvers (cg, jacobi, bicgstab, gmres, minres; also qmr/tfqmr/
// lsqr/lsmr/pcg below)
//
// SHARED CONTRACT
//   * @p b may have any number of columns. Each Krylov method here works on a
//     single right-hand side, so a multi-column b is solved COLUMN BY COLUMN
//     and the columns are assembled into an n x nrhs result -- the same shape
//     and the same values solve() would give. A failure on any one column
//     fails the whole call.
//   * A convergence claim is always confirmed against the TRUE residual
//     ||b - A x||, never against a recursively updated estimate alone; the
//     iterate is returned only when that residual is within tol of ||b||
//     (10*tol once the loop has ended). Otherwise the call returns
//     ConvergenceFail{iterations, residual} -- including on a breakdown, so a
//     NaN iterate can never be handed back as a success.
//   * b == 0 short-circuits to the exact zero vector.
//   * DimensionMismatch if A is not square or b's rows do not match it.
// ---------------------------------------------------------------------------

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> cg(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> jacobi(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> bicgstab(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> gmres(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t restart = 20,
    size_t max_iter = 1000,
    S tol = S(1e-10));

// --- New construction helpers ---
// (zeros, ones, eye are already declared in ms/core/operations.hpp)

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> kron(const Matrix<S, OA, Alloc>& A,
                           const Matrix<S, OA, Alloc>& B);

template<typename S>
std::vector<S> linspace(S a, S b, size_t n);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> repmat(const Matrix<S, OA, Alloc>& A, size_t p, size_t q);

// --- New basic operations ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> pinv(const Matrix<S, OA, Alloc>& A,
                                   S tol = S(0));

/// @brief Orthonormal basis of the null space of A, as an n x (n - rank)
///        matrix N with A*N == 0 and N^T*N == I.
///
/// @note  @p tol is a cutoff in SINGULAR-VALUE units (the same convention
///        rank(), pinv() and orth() use), not a relative or squared one: a
///        singular value is treated as zero when sigma_i <= tol. Passing 0
///        selects the default, 1e-10 * max(m,n) * sigma_max for m >= n --
///        which makes the result invariant under a uniform rescaling of A.
/// @note  For m >= n the rank comes from the SVD's singular values and the
///        basis is built by orthonormal completion of the row space, so every
///        returned column is a genuine unit vector even when the underlying
///        SVD leaves the right singular vector of an exactly zero singular
///        value unpopulated. For the wide case m < n the thin SVD's V (n x m)
///        cannot span an (n - m)-dimensional null space, so the eigenvectors
///        of A^T A are used instead; forming A^T A halves the available
///        digits, so the default cutoff there is widened to
///        sqrt(eps) * max(m,n) * sigma_max.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> null(const Matrix<S, OA, Alloc>& A,
                                   S tol = S(0));

/// @brief Orthonormal basis of the column space of A, as an m x rank matrix Q
///        with Q^T*Q == I and Q*Q^T*A == A.
///
/// @note  Computed from the SVD (the leading rank columns of U), because an
///        unpivoted QR is not rank revealing and its leading columns are the
///        wrong set whenever the deficiency is not in the trailing columns.
/// @note  @p tol is a cutoff in singular-value units; 0 selects the default
///        1e-10 * max(m,n) * sigma_max, matching rank(), pinv() and null().
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> orth(const Matrix<S, OA, Alloc>& A,
                                   S tol = S(0));

// --- New matrix functions ---
// See the "DOMAIN OF VALIDITY" block above logm/sqrtm: all three go through
// the same symmetric / Schur-Parlett split and reject the same inputs.

/// @brief Matrix sine. Satisfies sinm(A)^2 + cosm(A)^2 == I. Repeated and
///        clustered eigenvalues go through the blocked Schur-Parlett.
/// @return DimensionMismatch if A is not square; DomainError for a
///         non-symmetric A with complex-conjugate eigenvalues.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sinm(const Matrix<S, OA, Alloc>& A);

/// @brief Matrix cosine. Satisfies sinm(A)^2 + cosm(A)^2 == I. Repeated and
///        clustered eigenvalues go through the blocked Schur-Parlett.
/// @return DimensionMismatch if A is not square; DomainError for a
///         non-symmetric A with complex-conjugate eigenvalues.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> cosm(const Matrix<S, OA, Alloc>& A);

/// @brief f(A) for a scalar f, evaluated on the eigenvalues: exactly via the
///        orthogonal eigenbasis for a symmetric A, otherwise by the Parlett
///        recurrence on the real Schur factor (this is the "via Schur
///        decomposition" the name promises).
/// @note   f alone does not determine f(A) at a repeated eigenvalue, so this
///         overload reports that case rather than guessing; funm_taylor takes
///         the Taylor coefficients and handles it.
/// @return DimensionMismatch if A is not square; DomainError for an empty
///         function object, for an f that is not finite at some eigenvalue,
///         or for a non-symmetric A with complex-conjugate or clustered
///         eigenvalues.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> funm(const Matrix<S, OA, Alloc>& A,
                                   std::function<S(S)> func);

/// @brief f(A) from f's Taylor coefficients, so that a repeated or clustered
///        eigenvalue is handled rather than rejected.
///
/// f(A) at a repeated eigenvalue depends on the DERIVATIVES of f -- the
/// logarithm of the defective [[2,1],[0,2]] is [[ln 2, 1/2],[0, ln 2]], and that
/// 1/2 is (d/dx) ln x at 2 -- so no routine that only samples f on the spectrum
/// can produce it. Supplying the coefficients closes that gap: `logm`, `sinm`
/// and `cosm` already do it internally and accept defective matrices.
///
/// @param coefficients must return the k-th Taylor coefficient of f about x,
///        i.e. f^(k)(x)/k!, with k == 0 giving f(x) itself. The coefficient
///        rather than the bare derivative is asked for because the derivatives
///        of many ordinary functions overflow long before their series
///        converges.
/// @return DimensionMismatch if A is not square; DomainError for an empty
///         function object, for an f that is not finite at some eigenvalue, for
///         a non-symmetric A with complex-conjugate eigenvalues, or when the
///         Taylor series for a cluster does not converge.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> funm_taylor(const Matrix<S, OA, Alloc>& A,
                                          std::function<S(S, unsigned)> coefficients);

// --- New iterative solvers ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> minres(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Quasi-Minimal Residual for a general square (possibly
///        nonsymmetric) A, starting from x0 = 0.
///
/// @note  Freund & Nachtigal's QMR without look-ahead: unsymmetric (two-sided)
///        Lanczos builds biorthogonal bases from A and A^T, and the resulting
///        tridiagonal least-squares problem is smoothed by Givens rotations,
///        so the residual norm behaves far more smoothly than BiCG's. Costs
///        two matrix-vector products per iteration — one with A, one with A^T
///        (formed on the fly, A is never transposed) — and stores a fixed
///        handful of vectors regardless of iteration count. Unpreconditioned
///        (M1 = M2 = I). In exact arithmetic the Lanczos process spans the
///        whole Krylov space after n steps, so QMR terminates at k = n.
/// @note  Look-ahead is deliberately not implemented, so a serious Lanczos
///        breakdown (w^T v, q^T A p or the resulting beta vanishing — e.g. the
///        skew-symmetric A = [[0,1],[-1,0]] with b = (1,1)) stops the
///        iteration and is reported as ConvergenceFail rather than being
///        stepped over. The recursively updated residual is only an estimate,
///        so a convergence claim is always confirmed against the true
///        residual ||b - A x|| before x is returned.
/// @return x on convergence (relative residual ||b - A x|| <= tol*||b||, or
///         <= 10*tol*||b|| once the loop has ended); the zero vector when b is
///         zero; DimensionMismatch if A is not square or b's rows do not match
///         it; ConvergenceFail with the final true residual on breakdown or
///         when max_iter is exhausted.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> qmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsqr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief LSMR (Fong & Saunders): least-squares solver for min ||A x - b||
///        over a general m x n A, starting from x0 = 0.
///
/// @note  Golub-Kahan bidiagonalisation, as in lsqr(), but with a second
///        sequence of Givens rotations applied to R^T. The two solvers walk
///        the same Krylov space and differ in what they minimise there: LSQR
///        minimises ||r||, LSMR minimises ||A^T r|| — the normal-equation
///        residual — and does so monotonically. That makes LSMR the better
///        stopping citizen on inconsistent or ill-conditioned problems, where
///        ||r|| flattens out long before A^T r does. Two products per
///        iteration (one with A, one with A^T); A is never transposed.
/// @note  A is not required to be square: any m x n shape is accepted as long
///        as b has m rows, so overdetermined fits and underdetermined systems
///        both work. Because x0 = 0 every iterate stays in range(A^T), so on a
///        rank-deficient A the limit is the minimum-norm least-squares
///        solution. Undamped (lambda = 0).
/// @return x — always a value once the shape check passes, since the current
///         iterate is by construction the best approximation found so far;
///         this function never reports ConvergenceFail. The result is the zero
///         vector (A.cols() x 1) when b is zero or b is orthogonal to
///         range(A). DimensionMismatch if b's rows do not match A's.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

/// @brief Transpose-Free QMR (Freund 1993) for a general square A, starting
///        from x0 = 0.
///
/// @note  Builds the CGS polynomial recurrence and smooths it with the same
///        quasi-minimal-residual rotations QMR uses, so — unlike qmr() — only
///        products with A are needed and A^T never appears. Each iteration is
///        two matrix-vector products and advances the iteration index by two
///        "half steps"; @p max_iter counts the outer iterations, so up to
///        2*max_iter iterates are produced.
/// @note  The scalar tau tracked by the recurrence is a quasi-residual: the
///        only rigorous statement is ||b - A x_m|| <= tau_m * sqrt(m + 1), so
///        a small tau does not by itself mean x is right. tau can even reach
///        exactly zero in the very half step that first makes x correct (it
///        does on A = I). This implementation therefore applies the x update
///        before testing, treats tau <= tol*||b|| only as a screen, and
///        confirms it with the true residual ||b - A x|| — one extra product
///        per triggered check — before returning.
/// @return x on convergence (true relative residual <= tol, or <= 10*tol once
///         the loop has ended); the zero vector when b is zero;
///         DimensionMismatch if A is not square or b's rows do not match it;
///         ConvergenceFail with the half-step count and final residual on
///         breakdown, stagnation, or exhausted iterations.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> tfqmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

// --- Preconditioners ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
std::vector<S> precond_diag(const Matrix<S, OA, Alloc>& A);

/// @brief The SSOR splitting matrix of A, formed explicitly:
///        M = (D/omega + L) * (D/omega)^-1 * (D/omega + U), where D, L and U
///        are the diagonal, strictly lower and strictly upper parts of A.
///
/// @note  The customary scalar factor 1/(omega*(2 - omega)) is deliberately
///        NOT applied: it blows up at omega = 2, and a preconditioner is only
///        defined up to a positive scalar anyway (in a preconditioned Krylov
///        method the constant cancels out of alpha and beta, leaving the
///        iterates unchanged). M is symmetric whenever A is, positive definite
///        whenever A is SPD and 0 < omega < 2, and reduces to D itself when A
///        is diagonal and omega = 1.
/// @note  Operates on the leading min(rows, cols) x min(rows, cols) block, so
///        a rectangular A is truncated rather than read out of bounds. A zero
///        diagonal entry contributes nothing (its inverse is taken as 0) and
///        omega == 0 falls back to omega = 1, so no input divides by zero.
/// @return M, shaped n x n with n = min(rows, cols).
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ssor(const Matrix<S, OA, Alloc>& A, S omega = S(1.0));

/// @brief Apply the inverse of the SSOR preconditioner to a vector without
///        ever forming M = (D/omega + L) * (D/omega)^-1 * (D/omega + U).
///
/// @note  Runs three O(n^2) sweeps — a forward substitution with (D/omega + L),
///        a diagonal scaling by (D/omega), then a back substitution with
///        (D/omega + U) — which is what a preconditioned Krylov solver actually
///        needs; precond_ssor() materialises the same operator densely when a
///        caller wants to inspect it. Operates on the leading
///        min(rows, cols) x min(rows, cols) block, so a rectangular A is
///        accepted and truncated rather than read out of bounds.
/// @return z with M*z == r, shaped (n, 1). Returns a zero vector if r does not
///         have n rows. A zero diagonal entry (or omega == 0) degrades to a
///         unit pivot instead of dividing by zero.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ssor_apply(const Matrix<S, OA, Alloc>& A,
                                        S omega,
                                        const Matrix<S, OA, Alloc>& r);

/// @brief ILU(0) incomplete LU factorisation: Gaussian elimination restricted
///        to the sparsity pattern of A, so no fill-in is ever created.
///
/// @note  The factors are returned in one matrix in the usual compact layout:
///        the strictly lower triangle holds L (whose diagonal is an implied 1)
///        and the upper triangle including the diagonal holds U. Entries where
///        A is structurally zero stay zero. Exact for matrices whose pattern
///        admits no fill-in (tridiagonal, for instance), approximate otherwise.
///        A zero pivot is skipped rather than divided by.
/// @return The combined L\U factors, shaped n x n with n = min(rows, cols).
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ilu0(const Matrix<S, OA, Alloc>& A);

/// @brief Solve (L*U) z = r for the compact factors produced by precond_ilu0().
/// @note  Forward substitution against L's implied unit diagonal, then back
///        substitution against U; a zero pivot in U degrades to a unit pivot.
/// @return z, shaped (n, 1); a zero vector if r does not have n rows.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ilu0_apply(const Matrix<S, OA, Alloc>& LU,
                                        const Matrix<S, OA, Alloc>& r);

/// @brief Preconditioned conjugate gradient for a symmetric positive definite
///        A, with the preconditioner supplied as an operator z = M^-1 r.
///
/// @note  M_apply is taken as a std::function (the same convention funm() uses)
///        so the preconditioner never has to be materialised; pair it with
///        precond_ssor_apply() or precond_ilu0_apply(). A bare lambda will not
///        deduce against a std::function parameter, so build the std::function
///        first. M_apply must model an SPD operator: the iteration checks
///        r^T z > 0 every step and reports a DomainError if it is violated.
///        Passing the identity reproduces cg() iterate for iterate.
/// @return x on convergence (absolute residual ||b - A x||_2 < tol, the same
///         test cg() uses); DimensionMismatch if A is not square or b does not
///         match; DomainError if A is not symmetric, M_apply is empty, returns
///         the wrong shape, or is not positive definite; ConvergenceFail with
///         the final residual if max_iter is exhausted.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> pcg(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    const std::function<Matrix<S, OA, Alloc>(const Matrix<S, OA, Alloc>&)>& M_apply,
    size_t max_iter = 1000,
    S tol = S(1e-10));

} // namespace ms
