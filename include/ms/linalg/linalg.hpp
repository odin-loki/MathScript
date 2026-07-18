#pragma once

#include "ms/core/operations.hpp"
#include <functional>
#include <vector>

namespace ms {

struct EigResult {
    Matrix<double> values;
    Matrix<double> vectors;
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
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<LdlResult> ldl(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> hess(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<BidiagResult> bidiag(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<EigResult> eig(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<EigResult> eig_sym(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<SvdResult> svd(const Matrix<S, OA, Alloc>& A);

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
///       here because this file's schur() runs unshifted QR iteration and
///       is not guaranteed to fully triangularize matrices with complex
///       (non-real) eigenvalues, whereas the Kronecker approach is correct
///       for the fully general case (real or complex-conjugate eigenvalues)
///       and much simpler to get right.
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

// Matrix functions
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> logm(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sqrtm(const Matrix<S, OA, Alloc>& A);

// Iterative solvers
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

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> null(const Matrix<S, OA, Alloc>& A,
                                   S tol = S(0));

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> orth(const Matrix<S, OA, Alloc>& A,
                                   S tol = S(0));

// --- New matrix functions ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sinm(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> cosm(const Matrix<S, OA, Alloc>& A);

// funm: apply scalar function via Schur decomposition
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> funm(const Matrix<S, OA, Alloc>& A,
                                   std::function<S(S)> func);

// --- New iterative solvers ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> minres(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

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

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> tfqmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter = 1000,
    S tol = S(1e-10));

// --- Preconditioners (return diagonal scaling vector) ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
std::vector<S> precond_diag(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ssor(const Matrix<S, OA, Alloc>& A, S omega = S(1.0));

} // namespace ms
