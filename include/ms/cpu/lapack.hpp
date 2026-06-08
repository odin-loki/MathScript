#pragma once

namespace ms::cpu::lapack {

/// Cholesky factorization. uplo='L' stores L in the lower triangle of A.
/// Returns 0 on success, otherwise the 1-based index of the first non-positive pivot.
int dpotrf(char uplo, int n, double* A, int lda);

/// LU factorization with partial pivoting. On exit, A holds L (unit diagonal implicit) and U.
/// ipiv[k] is the row index swapped at step k. Returns 0 on success, otherwise a 1-based pivot index.
int dgetrf(int m, int n, double* A, int lda, int* ipiv);

/// Solve A x = b using dgetrf factors. trans='N' or 'T'. nrhs is the number of right-hand sides.
void dgetrs(char trans, int n, int nrhs, const double* A, int lda, const int* ipiv, double* B, int ldb);

/// Factor and solve A x = B for square A. Returns 0 on success, otherwise a 1-based pivot index.
int dgesv(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb);

/// QR factorization using Householder reflectors. tau has length min(m, n).
int dgeqrf(int m, int n, double* A, int lda, double* tau);

/// Generate the first k columns of Q from a dgeqrf factorization.
void dorgqr(int m, int n, int k, const double* A, int lda, const double* tau, double* Q, int ldq);

/// Apply Q or Q^T from a dgeqrf/dsytrd factorization to C.
void dormqr(
    char side,
    char trans,
    int m,
    int n,
    int k,
    const double* A,
    int lda,
    const double* tau,
    double* C,
    int ldc);

/// Generate Q or P**T from a dgebrd factorization.
void dorgbr(
    char vect,
    int m,
    int n,
    int k,
    const double* A,
    int lda,
    const double* tau,
    double* Q,
    int ldq);

/// Apply Q, Q**T, P, or P**T from a dgebrd factorization to C.
/// Returns 0 on success, 1 when arguments are invalid or unsupported.
int dormbr(
    char vect,
    char side,
    char trans,
    int m,
    int n,
    int k,
    const double* A,
    int lda,
    const double* tau,
    double* C,
    int ldc);

/// Least squares min ||A x - B|| for m >= n. On exit, first n rows of B hold x.
int dgels(int m, int n, int nrhs, double* A, int lda, double* B, int ldb);

/// Symmetric eigen-decomposition. jobz='N' for values only, 'V' for vectors in A.
int dsyev(char jobz, int n, double* A, int lda, double* w);

/// Symmetric tridiagonal eigenproblem. d[0..n-1] diagonal, e[0..n-2] subdiagonal.
/// On exit d holds eigenvalues (ascending); e is zeroed. Z column j is eigenvector j when non-null.
void dsteqr(int n, double* d, double* e, double* Z, int ldz);

/// Bidiagonal reduction Q^T A P = B. D diagonal, E off-diagonal, tauq/taup Householder scalars.
int dgebrd(int m, int n, double* A, int lda, double* D, double* E, double* tauq, double* taup);

/// SVD of a bidiagonal matrix. uplo='U' or 'L'. On exit D holds singular values (descending).
/// Optional U (n x n, left vectors as columns) and VT (n x n, V^T rows) when non-null.
/// Optional nru is the number of rows to update in U (defaults to n). Set init_vectors=false when U/VT are preset by dorgbr.
int dbdsqr(
    char uplo,
    int n,
    double* d,
    double* e,
    double* U = nullptr,
    int ldu = 0,
    double* VT = nullptr,
    int ldvt = 0,
    int nru = 0,
    bool init_vectors = true);

/// Thin SVD for column-major A (m x n). U is m x k, VT is k x n, S has length k = min(m, n).
int dgesvd(int m, int n, double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt);

} // namespace ms::cpu::lapack
