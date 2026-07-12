#pragma once

namespace ms::cpu::blas {

/// Dot product: sum_i x[i] * y[i] (respecting incx/incy strides).
double ddot(int n, const double* x, int incx, const double* y, int incy);

/// Vector update: y = alpha * x + y.
void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy);

/// Matrix-vector multiply: y = alpha * op(A) * x + beta * y (column-major).
void dgemv(
    char trans,
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* x,
    int incx,
    double beta,
    double* y,
    int incy);

/// Column-major DGEMM: C = alpha * op(A) * op(B) + beta * C.
/// transa/transb are 'N' or 'T'. lda/ldb/ldc are leading dimensions (column-major).
void dgemm(
    char transa,
    char transb,
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc);

/// Symmetric rank-k update: C = alpha * op(A) * op(A)^T + beta * C (column-major).
void dsyrk(
    char uplo,
    char trans,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    double beta,
    double* C,
    int ldc);

/// Triangular solve: op(A) * X = alpha * B (side='L') or X * op(A) = alpha * B (side='R').
void dtrsm(
    char side,
    char uplo,
    char transa,
    char diag,
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    double* B,
    int ldb);

/// Rank-1 update: A += alpha * x * y^T (column-major).
void dger(
    int m,
    int n,
    double alpha,
    const double* x,
    int incx,
    const double* y,
    int incy,
    double* A,
    int lda);

} // namespace ms::cpu::blas
