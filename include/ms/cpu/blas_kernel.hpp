#pragma once

namespace ms::cpu::blas::avx512 {

bool available();

void dgemm_nn(
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

} // namespace ms::cpu::blas::avx512
