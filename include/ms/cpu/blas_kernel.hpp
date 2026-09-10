// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

// Per-ISA dgemm kernels. Each namespace has the same three entry points:
//
//   available()   the CPU has the ISA *and* the OS preserves its register state
//   worthwhile()  the problem is big enough to repay packing two aligned panels
//   dgemm_nn()    C = alpha * A * B + beta * C, all column-major, no transpose
//
// A build with the kernel disabled links a stub whose available() returns false,
// so the dispatcher does not need to know which kernels were compiled in.
namespace ms::cpu::blas::avx512 {

bool available();
bool worthwhile(int m, int n, int k);

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

namespace ms::cpu::blas::avx2 {

bool available();
bool worthwhile(int m, int n, int k);

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

} // namespace ms::cpu::blas::avx2
