// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Stands in for avx2_dgemm.cpp when the build has AVX2 kernels disabled, so the
// dispatcher links unchanged and reports the path as unavailable at runtime.
#include "ms/cpu/blas_kernel.hpp"

namespace ms::cpu::blas::avx2 {

bool available() {
    return false;
}

bool worthwhile(int /*m*/, int /*n*/, int /*k*/) {
    return false;
}

void dgemm_nn(
    int /*m*/,
    int /*n*/,
    int /*k*/,
    double /*alpha*/,
    const double* /*A*/,
    int /*lda*/,
    const double* /*B*/,
    int /*ldb*/,
    double /*beta*/,
    double* /*C*/,
    int /*ldc*/) {}

} // namespace ms::cpu::blas::avx2
