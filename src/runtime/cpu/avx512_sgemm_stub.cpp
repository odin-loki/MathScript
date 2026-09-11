// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The AVX2 sgemm kernel, absent.
//
// Linked in place of avx512_sgemm.cpp when the build has no AVX-512 kernels, so the
// dispatcher in blas_sgemm.cpp needs no #if of its own: it asks available() and
// gets false. Same arrangement as the dgemm stubs next door.

#include "ms/cpu/blas_kernel.hpp"

namespace ms::cpu::blas::avx512 {

bool sgemm_available() {
    return false;
}

bool sgemm_worthwhile(int, int, int) {
    return false;
}

void sgemm_nn(int, int, int, float, const float*, int, const float*, int, float,
              float*, int) {}

} // namespace ms::cpu::blas::avx512
