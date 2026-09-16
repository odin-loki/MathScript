// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include <cstdint>
#include <string>

namespace ms::simd {

// True only where x86 feature detection is meaningful. Testing for GCC or
// Clang instead of for the architecture is the usual mistake and breaks every
// non-x86 target those compilers support — aarch64 as much as WebAssembly.
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#define MS_ISA_X86 1
#else
#define MS_ISA_X86 0
#endif

#if defined(__wasm_simd128__)
#define MS_ISA_WASM_SIMD 1
#else
#define MS_ISA_WASM_SIMD 0
#endif

struct IsaFeatures {
    bool sse2 = false;
    bool sse41 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool wasm_simd128 = false;   // the one vector ISA WebAssembly has
};

IsaFeatures detect_isa();
std::string isa_summary(const IsaFeatures& features);

} // namespace ms::simd
