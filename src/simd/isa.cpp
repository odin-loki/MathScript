// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/simd/isa.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace ms::simd {

namespace {

void cpuid(int info[4], int leaf, int subleaf = 0) {
#if defined(_MSC_VER)
    __cpuidex(info, leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
#else
    (void)leaf;
    (void)subleaf;
    info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

// Read the extended control register that says which register state the OS has
// agreed to save and restore across a context switch.
//
// This is the half of feature detection that CPUID does not cover, and leaving it
// out is not a portability nicety: CPUID reports what the silicon implements,
// XCR0 reports what the operating system has enabled. A hypervisor, a sandbox, a
// kernel booted with `noxsave`, or an older OS can leave AVX or AVX-512 masked off
// on a CPU that advertises them. Dispatching on the CPUID bit alone then selects a
// kernel compiled with -mavx512f and the first instruction faults, so the failure
// is SIGILL inside dgemm rather than a fall back to a slower path -- decided by the
// deployment environment, not by anything the caller passed in.
std::uint64_t xgetbv0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    // xgetbv with ecx=0. Written as .byte so it assembles without -mxsave.
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
    return 0;
#endif
}

// XCR0 bit 1 (SSE state) and bit 2 (AVX state, the upper half of YMM).
constexpr std::uint64_t kXcr0Ymm = 0x6;
// Additionally bit 5 (opmask), bit 6 (ZMM_Hi256) and bit 7 (Hi16_ZMM).
constexpr std::uint64_t kXcr0Zmm = 0xE6;

// MS_FORCE_ISA pins the reported level so every dispatch path can be exercised on
// one machine, and so a deployment can step down if a host turns out to be lying.
// It only ever narrows what was detected: forcing a level the CPU or OS does not
// support would reintroduce the fault this file exists to prevent.
const char* forced_isa_level() noexcept {
    const char* const raw = std::getenv("MS_FORCE_ISA");
    if (raw == nullptr || raw[0] == '\0') {
        return nullptr;
    }
    return raw;
}

void apply_isa_ceiling(IsaFeatures& f, const char* level) noexcept {
    // Ordered from most to least capable; each case falls through to clear
    // everything above the requested ceiling.
    if (std::strcmp(level, "scalar") == 0) {
        f = IsaFeatures{};
        return;
    }
    if (std::strcmp(level, "sse2") == 0) {
        f.sse41 = f.avx = f.avx2 = f.fma = f.avx512f = false;
        return;
    }
    if (std::strcmp(level, "sse41") == 0) {
        f.avx = f.avx2 = f.fma = f.avx512f = false;
        return;
    }
    if (std::strcmp(level, "avx") == 0) {
        f.avx2 = f.fma = f.avx512f = false;
        return;
    }
    if (std::strcmp(level, "avx2") == 0) {
        f.avx512f = false;
        return;
    }
    // "avx512" or anything unrecognised leaves the detected set alone. An
    // unrecognised value must not raise the ceiling.
}

} // namespace

IsaFeatures detect_isa() {
    IsaFeatures f;
    int info[4] = {0, 0, 0, 0};
    cpuid(info, 0);
    const int max_leaf = info[0];
    if (max_leaf == 0) {
        return f;
    }

    cpuid(info, 1);
    // SSE2 and SSE4.1 live in the legacy XMM state, which every OS that can run
    // this binary already saves, so they need no XCR0 agreement.
    f.sse2 = (info[3] & (1 << 26)) != 0;
    f.sse41 = (info[2] & (1 << 19)) != 0;

    const bool cpu_avx = (info[2] & (1 << 28)) != 0;
    const bool cpu_fma = (info[2] & (1 << 12)) != 0;
    const bool osxsave = (info[2] & (1 << 27)) != 0;

    // Without OSXSAVE the XGETBV instruction itself is unavailable, so there is no
    // way to ask, and the answer has to be no.
    std::uint64_t xcr0 = 0;
    if (osxsave) {
        xcr0 = xgetbv0();
    }
    const bool os_ymm = osxsave && ((xcr0 & kXcr0Ymm) == kXcr0Ymm);
    const bool os_zmm = osxsave && ((xcr0 & kXcr0Zmm) == kXcr0Zmm);

    f.avx = cpu_avx && os_ymm;
    // FMA operates on YMM, so it needs the same OS agreement as AVX even though its
    // CPUID bit sits apart from AVX's.
    f.fma = cpu_fma && os_ymm;

    if (max_leaf >= 7) {
        cpuid(info, 7, 0);
        f.avx2 = ((info[1] & (1 << 5)) != 0) && os_ymm;
        f.avx512f = ((info[1] & (1 << 16)) != 0) && os_zmm;
    }

    if (const char* const level = forced_isa_level()) {
        apply_isa_ceiling(f, level);
    }
    return f;
}

std::string isa_summary(const IsaFeatures& features) {
    std::string s;
    if (features.avx512f) {
        s += "AVX-512F ";
    }
    if (features.avx2) {
        s += "AVX2 ";
    }
    if (features.avx) {
        s += "AVX ";
    }
    if (features.fma) {
        s += "FMA ";
    }
    if (features.sse41) {
        s += "SSE4.1 ";
    }
    if (features.sse2) {
        s += "SSE2 ";
    }
    if (s.empty()) {
        return "scalar";
    }
    return s;
}

} // namespace ms::simd
