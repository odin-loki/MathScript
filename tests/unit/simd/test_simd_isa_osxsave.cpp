// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// ISA detection must agree with the operating system, not only with the silicon.
//
// CPUID reports what the CPU implements. It does not report whether the OS has
// enabled the extended register state, and dispatch built on the CPUID bit alone
// selects a kernel compiled with -mavx512f that faults on its first instruction --
// SIGILL inside dgemm, decided by the host rather than by any input. The fix reads
// CPUID.1:ECX.OSXSAVE and then XGETBV(0), requiring XCR0 & 0x6 for YMM state and
// XCR0 & 0xE6 for ZMM state.
//
// This host cannot be made to report AVX-512-without-OS-support on demand, so what
// is checked here is the property that survives either way: the reported set is
// internally consistent, it never exceeds what the OS agreed to, and MS_FORCE_ISA
// can only narrow it.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "ms/simd/isa.hpp"

using namespace ms::simd;

namespace {

// RAII around the environment variable so a failing expectation cannot leak a
// forced ceiling into the tests that run after it.
class ScopedForceIsa {
public:
    explicit ScopedForceIsa(const char* level) {
        if (const char* const prev = std::getenv("MS_FORCE_ISA")) {
            had_previous_ = true;
            previous_ = prev;
        }
        if (level == nullptr) {
            ::unsetenv("MS_FORCE_ISA");
        } else {
            ::setenv("MS_FORCE_ISA", level, 1);
        }
    }
    ~ScopedForceIsa() {
        if (had_previous_) {
            ::setenv("MS_FORCE_ISA", previous_.c_str(), 1);
        } else {
            ::unsetenv("MS_FORCE_ISA");
        }
    }
    ScopedForceIsa(const ScopedForceIsa&) = delete;
    ScopedForceIsa& operator=(const ScopedForceIsa&) = delete;

private:
    bool had_previous_ = false;
    std::string previous_;
};

// The hierarchy the dispatcher relies on: a wider path is never reported without
// every narrower one it is built on.
void expect_consistent(const IsaFeatures& f) {
    if (f.avx512f) {
        EXPECT_TRUE(f.avx2) << "AVX-512F reported without AVX2";
        EXPECT_TRUE(f.avx) << "AVX-512F reported without AVX";
    }
    if (f.avx2) {
        EXPECT_TRUE(f.avx) << "AVX2 reported without AVX";
    }
    if (f.fma) {
        EXPECT_TRUE(f.avx) << "FMA reported without AVX: both need OS YMM state";
    }
}

} // namespace

TEST(SimdIsaOsEnablement, DetectedSetIsInternallyConsistent) {
    const ScopedForceIsa clear(nullptr);
    expect_consistent(detect_isa());
}

TEST(SimdIsaOsEnablement, FmaIsGatedOnTheSameOsStateAsAvx) {
    // FMA operates on YMM registers, so it needs the OS to have agreed to save
    // them just as AVX does, even though its CPUID bit sits apart from AVX's.
    // Reporting FMA on a host where AVX is masked off would send callers down a
    // path whose registers the OS does not preserve.
    const ScopedForceIsa clear(nullptr);
    const IsaFeatures f = detect_isa();
    if (!f.avx) {
        EXPECT_FALSE(f.fma);
    }
}

TEST(SimdIsaOsEnablement, ForceIsaOnlyNarrows) {
    const IsaFeatures detected = [] {
        const ScopedForceIsa clear(nullptr);
        return detect_isa();
    }();

    struct Level {
        const char* name;
        bool IsaFeatures::* highest_allowed;
    };

    // Each forced level must leave the detected set unchanged or smaller, never
    // larger: a ceiling that could raise the level would reintroduce the fault.
    for (const char* level : {"avx512", "avx2", "avx", "sse41", "sse2", "scalar"}) {
        const ScopedForceIsa forced(level);
        const IsaFeatures f = detect_isa();
        SCOPED_TRACE(level);
        expect_consistent(f);
        EXPECT_FALSE(f.sse2 && !detected.sse2);
        EXPECT_FALSE(f.sse41 && !detected.sse41);
        EXPECT_FALSE(f.avx && !detected.avx);
        EXPECT_FALSE(f.avx2 && !detected.avx2);
        EXPECT_FALSE(f.fma && !detected.fma);
        EXPECT_FALSE(f.avx512f && !detected.avx512f);
    }
}

TEST(SimdIsaOsEnablement, ForceIsaCeilingsClearEverythingAbove) {
    {
        const ScopedForceIsa forced("scalar");
        const IsaFeatures f = detect_isa();
        EXPECT_FALSE(f.sse2);
        EXPECT_FALSE(f.avx512f);
        EXPECT_EQ(isa_summary(f), "scalar");
    }
    {
        const ScopedForceIsa forced("avx2");
        EXPECT_FALSE(detect_isa().avx512f);
    }
    {
        const ScopedForceIsa forced("avx");
        const IsaFeatures f = detect_isa();
        EXPECT_FALSE(f.avx2);
        EXPECT_FALSE(f.fma);
    }
    {
        const ScopedForceIsa forced("sse2");
        const IsaFeatures f = detect_isa();
        EXPECT_FALSE(f.sse41);
        EXPECT_FALSE(f.avx);
    }
}

TEST(SimdIsaOsEnablement, UnrecognisedForceIsaDoesNotRaiseTheCeiling) {
    const IsaFeatures detected = [] {
        const ScopedForceIsa clear(nullptr);
        return detect_isa();
    }();
    const ScopedForceIsa forced("definitely-not-an-isa");
    const IsaFeatures f = detect_isa();
    // An unrecognised value is ignored rather than treated as "everything".
    EXPECT_EQ(f.avx512f, detected.avx512f);
    EXPECT_EQ(f.avx2, detected.avx2);
    EXPECT_EQ(f.sse2, detected.sse2);
}

TEST(SimdIsaOsEnablement, EmptyForceIsaIsTreatedAsUnset) {
    const IsaFeatures detected = [] {
        const ScopedForceIsa clear(nullptr);
        return detect_isa();
    }();
    const ScopedForceIsa forced("");
    EXPECT_EQ(detect_isa().avx512f, detected.avx512f);
}
