#include "ms/simd/isa.hpp"
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

} // namespace

IsaFeatures detect_isa() {
    IsaFeatures f;
    int info[4] = {0, 0, 0, 0};
    cpuid(info, 0);
    if (info[0] == 0) {
        return f;
    }

    cpuid(info, 1);
    f.sse2 = (info[3] & (1 << 26)) != 0;
    f.sse41 = (info[2] & (1 << 19)) != 0;
    f.avx = (info[2] & (1 << 28)) != 0;
    f.fma = (info[2] & (1 << 12)) != 0;

    cpuid(info, 7, 0);
    f.avx2 = (info[1] & (1 << 5)) != 0;
    f.avx512f = (info[1] & (1 << 16)) != 0;
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
