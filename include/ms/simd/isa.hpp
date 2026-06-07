#pragma once

#include <cstdint>
#include <string>

namespace ms::simd {

struct IsaFeatures {
    bool sse2 = false;
    bool sse41 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
};

IsaFeatures detect_isa();
std::string isa_summary(const IsaFeatures& features);

} // namespace ms::simd
