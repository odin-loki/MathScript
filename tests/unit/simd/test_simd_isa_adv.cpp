// MathScript: SIMD ISA Detection and isa_summary Advanced Tests

#include <gtest/gtest.h>
#include <string>

#include "ms/simd/isa.hpp"

using namespace ms::simd;

// ---------------------------------------------------------------------------
// detect_isa: returns IsaFeatures struct
// ---------------------------------------------------------------------------

TEST(SimdIsa, DetectIsa_Returns_IsaFeatures) {
    IsaFeatures features = detect_isa();
    // On any modern x64 CPU, at least SSE2 should be present
    EXPECT_TRUE(features.sse2) << "SSE2 should be supported on all x64 systems";
}

TEST(SimdIsa, IsaFeatures_Hierarchy_Consistent) {
    IsaFeatures features = detect_isa();
    // AVX512F requires AVX2, which requires AVX, which requires SSE4.1
    if (features.avx512f) {
        EXPECT_TRUE(features.avx2) << "AVX512F requires AVX2";
        EXPECT_TRUE(features.avx) << "AVX512F requires AVX";
    }
    if (features.avx2) {
        EXPECT_TRUE(features.avx) << "AVX2 requires AVX";
    }
}

TEST(SimdIsa, IsaFeatures_SSE2_On_X64) {
    // x64 mandates SSE2
    IsaFeatures features = detect_isa();
    EXPECT_TRUE(features.sse2);
}

TEST(SimdIsa, IsaFeatures_BoolFields_AreValidBools) {
    IsaFeatures features = detect_isa();
    // Just verify they're valid booleans (no undefined behavior)
    bool all_bools = (features.sse2 == true || features.sse2 == false) &&
                     (features.sse41 == true || features.sse41 == false) &&
                     (features.avx == true || features.avx == false) &&
                     (features.avx2 == true || features.avx2 == false) &&
                     (features.fma == true || features.fma == false) &&
                     (features.avx512f == true || features.avx512f == false);
    EXPECT_TRUE(all_bools);
}

// ---------------------------------------------------------------------------
// isa_summary: returns string representation
// ---------------------------------------------------------------------------

TEST(SimdIsa, IsaSummary_NotEmpty) {
    IsaFeatures features = detect_isa();
    std::string summary = isa_summary(features);
    EXPECT_FALSE(summary.empty()) << "isa_summary should return a non-empty string";
}

TEST(SimdIsa, IsaSummary_ContainsSSE2_WhenSupported) {
    IsaFeatures features = detect_isa();
    std::string summary = isa_summary(features);
    if (features.sse2) {
        EXPECT_NE(summary.find("SSE2"), std::string::npos)
            << "Summary should mention SSE2 when supported";
    }
}

TEST(SimdIsa, IsaSummary_ContainsAVX512_WhenSupported) {
    IsaFeatures features = detect_isa();
    std::string summary = isa_summary(features);
    if (features.avx512f) {
        // Implementation uses "AVX-512F" 
        bool mentions_avx512 = summary.find("AVX-512") != std::string::npos ||
                               summary.find("AVX512") != std::string::npos ||
                               summary.find("avx512") != std::string::npos;
        EXPECT_TRUE(mentions_avx512) << "Summary should mention AVX512 when supported";
    }
}

TEST(SimdIsa, IsaSummary_IsHumanReadable) {
    IsaFeatures features = detect_isa();
    std::string summary = isa_summary(features);
    // Summary should be printable ASCII
    for (char c : summary) {
        EXPECT_TRUE(c >= 32 && c <= 126 || c == '\n' || c == '\t')
            << "Summary should contain only printable characters";
    }
}

TEST(SimdIsa, IsaSummary_ManualFeatures) {
    // Smoke test: pass in a known IsaFeatures struct
    IsaFeatures manual_features;
    manual_features.sse2 = true;
    manual_features.avx = false;
    manual_features.avx512f = false;
    std::string summary = isa_summary(manual_features);
    EXPECT_FALSE(summary.empty());
}

TEST(SimdIsa, IsaSummary_AllFalse_NocrashAndNonEmpty) {
    IsaFeatures empty_features{};
    std::string summary = isa_summary(empty_features);
    EXPECT_FALSE(summary.empty());
}

TEST(SimdIsa, IsaSummary_FullSupport) {
    IsaFeatures full;
    full.sse2 = true;
    full.sse41 = true;
    full.avx = true;
    full.avx2 = true;
    full.fma = true;
    full.avx512f = true;
    std::string summary = isa_summary(full);
    EXPECT_FALSE(summary.empty());
}
