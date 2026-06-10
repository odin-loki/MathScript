#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <span>

#include "ms/simd/simd.hpp"
#include "ms/simd/isa.hpp"

using namespace ms::simd;

// ---------------------------------------------------------------------------
// ISA detection
// ---------------------------------------------------------------------------

TEST(SimdVectorOpsTest, detect_isa_and_summary) {
    const auto info = dispatch_info();
    // isa_summary accepts the IsaFeatures struct
    const auto summary = isa_summary(info.isa);
    EXPECT_FALSE(summary.empty());
}

TEST(SimdVectorOpsTest, detect_isa_sse2_always_true) {
    // On any x86-64 target SSE2 is baseline
    const auto info = dispatch_info();
    EXPECT_TRUE(info.isa.sse2);
}

// ---------------------------------------------------------------------------
// add / mul
// ---------------------------------------------------------------------------

TEST(SimdVectorOpsTest, add_basic) {
    constexpr size_t n = 8;
    std::vector<double> a(n, 1.0), b(n, 2.0), out(n, 0.0);
    add(std::span<const double>{a}, std::span<const double>{b}, std::span<double>{out});
    for (size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(out[i], 3.0);
}

TEST(SimdVectorOpsTest, mul_basic) {
    constexpr size_t n = 8;
    std::vector<double> a(n, 3.0), b(n, 4.0), out(n, 0.0);
    mul(std::span<const double>{a}, std::span<const double>{b}, std::span<double>{out});
    for (size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(out[i], 12.0);
}

TEST(SimdVectorOpsTest, add_empty_does_not_crash) {
    std::vector<double> a, b, out;
    add(std::span<const double>{a}, std::span<const double>{b}, std::span<double>{out});
}

TEST(SimdVectorOpsTest, mul_identity_element) {
    constexpr size_t n = 5;
    std::vector<double> a(n), ones(n, 1.0), out(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        a[i] = static_cast<double>(i + 1);
    mul(std::span<const double>{a}, std::span<const double>{ones}, std::span<double>{out});
    for (size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(out[i], a[i]);
}

// ---------------------------------------------------------------------------
// scale
// ---------------------------------------------------------------------------

TEST(SimdVectorOpsTest, scale_by_two) {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    std::vector<double> out(4, 0.0);
    scale(2.0, std::span<const double>{x}, std::span<double>{out});
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_DOUBLE_EQ(out[i], 2.0 * x[i]);
}

// ---------------------------------------------------------------------------
// dot
// ---------------------------------------------------------------------------

TEST(SimdVectorOpsTest, dot_orthogonal_vectors) {
    std::vector<double> a{1.0, 0.0, 0.0};
    std::vector<double> b{0.0, 1.0, 0.0};
    EXPECT_DOUBLE_EQ(dot(std::span<const double>{a}, std::span<const double>{b}), 0.0);
}

TEST(SimdVectorOpsTest, dot_matches_scalar) {
    std::vector<double> a{1.0, 2.0, 3.0};
    std::vector<double> b{4.0, 5.0, 6.0};
    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_DOUBLE_EQ(dot(std::span<const double>{a}, std::span<const double>{b}), 32.0);
}
