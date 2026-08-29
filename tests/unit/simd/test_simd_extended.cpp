#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "ms/simd/simd.hpp"

using namespace ms::simd;

// ---------------------------------------------------------------------------
// simd::add
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, Add_Identity_AddZeroVector) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> zeros(a.size(), 0.0);
    std::vector<double> out(a.size());
    add(a, zeros, out);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(out[i], a[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Add_Commutativity) {
    const std::vector<double> a{3.0, 1.0, 4.0, 1.0, 5.0};
    const std::vector<double> b{2.0, 7.0, 1.0, 8.0, 2.0};
    std::vector<double> ab(a.size()), ba(a.size());
    add(a, b, ab);
    add(b, a, ba);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(ab[i], ba[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Add_NegativeValues) {
    const std::vector<double> a{-1.0, -2.0, -3.0, -4.0};
    const std::vector<double> b{ 1.0,  2.0,  3.0,  4.0};
    std::vector<double> out(4);
    add(a, b, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], 0.0, 1e-12);
    }
}

TEST(SimdExtendedTest, Add_SingleElement) {
    const std::vector<double> a{7.5};
    const std::vector<double> b{2.5};
    std::vector<double> out(1);
    add(a, b, out);
    EXPECT_NEAR(out[0], 10.0, 1e-12);
}

TEST(SimdExtendedTest, Add_LargeVector_Size64) {
    std::vector<double> a(64), b(64), out(64);
    std::iota(a.begin(), a.end(), 1.0);
    std::iota(b.begin(), b.end(), -31.0);
    add(a, b, out);
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(out[i], a[i] + b[i], 1e-12);
    }
}

// ---------------------------------------------------------------------------
// simd::mul
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, Mul_ByOnesPreservesInput) {
    const std::vector<double> a{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    const std::vector<double> ones(a.size(), 1.0);
    std::vector<double> out(a.size());
    mul(a, ones, out);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(out[i], a[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Mul_ByZeroGivesAllZeros) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<double> zeros(a.size(), 0.0);
    std::vector<double> out(a.size());
    mul(a, zeros, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], 0.0, 1e-12);
    }
}

TEST(SimdExtendedTest, Mul_NegativeTimesNegativeIsPositive) {
    const std::vector<double> a{-1.0, -2.0, -3.0, -4.0};
    const std::vector<double> b{-1.0, -2.0, -3.0, -4.0};
    std::vector<double> out(4);
    mul(a, b, out);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GT(out[i], 0.0);
        EXPECT_NEAR(out[i], a[i] * b[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Mul_ElementWiseKnownResult) {
    const std::vector<double> a{2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b{3.0, 4.0, 5.0, 6.0};
    std::vector<double> out(4);
    mul(a, b, out);
    EXPECT_NEAR(out[0], 6.0,  1e-12);
    EXPECT_NEAR(out[1], 12.0, 1e-12);
    EXPECT_NEAR(out[2], 20.0, 1e-12);
    EXPECT_NEAR(out[3], 30.0, 1e-12);
}

TEST(SimdExtendedTest, Mul_LargeVector_Size64) {
    std::vector<double> a(64), b(64), out(64);
    std::iota(a.begin(), a.end(), 1.0);
    std::iota(b.begin(), b.end(), 0.5);
    mul(a, b, out);
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(out[i], a[i] * b[i], 1e-10);
    }
}

// ---------------------------------------------------------------------------
// simd::scale
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, Scale_ByZeroGivesAllZeros) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> out(x.size());
    scale(0.0, x, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], 0.0, 1e-12);
    }
}

TEST(SimdExtendedTest, Scale_ByOnePreservesInput) {
    const std::vector<double> x{7.0, -3.0, 0.5, 100.0};
    std::vector<double> out(x.size());
    scale(1.0, x, out);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], x[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Scale_ByTwoDoubles) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> out(x.size());
    scale(2.0, x, out);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], 2.0 * x[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Scale_ByNegativeFlipsSigns) {
    const std::vector<double> x{1.0, -2.0, 3.0, -4.0};
    std::vector<double> out(x.size());
    scale(-1.0, x, out);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], -x[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Scale_LargeVector_Size512) {
    std::vector<double> x(512), out(512);
    std::iota(x.begin(), x.end(), 0.0);
    const double alpha = 3.14;
    scale(alpha, x, out);
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_NEAR(out[i], alpha * x[i], 1e-10);
    }
}

// ---------------------------------------------------------------------------
// simd::axpy
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, Axpy_ZeroAlphaLeavesYUnchanged) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> y{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    const std::vector<double> y_orig = y;
    axpy(0.0, x, y);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], y_orig[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Axpy_NegativeAlphaSubtracts) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    std::vector<double> y{5.0, 6.0, 7.0, 8.0};
    const std::vector<double> y_orig = y;
    axpy(-1.0, x, y);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], y_orig[i] - x[i], 1e-12);
    }
}

TEST(SimdExtendedTest, Axpy_UnitAlphaAddsX) {
    const std::vector<double> x{2.0, 4.0, 6.0, 8.0};
    std::vector<double> y{1.0, 1.0, 1.0, 1.0};
    axpy(1.0, x, y);
    EXPECT_NEAR(y[0], 3.0, 1e-12);
    EXPECT_NEAR(y[1], 5.0, 1e-12);
    EXPECT_NEAR(y[2], 7.0, 1e-12);
    EXPECT_NEAR(y[3], 9.0, 1e-12);
}

TEST(SimdExtendedTest, Axpy_AccumulatesTwoCalls) {
    const std::vector<double> x{1.0, 1.0, 1.0, 1.0};
    std::vector<double> y{0.0, 0.0, 0.0, 0.0};
    axpy(3.0, x, y);
    axpy(2.0, x, y);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], 5.0, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// simd::dot
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, Dot_OrthogonalVectorsIsZero) {
    const std::vector<double> a{1.0, 0.0, 0.0, 0.0};
    const std::vector<double> b{0.0, 1.0, 0.0, 0.0};
    EXPECT_NEAR(dot(a, b), 0.0, 1e-12);
}

TEST(SimdExtendedTest, Dot_CauchySchwarzInequality) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b{5.0, 4.0, 3.0, 2.0, 1.0};
    const double ab  = std::abs(dot(a, b));
    const double na  = std::sqrt(dot(a, a));
    const double nb  = std::sqrt(dot(b, b));
    EXPECT_LE(ab, na * nb + 1e-10);
}

TEST(SimdExtendedTest, Dot_SelfDotEqualsNormSquared) {
    const std::vector<double> a{3.0, 4.0};
    EXPECT_NEAR(dot(a, a), 25.0, 1e-12);
}

TEST(SimdExtendedTest, Dot_SingleElement) {
    const std::vector<double> a{3.0};
    const std::vector<double> b{4.0};
    EXPECT_NEAR(dot(a, b), 12.0, 1e-12);
}

TEST(SimdExtendedTest, Dot_AllNegativesGivesPositiveResult) {
    const std::vector<double> a{-1.0, -2.0, -3.0};
    const std::vector<double> b{-1.0, -2.0, -3.0};
    EXPECT_NEAR(dot(a, b), 14.0, 1e-12);
}

// ---------------------------------------------------------------------------
// simd::exp_map
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, ExpMap_AtZeroIsOne) {
    const std::vector<double> x{0.0, 0.0, 0.0, 0.0};
    std::vector<double> out(4);
    exp_map(x, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], 1.0, 1e-12);
    }
}

TEST(SimdExtendedTest, ExpMap_AtOneIsE) {
    const std::vector<double> x{1.0};
    std::vector<double> out(1);
    exp_map(x, out);
    EXPECT_NEAR(out[0], std::exp(1.0), 1e-9);
}

TEST(SimdExtendedTest, ExpMap_MonotonicallyIncreasing) {
    const std::vector<double> x{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    std::vector<double> out(x.size());
    exp_map(x, out);
    for (size_t i = 1; i < out.size(); ++i) {
        EXPECT_GT(out[i], out[i - 1]);
    }
}

TEST(SimdExtendedTest, ExpMap_NegativeInputGivesReciprocal) {
    const std::vector<double> x{-1.0};
    std::vector<double> out(1);
    exp_map(x, out);
    EXPECT_NEAR(out[0], std::exp(-1.0), 1e-9);
}

TEST(SimdExtendedTest, ExpMap_MatchesStdExpElementWise) {
    const std::vector<double> x{0.5, 1.5, 2.5, 3.5, -0.5, -1.5};
    std::vector<double> out(x.size());
    exp_map(x, out);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(out[i], std::exp(x[i]), 1e-9);
    }
}

// ---------------------------------------------------------------------------
// simd::gemv
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, Gemv_TwoByTwoNonIdentity) {
    // A = [[2, 1], [1, 3]], x = [1, 2]  => y = [4, 7]
    const std::vector<double> A{2.0, 1.0, 1.0, 3.0};
    const std::vector<double> x{1.0, 2.0};
    const auto y = gemv(A, 2, 2, x);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_NEAR(y[0], 4.0, 1e-12);
    EXPECT_NEAR(y[1], 7.0, 1e-12);
}

TEST(SimdExtendedTest, Gemv_ZeroMatrixGivesZeroVector) {
    const std::vector<double> A(9, 0.0);
    const std::vector<double> x{1.0, 2.0, 3.0};
    const auto y = gemv(A, 3, 3, x);
    ASSERT_EQ(y.size(), 3u);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-12);
    }
}

TEST(SimdExtendedTest, Gemv_ThreeByThreeKnownResult) {
    // A = [[1,2,3],[4,5,6],[7,8,9]], x = [1,0,0] => y = [1,4,7]
    const std::vector<double> A{1,2,3, 4,5,6, 7,8,9};
    const std::vector<double> x{1.0, 0.0, 0.0};
    const auto y = gemv(A, 3, 3, x);
    ASSERT_EQ(y.size(), 3u);
    EXPECT_NEAR(y[0], 1.0, 1e-12);
    EXPECT_NEAR(y[1], 4.0, 1e-12);
    EXPECT_NEAR(y[2], 7.0, 1e-12);
}

TEST(SimdExtendedTest, Gemv_RectangularTall_4x2) {
    // A = [[1,0],[0,1],[1,1],[2,3]], x = [2,3]  => y = [2,3,5,13]
    const std::vector<double> A{1,0, 0,1, 1,1, 2,3};
    const std::vector<double> x{2.0, 3.0};
    const auto y = gemv(A, 4, 2, x);
    ASSERT_EQ(y.size(), 4u);
    EXPECT_NEAR(y[0],  2.0, 1e-12);
    EXPECT_NEAR(y[1],  3.0, 1e-12);
    EXPECT_NEAR(y[2],  5.0, 1e-12);
    EXPECT_NEAR(y[3], 13.0, 1e-12);
}

TEST(SimdExtendedTest, Gemv_RectangularWide_2x4) {
    // A = [[1,2,3,4],[5,6,7,8]], x = [1,1,1,1] => y = [10, 26]
    const std::vector<double> A{1,2,3,4, 5,6,7,8};
    const std::vector<double> x{1.0, 1.0, 1.0, 1.0};
    const auto y = gemv(A, 2, 4, x);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_NEAR(y[0], 10.0, 1e-12);
    EXPECT_NEAR(y[1], 26.0, 1e-12);
}

// ---------------------------------------------------------------------------
// dispatch_info
// ---------------------------------------------------------------------------

TEST(SimdExtendedTest, DispatchInfo_KernelIsValidEnumValue) {
    const auto info = dispatch_info();
    const bool valid = (info.active == Kernel::Scalar ||
                        info.active == Kernel::Avx2   ||
                        info.active == Kernel::Avx512);
    EXPECT_TRUE(valid);
}

TEST(SimdExtendedTest, DispatchInfo_IsaStructIsAccessible) {
    const auto info = dispatch_info();
    // Fields must be booleans — just reading them must not crash
    (void)info.isa.sse2;
    (void)info.isa.sse41;
    (void)info.isa.avx;
    (void)info.isa.avx2;
    (void)info.isa.fma;
    (void)info.isa.avx512f;
    SUCCEED();
}

TEST(SimdExtendedTest, DispatchInfo_AvxImpliesAvx2OrScalar) {
    const auto info = dispatch_info();
    if (info.active == Kernel::Avx2) {
        EXPECT_TRUE(info.isa.avx2);
    }
    if (info.active == Kernel::Avx512) {
        EXPECT_TRUE(info.isa.avx512f);
    }
}
