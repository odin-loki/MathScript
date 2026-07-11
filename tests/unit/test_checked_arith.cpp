#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "ms/core/checked_arith.hpp"

using namespace ms;

namespace {

constexpr int8_t kInt8Max = std::numeric_limits<int8_t>::max();
constexpr int8_t kInt8Min = std::numeric_limits<int8_t>::min();
constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();

template<typename T>
void expect_overflow(const Result<T>& result) {
    ASSERT_FALSE(result.has_value());
    ASSERT_TRUE(std::holds_alternative<OverflowError>(result.error()));
}

template<typename T>
void expect_domain(const Result<T>& result) {
    ASSERT_FALSE(result.has_value());
    ASSERT_TRUE(std::holds_alternative<DomainError>(result.error()));
}

} // namespace

// ---------------------------------------------------------------------------
// checked_add
// ---------------------------------------------------------------------------

TEST(CheckedArith, CheckedAddInt8Success) {
    const auto r = checked_add(int8_t{10}, int8_t{20});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, int8_t{30});
}

TEST(CheckedArith, CheckedAddInt8PositiveOverflow) {
    expect_overflow(checked_add(kInt8Max, int8_t{1}));
}

TEST(CheckedArith, CheckedAddInt8NegativeOverflow) {
    expect_overflow(checked_add(kInt8Min, int8_t{-1}));
}

TEST(CheckedArith, CheckedAddInt32Success) {
    const auto r = checked_add(int32_t{100}, int32_t{200});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 300);
}

TEST(CheckedArith, CheckedAddInt32NegativeOverflow) {
    expect_overflow(checked_add(kInt32Min, int32_t{-1}));
}

TEST(CheckedArith, CheckedAddUint8Success) {
    const auto r = checked_add(uint8_t{100}, uint8_t{50});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, uint8_t{150});
}

TEST(CheckedArith, CheckedAddUint8Overflow) {
    expect_overflow(checked_add(uint8_t{255}, uint8_t{1}));
}

TEST(CheckedArith, CheckedAddInt64Success) {
    const auto r = checked_add(int64_t{1'000'000'000}, int64_t{2'000'000'000});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 3'000'000'000);
}

// ---------------------------------------------------------------------------
// checked_sub / checked_mul
// ---------------------------------------------------------------------------

TEST(CheckedArith, CheckedSubInt32Success) {
    const auto r = checked_sub(int32_t{50}, int32_t{20});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 30);
}

TEST(CheckedArith, CheckedSubInt32NegativeOverflow) {
    expect_overflow(checked_sub(kInt32Min, int32_t{1}));
}

TEST(CheckedArith, CheckedSubUint32Underflow) {
    expect_overflow(checked_sub(uint32_t{0}, uint32_t{1}));
}

TEST(CheckedArith, CheckedMulInt32Overflow) {
    expect_overflow(checked_mul(int32_t{1'000'000}, int32_t{1'000'000}));
}

TEST(CheckedArith, CheckedMulInt32NegativeOverflow) {
    expect_overflow(checked_mul(kInt32Min, int32_t{2}));
}

TEST(CheckedArith, CheckedMulUint32Success) {
    const auto r = checked_mul(uint32_t{1000}, uint32_t{1000});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 1'000'000u);
}

// ---------------------------------------------------------------------------
// checked_div / checked_mod
// ---------------------------------------------------------------------------

TEST(CheckedArith, CheckedDivInt32Success) {
    const auto r = checked_div(int32_t{20}, int32_t{4});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 5);
}

TEST(CheckedArith, CheckedDivByZero) {
    expect_domain(checked_div(int32_t{1}, int32_t{0}));
}

TEST(CheckedArith, CheckedDivIntMinOverNegOne) {
    expect_overflow(checked_div(kInt32Min, int32_t{-1}));
}

TEST(CheckedArith, CheckedModSuccess) {
    const auto r = checked_mod(int32_t{10}, int32_t{3});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 1);
}

TEST(CheckedArith, CheckedModByZero) {
    expect_domain(checked_mod(int32_t{10}, int32_t{0}));
}

// ---------------------------------------------------------------------------
// checked_neg / checked_abs / checked_pow
// ---------------------------------------------------------------------------

TEST(CheckedArith, CheckedNegSuccess) {
    const auto r = checked_neg(int32_t{42});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, -42);
}

TEST(CheckedArith, CheckedNegInt32Min) {
    expect_overflow(checked_neg(kInt32Min));
}

TEST(CheckedArith, CheckedAbsSuccess) {
    const auto r = checked_abs(int32_t{-42});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 42);
}

TEST(CheckedArith, CheckedAbsInt32Min) {
    expect_overflow(checked_abs(kInt32Min));
}

TEST(CheckedArith, CheckedPowInt8Overflow) {
    expect_overflow(checked_pow(int8_t{2}, 10u));
}

TEST(CheckedArith, CheckedPowInt8Success) {
    const auto r = checked_pow(int8_t{2}, 3u);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, int8_t{8});
}

// ---------------------------------------------------------------------------
// saturating arithmetic
// ---------------------------------------------------------------------------

TEST(CheckedArith, SaturatingAddSignedPositiveOverflow) {
    EXPECT_EQ(saturating_add(kInt8Max, int8_t{1}), kInt8Max);
}

TEST(CheckedArith, SaturatingAddSignedNegativeOverflow) {
    EXPECT_EQ(saturating_add(kInt8Min, int8_t{-1}), kInt8Min);
}

TEST(CheckedArith, SaturatingAddUnsignedOverflow) {
    EXPECT_EQ(saturating_add(uint8_t{255}, uint8_t{1}), uint8_t{255});
}

TEST(CheckedArith, SaturatingSubSignedUnderflow) {
    EXPECT_EQ(saturating_sub(kInt8Min, int8_t{1}), kInt8Min);
}

TEST(CheckedArith, SaturatingSubUnsignedUnderflow) {
    EXPECT_EQ(saturating_sub(uint8_t{0}, uint8_t{1}), uint8_t{0});
}

TEST(CheckedArith, SaturatingMulSignedPositiveOverflow) {
    EXPECT_EQ(saturating_mul(int8_t{64}, int8_t{4}), kInt8Max);
}

TEST(CheckedArith, SaturatingMulSignedNegativeOverflow) {
    EXPECT_EQ(saturating_mul(int8_t{-64}, int8_t{4}), kInt8Min);
}

TEST(CheckedArith, SaturatingMulUnsignedOverflow) {
    EXPECT_EQ(saturating_mul(uint8_t{255}, uint8_t{2}), uint8_t{255});
}

// ---------------------------------------------------------------------------
// wrapping arithmetic
// ---------------------------------------------------------------------------

TEST(CheckedArith, WrappingAddInt8) {
    EXPECT_EQ(wrapping_add(kInt8Max, int8_t{1}), kInt8Min);
}

TEST(CheckedArith, WrappingAddUint8) {
    EXPECT_EQ(wrapping_add(uint8_t{255}, uint8_t{1}), uint8_t{0});
}

TEST(CheckedArith, WrappingSubInt8) {
    EXPECT_EQ(wrapping_sub(kInt8Min, int8_t{1}), kInt8Max);
}

TEST(CheckedArith, WrappingMulInt8) {
    EXPECT_EQ(wrapping_mul(int8_t{32}, int8_t{4}), kInt8Min);
}

// ---------------------------------------------------------------------------
// float introspection
// ---------------------------------------------------------------------------

TEST(CheckedArith, FloatClassification) {
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const double inf_val = std::numeric_limits<double>::infinity();
    const double normal_val = 1.234;

    EXPECT_TRUE(is_nan(nan_val));
    EXPECT_FALSE(is_nan(normal_val));

    EXPECT_TRUE(is_inf(inf_val));
    EXPECT_FALSE(is_inf(normal_val));

    EXPECT_FALSE(is_finite(nan_val));
    EXPECT_FALSE(is_finite(inf_val));
    EXPECT_TRUE(is_finite(normal_val));

    EXPECT_FALSE(is_normal(nan_val));
    EXPECT_FALSE(is_normal(inf_val));
    EXPECT_TRUE(is_normal(normal_val));

    EXPECT_TRUE(signbit(-1.0));
    EXPECT_FALSE(signbit(1.0));
}

TEST(CheckedArith, FloatLimitsAndUlp) {
    EXPECT_EQ(eps<double>(), std::numeric_limits<double>::epsilon());
    EXPECT_EQ(huge<float>(), std::numeric_limits<float>::max());
    EXPECT_EQ(tiny<double>(), std::numeric_limits<double>::min());

    const double x = 1.0;
    EXPECT_NE(x + ulp(x), x);
}

// ---------------------------------------------------------------------------
// narrow / widen
// ---------------------------------------------------------------------------

TEST(CheckedArith, NarrowSuccess) {
    const auto r = narrow<int8_t>(int16_t{42});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, int8_t{42});
}

TEST(CheckedArith, NarrowOverflow) {
    expect_overflow(narrow<int8_t>(int16_t{200}));
}

TEST(CheckedArith, WidenUnchecked) {
    EXPECT_EQ(widen<int32_t>(int8_t{42}), 42);
}
