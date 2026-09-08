// Regression tests for ms::narrow (include/ms/core/checked_arith.hpp).
//
// The old implementation converted the destination bound INTO the source type
// before comparing (`x < static_cast<U>(std::numeric_limits<T>::min())`). That
// wraps whenever the bound is not representable in U, so the guard rejected valid
// values and accepted invalid ones:
//   * narrow<int32_t>(uint32_t{0})  -> OverflowError  (0 < 2147483648u)
//   * narrow<uint32_t>(int32_t{5})  -> OverflowError  (5 > (int32_t)4294967295 == -1)
//   * narrow<int64_t>(int8_t{-5})   -> OverflowError  (widening, can never fail)
//   * narrow<int64_t>(9223372036854775808.0) -> accepted, producing INT64_MIN
//   * narrow<float>(1e300)          -> accepted, producing +inf

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "ms/core/checked_arith.hpp"

namespace {

template<typename T, typename U>
void expect_narrow_eq(U x, T expected) {
    const auto r = ms::narrow<T>(x);
    ASSERT_TRUE(r.has_value()) << "narrow rejected a representable value";
    EXPECT_EQ(*r, expected);
}

template<typename T, typename U>
void expect_narrow_overflow(U x) {
    const auto r = ms::narrow<T>(x);
    EXPECT_FALSE(r.has_value()) << "narrow accepted a value outside the destination range";
}

} // namespace

TEST(NarrowContract, WideningConversionsNeverFail) {
    expect_narrow_eq<int64_t, int8_t>(int8_t{-5}, int64_t{-5});
    expect_narrow_eq<int64_t, int8_t>(std::numeric_limits<int8_t>::min(), int64_t{-128});
    expect_narrow_eq<int32_t, int8_t>(std::numeric_limits<int8_t>::min(), int32_t{-128});
    expect_narrow_eq<int64_t, int32_t>(std::numeric_limits<int32_t>::min(),
                                       int64_t{-2147483648LL});
    expect_narrow_eq<int64_t, int16_t>(std::numeric_limits<int16_t>::min(), int64_t{-32768});
    expect_narrow_eq<uint64_t, uint8_t>(std::numeric_limits<uint8_t>::max(), uint64_t{255});
    expect_narrow_eq<uint32_t, uint16_t>(std::numeric_limits<uint16_t>::max(), uint32_t{65535});
}

TEST(NarrowContract, UnsignedSourceToSignedDestinationBoundaries) {
    expect_narrow_eq<int32_t, uint32_t>(uint32_t{0}, int32_t{0});
    expect_narrow_eq<int32_t, uint32_t>(uint32_t{1}, int32_t{1});
    expect_narrow_eq<int32_t, uint32_t>(uint32_t{2147483647u},
                                        std::numeric_limits<int32_t>::max());
    expect_narrow_overflow<int32_t, uint32_t>(uint32_t{2147483648u});
    expect_narrow_overflow<int32_t, uint32_t>(std::numeric_limits<uint32_t>::max());

    expect_narrow_eq<int64_t, uint64_t>(uint64_t{9223372036854775807ull},
                                        std::numeric_limits<int64_t>::max());
    expect_narrow_overflow<int64_t, uint64_t>(uint64_t{9223372036854775808ull});
    expect_narrow_overflow<int64_t, uint64_t>(std::numeric_limits<uint64_t>::max());

    expect_narrow_eq<int8_t, uint64_t>(uint64_t{127}, int8_t{127});
    expect_narrow_overflow<int8_t, uint64_t>(uint64_t{128});
}

TEST(NarrowContract, SignedSourceToUnsignedDestinationBoundaries) {
    expect_narrow_eq<uint32_t, int32_t>(int32_t{0}, uint32_t{0});
    expect_narrow_eq<uint32_t, int32_t>(std::numeric_limits<int32_t>::max(),
                                        uint32_t{2147483647u});
    expect_narrow_overflow<uint32_t, int32_t>(int32_t{-1});
    expect_narrow_overflow<uint32_t, int32_t>(std::numeric_limits<int32_t>::min());

    expect_narrow_eq<uint64_t, int32_t>(int32_t{5}, uint64_t{5});
    expect_narrow_overflow<uint64_t, int32_t>(int32_t{-1});

    expect_narrow_eq<uint64_t, int64_t>(std::numeric_limits<int64_t>::max(),
                                        uint64_t{9223372036854775807ull});
    expect_narrow_overflow<uint64_t, int64_t>(int64_t{-1});

    expect_narrow_eq<uint8_t, int32_t>(int32_t{255}, uint8_t{255});
    expect_narrow_overflow<uint8_t, int32_t>(int32_t{256});
    expect_narrow_overflow<uint8_t, int32_t>(int32_t{-1});
}

TEST(NarrowContract, SameSignednessBoundaries) {
    expect_narrow_eq<int8_t, int16_t>(int16_t{127}, int8_t{127});
    expect_narrow_overflow<int8_t, int16_t>(int16_t{128});
    expect_narrow_eq<int8_t, int16_t>(int16_t{-128}, int8_t{-128});
    expect_narrow_overflow<int8_t, int16_t>(int16_t{-129});

    expect_narrow_eq<uint8_t, uint16_t>(uint16_t{255}, uint8_t{255});
    expect_narrow_overflow<uint8_t, uint16_t>(uint16_t{256});

    expect_narrow_eq<int32_t, int64_t>(int64_t{2147483647LL}, std::numeric_limits<int32_t>::max());
    expect_narrow_overflow<int32_t, int64_t>(int64_t{2147483648LL});
    expect_narrow_eq<int32_t, int64_t>(int64_t{-2147483648LL}, std::numeric_limits<int32_t>::min());
    expect_narrow_overflow<int32_t, int64_t>(int64_t{-2147483649LL});
}

TEST(NarrowContract, FloatingToIntegerBoundaries) {
    // static_cast truncates toward zero; narrow must agree.
    expect_narrow_eq<int32_t, double>(3.9, int32_t{3});
    expect_narrow_eq<int32_t, double>(-3.9, int32_t{-3});
    expect_narrow_eq<uint32_t, double>(-0.5, uint32_t{0});

    expect_narrow_eq<int32_t, double>(2147483647.0, std::numeric_limits<int32_t>::max());
    expect_narrow_overflow<int32_t, double>(2147483648.0);
    expect_narrow_eq<int32_t, double>(-2147483648.0, std::numeric_limits<int32_t>::min());
    expect_narrow_overflow<int32_t, double>(-2147483649.0);

    // 2^63 is exactly representable as a double but is NOT an int64_t. The largest
    // double strictly below it is 2^63 - 1024.
    expect_narrow_eq<int64_t, double>(9223372036854774784.0, int64_t{9223372036854774784LL});
    expect_narrow_overflow<int64_t, double>(9223372036854775808.0);
    expect_narrow_overflow<int64_t, double>(-9223372036854777856.0);
    expect_narrow_eq<int64_t, double>(-9223372036854775808.0,
                                      std::numeric_limits<int64_t>::min());

    // 2^64 likewise for uint64_t.
    expect_narrow_overflow<uint64_t, double>(18446744073709551616.0);
    expect_narrow_overflow<uint64_t, double>(-1.0);

    expect_narrow_overflow<int32_t, double>(std::numeric_limits<double>::quiet_NaN());
    expect_narrow_overflow<int32_t, double>(std::numeric_limits<double>::infinity());
    expect_narrow_overflow<int32_t, double>(-std::numeric_limits<double>::infinity());
}

TEST(NarrowContract, FloatingToFloatingOverflowIsCaught) {
    expect_narrow_overflow<float, double>(1e300);
    expect_narrow_overflow<float, double>(-1e300);
    const auto r = ms::narrow<float>(1.5);
    ASSERT_TRUE(r.has_value());
    EXPECT_FLOAT_EQ(*r, 1.5f);
    const auto widened = ms::narrow<double>(1.5f);
    ASSERT_TRUE(widened.has_value());
    EXPECT_DOUBLE_EQ(*widened, 1.5);
}

TEST(NarrowContract, IntegerToFloatingAlwaysSucceeds) {
    const auto a = ms::narrow<double>(std::numeric_limits<int64_t>::max());
    ASSERT_TRUE(a.has_value());
    EXPECT_DOUBLE_EQ(*a, static_cast<double>(std::numeric_limits<int64_t>::max()));
    const auto b = ms::narrow<float>(std::numeric_limits<uint64_t>::max());
    ASSERT_TRUE(b.has_value());
    EXPECT_TRUE(std::isfinite(*b));
}
