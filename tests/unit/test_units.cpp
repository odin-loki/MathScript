// Tests for the compile-time SI unit-dimension system (ms::core::TypedScalar).
// This is a separate, additive system alongside the runtime ms::Scalar class
// (covered by test_scalar_type.cpp); these tests do not touch ms::Scalar.

#include <gtest/gtest.h>
#include "ms/core/units.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace {

// SFINAE-based "can this expression compile" detection, used below to prove
// that TypedScalar rejects mismatched-dimension addition/subtraction at
// compile time. (A `requires(...) { a + b; }` C++20 requires-expression
// would express the same intent, but this project's MSVC toolchain was
// found to emit a hard compiler error -- rather than simply treating the
// requirement as unsatisfied -- for this particular shape of expression, so
// the classic std::void_t detection idiom is used instead for robustness.)
template <typename A, typename B, typename = void>
struct CanAdd : std::false_type {};
template <typename A, typename B>
struct CanAdd<A, B, std::void_t<decltype(std::declval<A>() + std::declval<B>())>>
    : std::true_type {};

template <typename A, typename B, typename = void>
struct CanSubtract : std::false_type {};
template <typename A, typename B>
struct CanSubtract<A, B, std::void_t<decltype(std::declval<A>() - std::declval<B>())>>
    : std::true_type {};

} // namespace

using namespace ms::core;

// --- 1. Basic construction and value retrieval --------------------------

TEST(UnitsTest, metres_construction_and_value) {
    constexpr Metres m(5.0);
    EXPECT_DOUBLE_EQ(m.value(), 5.0);
}

TEST(UnitsTest, kilograms_construction_and_value) {
    constexpr Kilograms kg(2.5);
    EXPECT_DOUBLE_EQ(kg.value(), 2.5);
}

TEST(UnitsTest, seconds_construction_and_value) {
    constexpr Seconds s(10.0);
    EXPECT_DOUBLE_EQ(s.value(), 10.0);
}

TEST(UnitsTest, base_unit_dimensions_are_distinct) {
    static_assert(Metres::units() == Units{.m = 1});
    static_assert(Kilograms::units() == Units{.kg = 1});
    static_assert(Seconds::units() == Units{.s = 1});
    static_assert(!(Metres::units() == Kilograms::units()));
}

// --- 2. Multiplication combining dimensions (F = m*a) --------------------

TEST(UnitsTest, multiplication_combines_dimensions_to_force) {
    constexpr MetresPerSecondSquared accel(2.0);
    constexpr Kilograms mass(3.0);
    constexpr auto force = accel * mass;

    static_assert(std::is_same_v<decltype(force), const Newtons>);
    static_assert(decltype(force)::units() == Newtons::units());
    EXPECT_DOUBLE_EQ(force.value(), 6.0);
}

TEST(UnitsTest, multiplication_exponents_add_elementwise) {
    static_assert((Metres::units() * Seconds::units()) == Units{.m = 1, .s = 1});
}

// --- 3. Division producing correctly inverted dimensions (velocity) ------

TEST(UnitsTest, division_produces_velocity_dimension) {
    constexpr Metres distance(9.0);
    constexpr Seconds time(3.0);
    constexpr auto velocity = distance / time;

    static_assert(std::is_same_v<decltype(velocity), const MetresPerSecond>);
    static_assert(decltype(velocity)::units() == (Units{.m = 1, .s = -1}));
    EXPECT_DOUBLE_EQ(velocity.value(), 3.0);
}

TEST(UnitsTest, division_exponents_subtract_elementwise) {
    static_assert((Metres::units() / Seconds::units()) == Units{.m = 1, .s = -1});
    static_assert((Seconds::units() / Seconds::units()) == Units{});
}

// --- 4. Compile-time rejection of mismatched-dimension addition ---------

static_assert(!CanAdd<Metres, Kilograms>::value,
              "adding Metres to Kilograms must be a compile error");
static_assert(!CanSubtract<Seconds, Newtons>::value,
              "subtracting Newtons from Seconds must be a compile error");
static_assert(CanAdd<Metres, Metres>::value,
              "adding two Metres values must compile");

TEST(UnitsTest, mismatched_dimension_addition_rejected_at_compile_time) {
    // The static_asserts above are the actual test; this TEST body just
    // gives the negative compile-time checks a home in the gtest output.
    SUCCEED();
}

// --- 5. Same-dimension addition/subtraction -------------------------------

TEST(UnitsTest, same_dimension_addition) {
    constexpr Metres a(1.5);
    constexpr Metres b(2.5);
    constexpr auto c = a + b;
    static_assert(std::is_same_v<decltype(c), const Metres>);
    EXPECT_DOUBLE_EQ(c.value(), 4.0);
}

TEST(UnitsTest, same_dimension_subtraction) {
    constexpr Seconds a(5.0);
    constexpr Seconds b(2.0);
    constexpr auto c = a - b;
    static_assert(std::is_same_v<decltype(c), const Seconds>);
    EXPECT_DOUBLE_EQ(c.value(), 3.0);
}

TEST(UnitsTest, derived_unit_aliases_have_expected_dimensions) {
    static_assert(Newtons::units() == (Units{.m = 1, .kg = 1, .s = -2}));
    static_assert(Joules::units() == (Units{.m = 2, .kg = 1, .s = -2}));
    static_assert(Watts::units() == (Units{.m = 2, .kg = 1, .s = -3}));
    static_assert(Pascals::units() == (Units{.m = -1, .kg = 1, .s = -2}));
    static_assert(Hertz::units() == Units{.s = -1});
}

// --- 6. is_dimensionless -------------------------------------------------

// A manually-constructed all-zero-exponent Units value is dimensionless...
static_assert(is_dimensionless(Units{}));
// ...and so is any other all-zero Units value spelled out explicitly.
static_assert(is_dimensionless(Units{.m = 0, .kg = 0, .s = 0, .A = 0, .K = 0, .mol = 0, .cd = 0}));
// ...but any single nonzero exponent is not.
static_assert(!is_dimensionless(Metres::units()));
static_assert(!is_dimensionless(Kilograms::units()));
static_assert(!is_dimensionless(Newtons::units()));

TEST(UnitsTest, is_dimensionless_units_value) {
    // The static_asserts above are the compile-time core of this test; this
    // TEST body gives them a home in the gtest output and additionally
    // exercises is_dimensionless() as an ordinary runtime call.
    constexpr Units dimensionless{};
    EXPECT_TRUE(is_dimensionless(dimensionless));
    EXPECT_FALSE(is_dimensionless(Metres::units()));
}

TEST(UnitsTest, is_dimensionless_true_for_ratio_of_equal_units) {
    // Metres / Metres already produces a Units{} result (see
    // division_exponents_subtract_elementwise above for the Seconds/Seconds
    // analogue); is_dimensionless must recognise that ratio as dimensionless.
    constexpr Metres a(6.0);
    constexpr Metres b(2.0);
    constexpr auto ratio = a / b;

    static_assert(is_dimensionless(decltype(ratio)::units()));
    static_assert(is_dimensionless(ratio));
    EXPECT_TRUE(is_dimensionless(ratio));
    EXPECT_DOUBLE_EQ(ratio.value(), 3.0);
}

TEST(UnitsTest, is_dimensionless_false_for_ratio_of_different_units) {
    // Newtons / Seconds is not dimensionless (it's kg*m/s^3), so
    // is_dimensionless must correctly reject it too.
    constexpr Newtons f(10.0);
    constexpr Seconds t(2.0);
    constexpr auto rate = f / t;

    static_assert(!is_dimensionless(rate));
    EXPECT_FALSE(is_dimensionless(rate));
}

// --- 7. sqrt halves dimension exponents -----------------------------------

TEST(UnitsTest, sqrt_of_squared_metres_recovers_metres_units) {
    // Squaring a Metres value via the existing operator* produces a
    // TypedScalar with Units{.m = 2}; sqrt should exactly invert that.
    constexpr Metres side(4.0);
    constexpr auto area = side * side;
    static_assert(decltype(area)::units() == Units{.m = 2});

    // sqrt()'s numeric part goes through std::sqrt(), which this toolchain
    // does not guarantee to be usable in a constant expression, so `length`
    // itself is not declared `constexpr` -- but its *type* (and therefore
    // its units) is still fully determined at compile time, which is what
    // decltype() below checks.
    const auto length = sqrt(area);
    static_assert(std::is_same_v<decltype(length), const Metres>);
    static_assert(decltype(length)::units() == Metres::units());
}

TEST(UnitsTest, sqrt_runtime_value_is_correct) {
    constexpr Metres side(4.0);
    constexpr auto area = side * side;  // value 16.0, units m^2
    EXPECT_DOUBLE_EQ(area.value(), 16.0);

    const auto length = sqrt(area);
    EXPECT_DOUBLE_EQ(length.value(), 4.0);
}

TEST(UnitsTest, sqrt_of_squared_seconds_recovers_seconds_units) {
    constexpr Seconds t(3.0);
    constexpr auto t_squared = t * t;
    static_assert(t_squared.units() == Units{.s = 2});

    const auto sqrt_t_squared = sqrt(t_squared);
    static_assert(std::is_same_v<decltype(sqrt_t_squared), const Seconds>);
    EXPECT_DOUBLE_EQ(sqrt_t_squared.value(), 3.0);
}

TEST(UnitsTest, sqrt_of_dimensionless_quantity_stays_dimensionless) {
    // A dimensionless ratio's units are all zero, hence trivially even, so
    // sqrt() must compile for it and produce another dimensionless result.
    constexpr Metres a(9.0);
    constexpr Metres b(1.0);
    constexpr auto ratio = a / b;  // 9.0, dimensionless
    static_assert(is_dimensionless(ratio));

    const auto root = sqrt(ratio);
    static_assert(std::is_same_v<decltype(root), const decltype(ratio)>);
    EXPECT_TRUE(is_dimensionless(root));
    EXPECT_DOUBLE_EQ(root.value(), 3.0);
}

TEST(UnitsTest, sqrt_physically_meaningful_velocity_squared_example) {
    // "energy / mass = velocity^2" dimensionally: Joules / Kilograms has
    // units (m^2 kg s^-2) / kg = m^2 s^-2, i.e. MetresPerSecond squared.
    // sqrt() of that should have exactly the MetresPerSecond dimension.
    constexpr Joules energy(50.0);
    constexpr Kilograms mass(2.0);
    constexpr auto velocity_squared = energy / mass;
    static_assert(decltype(velocity_squared)::units() == (Units{.m = 2, .s = -2}));

    const auto speed = sqrt(velocity_squared);
    static_assert(std::is_same_v<decltype(speed), const MetresPerSecond>);
    EXPECT_DOUBLE_EQ(speed.value(), std::sqrt(25.0));
}

// Note: sqrt() of an odd-exponent quantity is a compile error by design,
// e.g. the following would fail to compile with a static_assert if
// uncommented, since Metres has exponent m^1 (odd):
//
//   constexpr Metres m(4.0);
//   constexpr auto bad = sqrt(m);  // static_assert fires: "sqrt() requires
//                                  // every dimension exponent to be even..."
//
// Likewise for e.g. sqrt(Newtons_value), since Newtons has exponent s^-2 but
// also m^1 (odd) and kg^1 (odd).
