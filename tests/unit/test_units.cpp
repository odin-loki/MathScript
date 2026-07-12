// Tests for the compile-time SI unit-dimension system (ms::core::TypedScalar).
// This is a separate, additive system alongside the runtime ms::Scalar class
// (covered by test_scalar_type.cpp); these tests do not touch ms::Scalar.

#include <gtest/gtest.h>
#include "ms/core/units.hpp"

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
