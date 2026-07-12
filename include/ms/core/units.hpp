// MathScript Compile-Time SI Units
// Zero-overhead dimensional analysis via non-type template parameters

#pragma once

#include <cmath>

namespace ms::core {

/// Exponents of the 7 SI base dimensions (metre, kilogram, second, ampere,
/// kelvin, mole, candela). A `Units` value fully describes the physical
/// dimension of a `TypedScalar<T, U>` quantity, and is combined/compared
/// entirely at compile time -- there is no runtime representation of a
/// dimension anywhere in this system.
///
/// `Units` is a "structural type" (all members public, no virtual
/// functions, a defaulted `operator==`) as required by C++20/23 for use as
/// a non-type template parameter (NTTP), which is what makes it usable
/// directly as `TypedScalar<T, Units{.m = 1}>` below. (The underlying
/// per-dimension arithmetic in `TypedScalar` is deduced through seven
/// plain `int` NTTPs rather than through `Units` itself -- see the
/// `@note` on `TypedScalar` for why.)
struct Units {
    int m = 0;
    int kg = 0;
    int s = 0;
    int A = 0;
    int K = 0;
    int mol = 0;
    int cd = 0;

    constexpr bool operator==(const Units&) const = default;
};

/// Dimension of a product: exponents add element-wise.
constexpr Units operator*(Units a, Units b) {
    return Units{a.m + b.m, a.kg + b.kg, a.s + b.s, a.A + b.A,
                 a.K + b.K, a.mol + b.mol, a.cd + b.cd};
}

/// Dimension of a quotient: exponents subtract element-wise.
constexpr Units operator/(Units a, Units b) {
    return Units{a.m - b.m, a.kg - b.kg, a.s - b.s, a.A - b.A,
                 a.K - b.K, a.mol - b.mol, a.cd - b.cd};
}

/// True iff `u` carries no physical dimension at all, i.e. every one of its
/// seven SI base-dimension exponents is zero. A dimensionless `Units` value
/// typically arises from dividing two quantities of the same dimension
/// (e.g. `Metres{} / Metres{}` yields `Units{}`), and describes pure
/// ratios, counts, and angles (radians/steradians are dimensionless in SI).
///
/// Being a plain `constexpr` function of a structural-type value (rather
/// than, say, a template on seven separate `int` NTTPs), this works
/// uniformly whether `u` is a compile-time constant (e.g. inside a
/// `static_assert`) or only known at runtime.
constexpr bool is_dimensionless(Units u) {
    return u == Units{};
}

/// @cond MS_INTERNAL
namespace detail {

// The real implementation of TypedScalar. It is parameterized on seven
// plain `int` non-type template parameters -- one per SI base dimension --
// rather than on a single `Units` non-type template parameter, because
// this project's MSVC toolchain cannot reliably deduce a class-type
// ("structural type") NTTP from a function-template argument (e.g. when
// overload resolution for `operator*`/`operator/` needs to deduce the
// dimension of each operand from its type): deduction of plain `int` NTTPs
// in the same position is unambiguous and fully supported. `TypedScalar`
// (below) is a public alias template that presents the friendlier
// `Units`-based interface described in the module's design; user code
// never spells out `TypedScalarImpl` directly.
template <typename T, int M, int KG, int S, int A, int K, int MOL, int CD>
class TypedScalarImpl {
public:
    static constexpr Units units() { return Units{M, KG, S, A, K, MOL, CD}; }

    constexpr TypedScalarImpl() = default;
    constexpr explicit TypedScalarImpl(T value) : value_(value) {}

    constexpr T value() const { return value_; }

    /// Same-dimension addition. Only compiles when `other` shares this
    /// exact dimension -- there is no overload accepting a differently
    /// dimensioned quantity, so mismatched-dimension addition is a
    /// compile error rather than a runtime one.
    constexpr TypedScalarImpl operator+(TypedScalarImpl other) const {
        return TypedScalarImpl(value_ + other.value_);
    }

    /// Same-dimension subtraction; see `operator+` for the compile-time
    /// dimension check.
    constexpr TypedScalarImpl operator-(TypedScalarImpl other) const {
        return TypedScalarImpl(value_ - other.value_);
    }

private:
    T value_{};
};

/// Multiplication combines dimensions: each exponent of the result is the
/// sum of the operands' exponents for that dimension.
template <typename T, int M1, int KG1, int S1, int A1, int K1, int MOL1, int CD1,
                       int M2, int KG2, int S2, int A2, int K2, int MOL2, int CD2>
constexpr auto operator*(TypedScalarImpl<T, M1, KG1, S1, A1, K1, MOL1, CD1> a,
                          TypedScalarImpl<T, M2, KG2, S2, A2, K2, MOL2, CD2> b) {
    return TypedScalarImpl<T, M1 + M2, KG1 + KG2, S1 + S2, A1 + A2,
                            K1 + K2, MOL1 + MOL2, CD1 + CD2>(a.value() * b.value());
}

/// Division combines dimensions: each exponent of the result is the
/// difference of the operands' exponents for that dimension.
template <typename T, int M1, int KG1, int S1, int A1, int K1, int MOL1, int CD1,
                       int M2, int KG2, int S2, int A2, int K2, int MOL2, int CD2>
constexpr auto operator/(TypedScalarImpl<T, M1, KG1, S1, A1, K1, MOL1, CD1> a,
                          TypedScalarImpl<T, M2, KG2, S2, A2, K2, MOL2, CD2> b) {
    return TypedScalarImpl<T, M1 - M2, KG1 - KG2, S1 - S2, A1 - A2,
                            K1 - K2, MOL1 - MOL2, CD1 - CD2>(a.value() / b.value());
}

/// Square root of a dimensioned quantity: halves every dimension exponent
/// of the operand, e.g. `sqrt(Metres*Metres)` has the same dimension as
/// `Metres`, and `sqrt(Seconds*Seconds)` has the same dimension as
/// `Seconds`.
///
/// This is only well-defined -- and only compiles -- when every one of the
/// operand's seven exponents is even; taking the square root of e.g. plain
/// `Metres` (exponent m^1) cannot be expressed with an integer exponent, so
/// that call is rejected by the `static_assert` below at compile time
/// rather than silently truncating (e.g. to m^0) or producing wrong units.
/// The runtime value itself is simply `std::sqrt` of the underlying value.
template <typename T, int M, int KG, int S, int A, int K, int MOL, int CD>
auto sqrt(TypedScalarImpl<T, M, KG, S, A, K, MOL, CD> a) {
    static_assert(M % 2 == 0 && KG % 2 == 0 && S % 2 == 0 && A % 2 == 0 &&
                      K % 2 == 0 && MOL % 2 == 0 && CD % 2 == 0,
                  "sqrt() requires every dimension exponent to be even (e.g. "
                  "Metres*Metres, not Metres) so the result's units are "
                  "expressible with integer exponents");
    return TypedScalarImpl<T, M / 2, KG / 2, S / 2, A / 2, K / 2, MOL / 2, CD / 2>(
        std::sqrt(a.value()));
}

/// Convenience overload of `is_dimensionless(Units)` (see above) that
/// checks a `TypedScalarImpl` quantity directly, without the caller having
/// to spell out `decltype(a)::units()`.
template <typename T, int M, int KG, int S, int A, int K, int MOL, int CD>
constexpr bool is_dimensionless(TypedScalarImpl<T, M, KG, S, A, K, MOL, CD>) {
    return M == 0 && KG == 0 && S == 0 && A == 0 && K == 0 && MOL == 0 && CD == 0;
}

} // namespace detail
/// @endcond

/// A scalar physical quantity whose SI dimension `U` is encoded in its
/// type, checked entirely at compile time.
///
/// This is a deliberately minimal, `std::chrono::duration`-style strong
/// type -- NOT a full Boost.Units-style system with rational exponents or
/// unit-magnitude conversions. `TypedScalar` only tracks *SI base dimension
/// exponents*; it assumes values are already expressed in coherent SI base
/// units (metres, kilograms, seconds, ...), so no conversion factor is
/// stored or applied.
///
/// Two `TypedScalar` specializations with different `Units` are entirely
/// unrelated types with no implicit conversion between them, so:
///   - `operator+`/`operator-` only exist between two values of the exact
///     same dimension `U` -- adding `Metres` to `Kilograms` is a compile
///     error, not a runtime one.
///   - `operator*`/`operator/` combine the operands' dimensions and return
///     a new `TypedScalar` typed with the resulting `Units`, e.g.
///     `Metres{} / Seconds{}` has type `TypedScalar<double, Units{.m=1,.s=-1}>`.
///
/// @tparam T Underlying arithmetic representation (typically `double`).
/// @tparam U The compile-time SI dimension of this quantity.
///
/// @note This is a separate, additive system that lives alongside the
/// existing runtime `ms::Scalar` class (see `scalar.hpp`), which stores its
/// unit/prefix as `std::string` and checks compatibility at runtime.
/// `TypedScalar` does not replace, modify, or depend on `ms::Scalar` in any
/// way. Prefer `ms::Scalar` when units are only known at runtime (e.g.
/// parsed from user input); prefer `ms::core::TypedScalar` (or one of the
/// named aliases below) when units are known at compile time and you want
/// dimensionally-inconsistent code to fail to build.
///
/// @note `TypedScalar<T, U>` is an alias for an internal class template
/// parameterized on seven plain `int`s rather than directly on `U`. This is
/// a deliberate, MSVC-portability-driven deviation from a "single `Units`
/// NTTP" design: this compiler cannot reliably deduce a class-type NTTP
/// from a function-template argument, which `operator*`/`operator/` need
/// to do to combine two differently-dimensioned operands. The public
/// surface (construction from `Units{...}`, `operator+/-/*//`, `.units()`)
/// is unaffected -- callers never see the internal representation.
template <typename T, Units U>
using TypedScalar = detail::TypedScalarImpl<T, U.m, U.kg, U.s, U.A, U.K, U.mol, U.cd>;

// --- Base SI units -----------------------------------------------------

using Metres    = TypedScalar<double, Units{.m = 1}>;
using Kilograms = TypedScalar<double, Units{.kg = 1}>;
using Seconds   = TypedScalar<double, Units{.s = 1}>;
using Amperes   = TypedScalar<double, Units{.A = 1}>;
using Kelvin    = TypedScalar<double, Units{.K = 1}>;
using Moles     = TypedScalar<double, Units{.mol = 1}>;
using Candela   = TypedScalar<double, Units{.cd = 1}>;

// --- Common derived units -----------------------------------------------

/// Velocity: m/s.
using MetresPerSecond = TypedScalar<double, Units{.m = 1, .s = -1}>;
/// Acceleration: m/s^2.
using MetresPerSecondSquared = TypedScalar<double, Units{.m = 1, .s = -2}>;
/// Force: kg*m/s^2.
using Newtons = TypedScalar<double, Units{.m = 1, .kg = 1, .s = -2}>;
/// Energy: kg*m^2/s^2.
using Joules = TypedScalar<double, Units{.m = 2, .kg = 1, .s = -2}>;
/// Power: kg*m^2/s^3.
using Watts = TypedScalar<double, Units{.m = 2, .kg = 1, .s = -3}>;
/// Pressure: kg/(m*s^2).
using Pascals = TypedScalar<double, Units{.m = -1, .kg = 1, .s = -2}>;
/// Frequency: 1/s.
using Hertz = TypedScalar<double, Units{.s = -1}>;
/// Electric charge: A*s.
using Coulombs = TypedScalar<double, Units{.s = 1, .A = 1}>;
/// Electric potential: kg*m^2/(s^3*A).
using Volts = TypedScalar<double, Units{.m = 2, .kg = 1, .s = -3, .A = -1}>;

} // namespace ms::core
