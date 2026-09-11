// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once
#include "ms/error/error_types.hpp"
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ms {
namespace bignum {

// ========================== BigInt ==========================
// Arbitrary-precision signed integer.
// digits stored little-endian in base 10^9.
class BigInt {
public:
    static const int BASE = 1000000000;
    std::vector<uint32_t> digits;  // little-endian, each < BASE
    bool negative = false;

    BigInt() : digits(1, 0) {}
    BigInt(long long v);
    /// Defensive decimal parse: a non-digit anywhere in the significand yields zero
    /// (it used to reach std::stoul, which throws -- and under -fno-exceptions aborted).
    /// Use parse() to see the reason instead.
    explicit BigInt(const std::string& s);
    /// Parse `s` in `base` (2–36). On failure (bad base or digit) constructs zero.
    explicit BigInt(const std::string& s, int base);

    /// Parse `s` in `base` (2–36). Returns DomainError on a bad base, an invalid digit,
    /// or an empty significand ("" and a lone sign are not numerals). The constructor
    /// above is the defensive form that yields zero instead.
    static Result<BigInt> parse(const std::string& s, int base);
    /// Parse a base-10 string, with the same reporting as the base overload.
    static Result<BigInt> parse(const std::string& s);

    // Comparison
    int cmp_abs(const BigInt& o) const;  // -1 < 0 > +1
    bool operator==(const BigInt& o) const;
    bool operator!=(const BigInt& o) const { return !(*this==o); }
    bool operator< (const BigInt& o) const;
    bool operator<=(const BigInt& o) const { return !(o<*this); }
    bool operator> (const BigInt& o) const { return o<*this; }
    bool operator>=(const BigInt& o) const { return !(*this<o); }

    // Arithmetic
    BigInt operator+(const BigInt& o) const;
    BigInt operator-(const BigInt& o) const;
    BigInt operator*(const BigInt& o) const;
    BigInt operator/(const BigInt& o) const;
    BigInt operator%(const BigInt& o) const;
    BigInt operator-() const { BigInt r=*this; r.negative=!r.negative&&!is_zero(); return r; }

    BigInt& operator+=(const BigInt& o) { *this=*this+o; return *this; }
    BigInt& operator-=(const BigInt& o) { *this=*this-o; return *this; }
    BigInt& operator*=(const BigInt& o) { *this=*this*o; return *this; }

    // Shift by decimal digits
    BigInt shift10(int n) const;  // multiply by 10^n

    bool is_zero() const;
    bool is_one() const;
    std::string to_string() const;
    /// Convert to `base` (2–36). Invalid base returns an empty string.
    std::string to_string(int base) const;
    /// @brief Value as a `long long`, saturating at the ends of the range.
    /// @note It used to accumulate the low three base-1e9 limbs into a `long long`.
    ///   Three limbs reach 10^27, past 2^63, so that was undefined behaviour rather
    ///   than a wrapped number -- and a value with more limbs came back as its low 27
    ///   digits with nothing to mark it. Saturation is at least defined and monotone;
    ///   `to_ll_exact` reports instead of guessing.
    long long to_ll() const;
    /// @brief Exact value, or false when it does not fit in a `long long`.
    bool to_ll_exact(long long& out) const;
    double to_double() const;

    // Helpers
    static BigInt add_abs(const BigInt& a, const BigInt& b);
    static BigInt sub_abs(const BigInt& a, const BigInt& b);  // |a| >= |b|

    /// @brief Compute quotient and remainder of a/b in a single long-division pass.
    /// @param a Dividend.
    /// @param b Divisor.
    /// @return {q, r} such that a == q*b + r.
    /// @note Sign convention matches operator/ and operator%: truncating division
    ///   (quotient rounds toward zero), remainder has the same sign as `a` (or is zero).
    /// @note On b == 0, mirrors operator/'s defensive convention exactly: returns
    ///   {BigInt(0), a} with no exception thrown.
    /// @note This is the shared routine backing operator/ and operator%, so callers
    ///   needing both quotient and remainder should prefer this over calling both
    ///   operators separately, which would otherwise repeat the long-division work.
    static std::pair<BigInt, BigInt> divmod(const BigInt& a, const BigInt& b);

private:
    void trim();  // remove leading zeros
};

// ========================== Number theory ==========================

/// @brief Combined division: returns {quotient, remainder} of a/b from a single
///   long-division pass, instead of computing them via two separate passes
///   (as calling operator/ then operator% would).
/// @param a Dividend.
/// @param b Divisor.
/// @return {q, r} such that a == q*b + r.
/// @note Sign convention matches operator/ and operator%: truncating division
///   (quotient rounds toward zero), remainder has the same sign as `a` (or is zero),
///   and |r| < |b| whenever b != 0.
/// @note On b == 0, mirrors operator/'s defensive convention exactly: returns
///   {BigInt(0), a} with no exception thrown.
std::pair<BigInt, BigInt> bigint_divmod(const BigInt& a, const BigInt& b);

BigInt bigint_gcd(BigInt a, BigInt b);
// Returns {g, x, y} s.t. a*x + b*y = g and g == gcd(a,b)
std::tuple<BigInt, BigInt, BigInt> bigint_extended_gcd(BigInt a, BigInt b);
int  bigint_bit_length(const BigInt& a);
bool bigint_is_even(const BigInt& a);
bool bigint_is_odd(const BigInt& a);
BigInt bigint_lcm(const BigInt& a, const BigInt& b);
BigInt bigint_pow(const BigInt& base, long long exp);
BigInt bigint_pow_mod(BigInt base, BigInt exp, const BigInt& mod);

/// @brief Modular multiplicative inverse: find x such that a*x ≡ 1 (mod m).
/// @param a Value whose inverse is sought (need not be reduced mod m).
/// @param m Modulus; must be > 1 for a meaningful result.
/// @return x in [0, m) with a*x ≡ 1 (mod m) when gcd(a, m) == 1.
///   On degenerate modulus (m <= 1 or m negative) or when no inverse exists
///   (gcd(a, m) != 1), mirrors this module's defensive convention and returns
///   BigInt(0) rather than throwing/asserting.
/// @note Computed via bigint_extended_gcd(a, m): if (g, x, y) satisfies
///   a*x + m*y = g and g == 1, the inverse is x normalized into [0, m).
BigInt bigint_mod_inv(const BigInt& a, const BigInt& m);

BigInt bigint_factorial(int n);
BigInt bigint_fibonacci(int n);
bool   bigint_is_prime(const BigInt& n, int rounds = 10);  // Miller-Rabin

/// @brief Smallest prime p such that p >= n.
/// @param n Starting value (negative inputs are treated as 0).
/// @return The least prime >= n. If n is already prime, returns n unchanged
///   (e.g. next_prime(2) == 2, next_prime(97) == 97). For n <= 1, returns 2.
BigInt bigint_next_prime(const BigInt& n);

/// @brief Integer square root: floor(sqrt(n)) for a non-negative BigInt n, computed
///   exactly via Newton's method on BigInt arithmetic directly -- no floating-point
///   is involved at any point, so it stays exact for values far beyond what a
///   double's 53-bit mantissa could represent without precision loss.
/// @param n BigInt to take the square root of.
/// @return floor(sqrt(n)). On negative input, mirrors this module's defensive
///   convention (see BigInt::divmod's handling of b == 0) and returns BigInt(0)
///   rather than throwing/asserting.
/// @note Uses Newton's method (x_{k+1} = (x_k + n/x_k) / 2), seeded from n's
///   bit-length via bigint_bit_length (x0 = 2^ceil(bit_length(n)/2), which is
///   guaranteed >= the true root) so the iteration converges quadratically and
///   terminates in O(log(bit-length of n)) steps.
BigInt bigint_isqrt(const BigInt& n);

// ========================== Rational ==========================
class Rational {
public:
    BigInt num, den;  // den > 0, num/den in lowest terms

    Rational() : num(0LL), den(1LL) {}
    Rational(long long n, long long d = 1);
    Rational(BigInt n, BigInt d);
    explicit Rational(const std::string& s);  // "3/4" or "1.5"

    Rational operator+(const Rational& o) const;
    Rational operator-(const Rational& o) const;
    Rational operator*(const Rational& o) const;
    Rational operator/(const Rational& o) const;
    Rational operator-() const { return {-num, den}; }

    bool operator==(const Rational& o) const;
    bool operator< (const Rational& o) const;
    bool operator<=(const Rational& o) const { return !(*this>o); }
    bool operator> (const Rational& o) const { return o<*this; }

    std::string to_string() const;
    double to_double() const;

    BigInt floor() const;
    BigInt ceil() const;
    BigInt round() const;

private:
    void reduce();
};

// ========================== APFloat ==========================

/// @brief Arbitrary-precision floating point: an exact BigInt significand scaled by
///   a power of ten, `value == mantissa * 10^exponent`, carried to `prec`
///   significant DECIMAL digits.
/// @note The exponent is decimal, not binary, because BigInt stores base-10^9 limbs:
///   scaling by a power of ten is a limb slice, rounding to p significant digits is a
///   digit drop, and to_string() needs no base conversion at all. A binary exponent
///   would put a full O(d^2) base-2 to base-10 conversion on every print and every
///   rounding step (with BigInt::divmod's binary-search-per-limb constant on top of
///   it), and would force ap_exp's range reduction to materialise 2^k as a BigInt --
///   for exp(10^9) that is a 4.3*10^8-digit integer.
/// @note Every operation rounds exactly once, to nearest, ties to even. This
///   deliberately differs from Rational::round(), which rounds ties away from zero for
///   positive values and to even for negative ones; APFloat::round() is half-to-even
///   for both signs, so round(-x) == -round(x) holds everywhere.
/// @note The stored form is canonical: the mantissa never carries trailing decimal
///   zeros, so a value expressible in <= prec digits has exactly one representation
///   regardless of prec (APFloat(1000LL) is mantissa 1, exponent 3).
/// @note There is no NaN and no infinity. Degenerate inputs follow this module's
///   defensive convention (see BigInt::divmod's handling of b == 0) and return zero;
///   the ap_*_checked free functions return ms::Result errors instead. A result past
///   10^EXP10_LIMIT saturates (detectable with is_saturated()) and a result below
///   10^-EXP10_LIMIT becomes exactly zero, which is the correct limit.
class APFloat {
public:
    /// Significant decimal digits used when no precision is supplied.
    static constexpr int DEFAULT_PRECISION = 50;
    /// Upper clamp on any requested precision; larger requests are clamped to it.
    static constexpr int MAX_PRECISION = 100000;
    /// Largest representable |dec_exp()|. Past it values saturate or underflow.
    static constexpr int EXP10_LIMIT = 1000000000;
    /// Trig range reduction refuses arguments with dec_exp() beyond this.
    static constexpr int TRIG_ARG_LIMIT = 10000;
    /// Conversions that must materialise an exact integer (trunc/floor/ceil/round/
    /// to_rational/to_string_fixed) refuse to build more decimal digits than this and
    /// return zero instead of allocating gigabytes.
    static constexpr int MATERIALIZE_LIMIT = 1000000;

    BigInt mantissa;                ///< signed significand; value == mantissa*10^exponent
    int    exponent = 0;            ///< DECIMAL exponent
    int    prec     = DEFAULT_PRECISION;  ///< significant decimal digits

    // ---- construction ----
    APFloat();                                   ///< zero at DEFAULT_PRECISION
    /// @brief Zero carried to `precision` significant digits.
    /// @note The argument is a PRECISION, not a value: APFloat(5) is zero at five
    ///   digits, not the number five (that is APFloat(5LL)). APFloat::zero(p) says
    ///   the same thing unambiguously and is the preferred spelling.
    explicit APFloat(int precision);
    APFloat(long long v, int precision = DEFAULT_PRECISION);
    APFloat(BigInt m, int exp10, int precision = DEFAULT_PRECISION);

    /// @brief Exact conversion of the binary value a double actually holds.
    /// @note Every finite double is m*2^e with a 53-bit m, and for e < 0 that equals
    ///   m*5^|e| / 10^|e| -- a finite decimal -- so the conversion loses nothing:
    ///   APFloat(0.1, 60) is 0.1000000000000000055511151231257827021181583404541015625,
    ///   the true value of the double, not the literal the programmer typed. Use
    ///   APFloat("0.1", prec) to get the decimal 0.1.
    /// @note NaN and +-infinity have no representation here and yield zero.
    explicit APFloat(double v, int precision = DEFAULT_PRECISION);

    /// @brief Defensive decimal parse; unparsable input constructs zero.
    /// @note Mirrors BigInt::BigInt(const std::string&, int base) exactly: the
    ///   throwing-free failure mode is a zero value. Use parse() to see the reason.
    explicit APFloat(const std::string& s, int precision = DEFAULT_PRECISION);

    /// @brief Checked decimal parse of [+-]?(d[.d*] | .d+)([eE][+-]?d+)?.
    /// @param s Text to parse; no leading/trailing whitespace is accepted.
    /// @param precision Significant digits of the result (clamped, see set_precision).
    /// @return The parsed value, or DomainError on an empty/malformed significand,
    ///   a missing or out-of-range exponent field, or trailing characters.
    /// @note Mirrors BigInt::parse's Result convention. Parsing is exact: the digits
    ///   are read into a BigInt and only the final normalize() rounds.
    static Result<APFloat> parse(const std::string& s,
                                 int precision = DEFAULT_PRECISION);

    /// @brief Correctly rounded conversion of a Rational to `precision` digits.
    /// @note A zero denominator yields zero (BigInt::divmod's convention).
    static APFloat from_rational(const Rational& r,
                                 int precision = DEFAULT_PRECISION);
    static APFloat zero(int precision);
    static APFloat one(int precision);
    /// @brief The overflow sentinel: +-(10^precision - 1) * 10^(EXP10_LIMIT-precision).
    /// @param sign Negative for the negative sentinel; 0 and positive give the positive one.
    static APFloat saturated(int sign, int precision);

    // ---- precision ----
    int  precision() const { return prec; }
    /// @brief Set the significant-digit count and re-round to it.
    /// @note p <= 0 becomes DEFAULT_PRECISION and p > MAX_PRECISION is clamped, so a
    ///   precision argument is never invalid. Lowering precision re-rounds half-even;
    ///   raising it never invents digits.
    void set_precision(int p);
    APFloat with_precision(int p) const;

    // ---- predicates ----
    bool is_zero()     const;
    bool is_negative() const;
    /// True when the value has no fractional part (zero counts as an integer).
    bool is_integer()  const;
    /// True when the magnitude has reached the EXP10_LIMIT overflow sentinel.
    bool is_saturated() const;
    int  sign() const;  ///< -1, 0 or +1

    /// @brief Decimal exponent: 0 for zero, else the unique d with 10^(d-1) <= |x| < 10^d.
    /// @note O(1): the mantissa's digit count plus the stored exponent.
    int  dec_exp() const;

    // ---- arithmetic; result precision is max(prec, o.prec), one half-even rounding ----
    APFloat operator+(const APFloat& o) const;
    APFloat operator-(const APFloat& o) const;
    APFloat operator*(const APFloat& o) const;
    /// @brief Correctly rounded quotient.
    /// @note Division by zero returns zero, mirroring BigInt::divmod's b == 0 handling.
    /// @note Computes two extra digits plus a sticky digit derived from the exact
    ///   remainder, so the half-even tie case rounds the way the true quotient would.
    APFloat operator/(const APFloat& o) const;
    APFloat operator-() const;
    APFloat& operator+=(const APFloat& o);
    APFloat& operator-=(const APFloat& o);
    APFloat& operator*=(const APFloat& o);
    APFloat& operator/=(const APFloat& o);

    // ---- comparison, by value and never by representation ----
    int  cmp(const APFloat& o) const;  ///< -1, 0 or +1
    bool operator==(const APFloat& o) const { return cmp(o) == 0; }
    bool operator!=(const APFloat& o) const { return cmp(o) != 0; }
    bool operator< (const APFloat& o) const { return cmp(o) <  0; }
    bool operator<=(const APFloat& o) const { return cmp(o) <= 0; }
    bool operator> (const APFloat& o) const { return cmp(o) >  0; }
    bool operator>=(const APFloat& o) const { return cmp(o) >= 0; }

    APFloat abs() const;
    APFloat neg() const;

    // ---- integer extraction ----
    /// @brief Truncate toward zero.
    /// @note Materialises the exact integer, so its size grows with dec_exp();
    ///   a value needing more than MATERIALIZE_LIMIT digits returns zero instead.
    BigInt trunc() const;
    BigInt floor() const;  ///< largest integer <= value (same materialisation limit)
    BigInt ceil()  const;  ///< smallest integer >= value (same materialisation limit)
    /// @brief Nearest integer, ties to even (unlike Rational::round, see the class note).
    BigInt round() const;

    // ---- conversion / formatting ----
    /// @brief Nearest double, via a 18-digit decimal literal and std::strtod.
    /// @note Overflow gives +-HUGE_VAL and underflow gives +-0.0, exactly as strtod
    ///   documents; std::stod is deliberately not used because it throws.
    double      to_double() const;
    /// @brief trunc() converted with BigInt::to_ll (which truncates past ~3 limbs).
    long long   to_ll() const;
    /// @brief Exact rational value (mantissa/1 or mantissa/10^-exponent).
    Rational    to_rational() const;

    std::string to_string() const;              ///< == to_string(prec)
    /// @brief Render with exactly `digits` significant digits, half-even.
    /// @param digits Significant digits; <= 0 means the stored precision.
    /// @return Positional notation when -4 <= dec_exp() <= digits (trailing zeros are
    ///   kept: to_string(10) of 12345 is "12345.00000"), otherwise scientific with an
    ///   explicit exponent sign and no zero padding ("1.23e+4", "1.00e-7"). Zero is "0".
    std::string to_string(int digits) const;
    /// @brief Render with exactly `decimals` digits after the point, half-even, never
    ///   scientific. Negative `decimals` is treated as 0.
    std::string to_string_fixed(int decimals) const;

private:
    /// Strip trailing zeros, round to prec (half-even), then saturate/underflow.
    void normalize();
};

// --- constants; each is computed from scratch (no caching, so no shared state) ---

/// @brief pi to `prec` significant digits via Machin's 16*atan(1/5) - 4*atan(1/239).
/// @note Both arctangent series run on scaled BigInts with O(n) division by a small
///   divisor, so the cost is ~0.93*W iterations of O(W/9) limb work at W = prec plus
///   guard digits: about 0.1*W^2 limb operations, ~1 ms at 1000 digits.
APFloat ap_pi(int prec = APFloat::DEFAULT_PRECISION);
/// @brief e to `prec` significant digits from sum 1/k!, stopping when k! > 10^W.
APFloat ap_e(int prec = APFloat::DEFAULT_PRECISION);
/// @brief ln 2 to `prec` significant digits via 2*artanh(1/3).
APFloat ap_ln2(int prec = APFloat::DEFAULT_PRECISION);
/// @brief ln 10 to `prec` significant digits via 6*artanh(1/3) + 2*artanh(1/9),
///   i.e. ln 8 + ln 1.25.
APFloat ap_ln10(int prec = APFloat::DEFAULT_PRECISION);

// --- algebraic ---

APFloat ap_abs(const APFloat& x);
APFloat ap_neg(const APFloat& x);

/// @brief Square root to `prec` significant digits.
/// @param x Radicand.
/// @param prec Significant digits of the result (clamped).
/// @return sqrt(x). A negative or zero x returns zero, mirroring bigint_isqrt's
///   defensive convention on negative input; use ap_sqrt_checked for the error.
/// @note Shifts the mantissa by an even power of ten and calls bigint_isqrt, the
///   module's exact BigInt-native Newton iteration, so no floating point ever touches
///   the value and perfect squares come out exactly (ap_sqrt(1e100) == 1e50).
APFloat ap_sqrt(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);

/// @brief Real cube root, defined for negative input (ap_cbrt(-8) == -2 exactly).
/// @note Shifts the mantissa by a multiple-of-three power of ten and takes an exact
///   integer cube root by Newton's method on BigInt (x <- (2x + n/x^2)/3, then a
///   final +-1 correction), so perfect cubes come out exactly.
APFloat ap_cbrt(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);

/// @brief sqrt(a*a + b*b), with the sum of squares formed exactly before rounding.
APFloat ap_hypot(const APFloat& a, const APFloat& b,
                 int prec = APFloat::DEFAULT_PRECISION);

// --- transcendental ---

/// @brief exp(x) to `prec` significant digits.
/// @note Reduces x = k*ln10 + r with |r| <= ln10/2, halves r a further ~1.5*sqrt(W)
///   times, sums the Taylor series, squares back and finally shifts by 10^k -- which
///   with a decimal exponent is free, so exp(10^9) is as cheap as exp(1).
/// @note |x| >= 10^10 cannot land inside EXP10_LIMIT: it saturates (x > 0) or returns
///   zero (x < 0). ap_exp_checked reports the overflow instead.
APFloat ap_exp(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);

/// @brief Natural logarithm to `prec` significant digits.
/// @param x Argument; zero or negative returns zero (see ap_log_checked).
/// @note Splits x = f * 2^j * 10^E with f in [0.75, 1.5) using only exact operations
///   (a power-of-ten exponent shift and exact halvings), so f - 1 is exact and no
///   digits are lost as x approaches 1: log(1 + 1e-10) keeps all `prec` digits.
///   The reduced value then goes through 2*artanh((f-1)/(f+1)), |z| <= 0.2.
APFloat ap_log(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief Base-10 logarithm, ap_log(x)/ln 10; zero or negative x returns zero.
APFloat ap_log10(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);

/// @brief x^y for real y, as exp(y*log x).
/// @note An integer y is routed to ap_pow_int, which is exact for exact inputs and
///   handles a negative base. A non-integer y with x < 0 has no real value and
///   returns zero; x == 0 with y <= 0 likewise (see ap_pow_checked). 0^0 is 1,
///   matching bigint_pow(b, 0).
APFloat ap_pow(const APFloat& x, const APFloat& y,
               int prec = APFloat::DEFAULT_PRECISION);
/// @brief x^n by binary exponentiation; exact when x and n are (2^100 is the literal
///   1267650600228229401496703205376). Negative n gives 1/x^|n|; 0^0 is 1.
APFloat ap_pow_int(const APFloat& x, long long n,
                   int prec = APFloat::DEFAULT_PRECISION);

/// @brief sin(x) to `prec` significant digits.
/// @note Reduces modulo 2*pi and then by quadrant to |s| <= pi/4 before summing the
///   Taylor series, carrying dec_exp(x) extra guard digits to pay for the
///   cancellation in x - q*2*pi. An argument with dec_exp(x) > TRIG_ARG_LIMIT would
///   need a pi with more digits than the working precision can justify and returns
///   zero (see ap_sin_checked).
APFloat ap_sin(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief cos(x); same reduction and same argument limit as ap_sin.
APFloat ap_cos(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief tan(x) as sin/cos at working precision; a cos that rounds to exactly zero
///   returns zero (see ap_tan_checked).
APFloat ap_tan(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);

/// @brief asin(x) as 2*atan(x / (1 + sqrt(1 - x^2))), with 1 - x^2 formed exactly so
///   that the x -> +-1 cancellation never happens; |x| > 1 returns zero.
APFloat ap_asin(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief acos(x) as 2*atan(sqrt((1 - x)/(1 + x))), which (unlike pi/2 - asin x) does
///   not cancel as x -> 1; acos(-1) is pi and |x| > 1 returns zero.
APFloat ap_acos(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief atan(x) to `prec` significant digits.
/// @note Uses the sqrt-free rational reduction atan(a) = atan(1/n) + atan((n*a-1)/(n+a))
///   with n = round(1/|a|), whose arctan(1/n) pieces come from the same cheap scaled
///   integer series that drives ap_pi. Four levels take any |a| <= 1 below 1e-4.
APFloat ap_atan(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief Quadrant-correct atan(y/x) in (-pi, pi].
/// @note atan2(0, 0) has no value and returns zero, mirroring bigint_mod_inv's
///   zero-on-degenerate convention.
APFloat ap_atan2(const APFloat& y, const APFloat& x,
                 int prec = APFloat::DEFAULT_PRECISION);

/// @brief sinh(x). For |x| < 1 the Taylor series is used instead of (e^x - e^-x)/2,
///   which would cancel away dec_exp(x) digits.
APFloat ap_sinh(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief cosh(x) as (e^|x| + e^-|x|)/2, which never cancels.
APFloat ap_cosh(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief tanh(x) as sinh/cosh; saturating |x| gives exactly +-1.
APFloat ap_tanh(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1)), with extra guard digits for
///   small |x| so the relative accuracy of the result survives.
APFloat ap_asinh(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief acosh(x) = log(x + sqrt(x^2 - 1)) for x >= 1; x < 1 returns zero.
APFloat ap_acosh(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @brief atanh(x) = log((1 + x)/(1 - x))/2 for |x| < 1; |x| >= 1 returns zero.
APFloat ap_atanh(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);

// --- checked siblings; same maths, ms::Result instead of the defensive zero ---

/// @return sqrt(x), or DomainError when x < 0.
Result<APFloat> ap_sqrt_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return log(x), or DomainError when x <= 0.
Result<APFloat> ap_log_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return exp(x), or OverflowError when the result exceeds 10^EXP10_LIMIT.
///   Underflow is not an error: it returns exact zero, the correct limit.
Result<APFloat> ap_exp_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return asin(x), or DomainError when |x| > 1.
Result<APFloat> ap_asin_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return acos(x), or DomainError when |x| > 1.
Result<APFloat> ap_acos_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return acosh(x), or DomainError when x < 1.
Result<APFloat> ap_acosh_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return atanh(x), or DomainError when |x| >= 1.
Result<APFloat> ap_atanh_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return x^y, or DomainError for 0^(y<=0) and for a negative base with a
///   non-integer exponent, or OverflowError when exp overflows.
Result<APFloat> ap_pow_checked(const APFloat& x, const APFloat& y,
                               int prec = APFloat::DEFAULT_PRECISION);
/// @return sin(x), or DomainError when dec_exp(x) > TRIG_ARG_LIMIT.
Result<APFloat> ap_sin_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return cos(x), or DomainError when dec_exp(x) > TRIG_ARG_LIMIT.
Result<APFloat> ap_cos_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);
/// @return tan(x), or DomainError past the argument limit or at a pole.
Result<APFloat> ap_tan_checked(const APFloat& x, int prec = APFloat::DEFAULT_PRECISION);

// ========================== APComplex ==========================

/// @brief Arbitrary-precision complex number: two APFloats kept at one precision.
/// @note A constructor raises the lower of the two component precisions so both
///   components always agree; precision() reports it.
/// @note Every degenerate case mirrors APFloat's defensive convention: division by
///   zero and the logarithm of zero return zero rather than reporting an error, and
///   there is no NaN and no infinity.
class APComplex {
public:
    APFloat re, im;

    APComplex();
    explicit APComplex(int precision);
    explicit APComplex(APFloat r);              ///< imaginary part zero at r's precision
    APComplex(APFloat r, APFloat i);            ///< unifies the two precisions
    APComplex(long long r, long long i, int precision = APFloat::DEFAULT_PRECISION);
    static APComplex from_doubles(double r, double i,
                                  int precision = APFloat::DEFAULT_PRECISION);
    static APComplex zero(int precision);
    static APComplex one(int precision);
    static APComplex i_unit(int precision);

    int  precision() const { return re.precision(); }
    void set_precision(int p);
    APComplex with_precision(int p) const;

    bool is_zero() const;
    bool is_real() const;

    APComplex operator+(const APComplex& o) const;
    APComplex operator-(const APComplex& o) const;
    /// @note (ac - bd) + (ad + bc)i: four exact products, one rounding per component.
    APComplex operator*(const APComplex& o) const;
    /// @note Division by zero returns zero (BigInt::divmod's convention).
    APComplex operator/(const APComplex& o) const;
    APComplex operator-() const;
    bool operator==(const APComplex& o) const;
    bool operator!=(const APComplex& o) const { return !(*this == o); }

    APComplex conj() const;
    /// @brief "re + imi" / "re - imi", each component with `digits` significant digits.
    std::string to_string(int digits) const;
};

/// @brief |z| = sqrt(re^2 + im^2); the sum of squares is exact before the root, so
///   ap_cabs(3 + 4i) is exactly 5.
APFloat ap_cabs(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
/// @brief arg(z) in (-pi, pi]; arg(0) is zero (see ap_atan2).
APFloat ap_carg(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
APComplex ap_cconj(const APComplex& z);
/// @brief exp(z) = e^re * (cos im + i sin im), sharing a single sin/cos reduction.
APComplex ap_cexp(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
/// @brief Principal log: log|z| computed as log(re^2 + im^2)/2 (no square root, and
///   the sum of squares is exact) plus i*arg(z). log(0) returns zero.
APComplex ap_clog(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
/// @brief Principal square root by the stable |z| + |re| form, so ap_csqrt(3 + 4i) is
///   exactly 2 + i and ap_csqrt(-4) is exactly 2i (the branch takes sgn(0) as +1).
APComplex ap_csqrt(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
/// @brief sin(a + bi) = sin a cosh b + i cos a sinh b.
APComplex ap_csin(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
/// @brief cos(a + bi) = cos a cosh b - i sin a sinh b.
APComplex ap_ccos(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
/// @brief tan(z) = sin(z)/cos(z); a cos of exactly zero returns zero.
APComplex ap_ctan(const APComplex& z, int prec = APFloat::DEFAULT_PRECISION);
/// @brief z^w = exp(w * log z) on the principal branch; z == 0 returns zero.
APComplex ap_cpow(const APComplex& z, const APComplex& w,
                  int prec = APFloat::DEFAULT_PRECISION);
/// @brief z^n by binary exponentiation; exact for exact inputs ((1+i)^8 == 16).
APComplex ap_cpow_int(const APComplex& z, long long n,
                      int prec = APFloat::DEFAULT_PRECISION);

/// @return log(z), or DomainError when z is zero.
Result<APComplex> ap_clog_checked(const APComplex& z,
                                  int prec = APFloat::DEFAULT_PRECISION);
/// @return z^w, or DomainError when z is zero and w is not a positive real.
Result<APComplex> ap_cpow_checked(const APComplex& z, const APComplex& w,
                                  int prec = APFloat::DEFAULT_PRECISION);

} // namespace bignum
} // namespace ms
