#pragma once
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
    explicit BigInt(const std::string& s);
    explicit BigInt(const std::string& s, int base);

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
    std::string to_string(int base) const;
    long long to_ll() const;
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

} // namespace bignum
} // namespace ms
