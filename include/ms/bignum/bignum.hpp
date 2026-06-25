#pragma once
#include <cstdint>
#include <string>
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
    long long to_ll() const;
    double to_double() const;

    // Helpers
    static BigInt add_abs(const BigInt& a, const BigInt& b);
    static BigInt sub_abs(const BigInt& a, const BigInt& b);  // |a| >= |b|

private:
    void trim();  // remove leading zeros
};

// ========================== Number theory ==========================
BigInt bigint_gcd(BigInt a, BigInt b);
BigInt bigint_lcm(const BigInt& a, const BigInt& b);
BigInt bigint_pow(const BigInt& base, long long exp);
BigInt bigint_pow_mod(BigInt base, BigInt exp, const BigInt& mod);
BigInt bigint_factorial(int n);
BigInt bigint_fibonacci(int n);
bool   bigint_is_prime(const BigInt& n, int rounds = 10);  // Miller-Rabin

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

private:
    void reduce();
};

} // namespace bignum
} // namespace ms
