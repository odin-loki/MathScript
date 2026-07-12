#include "ms/bignum/bignum.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace ms::bignum;

// ---- BigInt Construction ----

TEST(BigIntBasic, Zero) {
    BigInt z(0LL);
    EXPECT_TRUE(z.is_zero());
    EXPECT_EQ(z.to_string(), "0");
}

TEST(BigIntBasic, Positive) {
    BigInt n(12345LL);
    EXPECT_EQ(n.to_string(), "12345");
    EXPECT_EQ(n.to_ll(), 12345LL);
}

TEST(BigIntBasic, Negative) {
    BigInt n(-9876LL);
    EXPECT_EQ(n.to_string(), "-9876");
    EXPECT_EQ(n.to_ll(), -9876LL);
}

TEST(BigIntBasic, FromString) {
    BigInt n("123456789012345678");
    EXPECT_EQ(n.to_string(), "123456789012345678");
    BigInt m("-42");
    EXPECT_EQ(m.to_string(), "-42");
}

TEST(BigIntBasic, Shift10) {
    BigInt n(123LL);
    auto s=n.shift10(2);
    EXPECT_EQ(s.to_string(), "12300");
}

TEST(BigIntBasic, IsOneAndToDouble) {
    BigInt one(1LL);
    EXPECT_TRUE(one.is_one());
    EXPECT_FALSE(BigInt(2LL).is_one());
    BigInt n(12345LL);
    EXPECT_NEAR(n.to_double(), 12345.0, 1e-6);
}

// ---- BigInt Arithmetic ----

TEST(BigIntArith, AddPositive) {
    BigInt a(999LL), b(1LL);
    auto c=a+b;
    EXPECT_EQ(c.to_string(), "1000");
}

TEST(BigIntArith, AddNegative) {
    BigInt a(-5LL), b(3LL);
    EXPECT_EQ((a+b).to_string(), "-2");
}

TEST(BigIntArith, Sub) {
    BigInt a(100LL), b(37LL);
    EXPECT_EQ((a-b).to_string(), "63");
    EXPECT_EQ((b-a).to_string(), "-63");
}

TEST(BigIntArith, Mul) {
    BigInt a(123LL), b(456LL);
    EXPECT_EQ((a*b).to_string(), "56088");
}

TEST(BigIntArith, LargeMul) {
    BigInt a("999999999"), b("999999999");
    auto c=a*b;
    EXPECT_EQ(c.to_string(), "999999998000000001");
}

TEST(BigIntArith, Div) {
    BigInt a(100LL), b(7LL);
    EXPECT_EQ((a/b).to_string(), "14");
}

TEST(BigIntArith, Mod) {
    BigInt a(100LL), b(7LL);
    EXPECT_EQ((a%b).to_string(), "2");
}

TEST(BigIntArith, Negate) {
    BigInt a(5LL);
    auto b=-a;
    EXPECT_EQ(b.to_string(), "-5");
}

TEST(BigIntArith, CompoundAssign) {
    BigInt a(10LL), b(3LL);
    a+=b;
    EXPECT_EQ(a.to_string(), "13");
    a-=BigInt(5LL);
    EXPECT_EQ(a.to_string(), "8");
    a*=BigInt(2LL);
    EXPECT_EQ(a.to_string(), "16");
}

TEST(BigIntArith, DivModIdentity) {
    BigInt a("123456789012345"), b("987654321");
    auto q=a/b;
    auto r=a%b;
    EXPECT_EQ((q*b+r).to_string(), a.to_string());
}

// ---- BigInt Combined Divmod ----

namespace {

// Verifies bigint_divmod against the invariant a == q*b + r, against the
// pre-existing operator/ and operator% (which must agree exactly), and
// against the |r| < |b| remainder-magnitude bound.
void expect_divmod_matches(const BigInt& a, const BigInt& b) {
    auto [q, r] = bigint_divmod(a, b);
    EXPECT_EQ(q * b + r, a) << "a=" << a.to_string() << " b=" << b.to_string();
    EXPECT_EQ(q, a / b) << "a=" << a.to_string() << " b=" << b.to_string();
    EXPECT_EQ(r, a % b) << "a=" << a.to_string() << " b=" << b.to_string();
    if (!b.is_zero()) {
        BigInt abs_r = r.negative ? -r : r;
        BigInt abs_b = b.negative ? -b : b;
        EXPECT_TRUE(abs_r < abs_b) << "|r| not < |b| for a=" << a.to_string() << " b=" << b.to_string();
    }
}

} // namespace

TEST(BigIntDivmod, BasicPositive) {
    // a > b
    expect_divmod_matches(BigInt(100LL), BigInt(7LL));
    // a < b
    expect_divmod_matches(BigInt(7LL), BigInt(100LL));
    // exact division, no remainder
    expect_divmod_matches(BigInt(42LL), BigInt(6LL));
}

TEST(BigIntDivmod, SignCombinations) {
    expect_divmod_matches(BigInt(-100LL), BigInt(7LL));   // a negative
    expect_divmod_matches(BigInt(100LL), BigInt(-7LL));   // b negative
    expect_divmod_matches(BigInt(-100LL), BigInt(-7LL));  // both negative
    expect_divmod_matches(BigInt(-7LL), BigInt(100LL));   // |a| < |b|, a negative
    expect_divmod_matches(BigInt(-6LL), BigInt(3LL));     // exact, a negative
}

TEST(BigIntDivmod, ReturnedValues) {
    BigInt a(100LL), b(7LL);
    auto [q, r] = bigint_divmod(a, b);
    EXPECT_EQ(q.to_string(), "14");
    EXPECT_EQ(r.to_string(), "2");
}

TEST(BigIntDivmod, TruncatingSignConvention) {
    // Truncating (C++-style) division: quotient rounds toward zero, and the
    // remainder takes the sign of the dividend `a` (matching operator%'s
    // existing convention, verified here rather than assumed).
    {
        auto [q, r] = bigint_divmod(BigInt(-7LL), BigInt(3LL));
        EXPECT_EQ(q.to_string(), "-2");
        EXPECT_EQ(r.to_string(), "-1");
    }
    {
        auto [q, r] = bigint_divmod(BigInt(7LL), BigInt(-3LL));
        EXPECT_EQ(q.to_string(), "-2");
        EXPECT_EQ(r.to_string(), "1");
    }
    {
        auto [q, r] = bigint_divmod(BigInt(-7LL), BigInt(-3LL));
        EXPECT_EQ(q.to_string(), "2");
        EXPECT_EQ(r.to_string(), "-1");
    }
}

TEST(BigIntDivmod, ConsistencyWithOperatorsRandomizedFixed) {
    // Fixed (non-random-seeded-at-runtime) set of varied a/b pairs, standing
    // in for a randomized sweep while staying deterministic across runs.
    std::vector<long long> as = {0, 1, -1, 17, -17, 12345, -12345, 999999999, -999999999,
                                  123456789, -123456789, 7, -7, 1000000000, -1000000000};
    std::vector<long long> bs = {1, -1, 3, -3, 7, -7, 100, -100, 999999999, -999999999,
                                  123457, -123457, 2, -2, 999999998};
    for (long long av : as) {
        for (long long bv : bs) {
            expect_divmod_matches(BigInt(av), BigInt(bv));
        }
    }
}

TEST(BigIntDivmod, LargeMultiLimb) {
    // Both operands well beyond a single 64-bit word / single base-1e9 limb,
    // exercising the actual long-division loop rather than a small-number path.
    BigInt a("123456789123456789123456789123456789123456789");
    BigInt b("987654321987654321987654321");
    expect_divmod_matches(a, b);
    expect_divmod_matches(-a, b);
    expect_divmod_matches(a, -b);
    expect_divmod_matches(-a, -b);

    auto [q, r] = bigint_divmod(a, b);
    EXPECT_EQ(q * b + r, a);
    EXPECT_EQ(q, a / b);
    EXPECT_EQ(r, a % b);

    // |a| < |b|, both multi-limb.
    expect_divmod_matches(b, a);
}

TEST(BigIntDivmod, DivideByZeroMatchesOperators) {
    BigInt a(42LL);
    BigInt zero(0LL);
    auto [q, r] = bigint_divmod(a, zero);
    // Must mirror operator/'s defensive (no-exception) convention exactly.
    EXPECT_EQ(q, a / zero);
    EXPECT_EQ(r, a % zero);
    EXPECT_EQ(q.to_string(), "0");
    EXPECT_EQ(r.to_string(), "42");

    BigInt neg(-42LL);
    auto [q2, r2] = bigint_divmod(neg, zero);
    EXPECT_EQ(q2, neg / zero);
    EXPECT_EQ(r2, neg % zero);
    EXPECT_EQ(q2.to_string(), "0");
    EXPECT_EQ(r2.to_string(), "-42");

    // Zero divided by zero, still no crash / matches operators.
    auto [q3, r3] = bigint_divmod(zero, zero);
    EXPECT_EQ(q3, zero / zero);
    EXPECT_EQ(r3, zero % zero);
    EXPECT_EQ(q3.to_string(), "0");
    EXPECT_EQ(r3.to_string(), "0");
}

// ---- BigInt Integer Square Root ----

namespace {

// Verifies bigint_isqrt against the definitive correctness property for
// floor(sqrt(n)): r*r <= n < (r+1)*(r+1). This is checked instead of hand-computed
// expected values so it stays meaningful at sizes where double-precision sqrt
// would already have lost precision.
void expect_isqrt_floor(const BigInt& n) {
    BigInt r = bigint_isqrt(n);
    EXPECT_TRUE(r * r <= n) << "r^2 > n for n=" << n.to_string() << " r=" << r.to_string();
    EXPECT_TRUE(n < (r + BigInt(1LL)) * (r + BigInt(1LL)))
        << "n >= (r+1)^2 for n=" << n.to_string() << " r=" << r.to_string();
}

} // namespace

TEST(BigIntIsqrt, SmallExactSquares) {
    EXPECT_EQ(bigint_isqrt(BigInt(0LL)).to_string(), "0");
    EXPECT_EQ(bigint_isqrt(BigInt(1LL)).to_string(), "1");
    EXPECT_EQ(bigint_isqrt(BigInt(4LL)).to_string(), "2");
    EXPECT_EQ(bigint_isqrt(BigInt(9LL)).to_string(), "3");
    EXPECT_EQ(bigint_isqrt(BigInt(144LL)).to_string(), "12");
    EXPECT_EQ(bigint_isqrt(BigInt(10000LL)).to_string(), "100");
}

TEST(BigIntIsqrt, SmallNonSquaresFloor) {
    EXPECT_EQ(bigint_isqrt(BigInt(2LL)).to_string(), "1");
    EXPECT_EQ(bigint_isqrt(BigInt(3LL)).to_string(), "1");
    EXPECT_EQ(bigint_isqrt(BigInt(143LL)).to_string(), "11");
    EXPECT_EQ(bigint_isqrt(BigInt(145LL)).to_string(), "12");
    EXPECT_EQ(bigint_isqrt(BigInt(99LL)).to_string(), "9");
    EXPECT_EQ(bigint_isqrt(BigInt(101LL)).to_string(), "10");
}

TEST(BigIntIsqrt, NegativeInputReturnsZero) {
    EXPECT_EQ(bigint_isqrt(BigInt(-1LL)).to_string(), "0");
    EXPECT_EQ(bigint_isqrt(BigInt(-144LL)).to_string(), "0");
    EXPECT_EQ(bigint_isqrt(BigInt("-123456789123456789123456789")).to_string(), "0");
}

TEST(BigIntIsqrt, LargePerfectSquareExactRecovery) {
    // Square a large (20+ digit) BigInt via operator*, then verify bigint_isqrt
    // of the square exactly recovers the original -- the case that would be
    // impossible to validate correctly via double-precision sqrt().
    BigInt original("12345678901234567890123");
    BigInt squared = original * original;
    EXPECT_EQ(bigint_isqrt(squared), original);
}

TEST(BigIntIsqrt, AnotherLargePerfectSquareExactRecovery) {
    BigInt original("99999999999999999999999999999999999999");
    BigInt squared = original * original;
    EXPECT_EQ(bigint_isqrt(squared), original);
}

TEST(BigIntIsqrt, PowerOfTenPerfectSquare) {
    // 10^20 = (10^10)^2, a perfect square with an exact, easy-to-state answer
    // even at a scale where double-precision sqrt would already be unreliable.
    BigInt n = bigint_pow(BigInt(10LL), 20);
    EXPECT_EQ(bigint_isqrt(n).to_string(), "10000000000");
}

TEST(BigIntIsqrt, LargeNonSquareFloorInvariant) {
    // Not a perfect square: original^2 + 1 and original^2 - 1 must both floor
    // down to original and original-1 respectively.
    BigInt original("123456789012345678901234567890");
    BigInt squared = original * original;
    expect_isqrt_floor(squared + BigInt(1LL));
    expect_isqrt_floor(squared - BigInt(1LL));
    EXPECT_EQ(bigint_isqrt(squared + BigInt(1LL)), original);
    EXPECT_EQ(bigint_isqrt(squared - BigInt(1LL)), original - BigInt(1LL));
}

TEST(BigIntIsqrt, FloorInvariantSweepSmall) {
    for (long long v = 0; v <= 2000; ++v) {
        expect_isqrt_floor(BigInt(v));
    }
}

TEST(BigIntIsqrt, FloorInvariantSweepLarge) {
    // A spread of large, non-perfect-square multi-limb values.
    std::vector<std::string> values = {
        "123456789123456789123456789123456789",
        "987654321987654321987654321987654321",
        "111111111111111111111111111111111111111",
        "700000000000000000000000000000000000007",
        "2718281828459045235360287471352662497757",
        "31415926535897932384626433832795028841971",
    };
    for (const auto& s : values) {
        expect_isqrt_floor(BigInt(s));
    }
}

TEST(BigIntIsqrt, ConsistentWithBitLengthSeed) {
    // Sanity check that the bit-length-seeded Newton iteration lands on the
    // same result regardless of how far the seed starts from the true root,
    // across a range of bit-lengths.
    for (int exp = 1; exp <= 40; ++exp) {
        BigInt n = bigint_pow(BigInt(2LL), exp);
        expect_isqrt_floor(n);
    }
}

TEST(BigIntIsqrt, OneIsItsOwnRoot) {
    EXPECT_EQ(bigint_isqrt(BigInt(1LL)), BigInt(1LL));
}

TEST(BigIntIsqrt, ZeroIsItsOwnRoot) {
    EXPECT_EQ(bigint_isqrt(BigInt(0LL)), BigInt(0LL));
}

// ---- BigInt Comparison ----

TEST(BigIntCmp, LessGreater) {
    BigInt a(5LL), b(10LL);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b > a);
}

TEST(BigIntCmp, Equal) {
    BigInt a(42LL), b(42LL);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
}

TEST(BigIntCmp, CmpAbs) {
    BigInt a(-100LL), b(50LL);
    EXPECT_EQ(a.cmp_abs(b), 1);
    EXPECT_EQ(BigInt(-50LL).cmp_abs(BigInt(100LL)), -1);
    EXPECT_EQ(BigInt(-7LL).cmp_abs(BigInt(-7LL)), 0);
}

// ---- BigInt Number Theory ----

TEST(BigIntNumThy, GCD) {
    auto g=bigint_gcd(BigInt(48LL), BigInt(18LL));
    EXPECT_EQ(g.to_string(), "6");
}

TEST(BigIntNumThy, LCM) {
    auto l=bigint_lcm(BigInt(4LL), BigInt(6LL));
    EXPECT_EQ(l.to_string(), "12");
}

TEST(BigIntNumThy, Pow) {
    auto p=bigint_pow(BigInt(2LL), 10);
    EXPECT_EQ(p.to_string(), "1024");
}

TEST(BigIntNumThy, Factorial) {
    auto f=bigint_factorial(10);
    EXPECT_EQ(f.to_string(), "3628800");
    auto f20=bigint_factorial(20);
    EXPECT_EQ(f20.to_string(), "2432902008176640000");
}

TEST(BigIntNumThy, Fibonacci) {
    EXPECT_EQ(bigint_fibonacci(0).to_string(), "0");
    EXPECT_EQ(bigint_fibonacci(1).to_string(), "1");
    EXPECT_EQ(bigint_fibonacci(10).to_string(), "55");
    EXPECT_EQ(bigint_fibonacci(20).to_string(), "6765");
}

TEST(BigIntNumThy, PrimalitySmall) {
    EXPECT_TRUE(bigint_is_prime(BigInt(2LL)));
    EXPECT_TRUE(bigint_is_prime(BigInt(3LL)));
    EXPECT_TRUE(bigint_is_prime(BigInt(7LL)));
    EXPECT_FALSE(bigint_is_prime(BigInt(1LL)));
    EXPECT_FALSE(bigint_is_prime(BigInt(4LL)));
    EXPECT_FALSE(bigint_is_prime(BigInt(9LL)));
    EXPECT_TRUE(bigint_is_prime(BigInt(97LL)));
}

TEST(BigIntNumThy, CompositeLarge) {
    EXPECT_FALSE(bigint_is_prime(BigInt("1000000000000000000")));
    EXPECT_TRUE(bigint_is_prime(BigInt("982451653")));
}

TEST(BigIntNumThy, PowMod) {
    // 3^10 mod 7 = 59049 mod 7 = 4
    auto r=bigint_pow_mod(BigInt(3LL),BigInt(10LL),BigInt(7LL));
    EXPECT_EQ(r.to_string(), "4");
}

TEST(BigIntNumThy, PowModSmall) {
    // 2^3 mod 5 = 3
    auto r=bigint_pow_mod(BigInt(2LL),BigInt(3LL),BigInt(5LL));
    EXPECT_EQ(r.to_string(), "3");
}

namespace {

void expect_bezout(const BigInt& a, const BigInt& b) {
    auto [g, x, y] = bigint_extended_gcd(a, b);
    EXPECT_EQ(g, bigint_gcd(a, b));
    EXPECT_EQ(a * x + b * y, g);
}

} // namespace

TEST(BigIntNumThy, ExtendedGcdBasic) {
    auto [g, x, y] = bigint_extended_gcd(BigInt(35LL), BigInt(15LL));
    EXPECT_EQ(g.to_string(), "5");
    EXPECT_EQ(BigInt(35LL) * x + BigInt(15LL) * y, g);
}

TEST(BigIntNumThy, ExtendedGcdBezout48_18) {
    expect_bezout(BigInt(48LL), BigInt(18LL));
}

TEST(BigIntNumThy, ExtendedGcdBezoutCoprime) {
    expect_bezout(BigInt(17LL), BigInt(13LL));
}

TEST(BigIntNumThy, ExtendedGcdBezoutNegativeA) {
    expect_bezout(BigInt(-35LL), BigInt(15LL));
}

TEST(BigIntNumThy, ExtendedGcdBezoutNegativeB) {
    expect_bezout(BigInt(35LL), BigInt(-15LL));
}

TEST(BigIntNumThy, ExtendedGcdBezoutBothNegative) {
    expect_bezout(BigInt(-35LL), BigInt(-15LL));
}

TEST(BigIntNumThy, ExtendedGcdBezoutLarge) {
    expect_bezout(BigInt("123456789012345"), BigInt("987654321098765"));
}

TEST(BigIntNumThy, BitLength) {
    EXPECT_EQ(bigint_bit_length(BigInt(0LL)), 0);
    EXPECT_EQ(bigint_bit_length(BigInt(1LL)), 1);
    EXPECT_EQ(bigint_bit_length(BigInt(255LL)), 8);
    EXPECT_EQ(bigint_bit_length(BigInt(256LL)), 9);
    EXPECT_EQ(bigint_bit_length(BigInt(1023LL)), 10);
    EXPECT_EQ(bigint_bit_length(BigInt(1024LL)), 11);
    EXPECT_EQ(bigint_bit_length(BigInt(-255LL)), 8);
}

TEST(BigIntNumThy, Parity) {
    EXPECT_TRUE(bigint_is_even(BigInt(0LL)));
    EXPECT_TRUE(bigint_is_even(BigInt(4LL)));
    EXPECT_TRUE(bigint_is_even(BigInt(-6LL)));
    EXPECT_FALSE(bigint_is_even(BigInt(255LL)));
    EXPECT_TRUE(bigint_is_odd(BigInt(3LL)));
    EXPECT_TRUE(bigint_is_odd(BigInt(-7LL)));
    EXPECT_TRUE(bigint_is_odd(BigInt(255LL)));
    EXPECT_FALSE(bigint_is_odd(BigInt(0LL)));
}

TEST(BigIntBasic, ToStringBaseKnown) {
    BigInt n(255LL);
    EXPECT_EQ(n.to_string(16), "ff");
    EXPECT_EQ(n.to_string(2), "11111111");
    EXPECT_EQ(n.to_string(8), "377");
    EXPECT_EQ(n.to_string(10), n.to_string());
}

TEST(BigIntBasic, FromStringBaseKnown) {
    EXPECT_EQ(BigInt("ff", 16).to_string(), "255");
    EXPECT_EQ(BigInt("FF", 16).to_string(), "255");
    EXPECT_EQ(BigInt("11111111", 2).to_string(), "255");
    EXPECT_EQ(BigInt("377", 8).to_string(), "255");
}

TEST(BigIntBasic, BaseRoundTripHex) {
    BigInt n("123456789012345678");
    auto s = n.to_string(16);
    EXPECT_EQ(BigInt(s, 16), n);
}

TEST(BigIntBasic, BaseRoundTripBinary) {
    BigInt n("987654321");
    auto s = n.to_string(2);
    EXPECT_EQ(BigInt(s, 2), n);
}

TEST(BigIntBasic, BaseRoundTripOctal) {
    BigInt n("-4095");
    auto s = n.to_string(8);
    EXPECT_EQ(BigInt(s, 8), n);
}

TEST(BigIntBasic, Base10Regression) {
    BigInt n("-123456789012345");
    EXPECT_EQ(n.to_string(10), n.to_string());
    EXPECT_EQ(BigInt(n.to_string(10), 10), n);
}

// ---- Rational ----

TEST(Rational, BasicConstruct) {
    Rational r(3LL,4LL);
    EXPECT_EQ(r.to_string(), "3/4");
    EXPECT_NEAR(r.to_double(), 0.75, 1e-10);
}

TEST(Rational, Reduction) {
    Rational r(6LL,8LL);
    EXPECT_EQ(r.to_string(), "3/4");
}

TEST(Rational, AddSub) {
    Rational a(1LL,2LL), b(1LL,3LL);
    auto s=a+b;
    EXPECT_EQ(s.to_string(), "5/6");
    auto d=a-b;
    EXPECT_EQ(d.to_string(), "1/6");
}

TEST(Rational, MulDiv) {
    Rational a(2LL,3LL), b(3LL,4LL);
    auto p=a*b;
    EXPECT_EQ(p.to_string(), "1/2");
    auto q=a/b;
    EXPECT_EQ(q.to_string(), "8/9");
}

TEST(Rational, NegativeRational) {
    Rational r(-1LL,3LL);
    EXPECT_EQ(r.to_string(), "-1/3");
    EXPECT_NEAR(r.to_double(), -1.0/3.0, 1e-10);
}

TEST(Rational, WholeNumber) {
    Rational r(6LL,2LL);
    EXPECT_EQ(r.to_string(), "3");
}

TEST(Rational, Comparison) {
    Rational a(1LL,2LL), b(2LL,3LL);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a == b);
}

TEST(Rational, Equality) {
    Rational a(2LL,4LL), b(1LL,2LL);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(b < a);
}

TEST(Rational, FromString) {
    Rational r("3/4");
    EXPECT_EQ(r.to_string(), "3/4");
}

TEST(Rational, FromDecimalString) {
    Rational r("1.5");
    EXPECT_EQ(r.to_string(), "3/2");
    EXPECT_NEAR(r.to_double(), 1.5, 1e-10);
}

TEST(Rational, BigRational) {
    // (1/7) * 7 = 1
    Rational r(1LL,7LL);
    Rational s(7LL,1LL);
    auto p=r*s;
    EXPECT_EQ(p.to_string(), "1");
}

TEST(Rational, FloorCeilRoundPositive) {
    Rational r(7LL, 3LL);
    EXPECT_EQ(r.floor().to_string(), "2");
    EXPECT_EQ(r.ceil().to_string(), "3");
    EXPECT_EQ(r.round().to_string(), "2");
    Rational half(5LL, 2LL);
    EXPECT_EQ(half.floor().to_string(), "2");
    EXPECT_EQ(half.ceil().to_string(), "3");
    EXPECT_EQ(half.round().to_string(), "3");
    Rational whole(6LL, 2LL);
    EXPECT_EQ(whole.floor().to_string(), "3");
    EXPECT_EQ(whole.ceil().to_string(), "3");
    EXPECT_EQ(whole.round().to_string(), "3");
}

TEST(Rational, FloorCeilRoundNegative) {
    Rational r(-7LL, 3LL);
    EXPECT_EQ(r.floor().to_string(), "-3");
    EXPECT_EQ(r.ceil().to_string(), "-2");
    EXPECT_EQ(r.round().to_string(), "-2");
    Rational half(-3LL, 2LL);
    EXPECT_EQ(half.floor().to_string(), "-2");
    EXPECT_EQ(half.ceil().to_string(), "-1");
    EXPECT_EQ(half.round().to_string(), "-2");
    Rational trunc_case(-5LL, 2LL);
    EXPECT_EQ(trunc_case.floor().to_string(), "-3");
    EXPECT_EQ(trunc_case.ceil().to_string(), "-2");
    EXPECT_EQ(trunc_case.round().to_string(), "-2");
}
