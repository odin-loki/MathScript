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
