// Public arbitrary-precision entry points that no test called.
//
// BigInt::parse's Result overload, APFloat's compound assignments, ap_abs/ap_neg,
// APFloat::to_string(), and most of APComplex's constructors plus ap_cconj were shipped
// API with no coverage at all -- a wrong sign or a lost precision in any of them would not
// have failed anything. Each expectation here is an exact decimal string or an identity
// that holds independently of how the operation is implemented.

#include <gtest/gtest.h>

#include <string>

#include "ms/bignum/bignum.hpp"

using ms::bignum::APComplex;
using ms::bignum::APFloat;
using ms::bignum::BigInt;

TEST(BignumApiSurface, BigIntParseReportsWhatItRejects) {
    const auto ok = BigInt::parse("-1234567890123456789012345678901234567890");
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(ok->to_string(), "-1234567890123456789012345678901234567890");

    const auto zero = BigInt::parse("0");
    ASSERT_TRUE(zero.has_value());
    EXPECT_TRUE(zero->is_zero());

    const auto plus = BigInt::parse("+42");
    ASSERT_TRUE(plus.has_value());
    EXPECT_EQ(plus->to_ll(), 42);

    // Malformed input is reported, not silently turned into zero.
    EXPECT_FALSE(BigInt::parse("").has_value());
    EXPECT_FALSE(BigInt::parse("12a3").has_value());
    EXPECT_FALSE(BigInt::parse("-").has_value());
    EXPECT_FALSE(BigInt::parse(" 12").has_value());

    // The base overload agrees with the decimal one.
    const auto hex = BigInt::parse("ff", 16);
    ASSERT_TRUE(hex.has_value());
    EXPECT_EQ(hex->to_ll(), 255);
    EXPECT_EQ(hex->to_string(16), "ff");
    const auto bin = BigInt::parse("1011", 2);
    ASSERT_TRUE(bin.has_value());
    EXPECT_EQ(bin->to_ll(), 11);
}

TEST(BignumApiSurface, ApFloatCompoundAssignmentsMatchTheBinaryOperators) {
    const APFloat a(7LL, 40);
    const APFloat b(3LL, 40);

    APFloat sum = a;
    sum += b;
    EXPECT_EQ(sum.to_string(), (a + b).to_string());

    APFloat diff = a;
    diff -= b;
    EXPECT_EQ(diff.to_string(), (a - b).to_string());

    APFloat prod = a;
    prod *= b;
    EXPECT_EQ(prod.to_string(), (a * b).to_string());

    APFloat quot = a;
    quot /= b;
    EXPECT_EQ(quot.to_string(), (a / b).to_string());

    // 7/3 to 40 significant digits, and the compound form must not lose any of them.
    EXPECT_EQ(quot.to_string(20).substr(0, 12), "2.3333333333");

    // Chaining returns *this, so the value accumulates.
    APFloat acc(1LL, 30);
    acc += APFloat(2LL, 30);
    acc *= APFloat(5LL, 30);
    acc -= APFloat(5LL, 30);
    acc /= APFloat(2LL, 30);
    EXPECT_EQ(acc.to_string(5), APFloat(5LL, 30).to_string(5));
}

TEST(BignumApiSurface, AbsAndNegate) {
    const APFloat neg = APFloat(-2.5, 30);
    const APFloat pos = APFloat(2.5, 30);

    EXPECT_EQ(ms::bignum::ap_abs(neg).to_string(10), pos.to_string(10));
    EXPECT_EQ(ms::bignum::ap_abs(pos).to_string(10), pos.to_string(10));
    EXPECT_EQ(ms::bignum::ap_neg(pos).to_string(10), neg.to_string(10));
    EXPECT_EQ(ms::bignum::ap_neg(neg).to_string(10), pos.to_string(10));

    // ap_neg is an involution and ap_abs is idempotent.
    EXPECT_EQ(ms::bignum::ap_neg(ms::bignum::ap_neg(neg)).to_string(10), neg.to_string(10));
    EXPECT_EQ(ms::bignum::ap_abs(ms::bignum::ap_abs(neg)).to_string(10), pos.to_string(10));

    // Zero has no sign to flip.
    const APFloat zero(0LL, 30);
    EXPECT_EQ(ms::bignum::ap_neg(zero).sign(), 0);
    EXPECT_EQ(ms::bignum::ap_abs(zero).sign(), 0);
}

TEST(BignumApiSurface, ApFloatDefaultToStringUsesItsOwnPrecision) {
    // to_string() is documented as to_string(prec).
    const APFloat x = ms::bignum::ap_sqrt(APFloat(2LL, 30), 30);
    EXPECT_EQ(x.to_string(), x.to_string(x.precision()));
    // sqrt(2) = 1.41421356237309504880168872420969807857...
    EXPECT_EQ(x.to_string(20).substr(0, 12), "1.4142135623");
}

TEST(BignumApiSurface, ApComplexConstructors) {
    const APComplex def;
    EXPECT_TRUE(def.is_zero());
    EXPECT_TRUE(def.is_real());

    const APComplex at_prec(60);
    EXPECT_EQ(at_prec.precision(), 60);
    EXPECT_TRUE(at_prec.is_zero());

    const APComplex from_ll(3LL, 4LL, 40);
    EXPECT_EQ(from_ll.precision(), 40);
    EXPECT_FALSE(from_ll.is_real());
    // |3 + 4i| is exactly 5: the sum of squares is exact before the root.
    EXPECT_EQ(ms::bignum::ap_cabs(from_ll, 40).to_string(6), APFloat(5LL, 40).to_string(6));

    const APComplex from_d = APComplex::from_doubles(1.5, -2.25, 40);
    EXPECT_EQ(from_d.re.to_string(6), APFloat(1.5, 40).to_string(6));
    EXPECT_EQ(from_d.im.to_string(6), APFloat(-2.25, 40).to_string(6));
    // 1.5 and -2.25 are exact in binary, so the conversion is exact.
    EXPECT_EQ(from_d.re.to_string_fixed(4), "1.5000");
    EXPECT_EQ(from_d.im.to_string_fixed(4), "-2.2500");

    const APComplex real_only{APFloat(7LL, 30)};
    EXPECT_TRUE(real_only.is_real());
    EXPECT_EQ(real_only.precision(), 30);
}

TEST(BignumApiSurface, ConjugateIsAnInvolutionAndFlipsOnlyTheImaginaryPart) {
    const APComplex z(3LL, 4LL, 40);
    const APComplex c = ms::bignum::ap_cconj(z);
    EXPECT_EQ(c.re.to_string(8), z.re.to_string(8));
    EXPECT_EQ(c.im.to_string(8), ms::bignum::ap_neg(z.im).to_string(8));
    EXPECT_TRUE(ms::bignum::ap_cconj(c) == z);
    // The free function and the member must agree.
    EXPECT_TRUE(ms::bignum::ap_cconj(z) == z.conj());
    // z * conj(z) == |z|^2 == 25, a real number.
    const APComplex prod = z * c;
    EXPECT_TRUE(prod.is_real());
    EXPECT_EQ(prod.re.to_string(6), APFloat(25LL, 40).to_string(6));
    // A real number is its own conjugate.
    const APComplex r(5LL, 0LL, 40);
    EXPECT_TRUE(ms::bignum::ap_cconj(r) == r);
}
