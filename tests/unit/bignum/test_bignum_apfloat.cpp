// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/bignum/bignum.hpp"
#include <gtest/gtest.h>
#include <string>
#include <variant>

using namespace ms::bignum;

// Every expected digit string below was cross-checked against an independent
// 400-digit python `decimal` computation with ROUND_HALF_EVEN. Values are always
// computed with at least ten more significant digits than are printed, so double
// rounding cannot turn a correct result into a failing one.

// ---- APFloat construction, formatting, parsing ----

TEST(APFloatBasic, ZeroCanonical) {
    APFloat z;
    EXPECT_TRUE(z.is_zero());
    EXPECT_EQ(z.sign(), 0);
    EXPECT_EQ(z.dec_exp(), 0);
    EXPECT_EQ(z.to_string(10), "0");
    EXPECT_EQ(z.to_string_fixed(3), "0.000");
    EXPECT_EQ(z.precision(), APFloat::DEFAULT_PRECISION);
    EXPECT_FALSE(z.is_negative());
    EXPECT_TRUE(z.is_integer());
}

TEST(APFloatBasic, FromLongLongFormatting) {
    APFloat a(12345LL, 20);
    EXPECT_EQ(a.to_string(10), "12345.00000");
    EXPECT_EQ(a.to_string(5), "12345");
    EXPECT_EQ(a.to_string(3), "1.23e+4");
    EXPECT_EQ(a.to_string_fixed(3), "12345.000");
    EXPECT_EQ(a.dec_exp(), 5);
    // Canonical form: trailing zeros live in the exponent, not the mantissa.
    APFloat k(1000LL, 20);
    EXPECT_EQ(k.mantissa.to_string(), "1");
    EXPECT_EQ(k.exponent, 3);
}

TEST(APFloatBasic, StringRoundTrip) {
    APFloat b("-0.00125", 50);
    EXPECT_EQ(b.to_string(3), "-0.00125");
    EXPECT_EQ(b.to_string_fixed(6), "-0.001250");
    EXPECT_EQ(APFloat("1.5e-7", 50).to_string(3), "1.50e-7");
    EXPECT_EQ(APFloat("2.5E3", 50).to_string(4), "2500");
    EXPECT_EQ(APFloat(".5", 50).to_string(3), "0.500");
    EXPECT_EQ(APFloat("5.", 50).to_string(3), "5.00");
    EXPECT_EQ(APFloat("+1e+3", 50).to_string(3), "1.00e+3");
}

TEST(APFloatBasic, ParseErrors) {
    auto r = APFloat::parse("1.2.3", 30);
    ASSERT_FALSE(r.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(r.error()));
    EXPECT_FALSE(APFloat::parse("", 30).has_value());
    EXPECT_FALSE(APFloat::parse("abc", 30).has_value());
    EXPECT_FALSE(APFloat::parse("  1", 30).has_value());
    EXPECT_FALSE(APFloat::parse("1e", 30).has_value());
    EXPECT_FALSE(APFloat::parse("1e2000000000", 30).has_value());
    EXPECT_TRUE(APFloat::parse("-12.5e-3", 30).has_value());
    EXPECT_TRUE(APFloat("abc", 30).is_zero());  // defensive constructor
}

TEST(APFloatBasic, ExactDoubleConversion) {
    EXPECT_EQ(APFloat(0.5, 30).to_string_fixed(10), "0.5000000000");
    // The exact binary value of the double, not the decimal literal.
    EXPECT_EQ(APFloat(0.1, 60).to_string_fixed(55),
              "0.1000000000000000055511151231257827021181583404541015625");
    EXPECT_EQ(APFloat("0.1", 60).to_string_fixed(20), "0.10000000000000000000");
}

TEST(APFloatBasic, ConversionsOut) {
    EXPECT_TRUE(APFloat("-2.7", 50).trunc() == BigInt(-2LL));
    EXPECT_TRUE(APFloat("-2.7", 50).floor() == BigInt(-3LL));
    EXPECT_TRUE(APFloat("-2.7", 50).ceil() == BigInt(-2LL));
    EXPECT_TRUE(APFloat("2.5", 50).round() == BigInt(2LL));   // ties to even,
    EXPECT_TRUE(APFloat("3.5", 50).round() == BigInt(4LL));   // unlike Rational::round
    EXPECT_TRUE(APFloat("-2.5", 50).round() == BigInt(-2LL));
    EXPECT_EQ(APFloat("-12345.9", 50).to_ll(), -12345LL);
    EXPECT_NEAR(APFloat("2.5", 50).to_double(), 2.5, 1e-15);
    EXPECT_NEAR(ap_pi(60).to_double(), 3.14159265358979312, 1e-15);
    EXPECT_EQ(APFloat("0.125", 50).to_rational().to_string(), "1/8");
    EXPECT_EQ(APFloat::from_rational(Rational(1, 3), 40).to_string(40),
              "0." + std::string(40, '3'));
}

// ---- arithmetic ----

TEST(APFloatArith, ExactDecimalAddition) {
    EXPECT_TRUE(APFloat("0.1", 50) + APFloat("0.2", 50) == APFloat("0.3", 50));
    EXPECT_TRUE(APFloat("1e-100", 50) < APFloat("1e100", 50));
    EXPECT_TRUE(APFloat("1.0", 50) == APFloat("1.000", 50));
    EXPECT_TRUE((APFloat("1e300", 50) - APFloat("1e300", 50)).is_zero());
    // A far smaller addend must not disturb the larger value, but must not blow up
    // the mantissa either.
    EXPECT_TRUE(APFloat("1e300", 50) + APFloat("1e-300", 50) == APFloat("1e300", 50));
    // ... and must still be visible when the larger value is exact and short.
    EXPECT_EQ((APFloat(1LL, 60) + APFloat("1e-40", 60)).to_string_fixed(40),
              "1." + std::string(39, '0') + "1");
}

TEST(APFloatArith, OneThirdAt100Digits) {
    const APFloat q = APFloat(1LL, 100) / APFloat(3LL, 100);
    EXPECT_EQ(q.to_string(100), "0." + std::string(100, '3'));
    EXPECT_EQ(q.to_string(50), "0." + std::string(50, '3'));
    EXPECT_EQ((APFloat(2LL, 50) / APFloat(3LL, 50)).to_string_fixed(50),
              "0.66666666666666666666666666666666666666666666666667");
    EXPECT_EQ((APFloat(1LL, 60) / APFloat(7LL, 60)).to_string_fixed(60),
              "0.142857142857142857142857142857142857142857142857142857142857");
    EXPECT_TRUE((APFloat(1LL, 50) / APFloat(0LL, 50)).is_zero());
}

TEST(APFloatArith, RoundHalfToEven) {
    EXPECT_EQ(APFloat("1.25", 2).to_string(2), "1.2");
    EXPECT_EQ(APFloat("1.35", 2).to_string(2), "1.4");
    EXPECT_EQ(APFloat("1.45", 2).to_string(2), "1.4");
    EXPECT_EQ(APFloat("1.26", 2).to_string(2), "1.3");
    EXPECT_EQ(APFloat("-1.25", 2).to_string(2), "-1.2");  // sign-symmetric
    // A carry out of the rounding position keeps the digit count right.
    EXPECT_EQ(APFloat("9.995", 3).to_string(3), "10.0");
}

TEST(APFloatArith, ComparisonIsByValue) {
    EXPECT_EQ(APFloat("1.0", 50).cmp(APFloat("1.000", 20)), 0);
    EXPECT_EQ(APFloat("0.9", 50).cmp(APFloat("1", 50)), -1);
    EXPECT_EQ(APFloat("-1", 50).cmp(APFloat("-2", 50)), 1);
    EXPECT_TRUE(APFloat("-0.0", 50).is_zero());
    EXPECT_TRUE(APFloat(-5LL, 50).abs() == APFloat(5LL, 50));
    EXPECT_TRUE((-APFloat(5LL, 50)) == APFloat(-5LL, 50));
}

// ---- roots ----

TEST(APFloatRoot, SqrtTwoTo50Places) {
    EXPECT_EQ(ap_sqrt(APFloat(2LL, 70), 60).to_string_fixed(50),
              "1.41421356237309504880168872420969807856967187537695");
    EXPECT_EQ(ap_sqrt(APFloat(3LL, 60), 50).to_string_fixed(40),
              "1.7320508075688772935274463415058723669428");
    EXPECT_EQ(ap_sqrt(APFloat("0.5", 70), 60).to_string_fixed(50),
              "0.70710678118654752440084436210484903928483593768847");
}

TEST(APFloatRoot, SqrtExactForPerfectSquares) {
    const APFloat big(BigInt("152415787532388367501905199875019052100"), 0, 60);
    EXPECT_EQ(ap_sqrt(big, 60).to_string_fixed(0), "12345678901234567890");
    EXPECT_TRUE(ap_sqrt(APFloat("1e100", 60), 60) == APFloat("1e50", 60));
    EXPECT_TRUE(ap_sqrt(APFloat(4LL, 50), 50) == APFloat(2LL, 50));
    EXPECT_TRUE(ap_hypot(APFloat(3LL, 50), APFloat(4LL, 50), 50) == APFloat(5LL, 50));
}

TEST(APFloatRoot, CbrtTwoAndNegativeCube) {
    EXPECT_EQ(ap_cbrt(APFloat(2LL, 70), 60).to_string_fixed(50),
              "1.25992104989487316476721060727822835057025146470151");
    EXPECT_TRUE(ap_cbrt(APFloat(-8LL, 50), 50) == APFloat(-2LL, 50));
    EXPECT_TRUE(ap_cbrt(APFloat(1000LL, 50), 50) == APFloat(10LL, 50));
    EXPECT_TRUE(ap_cbrt(APFloat(0LL, 50), 50).is_zero());
}

// ---- constants ----

TEST(APFloatConst, PiTo50And200Places) {
    EXPECT_EQ(ap_pi(60).to_string_fixed(50),
              "3.14159265358979323846264338327950288419716939937511");
    EXPECT_EQ(ap_pi(51).to_string(51),
              "3.14159265358979323846264338327950288419716939937511");
    EXPECT_EQ(ap_pi(220).to_string_fixed(200),
        "3.14159265358979323846264338327950288419716939937510582097494459230781640628"
        "62089986280348253421170679821480865132823066470938446095505822317253594081284"
        "8111745028410270193852110555964462294895493038196");
    EXPECT_EQ(ap_pi(1000).to_string_fixed(50),
              "3.14159265358979323846264338327950288419716939937511");
}

TEST(APFloatConst, ETo40And100Places) {
    EXPECT_EQ(ap_e(60).to_string_fixed(40),
              "2.7182818284590452353602874713526624977572");
    EXPECT_EQ(ap_e(120).to_string_fixed(100),
        "2.7182818284590452353602874713526624977572470936999595749669676277240766303535"
        "475945713821785251664274");
}

TEST(APFloatConst, Ln2AndLn10) {
    EXPECT_EQ(ap_ln2(60).to_string_fixed(40),
              "0.6931471805599453094172321214581765680755");
    EXPECT_EQ(ap_ln2(120).to_string_fixed(100),
        "0.6931471805599453094172321214581765680755001343602552541206800094933936219696"
        "947156058633269964186875");
    EXPECT_EQ(ap_ln10(60).to_string_fixed(40),
              "2.3025850929940456840179914546843642076011");
}

// ---- exp / log ----

TEST(APFloatExpLog, ExpValues) {
    EXPECT_EQ(ap_exp(APFloat(1LL, 60), 50).to_string_fixed(40),
              "2.7182818284590452353602874713526624977572");
    EXPECT_EQ(ap_exp(APFloat(2LL, 60), 50).to_string_fixed(40),
              "7.3890560989306502272304274605750078131803");
    EXPECT_EQ(ap_exp(APFloat(-1LL, 60), 50).to_string_fixed(40),
              "0.3678794411714423215955237701614608674458");
    EXPECT_EQ(ap_exp(APFloat(10LL, 60), 50).to_string_fixed(30),
              "22026.465794806716516957900645284244");
    EXPECT_EQ(ap_exp(APFloat("0.5", 60), 50).to_string_fixed(40),
              "1.6487212707001281468486507878141635716538");
    EXPECT_TRUE(ap_exp(APFloat(0LL, 50), 50) == APFloat(1LL, 50));
}

TEST(APFloatExpLog, ExpOneEqualsE) {
    EXPECT_TRUE(ap_exp(APFloat(1LL, 120), 100) == ap_e(100));
}

TEST(APFloatExpLog, LogValues) {
    EXPECT_EQ(ap_log(APFloat(2LL, 60), 50).to_string_fixed(40),
              "0.6931471805599453094172321214581765680755");
    EXPECT_EQ(ap_log(APFloat(10LL, 60), 50).to_string_fixed(40),
              "2.3025850929940456840179914546843642076011");
    EXPECT_EQ(ap_log(APFloat(1000LL, 60), 50).to_string_fixed(40),
              "6.9077552789821370520539743640530926228033");
    EXPECT_EQ(ap_log(APFloat("0.5", 60), 50).to_string_fixed(40),
              "-0.6931471805599453094172321214581765680755");
    EXPECT_EQ(ap_log(APFloat("1e-20", 60), 50).to_string_fixed(40),
              "-46.0517018598809136803598290936872841520220");
    EXPECT_TRUE(ap_log(APFloat(1LL, 50), 50).is_zero());
    EXPECT_EQ(ap_log10(APFloat(2LL, 60), 50).to_string_fixed(40),
              "0.3010299956639811952137388947244930267682");
}

TEST(APFloatExpLog, LogNearOneKeepsAllDigits) {
    // Fails with about ten correct digits if the [1, 10) split and the binary trim
    // are not carried out exactly.
    EXPECT_EQ(ap_log(APFloat("1.0000000001", 60), 40).to_string(40),
              "9.999999999500000000033333333330833333334e-11");
}

TEST(APFloatExpLog, ExpLogRoundTrip) {
    EXPECT_EQ(ap_log(ap_exp(APFloat("3.5", 80), 80), 50).to_string_fixed(40),
              "3.5000000000000000000000000000000000000000");
    EXPECT_TRUE(ap_exp(ap_log(APFloat(2LL, 120), 100), 100) == APFloat(2LL, 100));
    // The decimal exponent makes an enormous argument no more expensive than a
    // small one: 10^9/ln 10 = 434294481.9..., so the result sits at 10^434294482.
    EXPECT_EQ(ap_exp(APFloat("1e9", 50), 20).dec_exp(), 434294482);
}

// ---- trigonometric ----

TEST(APFloatTrig, SinCosTanOfOne) {
    EXPECT_EQ(ap_sin(APFloat(1LL, 60), 50).to_string_fixed(40),
              "0.8414709848078965066525023216302989996226");
    EXPECT_EQ(ap_cos(APFloat(1LL, 60), 50).to_string_fixed(40),
              "0.5403023058681397174009366074429766037323");
    EXPECT_EQ(ap_tan(APFloat(1LL, 60), 50).to_string_fixed(40),
              "1.5574077246549022305069748074583601730873");
}

TEST(APFloatTrig, SinPiOverSixIsExactlyHalf) {
    const APFloat s = ap_sin(ap_pi(80) / APFloat(6LL, 80), 40);
    EXPECT_EQ(s.to_string_fixed(40), "0.5000000000000000000000000000000000000000");
    EXPECT_TRUE(s == APFloat("0.5", 40));
    EXPECT_EQ(s.mantissa.to_string(), "5");   // canonical form
    EXPECT_EQ(s.exponent, -1);

    EXPECT_TRUE(ap_cos(ap_pi(80) / APFloat(3LL, 80), 40) == APFloat("0.5", 40));
    EXPECT_TRUE(ap_tan(ap_pi(80) / APFloat(4LL, 80), 40) == APFloat(1LL, 40));
}

TEST(APFloatTrig, LargeArgumentRangeReduction) {
    EXPECT_EQ(ap_sin(APFloat("1000000", 60), 50).to_string_fixed(40),
              "-0.3499935021712929521176524867807714690614");
    EXPECT_EQ(ap_cos(APFloat("1000000", 60), 50).to_string_fixed(40),
              "0.9367521275331447869385325350749187757081");
    EXPECT_EQ(ap_sin(APFloat("1e20", 60), 50).to_string_fixed(40),
              "-0.6452512852657808442058117113125230074069");
    EXPECT_EQ(ap_cos(APFloat("1e20", 60), 50).to_string_fixed(40),
              "0.7639704044417283004001468027378811228345");
}

TEST(APFloatTrig, SinOfPiIsTheResidual) {
    // ap_pi(120) is not pi, it is pi rounded to 120 digits, so its sine is the
    // difference between the two: 2.906...e-120. Getting that (rather than the
    // working precision's noise floor) is what the reduction's retry buys.
    const APFloat s = ap_sin(ap_pi(120), 50);
    EXPECT_TRUE(s.abs() < APFloat("1e-100", 50));
    EXPECT_FALSE(s.is_zero());
    EXPECT_EQ(s.to_string(8), "-2.9061554e-120");
    EXPECT_TRUE(ap_cos(ap_pi(120), 60) == APFloat(-1LL, 60));
}

TEST(APFloatTrig, PythagoreanIdentity) {
    const APFloat x("1.2345", 120);
    const APFloat s = ap_sin(x, 100);
    const APFloat c = ap_cos(x, 100);
    EXPECT_EQ((s * s + c * c).to_string_fixed(90), "1." + std::string(90, '0'));
    const APFloat s2 = ap_sin(x, 50);
    const APFloat c2 = ap_cos(x, 50);
    EXPECT_EQ((s2 * s2 + c2 * c2).to_string_fixed(45), "1." + std::string(45, '0'));
}

TEST(APFloatTrig, TinyArgument) {
    EXPECT_EQ(ap_sin(APFloat("0.001", 60), 50).to_string_fixed(50),
              "0.00099999983333334166666646825397100970015131473481");
    EXPECT_TRUE(ap_sin(APFloat(0LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_cos(APFloat(0LL, 50), 50) == APFloat(1LL, 50));
    EXPECT_TRUE(ap_tan(APFloat(0LL, 50), 50).is_zero());
}

// ---- inverse trigonometric ----

TEST(APFloatInvTrig, AtanOneIsPiOverFour) {
    EXPECT_EQ(ap_atan(APFloat(1LL, 70), 60).to_string_fixed(50),
              "0.78539816339744830961566084581987572104929234984378");
    EXPECT_TRUE(ap_atan(APFloat(1LL, 220), 200) * APFloat(4LL, 200) == ap_pi(200));
}

TEST(APFloatInvTrig, AtanAsinAcosHalf) {
    EXPECT_EQ(ap_atan(APFloat("0.5", 60), 50).to_string_fixed(40),
              "0.4636476090008061162142562314612144020285");
    EXPECT_EQ(ap_atan(APFloat(2LL, 60), 50).to_string_fixed(40),
              "1.1071487177940905030170654601785370400700");
    EXPECT_EQ(ap_asin(APFloat("0.5", 60), 50).to_string_fixed(40),
              "0.5235987755982988730771072305465838140329");   // == pi/6
    EXPECT_EQ(ap_acos(APFloat("0.5", 60), 50).to_string_fixed(40),
              "1.0471975511965977461542144610931676280657");   // == pi/3
    EXPECT_TRUE(ap_asin(APFloat("0.5", 100), 100) == ap_pi(100) / APFloat(6LL, 100));
}

TEST(APFloatInvTrig, EndpointsAndNearOne) {
    EXPECT_TRUE(ap_asin(APFloat(1LL, 50), 50) == ap_pi(50) / APFloat(2LL, 50));
    EXPECT_TRUE(ap_asin(APFloat(-1LL, 50), 50)
                == (ap_pi(50) / APFloat(2LL, 50)).neg());
    EXPECT_TRUE(ap_acos(APFloat(-1LL, 60), 50) == ap_pi(50));
    EXPECT_TRUE(ap_acos(APFloat(1LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_asin(APFloat(0LL, 50), 50).is_zero());
    EXPECT_EQ(ap_asin(APFloat("0.9999", 60), 50).to_string_fixed(40),
              "1.5566540733173837416350814658220953363742");
    EXPECT_EQ(ap_acos(APFloat("0.9999", 60), 50).to_string_fixed(40),
              "0.0141422534775128775962402258176561057244");
}

TEST(APFloatInvTrig, Atan2Quadrants) {
    const APFloat one(1LL, 60);
    const APFloat mone(-1LL, 60);
    const APFloat z0(0LL, 60);
    EXPECT_EQ(ap_atan2(one, mone, 50).to_string_fixed(40),
              "2.3561944901923449288469825374596271631479");     //  3pi/4
    EXPECT_EQ(ap_atan2(mone, mone, 50).to_string_fixed(40),
              "-2.3561944901923449288469825374596271631479");    // -3pi/4
    EXPECT_TRUE(ap_atan2(one, z0, 50) == ap_pi(50) / APFloat(2LL, 50));
    EXPECT_TRUE(ap_atan2(mone, z0, 50) == (ap_pi(50) / APFloat(2LL, 50)).neg());
    EXPECT_TRUE(ap_atan2(z0, z0, 50).is_zero());
}

// ---- hyperbolic ----

TEST(APFloatHyper, SinhCoshTanhOfOne) {
    EXPECT_EQ(ap_sinh(APFloat(1LL, 60), 50).to_string_fixed(40),
              "1.1752011936438014568823818505956008151557");
    EXPECT_EQ(ap_cosh(APFloat(1LL, 60), 50).to_string_fixed(40),
              "1.5430806348152437784779056207570616826015");
    EXPECT_EQ(ap_tanh(APFloat(1LL, 60), 50).to_string_fixed(40),
              "0.7615941559557648881194582826047935904128");
    EXPECT_EQ(ap_tanh(APFloat(10LL, 60), 50).to_string_fixed(40),
              "0.9999999958776927636195928371382757410508");
}

TEST(APFloatHyper, SinhSmallArgumentUsesTaylor) {
    // (e^x - e^-x)/2 would cancel away three digits here and return exactly 0.001.
    EXPECT_EQ(ap_sinh(APFloat("0.001", 70), 60).to_string_fixed(50),
              "0.00100000016666667500000019841270116843036014911031");
    EXPECT_EQ(ap_cosh(APFloat("0.001", 70), 60).to_string_fixed(50),
              "1.00000050000004166666805555558035714313271605147039");
    EXPECT_TRUE(ap_sinh(APFloat(0LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_cosh(APFloat(0LL, 50), 50) == APFloat(1LL, 50));
    EXPECT_TRUE(ap_sinh(APFloat(-1LL, 60), 50) == ap_sinh(APFloat(1LL, 60), 50).neg());
}

TEST(APFloatHyper, AreaFunctions) {
    EXPECT_EQ(ap_asinh(APFloat(1LL, 60), 50).to_string_fixed(40),
              "0.8813735870195430252326093249797923090282");   // log(1 + sqrt 2)
    EXPECT_EQ(ap_acosh(APFloat(2LL, 60), 50).to_string_fixed(40),
              "1.3169578969248167086250463473079684440270");   // log(2 + sqrt 3)
    EXPECT_EQ(ap_atanh(APFloat("0.5", 60), 50).to_string_fixed(40),
              "0.5493061443340548456976226184612628523237");   // (log 3)/2
    // Small arguments keep their relative accuracy instead of vanishing next to 1.
    EXPECT_EQ(ap_asinh(APFloat("0.001", 70), 60).to_string_fixed(50),
              "0.00099999983333340833328869050657239826277889676200");
    EXPECT_TRUE(ap_acosh(APFloat(1LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_atanh(APFloat(0LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_asinh(APFloat(-1LL, 60), 50) == ap_asinh(APFloat(1LL, 60), 50).neg());
}

// ---- powers ----

TEST(APFloatPow, IntegerExponentIsExact) {
    EXPECT_EQ(ap_pow_int(APFloat(2LL, 40), 100, 40).to_string(31),
              "1267650600228229401496703205376");
    EXPECT_TRUE(ap_pow(APFloat(2LL, 50), APFloat(10LL, 50), 50) == APFloat(1024LL, 50));
    EXPECT_TRUE(ap_pow(APFloat(-2LL, 50), APFloat(3LL, 50), 50) == APFloat(-8LL, 50));
    EXPECT_TRUE(ap_pow(APFloat(-2LL, 50), APFloat(4LL, 50), 50) == APFloat(16LL, 50));
    EXPECT_TRUE(ap_pow(APFloat(7LL, 50), APFloat(0LL, 50), 50) == APFloat(1LL, 50));
    EXPECT_TRUE(ap_pow(APFloat(0LL, 50), APFloat(0LL, 50), 50) == APFloat(1LL, 50));
    EXPECT_TRUE(ap_pow_int(APFloat(2LL, 50), -2, 50) == APFloat("0.25", 50));
    EXPECT_TRUE(ap_pow_int(APFloat(10LL, 20), -1000, 20) == APFloat("1e-1000", 20));
}

TEST(APFloatPow, NonIntegerExponent) {
    EXPECT_EQ(ap_pow(APFloat(2LL, 60), APFloat("0.5", 60), 50).to_string_fixed(40),
              "1.4142135623730950488016887242096980785697");
    EXPECT_EQ(ap_pow(APFloat("2.5", 60), APFloat("3.5", 60), 50).to_string_fixed(40),
              "24.7052942200654635312413558158806135446840");
    EXPECT_EQ(ap_pow(APFloat("2.5", 500), APFloat("3.5", 500), 500).to_string_fixed(40),
              "24.7052942200654635312413558158806135446840");
}

TEST(APFloatPow, RamanujanConstant) {
    const APFloat t = ap_exp(ap_pi(60) * ap_sqrt(APFloat(163LL, 60), 60), 40);
    EXPECT_EQ(t.to_string_fixed(10), "262537412640768744.0000000000");
    EXPECT_EQ(t.to_string_fixed(12), "262537412640768743.999999999999");
}

// ---- degenerate inputs ----

TEST(APFloatDegenerate, DivisionAndDomainErrors) {
    EXPECT_TRUE((APFloat(1LL, 50) / APFloat(0LL, 50)).is_zero());
    EXPECT_TRUE(ap_log(APFloat(0LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_log(APFloat(-1LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_sqrt(APFloat(-4LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_asin(APFloat(2LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_acos(APFloat(2LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_acosh(APFloat("0.5", 50), 50).is_zero());
    EXPECT_TRUE(ap_atanh(APFloat(1LL, 50), 50).is_zero());
    EXPECT_TRUE(ap_pow(APFloat(-2LL, 50), APFloat("0.5", 50), 50).is_zero());
    EXPECT_TRUE(ap_sin(APFloat("1e10001", 50), 50).is_zero());

    auto e1 = ap_log_checked(APFloat(0LL, 50), 50);
    ASSERT_FALSE(e1.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(e1.error()));
    auto e2 = ap_sqrt_checked(APFloat(-4LL, 50), 50);
    ASSERT_FALSE(e2.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(e2.error()));
    EXPECT_FALSE(ap_asin_checked(APFloat(2LL, 50), 50).has_value());
    EXPECT_FALSE(ap_acos_checked(APFloat(2LL, 50), 50).has_value());
    EXPECT_FALSE(ap_acosh_checked(APFloat("0.5", 50), 50).has_value());
    EXPECT_FALSE(ap_atanh_checked(APFloat(1LL, 50), 50).has_value());
    EXPECT_FALSE(ap_pow_checked(APFloat(-2LL, 50), APFloat("0.5", 50), 50).has_value());
    EXPECT_FALSE(ap_pow_checked(APFloat(0LL, 50), APFloat(-1LL, 50), 50).has_value());
    EXPECT_FALSE(ap_sin_checked(APFloat("1e10001", 50), 50).has_value());
    EXPECT_FALSE(ap_cos_checked(APFloat("1e10001", 50), 50).has_value());
    EXPECT_FALSE(ap_tan_checked(APFloat("1e10001", 50), 50).has_value());
    EXPECT_TRUE(ap_sqrt_checked(APFloat(9LL, 50), 50).has_value());
    EXPECT_TRUE(ap_log_checked(APFloat(9LL, 50), 50).has_value());
    EXPECT_TRUE(ap_tan_checked(APFloat(1LL, 50), 50).has_value());
}

TEST(APFloatDegenerate, PrecisionClamping) {
    APFloat a(1LL, 0);
    EXPECT_EQ(a.precision(), APFloat::DEFAULT_PRECISION);
    a.set_precision(-7);
    EXPECT_EQ(a.precision(), APFloat::DEFAULT_PRECISION);
    a.set_precision(10000000);
    EXPECT_EQ(a.precision(), APFloat::MAX_PRECISION);
    EXPECT_EQ(ap_pi(0).to_string_fixed(40),
              "3.1415926535897932384626433832795028841972");
    EXPECT_EQ(APFloat(1LL, 50).with_precision(10).precision(), 10);
}

TEST(APFloatDegenerate, OverflowSaturatesUnderflowsToZero) {
    EXPECT_TRUE(ap_exp(APFloat("1e10", 50), 50).is_saturated());
    EXPECT_TRUE(ap_exp(APFloat("-1e10", 50), 50).is_zero());
    EXPECT_TRUE(ap_pow(APFloat(10LL, 20), APFloat("1e9", 20), 20).is_saturated());
    EXPECT_TRUE((APFloat("1e-900000000", 20) * APFloat("1e-900000000", 20)).is_zero());
    EXPECT_TRUE((APFloat("1e900000000", 20) * APFloat("1e900000000", 20)).is_saturated());
    auto ov = ap_exp_checked(APFloat("1e10", 50), 50);
    ASSERT_FALSE(ov.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::OverflowError>(ov.error()));
    EXPECT_TRUE(ap_exp_checked(APFloat("-1e10", 50), 50).has_value());  // underflow is not an error
}

// ---- APComplex ----

TEST(APComplexBasic, ExactArithmetic) {
    const APComplex i1(APFloat(1LL, 50), APFloat(1LL, 50));   // 1 + i
    const APComplex sq = i1 * i1;
    EXPECT_TRUE(sq.re.is_zero());
    EXPECT_TRUE(sq.im == APFloat(2LL, 50));                   // (1+i)^2 == 2i exactly

    const APComplex j1(APFloat(1LL, 50), APFloat(-1LL, 50));  // 1 - i
    const APComplex d = i1 / j1;
    EXPECT_TRUE(d.re.is_zero());
    EXPECT_TRUE(d.im == APFloat(1LL, 50));                    // (1+i)/(1-i) == i exactly

    EXPECT_TRUE((i1 / APComplex::zero(50)).is_zero());        // defensive
    EXPECT_TRUE(i1.conj().im == APFloat(-1LL, 50));
    EXPECT_TRUE((i1 + j1) == APComplex(APFloat(2LL, 50), APFloat(0LL, 50)));
    EXPECT_TRUE((i1 - j1).re.is_zero());
    EXPECT_TRUE((-i1).re == APFloat(-1LL, 50));
    EXPECT_TRUE(APComplex(APFloat(3LL, 50)).is_real());
    EXPECT_EQ(APComplex(APFloat(1LL, 20), APFloat(1LL, 80)).precision(), 80);
    EXPECT_EQ(i1.to_string(3), "1.00 + 1.00i");
    EXPECT_EQ(j1.to_string(3), "1.00 - 1.00i");
}

TEST(APComplexBasic, AbsArgAndSqrt) {
    const APComplex z34(APFloat(3LL, 60), APFloat(4LL, 60));
    EXPECT_TRUE(ap_cabs(z34, 50) == APFloat(5LL, 50));          // exactly 5
    const APComplex r = ap_csqrt(z34, 50);
    EXPECT_TRUE(r.re == APFloat(2LL, 50));                      // exactly 2 + i
    EXPECT_TRUE(r.im == APFloat(1LL, 50));

    const APComplex m4(APFloat(-4LL, 60), APFloat(0LL, 60));
    const APComplex rm = ap_csqrt(m4, 50);
    EXPECT_TRUE(rm.re.is_zero());
    EXPECT_TRUE(rm.im == APFloat(2LL, 50));                     // sqrt(-4) == 2i

    const APComplex i1(APFloat(1LL, 60), APFloat(1LL, 60));
    EXPECT_EQ(ap_carg(i1, 50).to_string_fixed(50),
              "0.78539816339744830961566084581987572104929234984378");
    EXPECT_EQ(ap_cabs(i1, 60).to_string_fixed(50),
              "1.41421356237309504880168872420969807856967187537695");
    EXPECT_TRUE(ap_carg(APComplex::zero(50), 50).is_zero());
}

TEST(APComplexTranscendental, EulerIdentity) {
    const APComplex ipi(APFloat(0LL, 120), ap_pi(120));
    const APComplex e = ap_cexp(ipi, 60);
    const APComplex s = e + APComplex(APFloat(1LL, 60));
    EXPECT_TRUE(e.re == APFloat(-1LL, 60));   // cos of pi-to-120-digits rounds to -1
    EXPECT_TRUE(s.re.is_zero());
    // What is left is exactly the distance from ap_pi(120) to pi.
    EXPECT_TRUE(ap_cabs(s, 60) < APFloat("1e-100", 60));
}

TEST(APComplexTranscendental, ExpLogSinCos) {
    const APComplex z12(APFloat(1LL, 60), APFloat(2LL, 60));
    const APComplex e = ap_cexp(z12, 50);
    EXPECT_EQ(e.re.to_string_fixed(40), "-1.1312043837568136384312552555107947106289");
    EXPECT_EQ(e.im.to_string_fixed(40), "2.4717266720048189276169308935516645327362");

    const APComplex z34(APFloat(3LL, 60), APFloat(4LL, 60));
    const APComplex l = ap_clog(z34, 50);
    EXPECT_EQ(l.re.to_string_fixed(40), "1.6094379124341003746007593332261876395256");
    EXPECT_EQ(l.im.to_string_fixed(40), "0.9272952180016122324285124629224288040571");

    const APComplex z11(APFloat(1LL, 60), APFloat(1LL, 60));
    const APComplex sn = ap_csin(z11, 50);
    EXPECT_EQ(sn.re.to_string_fixed(40), "1.2984575814159772948260423658078156203134");
    EXPECT_EQ(sn.im.to_string_fixed(40), "0.6349639147847361082550822029915097815171");
    const APComplex cs = ap_ccos(z11, 50);
    EXPECT_EQ(cs.re.to_string_fixed(40), "0.8337300251311490488838853943350944798099");
    EXPECT_EQ(cs.im.to_string_fixed(40), "-0.9888977057628650963821295408926861886421");
    const APComplex tn = ap_ctan(z11, 50);
    EXPECT_EQ(tn.re.to_string_fixed(40), "0.2717525853195117165288437224985889207095");
    EXPECT_EQ(tn.im.to_string_fixed(40), "1.0839233273386945434757520612119717213450");
}

TEST(APComplexTranscendental, IntegerPowerIsExact) {
    const APComplex i1(APFloat(1LL, 50), APFloat(1LL, 50));
    const APComplex p8 = ap_cpow_int(i1, 8, 50);      // (1+i)^8 == 16
    EXPECT_TRUE(p8.re == APFloat(16LL, 50));
    EXPECT_TRUE(p8.im.is_zero());
    const APComplex p4 = ap_cpow_int(i1, 4, 50);      // == -4
    EXPECT_TRUE(p4.re == APFloat(-4LL, 50));
    EXPECT_TRUE(p4.im.is_zero());
    EXPECT_TRUE(ap_cpow_int(i1, 0, 50).re == APFloat(1LL, 50));
    EXPECT_TRUE(ap_cpow_int(i1, -2, 50) * ap_cpow_int(i1, 2, 50) == APComplex::one(50));

    auto bad = ap_clog_checked(APComplex::zero(50), 50);
    ASSERT_FALSE(bad.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(bad.error()));
    EXPECT_FALSE(ap_cpow_checked(APComplex::zero(50),
                                 APComplex(APFloat(-1LL, 50)), 50).has_value());
}

TEST(APComplexTranscendental, ITotheI) {
    // i^i == exp(-pi/2), a real number.
    const APComplex ii = ap_cpow(APComplex::i_unit(60), APComplex::i_unit(60), 40);
    EXPECT_EQ(ii.re.to_string_fixed(40),
              "0.2078795763507619085469556198349787700339");
    EXPECT_TRUE(ii.im.abs() < APFloat("1e-35", 40));
}
