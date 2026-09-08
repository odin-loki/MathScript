// Named regressions for four crashes the malformed-input sweep turned up. The sweep
// covers them too, but it probes by pattern, so these pin each specific input and the
// behaviour that replaced the crash.

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/finance/finance.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/numthy/numthy.hpp"

using namespace ms::interp;

TEST(ReplCrashRegressions, HistoricalCvarWithAConfidenceOutsideZeroOne) {
    // (1 - confidence) went negative, and converting a negative double to size_t is
    // undefined behaviour -- in practice ~2^64, so the summation loop read far past the
    // end of the sample and segfaulted.
    const std::vector<double> returns{-0.05, -0.02, 0.01, 0.03};

    for (const double c : {2.0, 1.0, 1.5, 100.0}) {
        const double v = ms::finance::historical_cvar(returns, c);
        EXPECT_TRUE(std::isfinite(v)) << "confidence " << c;
        // The tail collapses to the single worst observation.
        EXPECT_DOUBLE_EQ(v, 0.05) << "confidence " << c;
        EXPECT_TRUE(std::isfinite(ms::finance::historical_var(returns, c)));
    }
    for (const double c : {0.0, -1.0, -100.0}) {
        // The whole sample is in the tail: the mean of all four, negated.
        EXPECT_DOUBLE_EQ(ms::finance::historical_cvar(returns, c),
                         -(-0.05 + -0.02 + 0.01 + 0.03) / 4.0)
            << "confidence " << c;
        EXPECT_TRUE(std::isfinite(ms::finance::historical_var(returns, c)));
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isfinite(ms::finance::historical_cvar(returns, nan)));
    EXPECT_TRUE(std::isfinite(ms::finance::historical_var(returns, nan)));

    // The ordinary case is unchanged: 5% of four observations is a zero-length tail, so
    // the convention is the single worst return.
    EXPECT_DOUBLE_EQ(ms::finance::historical_cvar(returns, 0.95), 0.05);

    Interpreter interp;
    ASSERT_TRUE(interp.execute("R = [-0.05; -0.02; 0.01; 0.03]").has_value());
    EXPECT_TRUE(interp.execute("finance_historical_cvar(R, 2)").has_value());
    EXPECT_TRUE(interp.execute("finance_historical_var(R, -1)").has_value());
}

TEST(ReplCrashRegressions, ModPowWithAZeroModulus) {
    // base %= mod with mod == 0 is integer division by zero: SIGFPE.
    EXPECT_EQ(ms::numthy::mod_pow(2, 10, 0), 0u);
    EXPECT_EQ(ms::numthy::mod_pow(0, 0, 0), 0u);
    // Everything is 0 modulo 1.
    EXPECT_EQ(ms::numthy::mod_pow(2, 10, 1), 0u);
    // The ordinary case is unchanged: 2^10 = 1024 = 1 (mod 7).
    EXPECT_EQ(ms::numthy::mod_pow(2, 10, 7), 1024u % 7u);
    EXPECT_EQ(ms::numthy::mod_pow(3, 0, 7), 1u);

    Interpreter interp;
    EXPECT_TRUE(interp.execute("numthy_mod_pow(2, 1, 0)").has_value());
}

TEST(ReplCrashRegressions, MlMatMulWithNonConformingShapes) {
    // The inner index runs over A's columns and subscripts B's ROWS with it, so a 2x3
    // times 2x3 read past the end of B.
    const ms::ml::Mat a{{1, 2, 3}, {4, 5, 6}};      // 2x3
    const ms::ml::Mat b{{1, 2}, {3, 4}, {5, 6}};    // 3x2
    const ms::ml::Mat ab = ms::ml::mat_mul(a, b);
    ASSERT_EQ(ab.size(), 2u);
    ASSERT_EQ(ab[0].size(), 2u);
    EXPECT_DOUBLE_EQ(ab[0][0], 1 * 1 + 2 * 3 + 3 * 5);
    EXPECT_DOUBLE_EQ(ab[1][1], 4 * 2 + 5 * 4 + 6 * 6);

    EXPECT_TRUE(ms::ml::mat_mul(a, a).empty()) << "2x3 * 2x3 does not conform";
    EXPECT_TRUE(ms::ml::mat_mul({}, b).empty());
    EXPECT_TRUE(ms::ml::mat_mul(a, {}).empty());
    const ms::ml::Mat ragged{{1, 2, 3}, {4, 5}};
    EXPECT_TRUE(ms::ml::mat_mul(ragged, b).empty());

    // mat_vec, vec_add and vec_sub indexed with the other operand's length.
    EXPECT_EQ(ms::ml::mat_vec(a, {1.0}).size(), 2u);
    EXPECT_EQ(ms::ml::mat_vec(ragged, {1.0, 2.0, 3.0}).size(), 2u);
    EXPECT_EQ(ms::ml::vec_add({1.0, 2.0, 3.0}, {1.0}).size(), 1u);
    EXPECT_EQ(ms::ml::vec_sub({1.0}, {1.0, 2.0, 3.0}).size(), 1u);

    Interpreter interp;
    ASSERT_TRUE(interp.execute("R23 = [1, 2, 3; 4, 5, 6]").has_value());
    // A non-conforming product is reported, not an empty matrix and not a crash.
    EXPECT_FALSE(interp.execute("ml_mat_mul(R23, R23)").has_value());
    ASSERT_TRUE(interp.execute("R32 = [1, 2; 3, 4; 5, 6]").has_value());
    EXPECT_TRUE(interp.execute("ml_mat_mul(R23, R32)").has_value());
}

TEST(ReplCrashRegressions, BigIntStringConstructorOnNonNumericInput) {
    // The header documents "on failure constructs zero", but the body called std::stoul,
    // which throws std::invalid_argument -- and this library is built with
    // -fno-exceptions, so the throw called std::terminate and aborted the process.
    EXPECT_TRUE(ms::bignum::BigInt("x").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("-").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("12a3").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("1.5").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt(" 12").is_zero());

    // Valid input is unchanged, including across the 9-digit limb boundary.
    EXPECT_EQ(ms::bignum::BigInt("0").to_string(), "0");
    EXPECT_EQ(ms::bignum::BigInt("123456789").to_string(), "123456789");
    EXPECT_EQ(ms::bignum::BigInt("1234567890").to_string(), "1234567890");
    EXPECT_EQ(ms::bignum::BigInt("-98765432109876543210").to_string(),
              "-98765432109876543210");
    EXPECT_EQ(ms::bignum::BigInt("+7").to_string(), "7");

    Interpreter interp;
    EXPECT_TRUE(interp.execute("bigint_gcd(\"x\", \"x\")").has_value() ||
                true);  // must not abort; either answer is acceptable
    EXPECT_TRUE(interp.execute("bigint_gcd(\"12\", \"18\")").has_value());
}
