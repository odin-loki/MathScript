// Miller-Rabin: the cases the old implementation got wrong.
//
// It drew its witnesses from `nm1.to_ll()`, which truncates anything wider than
// three base-1e9 digits, so for exactly the inputs an arbitrary-precision test
// exists to handle the range came from a truncated number. That value then went
// through std::abs (undefined for LLONG_MIN) and became a modulus that could be
// zero. The generator was seeded with a constant, so the witness set was fixed and
// public, and composites that pass it are constructible by anyone who reads the
// source.
//
// Carmichael numbers are the ones that matter: composite, but Fermat-pseudoprime
// to every base coprime to them, so a weak test reports them as prime.

#include <gtest/gtest.h>

#include <string>

#include "ms/bignum/bignum.hpp"

using ms::BigInt;
using ms::bigint_is_prime;

namespace {
BigInt B(const char* s) { return BigInt(std::string(s)); }
} // namespace

TEST(BigIntPrimality, SmallPrimesAndComposites) {
    for (long long p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 97}) {
        EXPECT_TRUE(bigint_is_prime(BigInt(p), 20)) << p;
    }
    for (long long c : {0, 1, 4, 6, 8, 9, 15, 21, 25, 27, 33, 49, 51, 91, 100}) {
        EXPECT_FALSE(bigint_is_prime(BigInt(c), 20)) << c;
    }
}

TEST(BigIntPrimality, NegativeAndZeroAreNotPrime) {
    EXPECT_FALSE(bigint_is_prime(BigInt(-7LL), 20));
    EXPECT_FALSE(bigint_is_prime(BigInt(-1LL), 20));
    EXPECT_FALSE(bigint_is_prime(BigInt(0LL), 20));
}

TEST(BigIntPrimality, CarmichaelNumbersAreComposite) {
    // Fermat-pseudoprime to every coprime base; a weak test calls these prime.
    for (const char* c : {"561", "1105", "1729", "2465", "2821", "6601", "8911",
                          "10585", "15841", "29341", "41041", "62745", "63973",
                          "75361", "101101", "115921", "126217", "162401",
                          "172081", "188461", "252601", "278545", "294409"}) {
        EXPECT_FALSE(bigint_is_prime(B(c), 20)) << c;
    }
}

TEST(BigIntPrimality, StrongPseudoprimesToTheFirstBases) {
    // Each is a strong pseudoprime to every base up to the one noted, which is why
    // the deterministic set has to be the full first twelve primes rather than a
    // couple of small ones.
    EXPECT_FALSE(bigint_is_prime(B("2047"), 20));            // base 2
    EXPECT_FALSE(bigint_is_prime(B("1373653"), 20));         // bases 2, 3
    EXPECT_FALSE(bigint_is_prime(B("25326001"), 20));        // bases 2, 3, 5
    EXPECT_FALSE(bigint_is_prime(B("3215031751"), 20));      // bases 2, 3, 5, 7
    EXPECT_FALSE(bigint_is_prime(B("2152302898747"), 20));   // through 11
    EXPECT_FALSE(bigint_is_prime(B("3474749660383"), 20));   // through 13
    EXPECT_FALSE(bigint_is_prime(B("341550071728321"), 20)); // through 17
}

TEST(BigIntPrimality, WideValuesBeyondLongLong) {
    // The regression. These exceed what to_ll() can represent, so the old witness
    // range was derived from a truncated value.
    EXPECT_TRUE(bigint_is_prime(B("170141183460469231731687303715884105727"), 20))
        << "2^127 - 1 (Mersenne prime M127)";
    EXPECT_TRUE(bigint_is_prime(B("2305843009213693951"), 20)) << "2^61 - 1";
    EXPECT_TRUE(bigint_is_prime(B("618970019642690137449562111"), 20)) << "2^89 - 1";

    // Composite and far wider than long long.
    EXPECT_FALSE(bigint_is_prime(B("170141183460469231731687303715884105728"), 20));
    // 2^127-1 squared: obviously composite, no small factor.
    EXPECT_FALSE(
        bigint_is_prime(B("28948022309329048855892746252171976962977213799489202546"
                          "401021394546514198529"),
                        20));
}

TEST(BigIntPrimality, LargeSemiprimeIsComposite) {
    // Product of two 30-digit primes: no small factor, so trial division cannot
    // settle it and the witness loop has to.
    const BigInt p = B("671998030559713968361666935769");
    const BigInt q = B("282174488599599500573849980909");
    EXPECT_TRUE(bigint_is_prime(p, 20));
    EXPECT_TRUE(bigint_is_prime(q, 20));
    EXPECT_FALSE(bigint_is_prime(p * q, 20));
}

TEST(BigIntPrimality, ResultDoesNotDependOnRoundCount) {
    // Below the deterministic bound the answer is a proof, so extra rounds cannot
    // change it -- including rounds = 0, which used to mean "no witnesses at all"
    // and returned true for every odd number.
    for (const char* c : {"561", "2047", "25326001", "3215031751"}) {
        EXPECT_FALSE(bigint_is_prime(B(c), 0)) << c << " with rounds=0";
        EXPECT_FALSE(bigint_is_prime(B(c), 1)) << c << " with rounds=1";
        EXPECT_FALSE(bigint_is_prime(B(c), 50)) << c << " with rounds=50";
    }
    for (const char* p : {"97", "3474749660383", "2305843009213693951"}) {
        EXPECT_TRUE(bigint_is_prime(B(p), 0)) << p << " with rounds=0";
        EXPECT_TRUE(bigint_is_prime(B(p), 50)) << p << " with rounds=50";
    }
}

TEST(BigIntPrimality, RepeatedCallsAgree) {
    // Random witnesses above the deterministic bound must not make the answer
    // wobble between calls.
    const BigInt wide = B("170141183460469231731687303715884105727");
    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(bigint_is_prime(wide, 8)) << "call " << i;
    }
    const BigInt wide_composite = B("170141183460469231731687303715884105729");
    for (int i = 0; i < 8; ++i) {
        EXPECT_FALSE(bigint_is_prime(wide_composite, 8)) << "call " << i;
    }
}
