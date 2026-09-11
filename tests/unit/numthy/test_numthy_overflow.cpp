// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Number-theoretic results that were wrong rather than missing.
//
// A read-only audit swept the tree for values that a user would read as answers and
// that are not. These are the ones it found in `ms::numthy`, plus the two that were
// leaking their sentinels into the REPL. Each expected value here was computed in
// Python's arbitrary-precision integers, outside this program.
//
// The Jordan totient is the one worth reading twice. J_k(n) = n^k prod_{p|n}(1 - 1/p^k),
// and the implementation computed prod (p^(k*e) - p^(k*e - 1)) -- the Euler-totient
// shape, which agrees only at k = 1. The header comment said (1 - 1/p) and the only
// test asserted J_2(6) = 12 with a comment deriving it from that same wrong formula.
// Header, implementation and test all agreed with each other and none of them agreed
// with the Jordan totient. J_2(6) counts the pairs (a,b) in [1,6]^2 with gcd(a,b,6) = 1,
// which is 36 - 12 = 24 by inclusion-exclusion, and the direct count below says so
// without reference to any formula at all.

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/numthy/numthy.hpp"

using namespace ms::numthy;

namespace {

constexpr uint64_t kOverflow = UINT64_MAX;

/// J_k(n) by its combinatorial definition: the number of k-tuples in [1,n]^k whose
/// entries and n share no common factor. Brute force, so only small n and k -- which
/// is the point, since it uses no formula the implementation could also have wrong.
uint64_t jordan_by_counting(uint32_t k, uint64_t n) {
    uint64_t total = 0;
    std::vector<uint64_t> tuple(k, 1);
    while (true) {
        uint64_t g = n;
        for (const uint64_t entry : tuple) {
            g = std::gcd(g, entry);
        }
        if (g == 1) {
            ++total;
        }
        std::size_t position = 0;
        while (position < k && tuple[position] == n) {
            tuple[position] = 1;
            ++position;
        }
        if (position == k) {
            break;
        }
        ++tuple[position];
    }
    return total;
}

TEST(NumthyOverflow, JordanTotientMatchesADirectCount) {
    for (uint32_t k = 1; k <= 3; ++k) {
        for (uint64_t n = 1; n <= 12; ++n) {
            ASSERT_EQ(jordan_totient(k, n), jordan_by_counting(k, n))
                << "J_" << k << "(" << n << ")";
        }
    }
}

TEST(NumthyOverflow, JordanTotientHasTheValuesItsDefinitionGives) {
    // J_1 is the Euler totient, which is why k = 1 was the only case that ever passed.
    for (uint64_t n = 1; n <= 60; ++n) {
        ASSERT_EQ(jordan_totient(1, n), euler_phi(n)) << "n=" << n;
    }
    // These are the ones the old formula got wrong. The second number in each comment
    // is what it used to answer.
    EXPECT_EQ(jordan_totient(2, 6), 24U);    // was 12
    EXPECT_EQ(jordan_totient(2, 4), 12U);    // was 8
    EXPECT_EQ(jordan_totient(2, 3), 8U);     // was 6
    EXPECT_EQ(jordan_totient(2, 10), 72U);   // was 40
    EXPECT_EQ(jordan_totient(3, 12), 1456U); // was 576
    EXPECT_EQ(jordan_totient(4, 2), 15U);
    EXPECT_EQ(jordan_totient(2, 1000003), 1000006000008ULL);
    // Degenerate arguments keep their documented answers.
    EXPECT_EQ(jordan_totient(0, 6), 0U);
    EXPECT_EQ(jordan_totient(2, 0), 0U);
    EXPECT_EQ(jordan_totient(5, 1), 1U);
}

TEST(NumthyOverflow, JordanTotientReportsWhatItCannotRepresent) {
    // J_20(1000003) is about 1e120.
    EXPECT_EQ(jordan_totient(20, 1000003), kOverflow);
    EXPECT_EQ(jordan_totient(64, 3), kOverflow);
    // k*e used to be formed in uint32_t, so a huge k wrapped to a small exponent and
    // produced a small number instead of declining.
    EXPECT_EQ(jordan_totient(3000000000U, 6), kOverflow);
}

// sigma is superlinear, so it leaves uint64_t well before n does. The sum used to wrap
// to a value smaller than n itself.
TEST(NumthyOverflow, SumOfDivisorsReportsOverflow) {
    EXPECT_EQ(sum_divisors(1), 1U);
    EXPECT_EQ(sum_divisors(6), 12U);
    EXPECT_EQ(sum_divisors(28), 56U);
    EXPECT_EQ(sum_divisors(100), 217U);
    // sigma(2911175626334976000) is 18469238988275712000, past 2^64 - 1.
    EXPECT_EQ(sum_divisors(2911175626334976000ULL), kOverflow);
}

// The Bezout coefficient used to come from an int64_t extended gcd, so a modulus above
// 2^63 arrived negative and the answer was about a different pair of integers.
TEST(NumthyOverflow, ModularInverseHandlesLargeModuli) {
    // 9223372036854775837 is the first prime above 2^63.
    constexpr uint64_t kBigPrime = 9223372036854775837ULL;
    const auto three = mod_inv(3, kBigPrime);
    ASSERT_TRUE(three.has_value());
    EXPECT_EQ(*three, 6148914691236517225ULL);
    const auto other = mod_inv(1234567, kBigPrime);
    ASSERT_TRUE(other.has_value());
    EXPECT_EQ(*other, 2954053237778589169ULL);

    // The defining property, checked rather than the digits: a * a^-1 == 1 (mod m).
    // The product needs more than 64 bits, so it is formed exactly in BigInt rather
    // than in the type whose limits are the subject of this test.
    // Built from decimal text, not from a long long: kBigPrime is above INT64_MAX, so
    // the cast this test would otherwise make is the very narrowing it is testing for.
    const auto big = [](uint64_t value) {
        return ms::bignum::BigInt(std::to_string(value));
    };
    const ms::bignum::BigInt modulus = big(kBigPrime);
    const std::vector<uint64_t> bases{2ULL, 3ULL, 5ULL, 99991ULL, 4294967291ULL,
                                      kBigPrime - 1};
    for (const uint64_t a : bases) {
        const auto inverse = mod_inv(a, kBigPrime);
        ASSERT_TRUE(inverse.has_value()) << a;
        const ms::bignum::BigInt product = big(a) * big(*inverse);
        EXPECT_TRUE((product % modulus).is_one()) << "a=" << a << " inv=" << *inverse;
    }

    // Small moduli are unchanged.
    const auto small = mod_inv(3, 11);
    ASSERT_TRUE(small.has_value());
    EXPECT_EQ(*small, 4U); // 3*4 = 12 = 1 (mod 11)
    EXPECT_FALSE(mod_inv(2, 4).has_value());  // gcd != 1
    EXPECT_FALSE(mod_inv(1, 1).has_value());  // no meaningful inverse modulo 1
    EXPECT_FALSE(mod_inv(1, 0).has_value());
}

// The combined modulus is the product of all of them, and `M *= m[i]` wrapped past
// 2^64 -- so the answer came back reduced modulo a number that was not the product of
// anything, with no error.
TEST(NumthyOverflow, ChineseRemainderReportsAnUnrepresentableModulus) {
    const auto small = crt({2, 3, 2}, {3, 5, 7});
    ASSERT_TRUE(small.has_value());
    EXPECT_EQ(*small, 23U); // the classical Sun Tzu answer

    // 1000003 * 1000033 * 1000037 * 1000039 * 1000081 is about 1e30.
    const auto big = crt({1, 2, 3, 4, 5}, {1000003, 1000033, 1000037, 1000039, 1000081});
    EXPECT_FALSE(big.has_value());

    // A zero modulus is reported rather than divided by.
    EXPECT_FALSE(crt({1, 2}, {5, 0}).has_value());
    EXPECT_FALSE(crt({}, {}).has_value());
}

// Numerators and denominators grow at least as fast as the Fibonacci numbers, so a
// long coefficient list runs out of int64_t whatever its values are -- and signed
// overflow is undefined behaviour, not a wrapped number.
TEST(NumthyOverflow, ConvergentsStopWhereTheyStopBeingRepresentable) {
    // The continued fraction of the golden ratio: every convergent is a ratio of
    // consecutive Fibonacci numbers, and F(92) is the last one inside int64_t.
    const std::vector<int64_t> ones(200, 1);
    const auto conv = convergents(ones);
    EXPECT_LT(conv.size(), ones.size());
    EXPECT_GT(conv.size(), 80U);
    for (const auto& [p, q] : conv) {
        EXPECT_GT(p, 0);
        EXPECT_GT(q, 0);
    }
    // Everything returned is still exact: consecutive convergents satisfy
    // p_i * q_{i-1} - p_{i-1} * q_i = +-1. The products exceed int64_t long before the
    // convergents themselves do, so they are formed in BigInt.
    for (std::size_t i = 1; i < conv.size(); ++i) {
        const ms::bignum::BigInt lhs = ms::bignum::BigInt(conv[i].first) *
                                       ms::bignum::BigInt(conv[i - 1].second);
        const ms::bignum::BigInt rhs = ms::bignum::BigInt(conv[i - 1].first) *
                                       ms::bignum::BigInt(conv[i].second);
        const ms::bignum::BigInt difference = lhs - rhs;
        ASSERT_TRUE(difference.is_one() || (-difference).is_one()) << "i=" << i;
    }
    EXPECT_TRUE(convergents({}).empty());
}

// U_k grows geometrically, so it leaves int64_t quickly -- at k = 92 for P = Q = 1,
// which is the Fibonacci sequence.
TEST(NumthyOverflow, LucasSequencesReportWhatTheyCannotHold) {
    const auto small = lucas_sequence(10, 1, -1);
    EXPECT_EQ(small.first, 55);  // F(10)
    EXPECT_EQ(small.second, 123); // L(10)
    // F(90) = 2880067194370816120 and L(90) = 6440026026380244498, both inside
    // int64_t -- but 2*F(91), which the old associativity formed on the way to L(90),
    // is not.
    const auto fib = lucas_sequence(90, 1, -1);
    EXPECT_EQ(fib.first, 2880067194370816120LL);  // F(90)
    EXPECT_EQ(fib.second, 6440026026380244498LL); // L(90)
    // F(93) is 12200160415121876738, past 2^63 - 1.
    const auto over = lucas_sequence(93, 1, -1);
    EXPECT_EQ(over.first, 0);
    EXPECT_EQ(over.second, 0);
    const auto way_over = lucas_sequence(1000, 3, -7);
    EXPECT_EQ(way_over.first, 0);
    EXPECT_EQ(way_over.second, 0);
}

} // namespace
