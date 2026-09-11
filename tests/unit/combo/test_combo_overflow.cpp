// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The counting functions, checked against values computed independently in exact
// arithmetic rather than against what the implementation happens to produce.
//
// This distinction is the whole point of the file. binomial() used to advance with
//
//     r = r * (n - i) / (i + 1);
//
// which is exact as a rational step but overflows one bit before the answer does: the
// product r * (n - i) is the result times (i + 1). So C(67,33) -- 14226520737620288370,
// comfortably inside uint64_t -- came back as 8829174638479413, a sixteen-digit number
// with nothing to mark it as wrong. A test that pinned the implementation's own output
// would have agreed with it.
//
// Every expected value below was computed with Python's arbitrary-precision integers
// (math.comb, math.perm, math.factorial) and is written out in full.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "ms/combo/combo.hpp"

using namespace ms::combo;

namespace {

constexpr uint64_t kOverflow = UINT64_MAX;

// C(2m, m) is the largest binomial coefficient of its row, so it is where the
// intermediate product first exceeds the answer. m = 33 still fits; m = 34 does not.
TEST(ComboOverflow, BinomialIsExactUpToTheLimitOfTheType) {
    EXPECT_EQ(binomial(62, 31), 465428353255261088ULL);
    EXPECT_EQ(binomial(64, 32), 1832624140942590534ULL);
    EXPECT_EQ(binomial(66, 33), 7219428434016265740ULL);
    EXPECT_EQ(binomial(67, 33), 14226520737620288370ULL);
}

// C(68,34) is 28453041475240576740, past 2^64 - 1 = 18446744073709551615.
TEST(ComboOverflow, BinomialReportsWhatItCannotRepresent) {
    EXPECT_EQ(binomial(68, 34), kOverflow);
    EXPECT_EQ(binomial(100, 50), kOverflow);
    EXPECT_EQ(binomial(1000, 500), kOverflow);
}

TEST(ComboOverflow, BinomialKeepsItsSmallCases) {
    EXPECT_EQ(binomial(0, 0), 1ULL);
    EXPECT_EQ(binomial(5, 2), 10ULL);
    EXPECT_EQ(binomial(5, 6), 0ULL);
    EXPECT_EQ(binomial(52, 5), 2598960ULL);
    // Symmetry has to survive the new cancellation, including on the overflow side.
    for (uint32_t n = 0; n <= 70; ++n) {
        for (uint32_t k = 0; k <= n; ++k) {
            ASSERT_EQ(binomial(n, k), binomial(n, n - k)) << "n=" << n << " k=" << k;
        }
    }
}

// Pascal's rule is an identity the implementation never uses, so it is an independent
// check: C(n,k) = C(n-1,k-1) + C(n-1,k) wherever all three terms are representable.
TEST(ComboOverflow, BinomialSatisfiesPascalsRule) {
    for (uint32_t n = 1; n <= 66; ++n) {
        for (uint32_t k = 1; k < n; ++k) {
            const uint64_t whole = binomial(n, k);
            const uint64_t left = binomial(n - 1, k - 1);
            const uint64_t right = binomial(n - 1, k);
            if (whole == kOverflow || left == kOverflow || right == kOverflow) {
                continue;
            }
            ASSERT_EQ(whole, left + right) << "n=" << n << " k=" << k;
        }
    }
}

TEST(ComboOverflow, PermutationsAreExactThenReportOverflow) {
    EXPECT_EQ(permutations(5, 2), 20ULL);
    EXPECT_EQ(permutations(5, 0), 1ULL);
    EXPECT_EQ(permutations(3, 5), 0ULL);
    EXPECT_EQ(permutations(20, 20), 2432902008176640000ULL); // 20!
    // 21! is 51090942171709440000, past the type. The bare product used to answer
    // 14197454024290336768.
    EXPECT_EQ(permutations(21, 21), kOverflow);
    EXPECT_EQ(permutations(30, 30), kOverflow);
}

TEST(ComboOverflow, PermutationsAgreeWithBinomialTimesFactorial) {
    for (uint32_t n = 0; n <= 20; ++n) {
        for (uint32_t k = 0; k <= n; ++k) {
            const uint64_t p = permutations(n, k);
            const uint64_t c = binomial(n, k);
            const uint64_t f = factorial(k);
            if (p == kOverflow || c == kOverflow || f == kOverflow) {
                continue;
            }
            ASSERT_EQ(p, c * f) << "n=" << n << " k=" << k;
        }
    }
}

// n!/(k1!k2!...) used to be factorial(n) divided down, so any n past 20 divided the
// overflow sentinel by real factorials and returned a number.
TEST(ComboOverflow, MultinomialSurvivesFactorialsItCannotFormDirectly) {
    EXPECT_EQ(multinomial(6, {2, 2, 2}), 90ULL);
    EXPECT_EQ(multinomial(3, {1, 1, 1}), 6ULL);
    EXPECT_EQ(multinomial(24, {8, 8, 8}), 9465511770ULL);
    EXPECT_EQ(multinomial(25, {5, 5, 5, 5, 5}), 623360743125120ULL);
    EXPECT_EQ(multinomial(30, {10, 10, 10}), 5550996791340ULL);
    // Mismatched parts are not a multinomial at all.
    EXPECT_EQ(multinomial(4, {1, 1}), 0ULL);
}

TEST(ComboOverflow, MultinomialReportsWhatItCannotRepresent) {
    // 60!/(20!)^3 is about 5.6e26.
    EXPECT_EQ(multinomial(60, {20, 20, 20}), kOverflow);
}

TEST(ComboOverflow, MultisetCountsHandleAnEmptyAlphabet) {
    EXPECT_EQ(combinations_with_rep(3, 2), 6ULL);  // C(4,2)
    EXPECT_EQ(combinations_with_rep(5, 3), 35ULL); // C(7,3)
    EXPECT_EQ(combinations_with_rep(50, 10), 62828356305ULL);
    // There is one empty multiset, whatever the alphabet.
    EXPECT_EQ(combinations_with_rep(0, 0), 1ULL);
    EXPECT_EQ(combinations_with_rep(7, 0), 1ULL);
    // ...and no non-empty multiset over no symbols. n + k - 1 used to wrap through
    // zero here and answer with a product over 4294967295.
    EXPECT_EQ(combinations_with_rep(0, 5), 0ULL);
}

TEST(ComboOverflow, RankingStopsWhereRanksStopFitting) {
    // 20 elements is the last permutation count inside the type.
    std::vector<int> identity20(20);
    for (int i = 0; i < 20; ++i) {
        identity20[static_cast<std::size_t>(i)] = i;
    }
    EXPECT_EQ(rank_permutation(identity20), 0ULL);

    std::vector<int> identity21(21);
    for (int i = 0; i < 21; ++i) {
        identity21[static_cast<std::size_t>(i)] = i;
    }
    EXPECT_EQ(rank_permutation(identity21), kOverflow);
    EXPECT_TRUE(unrank_permutation(21, 0).empty());
    // A rank at or past n! names no permutation.
    EXPECT_TRUE(unrank_permutation(3, 6).empty());
    EXPECT_EQ(unrank_permutation(3, 5).size(), 3U);
}

TEST(ComboOverflow, CombinationRankingRoundTrips) {
    const uint64_t count = binomial(6, 3);
    ASSERT_EQ(count, 20ULL);
    for (uint64_t r = 0; r < count; ++r) {
        const auto v = unrank_combination(6, 3, r);
        ASSERT_EQ(v.size(), 3U) << "r=" << r;
        ASSERT_EQ(rank_combination(v, 6), r) << "r=" << r;
    }
    // Past the last one there is nothing to name.
    EXPECT_TRUE(unrank_combination(6, 3, count).empty());
    // And where the count itself does not fit, no rank in it does either.
    EXPECT_TRUE(unrank_combination(100, 50, 0).empty());
}

} // namespace
