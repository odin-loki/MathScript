#include "ms/combo/combo.hpp"
#include <gtest/gtest.h>
#include <numeric>

using namespace ms::combo;

TEST(ComboFactorials, Factorial) {
    EXPECT_EQ(factorial(0), 1u);
    EXPECT_EQ(factorial(1), 1u);
    EXPECT_EQ(factorial(5), 120u);
    EXPECT_EQ(factorial(10), 3628800u);
}

TEST(ComboFactorials, DoubleFactorial) {
    EXPECT_EQ(double_factorial(5), 15u);  // 5*3*1
    EXPECT_EQ(double_factorial(6), 48u);  // 6*4*2
    EXPECT_EQ(double_factorial(0), 1u);
    EXPECT_EQ(double_factorial(1), 1u);
}

TEST(ComboFactorials, Subfactorial) {
    EXPECT_EQ(subfactorial(0), 1u);
    EXPECT_EQ(subfactorial(1), 0u);
    EXPECT_EQ(subfactorial(2), 1u);
    EXPECT_EQ(subfactorial(3), 2u);
    EXPECT_EQ(subfactorial(4), 9u);
}

TEST(ComboCounting, Binomial) {
    EXPECT_EQ(binomial(5, 2), 10u);
    EXPECT_EQ(binomial(10, 3), 120u);
    EXPECT_EQ(binomial(0, 0), 1u);
    EXPECT_EQ(binomial(5, 6), 0u);
}

TEST(ComboCounting, Multinomial) {
    EXPECT_EQ(multinomial(6, {2, 2, 2}), 90u); // 6!/(2!2!2!)
    EXPECT_EQ(multinomial(3, {1, 1, 1}), 6u);  // 3!
}

TEST(ComboCounting, Permutations) {
    EXPECT_EQ(permutations(5, 2), 20u);
    EXPECT_EQ(permutations(5, 0), 1u);
    EXPECT_EQ(permutations(4, 4), 24u);
}

TEST(ComboCounting, CombinationsWithRep) {
    EXPECT_EQ(combinations_with_rep(3, 2), 6u);  // C(4,2)
    EXPECT_EQ(combinations_with_rep(5, 3), 35u); // C(7,3)
}

TEST(ComboEnum, NextPermutation) {
    std::vector<int> v = {1, 2, 3};
    EXPECT_TRUE(next_perm(v));
    EXPECT_EQ(v, (std::vector<int>{1, 3, 2}));
    v = {3, 2, 1};
    EXPECT_FALSE(next_perm(v));
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3}));
}

TEST(ComboEnum, NextCombination) {
    std::vector<int> v = {0, 1};
    EXPECT_TRUE(next_comb(v, 4));
    EXPECT_EQ(v, (std::vector<int>{0, 2}));
    v = {2, 3};
    EXPECT_FALSE(next_comb(v, 4));
}

TEST(ComboEnum, RankPermutation) {
    std::vector<int> v = {0, 1, 2};
    EXPECT_EQ(rank_permutation(v), 0u);
    v = {2, 1, 0};
    EXPECT_EQ(rank_permutation(v), 5u);  // last permutation = n!-1 = 5
}

TEST(ComboEnum, UnrankPermutation) {
    auto v = unrank_permutation(3, 0);
    EXPECT_EQ(v, (std::vector<int>{0, 1, 2}));
    v = unrank_permutation(3, 5);
    EXPECT_EQ(v, (std::vector<int>{2, 1, 0}));
}

TEST(ComboEnum, UnrankCombination) {
    auto v = unrank_combination(4, 2, 0);
    EXPECT_EQ(v, (std::vector<int>{0, 1}));
    v = unrank_combination(4, 2, 1);
    EXPECT_EQ(v, (std::vector<int>{0, 2}));
}

TEST(ComboEnum, AllPermutations) {
    auto perms = all_permutations(3);
    EXPECT_EQ(perms.size(), 6u);
}

TEST(ComboEnum, AllSubsets) {
    auto subs = all_subsets(3);
    EXPECT_EQ(subs.size(), 8u);
}

TEST(ComboEnum, AllPartitions) {
    auto parts = all_partitions(4);
    EXPECT_EQ(parts.size(), 5u);  // p(4)=5
}

TEST(ComboEnum, AllCompositions) {
    auto comps = all_compositions(3);
    EXPECT_EQ(comps.size(), 4u);
}

TEST(ComboEnum, Derangements) {
    auto d = derangements(3);
    EXPECT_EQ(d.size(), 2u);  // D(3) = 2
    for (auto& p : d)
        for (int i = 0; i < 3; ++i)
            EXPECT_NE(p[i], i);
}

TEST(ComboSpecial, CatalanNum) {
    EXPECT_EQ(catalan_num(0), 1u);
    EXPECT_EQ(catalan_num(1), 1u);
    EXPECT_EQ(catalan_num(2), 2u);
    EXPECT_EQ(catalan_num(3), 5u);
    EXPECT_EQ(catalan_num(4), 14u);
    EXPECT_EQ(catalan_num(5), 42u);
}

TEST(ComboSpecial, Stirling) {
    EXPECT_EQ(stirling1(4, 2), 11u);  // unsigned S1
    EXPECT_EQ(stirling2(4, 2), 7u);
    EXPECT_EQ(stirling2(0, 0), 1u);
}

TEST(ComboSpecial, BellNum) {
    EXPECT_EQ(bell_num(0), 1u);
    EXPECT_EQ(bell_num(1), 1u);
    EXPECT_EQ(bell_num(2), 2u);
    EXPECT_EQ(bell_num(3), 5u);
    EXPECT_EQ(bell_num(4), 15u);
    EXPECT_EQ(bell_num(5), 52u);
}

TEST(ComboSpecial, MotzkinNum) {
    EXPECT_EQ(motzkin_num(0), 1u);
    EXPECT_EQ(motzkin_num(1), 1u);
    EXPECT_EQ(motzkin_num(2), 2u);
    EXPECT_EQ(motzkin_num(3), 4u);
    EXPECT_EQ(motzkin_num(4), 9u);
}
