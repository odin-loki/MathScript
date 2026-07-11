#include "ms/combo/combo.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <set>

using namespace ms::combo;

namespace {

void assert_valid_set_partition(const std::vector<std::vector<int>>& part, int n) {
    std::vector<bool> seen(n, false);
    for (const auto& block : part) {
        ASSERT_FALSE(block.empty());
        for (int elem : block) {
            ASSERT_GE(elem, 0);
            ASSERT_LT(elem, n);
            ASSERT_FALSE(seen[elem]);
            seen[elem] = true;
        }
    }
    for (int i = 0; i < n; ++i)
        ASSERT_TRUE(seen[i]);
}

bool is_balanced_dyck(const std::string& s) {
    int depth = 0;
    for (char c : s) {
        if (c == '(') ++depth;
        else {
            --depth;
            if (depth < 0) return false;
        }
    }
    return depth == 0;
}

bool is_valid_motzkin_path(const std::string& s) {
    int height = 0;
    for (char c : s) {
        if (c == 'U') ++height;
        else if (c == 'D') {
            --height;
            if (height < 0) return false;
        } else if (c != 'F')
            return false;
    }
    return height == 0;
}

} // namespace

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

TEST(ComboSetPartitions, CountMatchesBellNum) {
    for (int n = 0; n <= 6; ++n)
        EXPECT_EQ(set_partitions(n).size(), bell_num(static_cast<uint32_t>(n)));
}

TEST(ComboSetPartitions, ValidPartitionsSmallN) {
    for (int n = 0; n <= 5; ++n) {
        auto parts = set_partitions(n);
        std::set<std::vector<std::vector<int>>> unique(parts.begin(), parts.end());
        EXPECT_EQ(unique.size(), parts.size());
        for (const auto& part : parts)
            assert_valid_set_partition(part, n);
    }
}

TEST(ComboSetPartitions, KnownSmallCases) {
    EXPECT_EQ(set_partitions(0), (std::vector<std::vector<std::vector<int>>>{{}}));
    EXPECT_EQ(set_partitions(1), (std::vector<std::vector<std::vector<int>>>{{{0}}}));
    auto parts2 = set_partitions(2);
    EXPECT_EQ(parts2.size(), 2u);
}

TEST(ComboInvolutions, HandVerifiedValues) {
    EXPECT_EQ(involutions(0), 1u);
    EXPECT_EQ(involutions(1), 1u);
    EXPECT_EQ(involutions(2), 2u);
    EXPECT_EQ(involutions(3), 4u);
    EXPECT_EQ(involutions(4), 10u);
}

TEST(ComboInvolutions, RecurrenceValues) {
    EXPECT_EQ(involutions(5), 26u);
    EXPECT_EQ(involutions(6), 76u);
    EXPECT_EQ(involutions(7), 232u);
}

TEST(ComboDyckPaths, CountMatchesCatalanNum) {
    for (int n = 0; n <= 5; ++n)
        EXPECT_EQ(dyck_paths(n).size(), catalan_num(static_cast<uint32_t>(n)));
}

TEST(ComboDyckPaths, ValidBalancedPaths) {
    for (int n = 0; n <= 4; ++n) {
        auto paths = dyck_paths(n);
        for (const auto& p : paths) {
            EXPECT_EQ(static_cast<int>(p.size()), 2 * n);
            EXPECT_TRUE(is_balanced_dyck(p));
            int depth = 0;
            for (char c : p) {
                if (c == '(') ++depth;
                else --depth;
                EXPECT_GE(depth, 0);
            }
        }
        std::set<std::string> unique(paths.begin(), paths.end());
        EXPECT_EQ(unique.size(), paths.size());
    }
}

TEST(ComboDyckPaths, KnownSmallCases) {
    EXPECT_EQ(dyck_paths(0), (std::vector<std::string>{""}));
    EXPECT_EQ(dyck_paths(1), (std::vector<std::string>{"()"}));
    auto paths3 = dyck_paths(3);
    EXPECT_EQ(paths3.size(), 5u);
}

TEST(ComboMotzkinPaths, CountMatchesMotzkinNum) {
    for (int n = 0; n <= 6; ++n)
        EXPECT_EQ(motzkin_paths(n).size(), motzkin_num(static_cast<uint32_t>(n)));
}

TEST(ComboMotzkinPaths, ValidPaths) {
    for (int n = 0; n <= 5; ++n) {
        auto paths = motzkin_paths(n);
        for (const auto& p : paths) {
            EXPECT_EQ(static_cast<int>(p.size()), n);
            EXPECT_TRUE(is_valid_motzkin_path(p));
        }
        std::set<std::string> unique(paths.begin(), paths.end());
        EXPECT_EQ(unique.size(), paths.size());
    }
}

TEST(ComboMotzkinPaths, KnownSmallCases) {
    EXPECT_EQ(motzkin_paths(0), (std::vector<std::string>{""}));
    EXPECT_EQ(motzkin_paths(1), (std::vector<std::string>{"F"}));
    auto paths3 = motzkin_paths(3);
    EXPECT_EQ(paths3.size(), 4u);
}

TEST(ComboNecklaces, BinaryLength3Count) {
    auto necks = necklaces(3, 2);
    EXPECT_EQ(necks.size(), 4u);
}

TEST(ComboNecklaces, BinaryLength3Representatives) {
    auto necks = necklaces(3, 2);
    std::set<std::vector<int>> expected{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    std::set<std::vector<int>> actual(necks.begin(), necks.end());
    EXPECT_EQ(actual, expected);
}

TEST(ComboNecklaces, SmallCounts) {
    EXPECT_EQ(necklaces(0, 2).size(), 1u);
    EXPECT_EQ(necklaces(1, 3).size(), 3u);
    EXPECT_EQ(necklaces(2, 2).size(), 3u);
    EXPECT_EQ(necklaces(4, 2).size(), 6u);
}

TEST(ComboLyndonWords, BinaryLength3) {
    auto words = lyndon_words(3, 2);
    EXPECT_EQ(words.size(), 2u);
    std::set<std::vector<int>> expected{{0, 0, 1}, {0, 1, 1}};
    std::set<std::vector<int>> actual(words.begin(), words.end());
    EXPECT_EQ(actual, expected);
}

TEST(ComboLyndonWords, BinaryLength2) {
    auto words = lyndon_words(2, 2);
    EXPECT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0], (std::vector<int>{0, 1}));
}

TEST(ComboLyndonWords, Length1OverKAlphabet) {
    EXPECT_EQ(lyndon_words(1, 1).size(), 1u);
    EXPECT_EQ(lyndon_words(1, 2).size(), 2u);
    EXPECT_EQ(lyndon_words(1, 4).size(), 4u);
    EXPECT_EQ(lyndon_words(1, 3)[1], (std::vector<int>{1}));
}

TEST(ComboLyndonWords, AdditionalSmallCounts) {
    EXPECT_EQ(lyndon_words(2, 3).size(), 3u);
    EXPECT_EQ(lyndon_words(4, 2).size(), 3u);
}
