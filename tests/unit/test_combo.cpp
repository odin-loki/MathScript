#include "ms/combo/combo.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <set>
#include <utility>

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

// All 2n dihedral images of v: its n rotations plus the n rotations of its
// reversal. Used to test bracelets() for uniqueness/closure under D_n.
std::vector<std::vector<int>> dihedral_images(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    std::vector<std::vector<int>> images;
    for (int shift = 0; shift < n; ++shift) {
        std::vector<int> rotated(n);
        for (int i = 0; i < n; ++i) rotated[i] = v[(i + shift) % n];
        images.push_back(std::move(rotated));
    }
    std::vector<int> rev(v.rbegin(), v.rend());
    for (int shift = 0; shift < n; ++shift) {
        std::vector<int> rotated(n);
        for (int i = 0; i < n; ++i) rotated[i] = rev[(i + shift) % n];
        images.push_back(std::move(rotated));
    }
    return images;
}

std::vector<int> canonical_bracelet(const std::vector<int>& v) {
    auto images = dihedral_images(v);
    return *std::min_element(images.begin(), images.end());
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

// Hamming distance between two equal-length 0/1 vectors.
int hamming_distance(const std::vector<int>& a, const std::vector<int>& b) {
    int dist = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++dist;
    return dist;
}

std::size_t ipow(int base, int exp) {
    std::size_t r = 1;
    for (int i = 0; i < exp; ++i) r *= static_cast<std::size_t>(base);
    return r;
}

// Extracts every cyclic length-n window of `seq` (wrapping past the end
// back to the beginning), encoding each window as a base-k integer for
// easy deduplication/hashing.
std::set<uint64_t> cyclic_windows(const std::vector<int>& seq, int k, int n) {
    std::set<uint64_t> windows;
    int len = static_cast<int>(seq.size());
    for (int start = 0; start < len; ++start) {
        uint64_t code = 0;
        for (int j = 0; j < n; ++j) {
            code = code * static_cast<uint64_t>(k) +
                   static_cast<uint64_t>(seq[(start + j) % len]);
        }
        windows.insert(code);
    }
    return windows;
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

TEST(ComboBracelets, BinaryLength3Count) {
    auto bracs = bracelets(3, 2);
    EXPECT_EQ(bracs.size(), 4u);
}

TEST(ComboBracelets, BinaryLength3Representatives) {
    // n=3 is small enough that every necklace class over {0,1} is already
    // its own reflection, so the bracelet reps match the necklace reps.
    auto bracs = bracelets(3, 2);
    std::set<std::vector<int>> expected{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    std::set<std::vector<int>> actual(bracs.begin(), bracs.end());
    EXPECT_EQ(actual, expected);
}

TEST(ComboBracelets, SmallKnownCounts) {
    // n=1: every single symbol is its own reflection, same as necklaces.
    EXPECT_EQ(bracelets(1, 2).size(), 2u);
    EXPECT_EQ(bracelets(1, 3).size(), 3u);
    // n=2: reflection just swaps the two positions, so bracelets(2,k) is
    // exactly the number of unordered pairs with repetition, k*(k+1)/2.
    EXPECT_EQ(bracelets(2, 2).size(), 3u);
    EXPECT_EQ(bracelets(2, 3).size(), 6u);
    EXPECT_EQ(bracelets(2, 4).size(), 10u);
    // Brute-force-verified counts for slightly larger (n,k), including
    // cases where bracelets strictly merges necklace classes together.
    EXPECT_EQ(bracelets(3, 3).size(), 10u);   // necklaces(3,3) == 11
    EXPECT_EQ(bracelets(4, 2).size(), 6u);    // == necklaces(4,2)
    EXPECT_EQ(bracelets(4, 3).size(), 21u);   // necklaces(4,3) == 24
    EXPECT_EQ(bracelets(6, 2).size(), 13u);   // necklaces(6,2) == 14
}

TEST(ComboBracelets, CountNeverExceedsNecklaceCount) {
    // Bracelets merge necklace classes whose reflection lands in a
    // different rotation class, so the count can only shrink or stay
    // equal, never grow, relative to necklaces() for the same (n, k).
    for (int n = 0; n <= 7; ++n) {
        for (int k = 1; k <= 4; ++k) {
            EXPECT_LE(bracelets(n, k).size(), necklaces(n, k).size())
                << "n=" << n << " k=" << k;
        }
    }
}

TEST(ComboBracelets, NoDuplicateEquivalenceClassesInOutput) {
    // No two distinct returned representatives should be related by any
    // rotation or reflection of each other.
    const std::vector<std::pair<int, int>> cases{{3, 3}, {4, 3}, {5, 2}, {6, 2}};
    for (const auto& [n, k] : cases) {
        auto bracs = bracelets(n, k);
        for (std::size_t i = 0; i < bracs.size(); ++i) {
            std::set<std::vector<int>> orbit_i;
            for (auto& img : dihedral_images(bracs[i])) orbit_i.insert(std::move(img));
            for (std::size_t j = 0; j < bracs.size(); ++j) {
                if (i == j) continue;
                EXPECT_EQ(orbit_i.count(bracs[j]), 0u)
                    << "n=" << n << " k=" << k << " i=" << i << " j=" << j;
            }
        }
    }
}

TEST(ComboBracelets, EveryRepresentativeIsItsOwnCanonicalForm) {
    // Each returned representative must already be the lexicographically
    // smallest among its own 2n dihedral images (self-consistency of the
    // canonicalization).
    for (auto n : {3, 4, 5}) {
        for (auto k : {2, 3}) {
            auto bracs = bracelets(n, k);
            for (const auto& b : bracs)
                EXPECT_EQ(canonical_bracelet(b), b);
        }
    }
}

TEST(ComboBracelets, ClosedUnderRotationAndReflection) {
    // Applying any rotation or reflection to any returned representative
    // and re-canonicalizing must land back on an element already present
    // in the returned set (completeness of the enumeration).
    auto bracs = bracelets(4, 3);
    std::set<std::vector<int>> repset(bracs.begin(), bracs.end());
    ASSERT_FALSE(repset.empty());
    for (const auto& b : bracs) {
        for (const auto& img : dihedral_images(b)) {
            auto canon = canonical_bracelet(img);
            EXPECT_TRUE(repset.count(canon) > 0)
                << "canonical form of a dihedral image missing from output";
        }
    }
}

TEST(ComboBracelets, DegenerateCasesMatchNecklacesConvention) {
    EXPECT_EQ(bracelets(-1, 2), necklaces(-1, 2));
    EXPECT_EQ(bracelets(3, 0), necklaces(3, 0));
    EXPECT_EQ(bracelets(3, -2), necklaces(3, -2));
    EXPECT_EQ(bracelets(0, 3), necklaces(0, 3));
    EXPECT_EQ(bracelets(0, 3).size(), 1u);

    // k == 1: only one possible string, regardless of n.
    EXPECT_EQ(bracelets(1, 1).size(), 1u);
    EXPECT_EQ(bracelets(5, 1).size(), 1u);

    // n == 1: every single symbol is trivially its own bracelet.
    EXPECT_EQ(bracelets(1, 5).size(), 5u);
}

TEST(ComboGrayCode, DegenerateCases) {
    EXPECT_EQ(gray_code(0), (std::vector<std::vector<int>>{{}}));
    EXPECT_EQ(gray_code(-1), (std::vector<std::vector<int>>{}));
    EXPECT_EQ(gray_code(-5), (std::vector<std::vector<int>>{}));
}

TEST(ComboGrayCode, OneBitCase) {
    EXPECT_EQ(gray_code(1), (std::vector<std::vector<int>>{{0}, {1}}));
}

TEST(ComboGrayCode, CountIsPowerOfTwo) {
    for (int n : {2, 3, 4, 5}) {
        EXPECT_EQ(gray_code(n).size(), static_cast<std::size_t>(1) << n)
            << "n=" << n;
    }
}

TEST(ComboGrayCode, EveryCodeHasExactlyNBits) {
    for (int n : {0, 1, 2, 3, 4, 5}) {
        auto codes = gray_code(n);
        for (const auto& c : codes)
            EXPECT_EQ(static_cast<int>(c.size()), n) << "n=" << n;
    }
}

TEST(ComboGrayCode, AllCodesDistinct) {
    for (int n : {1, 2, 3, 4, 5}) {
        auto codes = gray_code(n);
        std::set<std::vector<int>> unique(codes.begin(), codes.end());
        EXPECT_EQ(unique.size(), codes.size()) << "n=" << n;
    }
}

TEST(ComboGrayCode, ConsecutivePairsDifferByExactlyOneBit) {
    // The definitive Gray code correctness check: every consecutive pair
    // (including the wrap-around from last back to first) must have
    // Hamming distance exactly 1.
    for (int n : {1, 2, 3, 4, 5, 6}) {
        auto codes = gray_code(n);
        std::size_t count = codes.size();
        for (std::size_t i = 0; i < count; ++i) {
            const auto& cur = codes[i];
            const auto& next = codes[(i + 1) % count];
            EXPECT_EQ(hamming_distance(cur, next), 1)
                << "n=" << n << " i=" << i;
        }
    }
}

TEST(ComboDeBruijnSequence, DegenerateCases) {
    EXPECT_EQ(de_bruijn_sequence(0, 3), (std::vector<int>{}));
    EXPECT_EQ(de_bruijn_sequence(2, 0), (std::vector<int>{}));
    EXPECT_EQ(de_bruijn_sequence(-1, 3), (std::vector<int>{}));
    EXPECT_EQ(de_bruijn_sequence(2, -1), (std::vector<int>{}));
}

TEST(ComboDeBruijnSequence, LengthIsKToTheN) {
    EXPECT_EQ(de_bruijn_sequence(2, 3).size(), 8u);
    EXPECT_EQ(de_bruijn_sequence(2, 4).size(), 16u);
    EXPECT_EQ(de_bruijn_sequence(3, 2).size(), 9u);
    EXPECT_EQ(de_bruijn_sequence(2, 5).size(), 32u);
}

TEST(ComboDeBruijnSequence, SymbolsWithinAlphabetRange) {
    for (int sym : de_bruijn_sequence(4, 3)) {
        EXPECT_GE(sym, 0);
        EXPECT_LT(sym, 4);
    }
}

TEST(ComboDeBruijnSequence, EveryWindowAppearsExactlyOnceCyclically) {
    // The definitive De Bruijn sequence correctness check: treating the
    // sequence as cyclic, every one of the k^n possible length-n strings
    // must appear as a contiguous window exactly once.
    const std::vector<std::pair<int, int>> cases{{2, 3}, {3, 2}, {2, 5}};
    for (const auto& [k, n] : cases) {
        auto seq = de_bruijn_sequence(k, n);
        ASSERT_EQ(seq.size(), ipow(k, n)) << "k=" << k << " n=" << n;
        auto windows = cyclic_windows(seq, k, n);
        EXPECT_EQ(windows.size(), seq.size())
            << "k=" << k << " n=" << n
            << ": expected every length-n window to be distinct";
    }
}

TEST(ComboDeBruijnSequence, LargerCaseExercisesAlgorithm) {
    // k=2, n=5 gives a 32-symbol sequence, meaningfully exercising the
    // FKM construction beyond the trivial small cases above.
    auto seq = de_bruijn_sequence(2, 5);
    ASSERT_EQ(seq.size(), 32u);
    auto windows = cyclic_windows(seq, 2, 5);
    EXPECT_EQ(windows.size(), 32u);
}
