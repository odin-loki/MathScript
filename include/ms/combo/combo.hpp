#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ms {
namespace combo {

// --- Factorials ---
uint64_t factorial(uint32_t n);             // n!
uint64_t double_factorial(uint32_t n);      // n!!
uint64_t subfactorial(uint32_t n);          // D(n) — derangements count

// --- Counting ---
uint64_t binomial(uint32_t n, uint32_t k);  // C(n,k)
uint64_t multinomial(uint32_t n, const std::vector<uint32_t>& ks);
uint64_t permutations(uint32_t n, uint32_t k);   // P(n,k) = n!/(n-k)!
uint64_t combinations(uint32_t n, uint32_t k);   // alias for binomial
uint64_t combinations_with_rep(uint32_t n, uint32_t k);  // C(n+k-1, k)

// --- Enumeration ---
// next/prev permutation: modifies in-place, returns false at end/start
bool next_perm(std::vector<int>& v);
bool prev_perm(std::vector<int>& v);
// next combination: v is a k-subset of {0..n-1}, modifies in-place
bool next_comb(std::vector<int>& v, int n);

// Ranking / unranking
uint64_t rank_permutation(const std::vector<int>& v);
std::vector<int> unrank_permutation(int n, uint64_t rank);
uint64_t rank_combination(const std::vector<int>& v, int n);
std::vector<int> unrank_combination(int n, int k, uint64_t rank);

// Generate all permutations of 0..n-1
std::vector<std::vector<int>> all_permutations(int n);
// Generate all subsets of {0..n-1}
std::vector<std::vector<int>> all_subsets(int n);
// Generate all integer compositions of n into at most k parts
std::vector<std::vector<int>> all_compositions(int n, int max_parts = -1);
// Generate all integer partitions of n (non-increasing order)
std::vector<std::vector<int>> all_partitions(int n);

// Derangements of 0..n-1
std::vector<std::vector<int>> derangements(int n);

// Catalan number C_n = C(2n,n)/(n+1)
uint64_t catalan_num(uint32_t n);

// Stirling numbers
uint64_t stirling1(uint32_t n, uint32_t k);  // unsigned, first kind |s(n,k)|
uint64_t stirling2(uint32_t n, uint32_t k);  // second kind S(n,k)

// Bell numbers B_n
uint64_t bell_num(uint32_t n);

// Motzkin numbers M_n
uint64_t motzkin_num(uint32_t n);

// Set partitions of {0..n-1} (enumeration analogue of bell_num)
std::vector<std::vector<std::vector<int>>> set_partitions(int n);

// Involution count I(n): permutations that are their own inverse
uint64_t involutions(uint32_t n);

// All Dyck paths of semilength n (balanced '(' ')' strings; count = catalan_num(n))
std::vector<std::string> dyck_paths(int n);

// All Motzkin paths of length n over 'U'/'D'/'F' (count = motzkin_num(n))
std::vector<std::string> motzkin_paths(int n);

// Distinct necklaces of length n over k colors (rotation equivalence)
std::vector<std::vector<int>> necklaces(int n, int k);

// Lyndon words of length n over k symbols (aperiodic, strictly minimal rotation)
std::vector<std::vector<int>> lyndon_words(int n, int k);

} // namespace combo
} // namespace ms
