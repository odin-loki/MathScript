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
// next/prev combination: v is a k-subset of {0..n-1}, modifies in-place
bool next_comb(std::vector<int>& v, int n);
bool prev_comb(std::vector<int>& v, int n);

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
// Generate all integer partitions of n into exactly k positive parts (non-increasing order)
std::vector<std::vector<int>> restricted_partitions(int n, int k);

// Derangements of 0..n-1
std::vector<std::vector<int>> derangements(int n);

// Catalan number C_n = C(2n,n)/(n+1)
uint64_t catalan_num(uint32_t n);

// Stirling numbers
uint64_t stirling1(uint32_t n, uint32_t k);  // unsigned, first kind |s(n,k)|
uint64_t stirling2(uint32_t n, uint32_t k);  // second kind S(n,k)

// Eulerian numbers A(n,k): permutations of {1..n} with exactly k ascents
uint64_t eulerian_number(uint32_t n, uint32_t k);

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

/// Distinct bracelets of length n over k colors (rotation + reflection
/// equivalence, i.e. the dihedral group D_n of order 2n rather than just
/// the cyclic group C_n of order n used by necklaces()).
///
/// Two length-n strings over a k-color alphabet are the same bracelet if
/// one can be reached from the other by a rotation, a reflection, or a
/// rotation composed with a reflection. Each returned entry is the
/// lexicographically smallest string among all 2n rotations of the string
/// and of its reversal (its canonical representative), mirroring how
/// necklaces() picks the lexicographically smallest of the n rotations.
///
/// @param n  Length of the strings (bracelet size).
/// @param k  Number of available colors/symbols (alphabet size).
/// @return   One canonical representative per distinct bracelet, in the
///           order their representatives are produced by the internal
///           k-ary odometer. Since every necklace-equivalence-class is
///           either its own reflection or gets merged with exactly one
///           other necklace's class, bracelets(n, k).size() <=
///           necklaces(n, k).size() always holds, with equality only in
///           degenerate cases (e.g. n <= 2, k == 1).
/// @note Defensive: returns {} for n < 0 or k <= 0, and {{}} (one empty
///       representative) for n == 0 — matching necklaces()'s convention.
/// @note Brute-force over all k^n strings, each checked against its 2n
///       dihedral images: O(k^n * n) time. Practical only for small n
///       (the same small-n range as necklaces()/lyndon_words(), e.g.
///       n up to roughly 15-20 for small k); not suitable for large n.
std::vector<std::vector<int>> bracelets(int n, int k);

// Lyndon words of length n over k symbols (aperiodic, strictly minimal rotation)
std::vector<std::vector<int>> lyndon_words(int n, int k);

/// Standard binary reflected Gray code of n bits: a sequence of all 2^n
/// distinct n-bit binary strings such that consecutive strings (including
/// wrapping from the last back to the first) differ in exactly one bit.
///
/// Each code is returned as a length-n vector of 0/1 ints in
/// most-significant-bit-first order (i.e. code[0] is the highest-order
/// bit), matching how a binary string literal reads left-to-right. Codes
/// are produced in Gray-code order via the standard closed-form
/// construction: entry i is the binary representation of `i XOR (i >> 1)`.
///
/// @param n  Number of bits. n < 0 returns {}. n == 0 returns {{}} (one
///           empty representative), matching this module's convention for
///           a "trivial" n == 0 case (see necklaces(), bracelets()).
/// @return   2^n codes, each a length-n 0/1 vector, in Gray-code order.
/// @note 2^n grows exponentially; intended for small n (e.g. n <= 20), like
///       the other brute-force enumeration functions in this module.
std::vector<std::vector<int>> gray_code(int n);

/// De Bruijn sequence B(k, n): a cyclic sequence over a k-symbol alphabet
/// (symbols 0..k-1) of length k^n such that every one of the k^n possible
/// length-n strings over that alphabet appears exactly once as a
/// contiguous, cyclically-wrapping substring.
///
/// Constructed via the standard FKM (Fredricksen-Kessler-Maiorana)
/// algorithm: the sequence is the concatenation, in lexicographic order,
/// of all Lyndon words over {0,...,k-1} whose length divides n.
///
/// @param k  Alphabet size. k <= 0 returns {}.
/// @param n  Substring length. n <= 0 returns {}.
/// @return   The k^n symbols of the sequence, each in [0, k), NOT
///           including the wrap-around repetition of the first n-1
///           symbols at the end — the caller must conceptually wrap
///           around to the front to read the final few length-n windows.
/// @note k^n grows exponentially; intended for small k, n (e.g. k^n up to
///       a few thousand).
std::vector<int> de_bruijn_sequence(int k, int n);

} // namespace combo
} // namespace ms
