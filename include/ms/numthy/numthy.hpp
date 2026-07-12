#pragma once
#include "ms/error/error_types.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace ms {
namespace numthy {

// --- Primality & primes ---
bool isprime(uint64_t n);
uint64_t nextprime(uint64_t n);
uint64_t prevprime(uint64_t n);          // returns 0 if none
std::vector<uint64_t> primes(uint64_t lo, uint64_t hi);  // sieve
uint64_t prime_pi(uint64_t n);           // count of primes <= n
uint64_t prime_nth(uint64_t n);          // nth prime (1-indexed)

// --- Factorisation ---
std::vector<uint64_t> factor(uint64_t n);          // prime factors, sorted
std::vector<std::pair<uint64_t,int>> factor_exp(uint64_t n); // {p, e} pairs

// --- Divisor functions ---
std::vector<uint64_t> divisors(uint64_t n);
uint64_t num_divisors(uint64_t n);       // tau(n)
uint64_t sum_divisors(uint64_t n);       // sigma(n)
uint64_t euler_phi(uint64_t n);          // Euler totient
// Jordan totient J_k(n) = n^k * prod_{p|n}(1 - 1/p).  J_1(n) == euler_phi(n).
// Returns 0 if k == 0, n == 0, or the result would overflow uint64_t.
uint64_t jordan_totient(uint32_t k, uint64_t n);
int64_t  mobius(uint64_t n);             // Möbius function: -1,0,1
int      liouville(uint64_t n);          // Liouville lambda: ±1
// von Mangoldt: ln(p) if n = p^k (p prime, k >= 1), else 0
double   von_mangoldt(uint64_t n);

// --- Arithmetic ---
uint64_t gcd(uint64_t a, uint64_t b);
uint64_t lcm(uint64_t a, uint64_t b);
// Returns {g, x, y} s.t. a*x + b*y = g
std::tuple<int64_t,int64_t,int64_t> extended_gcd(int64_t a, int64_t b);
Result<uint64_t> mod_inv(uint64_t a, uint64_t m);   // modular inverse
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod);

// --- Modular arithmetic ---
// Chinese Remainder Theorem: find x s.t. x ≡ r[i] (mod m[i])
Result<uint64_t> crt(const std::vector<uint64_t>& r,
                     const std::vector<uint64_t>& m);

// Legendre / Jacobi / Kronecker symbols
int legendre_symbol(int64_t a, uint64_t p);   // p must be odd prime
// Quadratic residues mod odd prime p; empty if p <= 2 or p not prime
std::vector<uint64_t> quadratic_residues(uint64_t p);
int jacobi_symbol(int64_t a, uint64_t n);
int kronecker_symbol(int64_t a, int64_t n);

// --- Discrete log (baby-step giant-step): g^x ≡ h (mod p) ---
Result<uint64_t> discrete_log(uint64_t g, uint64_t h, uint64_t p);
bool is_primitive_root(uint64_t g, uint64_t p);
uint64_t primitive_root(uint64_t p);

// --- Tonelli-Shanks: modular square root, x^2 ≡ n (mod p) ---
Result<uint64_t> tonelli_shanks(uint64_t n, uint64_t p);

// Cornacchia's algorithm: find non-negative (x, y) with x^2 + d*y^2 = p,
// for prime p and 0 < d < p. Error if p is not prime, d == 0, d >= p, or
// no representation exists.
Result<std::pair<uint64_t,uint64_t>> cornacchia(uint64_t d, uint64_t p);

// --- Continued fractions ---
std::vector<int64_t> continued_fraction(double x, int max_terms = 20);
// Convergents p/q from CF coefficients
std::vector<std::pair<int64_t,int64_t>> convergents(
    const std::vector<int64_t>& cf);

// Farey sequence F_n: reduced fractions p/q in [0,1], 1 <= q <= n, ascending
std::vector<std::pair<uint64_t,uint64_t>> farey(uint32_t n);

// Stern-Brocot tree: first n nodes (numerator, denominator) in breadth-first
// order, starting from the root 1/1. Convention matches farey(n): n bounds
// the *output size* (node count), not the tree depth. Node p/q has left
// child p/(p+q) and right child (p+q)/q (mediant construction); every node
// is automatically in reduced form. n == 0 returns an empty vector.
std::vector<std::pair<uint64_t,uint64_t>> stern_brocot(uint64_t n);

// Fundamental Pell solution (x,y) with x^2 - D*y^2 = 1; error if D is a perfect square.
// Uses integer sqrt(D) continued-fraction period + convergents (not double CF).
Result<std::pair<uint64_t,uint64_t>> pell_solve(uint64_t D);

// --- Partition function ---
uint64_t partition(uint32_t n);   // number of integer partitions of n

// Carmichael function λ(n): the smallest positive integer m such that a^m ≡ 1 (mod n) for
// every integer a coprime to n. Equivalently the exponent of the multiplicative group
// (Z/nZ)*; used in RSA key selection because λ(n) is typically tighter than Euler's
// totient φ(n) (always λ(n) ≤ φ(n), with equality when (Z/nZ)* is cyclic). Computed as
// lcm over prime power factors p^k of n of λ(p^k), where λ(2)=1, λ(4)=2,
// λ(2^k)=2^(k-2) for k≥3 (the group mod 2^k is not cyclic for k≥3), and λ(p^k)=φ(p^k)
// for odd prime p (the group mod p^k is cyclic). Uses factor_exp and lcm internally.
// @param n positive integer. λ(1)=1 by convention.
// @return λ(n), or 0 if n==0 (invalid input).
uint64_t carmichael_lambda(uint64_t n);

// Korselt's criterion Carmichael number test: n is a Carmichael number iff n is composite,
// squarefree, and for every prime p dividing n, (p-1) divides (n-1). Carmichael numbers are the
// classic counterexamples that make Fermat's little theorem primality tests unreliable (they
// pass Fermat's test for EVERY base coprime to n, despite being composite) -- the smallest is
// 561 = 3*11*17. This function checks the criterion directly via existing factor_exp (squarefree
// iff every exponent is exactly 1) rather than testing many Fermat bases, since Korselt's
// criterion is both necessary AND sufficient (no probabilistic uncertainty, unlike a
// finite-base Fermat test).
// @param n candidate. Returns false for n < 3, prime n, or n failing any part of the criterion.
bool is_carmichael(uint64_t n);

// Lucas sequences U_k(P,Q) and V_k(P,Q): the generalizations of Fibonacci (U_k with P=1,Q=-1)
// and Lucas numbers (V_k with P=1,Q=-1) satisfying the shared recurrence
//   U_0=0, U_1=1, U_k = P*U_{k-1} - Q*U_{k-2}
//   V_0=2, V_1=P, V_k = P*V_{k-1} - Q*V_{k-2}
// Computed via binary (matrix) exponentiation of the companion matrix [[P,-Q],[1,0]] -- the
// same O(log k) binary-exponentiation strategy mod_pow uses for modular exponentiation --
// rather than naive O(k) iteration.
// @param k index. k < 0 is clamped to k == 0 behavior (negative-index Lucas sequences require
//        division and are out of scope here; documented limitation, not silently wrong output).
// @param P, Q sequence parameters (may be negative).
// @return {U_k, V_k} as a pair.
// @note Uses int64_t arithmetic with no overflow checking; large k/P/Q combinations can overflow
//       silently. BigInt support is out of scope for this function.
std::pair<int64_t, int64_t> lucas_sequence(int64_t k, int64_t P, int64_t Q);

// Multiplicative order of a modulo n: the smallest positive integer k such that
// a^k ≡ 1 (mod n). Exists iff gcd(a, n) == 1. By Euler's theorem the order always
// divides φ(n), so it is found by starting at k = φ(n) and, for each prime factor p
// of φ(n), repeatedly dividing k by p while p | k and a^(k/p) ≡ 1 (mod n) still holds
// (mirrors the classic "reduce Euler's exponent down to the true order" algorithm).
// Uses euler_phi, factor_exp, and mod_pow internally.
// @param a base. @param n modulus.
// @return the multiplicative order of a mod n, or an error (mirroring mod_inv's
//         "gcd(a,m) != 1" convention) if n == 0 or gcd(a, n) != 1. n == 1 returns 1
//         (the trivial group has order 1, and gcd(a,1) == 1 for every a).
Result<uint64_t> multiplicative_order(uint64_t a, uint64_t n);

} // namespace numthy
} // namespace ms
