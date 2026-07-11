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

} // namespace numthy
} // namespace ms
