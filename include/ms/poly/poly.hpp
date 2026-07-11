#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace ms {
namespace poly {

// Evaluate polynomial (Horner) at x — returns {value}
std::vector<double> poly_eval(const std::vector<double>& coeffs, double x);

// Arithmetic
std::vector<double> poly_add(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> poly_sub(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> poly_mul(const std::vector<double>& a, const std::vector<double>& b);

// Division: returns {quotient, ..., remainder starts at n+1}
// Convention: result = quotient coeffs; use poly_mod for remainder
std::vector<double> poly_div_quot(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> poly_mod(const std::vector<double>& a, const std::vector<double>& b);

// Calculus
std::vector<double> poly_deriv(const std::vector<double>& coeffs);
std::vector<double> poly_integ(const std::vector<double>& coeffs, double c = 0.0);

// Composition: p(q(x))
std::vector<double> poly_compose(const std::vector<double>& p,
                                  const std::vector<double>& q);

// GCD / LCM
std::vector<double> poly_gcd(const std::vector<double>& a,
                               const std::vector<double>& b);
std::vector<double> poly_lcm(const std::vector<double>& a,
                               const std::vector<double>& b);

// Exponentiation (n >= 0; n=0 returns {1}; negative n unsupported)
std::vector<double> poly_pow(const std::vector<double>& p, int n);

// Normalize leading coefficient to 1 (zero poly returned unchanged)
std::vector<double> poly_monic(const std::vector<double>& p);

// Reverse coefficient order (reciprocal polynomial)
std::vector<double> poly_reverse(const std::vector<double>& p);

// Taylor shift: returns q with q(x) = p(x - a)
std::vector<double> poly_shift(const std::vector<double>& p, double a);

// Scale argument: returns q with q(x) = p(a * x)
std::vector<double> poly_scale(const std::vector<double>& p, double a);

// Sylvester matrix (columns x^{m+n-1} .. x^0, descending power left-to-right)
ColMatrix<double> poly_sylvester(const std::vector<double>& p,
                                  const std::vector<double>& q);

// Resultant Res(p, q) = det(Sylvester(p, q)); zero iff p and q share a root
double poly_resultant(const std::vector<double>& p,
                       const std::vector<double>& q);

// Discriminant of p (zero iff p has a repeated root)
double poly_discriminant(const std::vector<double>& p);

// Square-free part: p / gcd(p, p')
std::vector<double> poly_squarefree(const std::vector<double>& p);

// Bernstein basis B_{n,i}(x) = C(n,i) x^i (1-x)^{n-i}
double bernstein(int n, int i, double x);

// Roots via companion matrix eigenvalues (real coefficients)
std::vector<std::complex<double>> poly_roots(const std::vector<double>& coeffs);

// Polynomial fitting: find coefficients of degree n poly through (xs, ys)
std::vector<double> poly_fit(const std::vector<double>& xs,
                              const std::vector<double>& ys, int degree);

// Lagrange interpolation
std::vector<double> poly_lagrange(const std::vector<double>& xs,
                                   const std::vector<double>& ys);

// Newton's divided-difference interpolation; same interpolating polynomial as
// poly_lagrange (up to numerical error), same ascending-coefficient convention.
// Returns {} on size mismatch or duplicate xs entries.
std::vector<double> interp_newton(const std::vector<double>& xs,
                                   const std::vector<double>& ys);

// Hermite interpolation: unique degree-<2n polynomial p with p(xs[i]) == ys[i]
// and p'(xs[i]) == dys[i] for every i. Returns {} on size mismatch or
// duplicate xs entries.
std::vector<double> interp_hermite(const std::vector<double>& xs,
                                    const std::vector<double>& ys,
                                    const std::vector<double>& dys);

// Chebyshev evaluation
double poly_cheb_eval(const std::vector<double>& cheb_coeffs, double x);

// Sturm chain: count real roots in [a, b]
int poly_root_count(const std::vector<double>& p, double a, double b);

// --- Rational root extraction / factorization ---
//
// Coefficient convention: same as the rest of this module (ascending powers,
// coeffs[0] is the constant term). Coefficients are treated as exact integers
// by rounding to the nearest integer; this module does NOT accept arbitrary
// rational coefficients or integrate with ms::bignum::Rational. Rationale:
// every other function in ms::poly operates on std::vector<double>, and
// pulling ms::bignum::Rational (BigInt-backed) into this otherwise
// double-only module would be inconsistent with that existing convention and
// would force every caller through a conversion. Callers with genuinely
// rational (non-integer) coefficients should clear denominators themselves
// (multiply through by the LCM of denominators) before calling; if any
// coefficient is not within `tol` of an integer after that, a Result error
// is returned rather than silently truncating.
//
// Finds all RATIONAL roots of a polynomial with (near-)integer coefficients,
// via the Rational Root Theorem: any rational root p/q (lowest terms) of
// a_n*x^n + ... + a_0 must have p | a_0 and q | a_n. Candidate p/q pairs are
// generated from ms::numthy::divisors(|a_0|) and ms::numthy::divisors(|a_n|),
// tested via poly_eval and confirmed via exact-if-possible synthetic
// division (poly_div_quot / poly_mod), then divided out to expose higher
// multiplicities of the same root. Returns each root as a reduced
// {numerator, denominator} pair with denominator > 0, with multiplicity
// (i.e. a double root appears twice), in the order found.
//
// Errors (Result):
//  - DomainError if any input coefficient is not within `tol` of an integer.
//  - DomainError if the (stripped) input is the zero polynomial (every x is
//    a root -- not representable as a finite list).
//  - DomainError if a leading/trailing (after stripping x=0 roots) integer
//    coefficient magnitude exceeds a safety limit (see kRationalRootSafetyLimit
//    in poly.cpp), to avoid an expensive/unbounded divisor search.
// A constant (degree-0) nonzero polynomial is valid input and simply yields
// an empty root list (not an error).
Result<std::vector<std::pair<int64_t,int64_t>>> poly_rational_roots(
    const std::vector<double>& p, double tol = 1e-6);

// Full rational factorization built on poly_rational_roots: repeatedly
// divides out each rational root p/q via synthetic division by the linear
// factor (q*x - p) until no further rational roots remain.
struct RationalFactorization {
    // One entry per linear factor found (q*x - p for root p/q), with
    // multiplicity, same order as poly_rational_roots.
    std::vector<std::pair<int64_t,int64_t>> linear_roots;
    // What remains after dividing out every rational root: an
    // irreducible-over-Q (as far as rational roots go) remainder, or a
    // nonzero constant {c} if the polynomial fully factors into linear
    // terms. Same ascending-coefficient convention as the input. Multiplying
    // `remainder` by (den*x - num) for every entry in `linear_roots` (via
    // poly_mul) reconstructs the original polynomial (up to floating-point
    // tolerance).
    std::vector<double> remainder;
};

Result<RationalFactorization> poly_factor_rational(const std::vector<double>& p,
                                                     double tol = 1e-6);

} // namespace poly

// Backward-compat aliases (unqualified namespace ms::)
using poly::poly_eval;
using poly::poly_add;
using poly::poly_sub;
using poly::poly_mul;
using poly::poly_deriv;

} // namespace ms
