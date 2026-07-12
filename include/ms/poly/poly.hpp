#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <complex>
#include <cstdint>
#include <functional>
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

// Chebyshev series expansion: computes the coefficients c_0..c_n such that
// poly_cheb_eval(poly_cheb_expand(f, n, a, b), x) approximates f(x) for x in
// [a, b]. This is the dual/inverse operation of poly_cheb_eval: that function
// consumes a coefficient vector and evaluates the resulting series at a
// point, while this function produces the coefficient vector from an
// arbitrary callable f.
//
// Algorithm (discrete-cosine-transform / Chebyshev-node-sampling, closely
// related to Clenshaw-Curtis quadrature):
//  1. Sample f at the n+1 Chebyshev points of the first kind on [-1, 1],
//     x_k = cos((2k+1)*pi / (2*(n+1))) for k = 0..n, mapped onto [a, b] via
//     x_mapped_k = (a+b)/2 + (b-a)/2 * x_k.
//  2. For each order j = 0..n, use the discrete orthogonality of the
//     Chebyshev polynomials over these nodes:
//       c_j = (2/(n+1)) * sum_{k=0}^{n} f(x_mapped_k) * T_j(x_k),
//     evaluating T_j(x_k) = cos(j * acos(x_k)) directly.
//  3. c_0 is then halved so the result matches poly_cheb_eval's convention,
//     which evaluates the series as c_0 + sum_{j=1}^{n} c_j*T_j(x) (i.e.
//     poly_cheb_eval does NOT itself halve coeffs[0] -- see PolyChebEval.
//     T0IsOne, where coeffs={1.0} evaluates to exactly 1.0). Folding the
//     usual 1/2 factor into c_0 here (rather than leaving it to the caller
//     or to poly_cheb_eval) keeps the two functions a true round-trip pair.
//
// @param f function to expand; sampled at n+1 points, no derivatives needed.
// @param n degree of the expansion; returns n+1 coefficients c_0..c_n.
//          n < 0 returns {} (invalid). n == 0 returns a single coefficient
//          equal to f evaluated at the midpoint (a+b)/2 (the one Chebyshev
//          node for degree 0), i.e. a constant approximation.
// @param a left endpoint of the interval (default -1.0).
// @param b right endpoint of the interval (default 1.0).
// @return vector of n+1 coefficients c_0..c_n, directly usable with
//         poly_cheb_eval (after mapping the evaluation point x in [a, b]
//         back to [-1, 1] via t = (2*x - (a+b)) / (b-a), since poly_cheb_eval
//         itself always operates on the canonical [-1, 1] argument).
// @note If f is itself a polynomial of degree <= n, the expansion reproduces
//       it exactly (up to floating-point error), since {T_0..T_n} exactly
//       spans that space. For smooth-but-non-polynomial f, the approximation
//       error shrinks rapidly (typically geometrically) as n grows, provided
//       f is analytic in a neighborhood of [a, b]; for less smooth f
//       convergence is slower but still improves with n.
std::vector<double> poly_cheb_expand(std::function<double(double)> f, int n,
                                      double a = -1.0, double b = 1.0);

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

// --- Partial fraction decomposition ---
//
// One term of a partial fraction decomposition: either a real-pole term
// A / (x - r)^k, or (when is_quadratic is true) a term for a
// complex-conjugate pole pair combined into a real quadratic factor,
// (B*x + C) / (x^2 + p*x + q)^k with discriminant p^2 - 4*q < 0.
struct PartialFractionTerm {
    double A = 0.0, B = 0.0, C = 0.0;   // A/(x-r)^k  or  (B*x+C)/(x^2+p*x+q)^k
    double r = 0.0;                      // real pole (for A-type terms)
    double p = 0.0, q = 0.0;             // quadratic factor coefficients (for B,C-type terms)
    int k = 1;                           // multiplicity
    bool is_quadratic = false;           // true => (B*x+C)/(x^2+p*x+q)^k, false => A/(x-r)^k
};

// Result of a partial fraction decomposition: a polynomial quotient part
// (nonempty only if deg(numerator) >= deg(denominator) to begin with) plus
// the list of proper-fraction terms for the remainder.
struct PartialFractionResult {
    std::vector<double> quotient;        // polynomial quotient part (empty if deg(N)<deg(D))
    std::vector<PartialFractionTerm> terms;
};

// Partial fraction decomposition of the real rational function
// numerator(x) / denominator(x). Coefficients use this module's usual
// ascending-power convention (index 0 = constant term; see poly_eval).
//
// Algorithm:
//  1. If deg(numerator) >= deg(denominator), polynomial long division
//     (poly_div_quot / poly_mod) splits off a polynomial `quotient`, leaving
//     a proper-fraction remainder numerator N_r(x) with deg(N_r) <
//     deg(denominator). Only N_r(x) / denominator(x) is decomposed into
//     terms below.
//  2. denominator's roots are found via poly_roots (Durand-Kerner; returns
//     one complex root per degree, with algebraic multiplicity). Roots with
//     near-zero imaginary part are treated as real poles; the remainder are
//     paired up as complex-conjugate pairs and combined into real quadratic
//     factors x^2 + p*x + q (p = -2*Re(root), q = |root|^2), since this
//     module works in real (not ms::cplx) arithmetic.
//  3. Roots are grouped into distinct real poles / distinct quadratic
//     factors, each with a multiplicity, by clustering within a small
//     relative tolerance (see kPartialFractionRootTol in poly.cpp) -- a
//     numerical root-finder applied to a polynomial with an exact repeated
//     root of multiplicity k will not return k EXACTLY equal roots in
//     floating point, so nearby roots are merged rather than compared for
//     exact equality.
//  4. Rather than the classical calculus-based approach (where a repeated
//     real pole's k coefficients come from Taylor-expanding N_r(x)/D_rest(x)
//     around that pole -- only the simple-pole case reduces to the familiar
//     residue formula A = N_r(r) / D'(r)), this implementation clears
//     denominators (multiplies the assumed decomposition through by
//     denominator(x)) and matches coefficients of every power of x, giving
//     a square linear system of exactly deg(denominator) equations in
//     deg(denominator) unknowns (the A / B,C coefficients), solved via
//     straightforward Gaussian elimination with partial pivoting. This
//     "coefficient matching" method is mathematically equivalent to the
//     classical residue/Taylor-coefficient method for every pole shape
//     (simple real, repeated real, complex-conjugate pair) but avoids
//     needing separate case-by-case calculus for repeated poles, at the
//     cost of a (typically small, at most deg(denominator) x
//     deg(denominator)) dense linear solve.
//
// @param numerator N(x), ascending powers.
// @param denominator D(x), ascending powers. A zero or empty denominator
//        yields an empty result (no quotient, no terms) -- there is no
//        finite decomposition to return.
// @return quotient (empty if deg(N) < deg(D) initially) plus the list of
//         partial fraction terms for the proper-fraction remainder.
PartialFractionResult poly_partial_fractions(const std::vector<double>& numerator,
                                              const std::vector<double>& denominator);

} // namespace poly

// Backward-compat aliases (unqualified namespace ms::)
using poly::poly_eval;
using poly::poly_add;
using poly::poly_sub;
using poly::poly_mul;
using poly::poly_deriv;

} // namespace ms
