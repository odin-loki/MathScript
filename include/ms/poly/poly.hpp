#pragma once

#include "ms/core/matrix.hpp"
#include <complex>
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

} // namespace poly

// Backward-compat aliases (unqualified namespace ms::)
using poly::poly_eval;
using poly::poly_add;
using poly::poly_sub;
using poly::poly_mul;
using poly::poly_deriv;

} // namespace ms
