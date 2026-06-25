#pragma once

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

// Roots via companion matrix eigenvalues (real coefficients)
std::vector<std::complex<double>> poly_roots(const std::vector<double>& coeffs);

// Polynomial fitting: find coefficients of degree n poly through (xs, ys)
std::vector<double> poly_fit(const std::vector<double>& xs,
                              const std::vector<double>& ys, int degree);

// Lagrange interpolation
std::vector<double> poly_lagrange(const std::vector<double>& xs,
                                   const std::vector<double>& ys);

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
