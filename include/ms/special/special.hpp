// MathScript Special Functions Header

#pragma once

#include <complex>

namespace ms {

// Error functions and related
double erf(double x);
double erfc(double x);
/// Inverse error function erf⁻¹(x), x ∈ (−1, 1) (DLMF §7.17).
double erfinv(double x);
/// Inverse complementary error function erfc⁻¹(x), x ∈ (0, 2) (DLMF §7.17).
double erfcinv(double x);
double erfi(double x);
double erfcx(double x);
double dawson(double x);
double dawsonx(double x);

// Fresnel integrals
double fresnel_c(double x);
double fresnel_s(double x);

// Voigt profile and pseudo-Voigt approximation (spectroscopy line shapes)
/// Voigt profile: normalized convolution of a Gaussian (std dev sigma) and a Lorentzian
/// (half-width gamma), V(x; sigma, gamma) = Re[w(z)] / (sigma sqrt(2 pi)), z = (x + i gamma) /
/// (sigma sqrt(2)), where w is the Faddeeva function. Integrates to 1 over all x.
/// Requires sigma > 0; gamma < 0 is clamped to 0. gamma = 0 reduces to a pure Gaussian; as
/// sigma -> 0 with gamma > 0 fixed, V approaches a pure Lorentzian. sigma <= 0 falls back to
/// the pure Lorentzian (gamma > 0) or a Dirac-delta-like spike at x = 0 (gamma <= 0 too).
double voigt(double x, double sigma, double gamma);
/// Pseudo-Voigt approximation: eta * L(x; gamma_pv) + (1 - eta) * G(x; sigma_pv), a cheap
/// linear-combination approximation to voigt() using a shared FWHM derived from sigma and
/// gamma via the Thompson-Cox-Hastings mixing formula. eta in [0, 1] is the caller-supplied
/// Lorentzian mixing fraction (see pseudo_voigt_auto() to derive eta automatically).
double pseudo_voigt(double x, double sigma, double gamma, double eta);
/// Pseudo-Voigt with eta derived automatically from sigma/gamma via the Thompson-Cox-Hastings
/// formula eta = 1.36603 r - 0.47719 r^2 + 0.11116 r^3, r = f_L / f (shared FWHM f).
double pseudo_voigt_auto(double x, double sigma, double gamma);

// Gamma family
double gamma_func(double x);
double log_gamma(double x);
/// Reciprocal gamma 1/Γ(x); zero at non-positive integers (pole cancellation).
double rgamma(double x);
double beta_func(double a, double b);
double digamma(double x);
/// Trigamma ψ′(x) = ψ⁽¹⁾(x), the first derivative of digamma (DLMF §5.15).
double trigamma(double x);
/// Polygamma ψ⁽ⁿ⁾(x); n = 0 delegates to digamma, n ≥ 1 via Hurwitz zeta (DLMF §5.15).
double polygamma(int n, double x);
/// Rising factorial (a)_n = a(a+1)…(a+n−1) for integer n ≥ 0.
double pochhammer(double a, int n);
/// Falling factorial a^{(n)} = a(a−1)…(a−n+1) for integer n ≥ 0.
double falling_factorial(double a, int n);
/// Regularized lower incomplete gamma P(a, x) = γ(a, x)/Γ(a) (DLMF §8.2).
double gamma_inc_reg(double a, double x);
/// Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x) (DLMF §8.2).
double gamma_inc_reg_upper(double a, double x);
/// Lower incomplete gamma γ(a, x) = P(a, x) Γ(a) (DLMF §8.2).
double gamma_inc(double a, double x);
/// Regularized incomplete beta I_x(a, b) (DLMF §8.17).
double beta_inc_reg(double x, double a, double b);
/// Incomplete beta B_x(a, b) = I_x(a, b) B(a, b) (DLMF §8.17).
double beta_inc(double x, double a, double b);

// Combinatorial constants
double bernoulli_number(int n);
double euler_number(int n);

// Airy functions
double airy_ai(double x);
double airy_bi(double x);
double airy_aip(double x);
double airy_bip(double x);

// Bessel functions (integer order)
double bessel_j(int nu, double x);
double bessel_y(int nu, double x);
double bessel_i(int nu, double x);
double bessel_k(int nu, double x);
double bessel_j0(double x);
double bessel_j1(double x);
double bessel_y0(double x);
double bessel_y1(double x);

// Legacy aliases
double bessel_h(int nu, double x);
double bessel_hy(int nu, double x);
double bessel_l(int nu, double x);
double bessel_lu(int nu, double x);

// Spherical Bessel functions
double spherical_jn(int n, double x);
double spherical_yn(int n, double x);
double spherical_in(int n, double x);
double spherical_kn(int n, double x);

// Spherical Bessel functions of the first kind, j_n(x): solutions to the spherical Bessel
// ODE arising in 3D wave/Helmholtz problems in spherical coordinates (radial part of the
// solution). Closed forms for the lowest orders:
//   j_0(x) = sin(x)/x
//   j_1(x) = sin(x)/x^2 - cos(x)/x
// Higher orders via the standard stable upward recurrence:
//   j_{n+1}(x) = ((2n+1)/x) * j_n(x) - j_{n-1}(x)
// (This upward recurrence is numerically stable for j_n specifically because j_n decays with
// increasing n for fixed x -- unlike the OTHER direction, which would be unstable.)
// @param n order (n >= 0; n < 0 returns NaN). @param x argument. x == 0 is a
//        removable-singularity special case: j_0(0) = 1 (the limit of sin(x)/x as x->0),
//        j_n(0) = 0 for n >= 1 (all higher orders vanish at the origin).
// @return j_n(x).
double sph_bessel_j(int n, double x);

// Spherical Bessel functions of the second kind, y_n(x) (also called spherical Neumann
// functions): the second linearly independent solution, singular at x=0. Closed forms:
//   y_0(x) = -cos(x)/x
//   y_1(x) = -cos(x)/x^2 - sin(x)/x
// Higher orders via the SAME upward recurrence form as j_n (this direction is stable for y_n
// too, for a different reason -- y_n GROWS with n, so accumulated round-off from the recurrence
// is relatively small compared to the answer):
//   y_{n+1}(x) = ((2n+1)/x) * y_n(x) - y_{n-1}(x)
// @param n order (n >= 0; n < 0 returns NaN). @param x argument. x <= 0 is a domain error
//        (y_n(0) is a genuine singularity for all n, and this module's other second-kind
//        Bessel functions of this signature, e.g. bessel_y() and spherical_yn(), likewise
//        restrict to x > 0) -- returns NaN, matching that convention.
// @return y_n(x).
double sph_bessel_y(int n, double x);

// Bessel function zeros (n is 1-based index)
double bessel_zero_jnu(int nu, int n);
double bessel_zero_ynu(int nu, int n);

// Struve functions
double struve_h(int nu, double x);
double struve_l(int nu, double x);
double struve_k(int nu, double x);
double struve_hn(int nu, double x);
double struve_yn(int nu, double x);

// Anger and Weber functions
double anger_j(int nu, double x);
double weber_e(int nu, double x);

// Kelvin functions
double kelvin_ber(int nu, double x);
double kelvin_bei(int nu, double x);
double kelvin_ker(int nu, double x);
double kelvin_kei(int nu, double x);

// Orthogonal polynomials
double legendre_p(int n, double x);
double legendre_q(int n, double x);
double legendre_pn(int n, int m, double x);
/// Associated Legendre polynomial P_l^m(x), m may be negative (DLMF §14.9).
/// For m >= 0 this matches legendre_pn(l, m, x); negative m uses
/// P_l^{-m}(x) = (-1)^m (l-m)!/(l+m)! P_l^m(x).
double assoc_legendre_p(int l, int m, double x);
double hermite_h(int n, double x);
double hermite_hf(int n, double x);
double hermite_hn(int n, double x);
double laguerre_l(int n, double x);
double laguerre_ln(int n, int k, double x);
double chebyshev_t(int n, double x);
double chebyshev_u(int n, double x);
double chebyshev_tn(int n, int k, double x);
double chebyshev_un(int n, int k, double x);

// Extended orthogonal polynomials
double hermite_he(int n, double x);
double laguerre_la(int n, double a, double x);
double chebyshev_v(int n, double x);
double chebyshev_w(int n, double x);
double gegenbauer_c(int n, double lambda, double x);
double jacobi_p(int n, double alpha, double beta, double x);
/// Complex orthonormal spherical harmonic Y_l^m(θ, φ) (DLMF §14.30):
/// Y_l^m(θ, φ) = √((2l+1)/(4π) (l-m)!/(l+m)!) P_l^m(cos θ) e^{imφ},
/// with Condon–Shortley phase in P_l^m. θ is colatitude [0, π], φ azimuth.
/// @param l degree (l >= 0; |m| > l returns NaN+NaN i). @param m order.
/// @param theta colatitude. @param phi azimuth.
/// @return Y_l^m(θ, φ) as std::complex<double>.
std::complex<double> sph_harm_y(int l, int m, double theta, double phi);
/// Real part of sph_harm_y(l, m, theta, phi); legacy convenience wrapper.
double sph_harm(int l, int m, double theta, double phi);

// Elliptic integrals (k is the modulus, |k| < 1)
double ellip_k(double k);
double ellip_e(double k);
double ellip_pi(double n, double k);
double ellip_f(double phi, double k);
double ellip_e_inc(double phi, double k);
double ellip_d(double k);

// Jacobi elliptic functions
double jacobi_sn(double u, double k);
double jacobi_cn(double u, double k);
double jacobi_dn(double u, double k);
double jacobi_am(double u, double k);
double jacobi_sc(double u, double k);
double jacobi_sd(double u, double k);
double jacobi_nd(double u, double k);
double jacobi_nc(double u, double k);
double jacobi_dc(double u, double k);
double jacobi_cs(double u, double k);
double jacobi_ns(double u, double k);
double jacobi_ds(double u, double k);
double jacobi_cd(double u, double k);

// Theta functions (q is the nome, |q| < 1)
double theta1(double z, double q);
double theta2(double z, double q);
double theta3(double z, double q);
double theta4(double z, double q);
double theta1_prime(double z, double q);
double jacobi_theta(int n, double z, double tau);

// Weierstrass elliptic functions (short series near the origin)
double weierstrass_p(double z, double g2, double g3);
double weierstrass_pprime(double z, double g2, double g3);
double weierstrass_zeta(double z, double g2, double g3);
double weierstrass_sigma(double z, double g2, double g3);

// Zeta and related functions (DLMF Ch. 25)
/// Riemann zeta function ζ(s) for real s; pole at s=1 (DLMF §25.2).
double zeta(double s);
double zeta_hurwitz(double s, double a);
double lerch_phi(double z, double s, double a);
/// Dirichlet eta function η(s) = (1 - 2^{1-s}) ζ(s) (DLMF §25.3).
double eta_dirichlet(double s);
double beta_dirichlet(double s);
/// Polylogarithm Li_n(z) = Σ_{k=1}^∞ z^k / k^n, |z| < 1 (DLMF §25.12).
double polylog(int n, double z);
/// Clausen function Cl_2(θ) = -∫_0^θ ln|2 sin(t/2)| dt = Σ sin(kθ)/k² (DLMF §27.8).
double clausen(double x);
/// Debye function D_n(x) = (n/x^n) ∫_0^x t^n/(e^t - 1) dt, n ≥ 1, x > 0 (DLMF §6.3).
double debye(int n, double x);
/// Lambert W function W_k(z), branch k = 0 (principal, z ≥ −1/e) or k = −1 (z ∈ [−1/e, 0)).
/// Satisfies W_k(z) exp(W_k(z)) = z; evaluated via Fritsch–Halley iteration.
double lambert_w(int branch, double z);

// Mathieu functions (x in radians)
double mathieu_a(int n, double q);
double mathieu_b(int n, double q);
double mathieu_ce(int n, double q, double x);
double mathieu_se(int n, double q, double x);
double mathieu_mc(int n, double q, double x);
double mathieu_ms(int n, double q, double x);

// Spheroidal wave functions (prolate, m >= 0, n >= m)
double spheroidal_lambda(int n, int m, double c);
double spheroidal_s1(int n, int m, double c, double x);
double spheroidal_s2(int n, int m, double c, double x);

// Parabolic cylinder functions
double pcf_u(double a, double x);
double pcf_v(double a, double x);
double pcf_w(double a, double x);

// Hypergeometric functions
double hypergeo_0f1(double b, double z);
double hypergeo_1f1(double a, double z);
double hypergeo_2f1(double a, double b, double c, double z);
double kummer_m(double a, double b, double z);
/// Kummer confluent hypergeometric function of the second kind U(a,b,z) (DLMF §13.2).
/// Linearly independent from kummer_m(a,b,z); decays as z^(-a) for large z when Re(a)>0.
/// Evaluated via the connection formula in terms of kummer_m() for z in (0, 20], and via
/// the leading large-z asymptotic U(a,b,z) ~ z^(-a) for z > 20.
/// @param a upper parameter. @param b lower parameter (non-integer; integer b returns NaN).
/// @param z argument (z > 0).
/// @return U(a,b,z).
/// @note Returns NaN for z <= 0 or integer b (except the exact U(a,a+1,z)=z^(-a)
///       identity). For b >= 2 the connection formula is bypassed in favour of the
///       existing Tricomi U series because M(a-b+1,2-b,z) is ill-conditioned there.
/// @accuracy Relative error typically below ~1e-3 for non-integer b and z in (0, 20]; the
///           z > 20 branch matches the leading asymptotic z^(-a) (O(1/z) corrections omitted).
double kummer_u(double a, double b, double z);
double tricomi_u(double a, double b, double z);
double whittaker_m(double kappa, double mu, double z);
double whittaker_w(double kappa, double mu, double z);
double meijer_g(double a, double b, double z);
double fox_h(double a, double b, double z);
double hypergeo_0f1n(int n, double a, double z);
double hypergeo_1f1n(int n, double a, double z);

// Heun functions (fundamental solution y(z0)=1, y'(z0)=0 via numerical ODE)
double heun_g(double a, double q, double alpha, double beta, double gamma, double delta, double z);
double heun_c(double q, double alpha, double beta, double gamma, double delta, double z);
double heun_d(double q, double alpha, double gamma, double delta, double z);
double heun_b(double q, double alpha, double beta, double delta, double z);
double heun_t(double q, double alpha, double beta, double gamma, double z);

// Painlevé transcendents (y0, yp0 are initial data at the regular point x0)
double painleve1(double x, double y0, double yp0);
double painleve2(double x, double y0, double yp0, double alpha);
double painleve3(double x, double y0, double yp0, double alpha, double beta);
double painleve4(double x, double y0, double yp0, double alpha, double beta);
double painleve5(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta);
double painleve6(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta);

} // namespace ms
