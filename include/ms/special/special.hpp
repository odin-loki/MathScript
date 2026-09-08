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
/// Imaginary error function erfi(x) = (2/√π) Σ_{n≥0} x^{2n+1}/(n!(2n+1)) = (2/√π)∫₀ˣ e^{t²}dt.
/// All series terms share the sign of x, so there is no cancellation; relative error ≲1e-15
/// until the result overflows (|x| ≳ 26.6).
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

// Airy functions Ai, Bi and their derivatives, solutions of y'' = x y (DLMF §9).
// Evaluated from the convergent Maclaurin series for |x| < 1 and from the exact Bessel
// connection formulas (DLMF 9.6.2-9.6.9) in ζ = (2/3)|x|^{3/2} for |x| ≥ 1, so both the
// oscillatory region x < 0 and the exponential region x > 0 are covered.
// @accuracy Relative error ≲1e-14 over the whole real line (measured against the DLMF 9.7
//           asymptotic expansions and a long-double Taylor march of y'' = x y); the Wronskian
//           Ai(x)Bi'(x) − Ai'(x)Bi(x) = 1/π holds to ~1e-16. Ai underflows to 0 beyond x ≈ 105
//           and Bi overflows beyond x ≈ 104, as the true functions do.
double airy_ai(double x);
double airy_bi(double x);
double airy_aip(double x);
double airy_bip(double x);

// Bessel functions (integer order nu ≥ 0).
// J: ascending series or the stable upward recurrence below the turning point, Miller's
//    backward recurrence above it, and the Hankel asymptotic expansion for large x.
// Y: ascending log-series (Y₀, Y₁) plus the stable upward recurrence, Hankel for large x.
// I: ascending series at every order — the upward recurrence is the UNSTABLE direction for I.
// K: the integral representation ∫₀^∞ e^{-x cosh t} cosh(nu t) dt by exponentially convergent
//    trapezoid, carried to high order by the stable upward recurrence of DLMF 10.29.1.
// @accuracy Relative error ≲1e-13 across the tested range; the Wronskians
//           J_{n+1}(x)Y_n(x) − J_n(x)Y_{n+1}(x) = 2/(πx) and
//           I_n(x)K_{n+1}(x) + I_{n+1}(x)K_n(x) = 1/x both hold to ~1e-15.
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

// Bessel function zeros (n is 1-based index): the n-th POSITIVE zero of J_nu resp. Y_nu.
// The bracket is found by marching from the origin with a step below the minimum spacing of
// consecutive zeros, so the returned root really is the n-th one, is strictly positive and is
// strictly increasing in n; it is then bisected and polished by Newton.
double bessel_zero_jnu(int nu, int n);
double bessel_zero_ynu(int nu, int n);

// Struve functions (DLMF §11).
/// Struve H_nu(x) = Σ_m (−1)^m (x/2)^{2m+nu+1}/(Γ(m+3/2)Γ(m+nu+3/2)); for |x| > 20 evaluated as
/// Y_nu(x) plus the DLMF 11.6.1 expansion, where the ascending series would cancel.
double struve_h(int nu, double x);
/// MODIFIED Struve L_nu(x): the same series without the (−1)^m sign (DLMF 11.2.2).
double struve_l(int nu, double x);
/// Struve function of the second kind K_nu(x) = H_nu(x) − Y_nu(x) (DLMF 11.2.5).
double struve_k(int nu, double x);
/// Integer-order alias of struve_h.
double struve_hn(int nu, double x);
/// Alias of struve_k (H_nu − Y_nu).
double struve_yn(int nu, double x);

// Anger and Weber functions
double anger_j(int nu, double x);
double weber_e(int nu, double x);

// Kelvin functions of general integer order nu ≥ 0 (DLMF §10.61):
//   ber_nu(x) + i bei_nu(x) = J_nu(x e^{3πi/4})   — ascending series DLMF 10.65.1
//   ker_nu(x) + i kei_nu(x) = e^{−nuπi/2} K_nu(x e^{iπ/4}) — complex K by the same
//                                                            exponentially convergent quadrature
// @accuracy ber/bei lose ≈0.29x/ln10 digits to cancellation (≲1e-13 at x = 10, ≲1e-10 at
//           x = 30); ker/kei are accurate to ~1e-15 throughout.
double kelvin_ber(int nu, double x);
double kelvin_bei(int nu, double x);
double kelvin_ker(int nu, double x);
double kelvin_kei(int nu, double x);

// Orthogonal polynomials
double legendre_p(int n, double x);
/// Legendre function of the second kind Q_n(x) on (−1, 1) (DLMF §14.3): the FULL function,
/// Q_n(x) = ½P_n(x)ln((1+x)/(1−x)) − W_{n−1}(x), generated from Q₀ = atanh x, Q₁ = xQ₀ − 1 by
/// the three-term recurrence P and Q share. NaN for |x| ≥ 1 or n < 0.
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
/// Associated (generalised) Laguerre polynomial L_n^{(k)}(x) for integer k ≥ 0.
/// Returns 0 for k < 0 or n < 0.
double laguerre_ln(int n, int k, double x);
double chebyshev_t(int n, double x);
double chebyshev_u(int n, double x);
/// k-th derivative of the Chebyshev polynomials, dᵏT_n/dxᵏ and dᵏU_n/dxᵏ, via the Gegenbauer
/// differentiation formula (DLMF 18.9.19-18.9.21). k = 0 reproduces chebyshev_t/chebyshev_u,
/// k > n gives 0, k < 0 or |x| > 1 gives NaN.
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

// Weierstrass elliptic functions from the Laurent expansion about z = 0.  All four are derived
// from ONE coefficient array generated by the DLMF 23.9.3 recurrence
//   ℘(z) = z⁻² + Σ_{n≥2} c_n z^{2n−2},  c₂ = g₂/20, c₃ = g₃/28,
//   c_n = 3/((2n+1)(n−3)) Σ_{m=2}^{n−2} c_m c_{n−m},
// so ℘′ = −2z⁻³ + Σ(2n−2)c_n z^{2n−3}, ζ = 1/z − Σ c_n z^{2n−1}/(2n−1) and
// σ = z exp(−Σ c_n z^{2n}/(2n(2n−1))) — making ζ′ = −℘ and σ′/σ = ζ exact by construction, and
// ℘′² = 4℘³ − g₂℘ − g₃ exact to round-off.
// The series converge only for |z| below the modulus of the nearest lattice point; outside that
// disc the truncated terms diverge and NaN is returned rather than a meaningless partial sum.
double weierstrass_p(double z, double g2, double g3);
double weierstrass_pprime(double z, double g2, double g3);
double weierstrass_zeta(double z, double g2, double g3);
double weierstrass_sigma(double z, double g2, double g3);

// Zeta and related functions (DLMF Ch. 25)
/// Riemann zeta function ζ(s) for all real s; +∞ at the pole s = 1 (DLMF §25.2).
/// s > 1: Euler-Maclaurin (DLMF 25.2.9). 0 < s < 1: η(s)/(1−2^{1−s}) with η by Borwein
/// acceleration. s = 0: −1/2. s < 0: the reflection formula ζ(s) = 2^s π^{s−1} sin(πs/2)
/// Γ(1−s) ζ(1−s), with the trivial zeros at negative even integers returned as exactly 0.
double zeta(double s);
double zeta_hurwitz(double s, double a);
double lerch_phi(double z, double s, double a);
/// Dirichlet eta function η(s) = (1 − 2^{1−s}) ζ(s) (DLMF §25.3), defined for ALL real s.
/// For s ≤ 1 it is summed by Borwein's Chebyshev acceleration, whose error bound
/// 3/(3+√8)^N is below 1e-22 at the N = 30 used here.
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

// Mathieu functions (x in radians), DLMF §28.
/// Characteristic values a_n(q) and b_n(q): the n-th eigenvalue of the appropriate one of the
/// FOUR symmetric tridiagonal recurrence matrices (√2-enhanced first coupling for the even
/// cosine family, ±q on the first diagonal entry for the odd families). b_0 does not exist
/// (se_0 ≡ 0) and returns NaN.
double mathieu_a(int n, double q);
double mathieu_b(int n, double q);
/// Periodic Mathieu functions ce_n(x,q), se_n(x,q) in the standard A&S 20.2.27 normalisation
/// (∫₀^{2π} ce_n² dx = ∫₀^{2π} se_n² dx = π), with ce_n → cos nx and se_n → sin nx as q → 0.
double mathieu_ce(int n, double q, double x);
double mathieu_se(int n, double q, double x);
/// Modified (radial) Mathieu functions Mc_n(z,q) = ce_n(iz,q) and Ms_n(z,q) = −i se_n(iz,q),
/// solutions of y'' = (a − 2q cosh 2z) y in the same normalisation as ce_n / se_n above; in
/// particular Mc_n(0,q) = ce_n(0,q) and Ms_n(0,q) = 0.
/// Evaluated from the cosh/sinh Fourier series while √q e^{|z|} ≲ 25 and continued with RK4 on
/// the defining equation beyond that; NaN once the continuation would exceed its step budget
/// (roughly √q e^{|z|} > 8e4).
double mathieu_mc(int n, double q, double x);
double mathieu_ms(int n, double q, double x);

// Prolate spheroidal ANGULAR functions (m ≥ 0, n ≥ m), for the equation
//   d/dx[(1−x²)dS/dx] + [λ − c²x² − m²/(1−x²)]S = 0  on −1 < x < 1.
/// Eigenvalue λ_mn(c): the (n−m)/2-th eigenvalue of the symmetric tridiagonal matrix of that
/// operator in the orthonormal Ferrers basis. Exactly n(n+1) at c = 0, and
/// n(n+1) + c²⟨x²⟩_{n,m} + O(c⁴) for small c. NaN for n < m or m < 0.
double spheroidal_lambda(int n, int m, double c);
/// Angular function of the first kind S1_mn(c,x) = Σ_r d_r P_{m+r}^m(x) in the FLAMMER
/// normalisation (S1 → P_n^m(x) as c → 0, imposed by matching the value at x = 0 when n−m is
/// even and the derivative there when it is odd).
double spheroidal_s1(int n, int m, double c, double x);
/// Angular function of the second kind S2_mn(c,x): the solution of the same equation with the
/// parity opposite to S1 and the Legendre-matched Wronskian
/// (1−x²)(S1 S2′ − S1′ S2) = (−1)^m (n+m)!/(n−m)!, so that S2 → Q_n^m(x) as c → 0. Obtained by
/// RK4 from x = 0; the equation degenerates at x = ±1, so accuracy falls off as |x| → 1 and
/// |x| ≥ 1 returns NaN.
double spheroidal_s2(int n, int m, double c, double x);

// Parabolic cylinder functions (DLMF §12).
/// U(a,x) = D_{−a−1/2}(x), the recessive solution of y'' − (x²/4 + a)y = 0.
/// x > 0 via D_nu(x) = 2^{nu/2}e^{−x²/4}U(−nu/2, 1/2, x²/2); x ≤ 0 via the two Kummer M series
/// of DLMF 12.4.1-12.4.2, which carry no cancellation there.
double pcf_u(double a, double x);
/// V(a,x) = (Γ(½+a)/π)(sin(πa)U(a,x) + U(a,−x)) (DLMF 12.2.20). At a = −½, −3/2, … the Γ pole
/// cancels a vanishing bracket; those points are filled in by averaging two nearby a (≈1e-9).
double pcf_v(double a, double x);
/// W(a,x): the OSCILLATORY parabolic cylinder function, solution of y'' + (x²/4 − a)y = 0,
/// from DLMF 12.14.4 with G₁ = |Γ(¼ + ia/2)|, G₃ = |Γ(¾ + ia/2)|. It is not a real-coefficient
/// combination of U and V.
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
/// Tricomi confluent hypergeometric U(a,b,z), z > 0. For a > 0 the cancellation-free integral
/// representation U = (1/Γ(a))∫₀^∞ e^{−zt}t^{a−1}(1+t)^{b−a−1}dt is evaluated with a
/// double-exponential trapezoid rule (valid for every b, integer b included); for large z the
/// optimally truncated DLMF 13.7.3 expansion in −1/z is used, and a ≤ 0 is reached by the
/// recurrence DLMF 13.3.7 from shifted values. @accuracy relative error ≲1e-13.
double tricomi_u(double a, double b, double z);
double whittaker_m(double kappa, double mu, double z);
double whittaker_w(double kappa, double mu, double z);
/// The single Meijer G case this signature can express, in closed form:
///   G^{1,1}_{1,1}(z | a ; b) = Γ(1−a+b) z^b (1+z)^{a−b−1}   (z > 0; 0 for z ≤ 0).
double meijer_g(double a, double b, double z);
/// Fox H-function with unit exponents, H^{1,1}_{1,1}[z | (a,1) ; (b,1)]. With A₁ = B₁ = 1 the
/// H-function reduces exactly to the Meijer G above, so this returns the same value; the
/// signature carries no A_j/B_j and therefore cannot express any other H-function.
double fox_h(double a, double b, double z);
double hypergeo_0f1n(int n, double a, double z);
double hypergeo_1f1n(int n, double a, double z);

// Heun equations (DLMF 31.1.1 and 31.12.1-31.12.4), integrated with RK4 from the base point
// noted per function. Where the shipped signature has fewer or differently named slots than
// DLMF, the mapping is given below.
/// General Heun: y'' + (γ/z + δ/(z−1) + ε/(z−a))y' + (αβz − q)/(z(z−1)(z−a))y = 0 with ε fixed
/// by the Fuchs relation ε = α+β−γ−δ+1. Local exponent-zero (Frobenius) solution at z = 0,
/// y(0) = 1, y'(0) = q/(aγ). Defined only up to the first singular point to the right of the
/// origin — NaN for z ≥ min(1, a) (a > 0) resp. z ≥ 1.
double heun_g(double a, double q, double alpha, double beta, double gamma, double delta, double z);
/// Confluent Heun (DLMF 31.12.1): y'' + (γ/z + δ/(z−1) + ε)y' + (αz − q)/(z(z−1))y = 0, with
/// `beta` carrying ε. Frobenius solution at z = 0 (y(0) = 1, y'(0) = −q/γ); NaN for z ≥ 1.
double heun_c(double q, double alpha, double beta, double gamma, double delta, double z);
/// Doubly confluent Heun (DLMF 31.12.2): z²y'' + (−z² + δz + γ)y' + (αz − q)y = 0. z = 0 is an
/// irregular singular point, so the base point is the ordinary point z = 1 with y(1) = 1,
/// y'(1) = 0 and the integration runs in either direction. NaN for z ≤ 0.
double heun_d(double q, double alpha, double gamma, double delta, double z);
/// Biconfluent Heun (DLMF 31.12.3): zy'' + (1+α−βz−2z²)y' + ((γ−α−2)z − (δ+(1+α)β)/2)y = 0,
/// with `q` carrying DLMF's γ. Frobenius solution at z = 0.
double heun_b(double q, double alpha, double beta, double delta, double z);
/// Triconfluent Heun (DLMF 31.12.4): y'' − (γ + 3z²)y' + (αz − q)y = 0, normalised by y(0) = 1,
/// y'(0) = 0 at the ordinary point z = 0. The THE has only three parameters; `beta` is accepted
/// for signature symmetry with the other members and does not appear in the equation.
double heun_t(double q, double alpha, double beta, double gamma, double z);

// Painlevé transcendents, the DLMF 32.2.1-32.2.6 equations integrated with RK4 from the base
// point x0 given per function with the caller's initial data (y0, yp0); x ≤ x0 returns y0.
double painleve1(double x, double y0, double yp0);                       // x0 = 0
double painleve2(double x, double y0, double yp0, double alpha);         // x0 = 0
/// PIII (DLMF 32.2.3): w'' = w'²/w − w'/z + (αw² + β)/z + γw³ + δ/w, x0 = 0.1. The signature
/// carries only (α, β), so γ = 1 and δ = −1 are fixed — the standard generic normalisation
/// PIII(α, β, 1, −1) to which every non-degenerate PIII can be rescaled.
double painleve3(double x, double y0, double yp0, double alpha, double beta);
/// PIV (DLMF 32.2.4): w'' = w'²/(2w) + (3/2)w³ + 4zw² + 2(z²−α)w + β/w, x0 = 0.1.
double painleve4(double x, double y0, double yp0, double alpha, double beta);
/// PV (DLMF 32.2.5): w'' = (1/(2w)+1/(w−1))w'² − w'/z + (w−1)²/z²(αw + β/w) + γw/z
///                       + δw(w+1)/(w−1), x0 = 0.2.
double painleve5(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta);
/// PVI (DLMF 32.2.6): w'' = ½(1/w+1/(w−1)+1/(w−z))w'² − (1/z+1/(z−1)+1/(w−z))w'
///     + w(w−1)(w−z)/(z²(z−1)²)·[α + βz/w² + γ(z−1)/(w−1)² + δz(z−1)/(w−z)²], x0 = 2.1
/// (an ordinary point; the equation is singular at z = 0, 1, ∞).
double painleve6(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta);

} // namespace ms
