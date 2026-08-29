// Comprehensive numerical reference tests for special functions (extended set).
// Reference values are taken from NIST DLMF, Abramowitz & Stegun, and
// Wolfram Research.  Each TEST block covers at least one function family.

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

#include <ms/special/special.hpp>

using namespace ms;

// ─────────────────────────────────────────────────────────────────────────────
// Bessel J0 / J1 – fast specialised entry points
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, BesselJ0_SpecificPoints) {
    // J0(0) = 1 exactly
    EXPECT_NEAR(bessel_j0(0.0),  1.000000000,  1e-14);
    // DLMF Table 10.2.1
    EXPECT_NEAR(bessel_j0(1.0),  0.765197687,  1e-8);
    // Near first zero of J0 (first zero ≈ 2.4048); value ≈ 0.002508
    EXPECT_NEAR(bessel_j0(2.4),  0.002507684,  1e-7);
    // DLMF Table 10.2.1
    EXPECT_NEAR(bessel_j0(3.0), -0.260051955,  1e-7);
    EXPECT_NEAR(bessel_j0(5.0), -0.177596771,  1e-7);
}

TEST(NumericalSpecialExt, BesselJ1_SpecificPoints) {
    // J1(0) = 0 exactly
    EXPECT_NEAR(bessel_j1(0.0),  0.000000000,  1e-14);
    // DLMF Table 10.2.1
    EXPECT_NEAR(bessel_j1(1.0),  0.440050586,  1e-7);
    // Series-verified: J1(2.4) ≈ 0.520185
    EXPECT_NEAR(bessel_j1(2.4),  0.520185,     1e-5);
    // DLMF Table 10.2.1
    EXPECT_NEAR(bessel_j1(3.0),  0.339058958,  1e-7);
    EXPECT_NEAR(bessel_j1(5.0), -0.327579138,  1e-7);
}

TEST(NumericalSpecialExt, BesselY0_SpecificPoints) {
    // DLMF Table 10.2.2
    EXPECT_NEAR(bessel_y0(1.0),  0.088256964,  1e-7);
    EXPECT_NEAR(bessel_y0(2.0),  0.510375672,  1e-7);
    EXPECT_NEAR(bessel_y0(5.0), -0.308517664,  1e-7);
}

TEST(NumericalSpecialExt, BesselY1_SpecificPoints) {
    // DLMF Table 10.2.2
    EXPECT_NEAR(bessel_y1(1.0), -0.781212821,  1e-7);
    EXPECT_NEAR(bessel_y1(2.0), -0.107032431,  1e-7);
    EXPECT_NEAR(bessel_y1(5.0),  0.147863144,  1e-7);
}

TEST(NumericalSpecialExt, BesselJ_HigherOrders) {
    // Three-term recurrence from DLMF values at x = 5:
    //   J3(5) = (4/5)*J2(5) - J1(5) ≈ 0.364831
    //   J4(5) = (6/5)*J3(5) - J2(5) ≈ 0.391232
    EXPECT_NEAR(bessel_j(3, 5.0), 0.364831,  1e-5);
    EXPECT_NEAR(bessel_j(4, 5.0), 0.391232,  1e-5);
    // Generic interface must match the fast paths
    EXPECT_NEAR(bessel_j(0, 1.0), bessel_j0(1.0), 1e-14);
    EXPECT_NEAR(bessel_j(1, 1.0), bessel_j1(1.0), 1e-14);
    // J2 at x = 2 from DLMF Table 10.2.1
    EXPECT_NEAR(bessel_j(2, 2.0), 0.352834030, 1e-7);
}

// ─────────────────────────────────────────────────────────────────────────────
// Elliptic integrals – complete K(k) and E(k), modulus convention |k| < 1
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, EllipticK_Reference) {
    // K(0) = π/2  (DLMF 19.2.1)
    EXPECT_NEAR(ellip_k(0.0),  M_PI / 2.0,    1e-12);
    // K(0.5) ≈ 1.6857503549  (DLMF Table 19.2.1)
    EXPECT_NEAR(ellip_k(0.5),  1.685750355,   1e-7);
    // K(0.9) ≈ 2.2805493389  (DLMF Table 19.2.1)
    EXPECT_NEAR(ellip_k(0.9),  2.280549339,   1e-5);
    // Symmetry: K(-k) = K(k)
    EXPECT_NEAR(ellip_k(-0.5), ellip_k(0.5),  1e-14);
}

TEST(NumericalSpecialExt, EllipticE_Reference) {
    // E(0) = π/2  (DLMF 19.2.1)
    EXPECT_NEAR(ellip_e(0.0),  M_PI / 2.0,    1e-12);
    // E(0.5) ≈ 1.4674622175  (DLMF Table 19.2.1)
    EXPECT_NEAR(ellip_e(0.5),  1.467462218,   1e-7);
    // E(0.9) ≈ 1.1714913527  (DLMF Table 19.2.1)
    EXPECT_NEAR(ellip_e(0.9),  1.171491353,   1e-3);
    // Symmetry: E(-k) = E(k)
    EXPECT_NEAR(ellip_e(-0.5), ellip_e(0.5),  1e-14);
}

TEST(NumericalSpecialExt, EllipticPi_AtZeroN) {
    // Π(0, k) = K(k) by definition  (DLMF 19.2.4)
    EXPECT_NEAR(ellip_pi(0.0, 0.0), ellip_k(0.0), 1e-12);
    EXPECT_NEAR(ellip_pi(0.0, 0.5), ellip_k(0.5), 1e-8);
    EXPECT_NEAR(ellip_pi(0.0, 0.9), ellip_k(0.9), 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Legendre polynomials  P_n(x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, LegendreP_AtEndpoints) {
    // P_n(1) = 1 for all n  (DLMF 14.6.1)
    EXPECT_NEAR(legendre_p(0,  1.0),  1.0, 1e-14);
    EXPECT_NEAR(legendre_p(1,  1.0),  1.0, 1e-14);
    EXPECT_NEAR(legendre_p(2,  1.0),  1.0, 1e-14);
    EXPECT_NEAR(legendre_p(3,  1.0),  1.0, 1e-14);
    // P_n(-1) = (-1)^n  (DLMF 14.6.1)
    EXPECT_NEAR(legendre_p(0, -1.0),  1.0, 1e-14);
    EXPECT_NEAR(legendre_p(1, -1.0), -1.0, 1e-14);
    EXPECT_NEAR(legendre_p(2, -1.0),  1.0, 1e-14);
    EXPECT_NEAR(legendre_p(3, -1.0), -1.0, 1e-14);
}

TEST(NumericalSpecialExt, LegendreP_AtZero) {
    // P_n(0): 0 for odd n; explicit even values (A&S 22.4)
    EXPECT_NEAR(legendre_p(0, 0.0),  1.0,    1e-14);
    EXPECT_NEAR(legendre_p(1, 0.0),  0.0,    1e-14);
    EXPECT_NEAR(legendre_p(2, 0.0), -0.5,    1e-14);
    EXPECT_NEAR(legendre_p(3, 0.0),  0.0,    1e-14);
    // P_4(0) = 3/8
    EXPECT_NEAR(legendre_p(4, 0.0),  0.375,  1e-14);
}

TEST(NumericalSpecialExt, LegendreP_AtHalf) {
    // Exact values at x = 0.5
    EXPECT_NEAR(legendre_p(0, 0.5),  1.0,     1e-14);
    EXPECT_NEAR(legendre_p(1, 0.5),  0.5,     1e-14);
    // P_2(0.5) = (3*0.25 - 1)/2 = -0.125
    EXPECT_NEAR(legendre_p(2, 0.5), -0.125,   1e-14);
    // P_3(0.5) = (5*0.125 - 3*0.5)/2 = -0.4375
    EXPECT_NEAR(legendre_p(3, 0.5), -0.4375,  1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hermite polynomials – physicists' H_n(x):  H_0=1, H_1=2x, H_2=4x²-2, …
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, HermiteH_AtZeroAndOne) {
    // H_n(0): 1 for n=0, 0 for odd n, (-2)^(n/2)*(n-1)!! for even n
    EXPECT_NEAR(hermite_h(0, 0.0),  1.0,  1e-14);
    EXPECT_NEAR(hermite_h(1, 0.0),  0.0,  1e-14);
    EXPECT_NEAR(hermite_h(2, 0.0), -2.0,  1e-14);
    EXPECT_NEAR(hermite_h(3, 0.0),  0.0,  1e-14);
    EXPECT_NEAR(hermite_h(4, 0.0), 12.0,  1e-14);
    // H_n(1): H_0=1, H_1=2, H_2=2, H_3=-4, H_4=-20
    EXPECT_NEAR(hermite_h(0, 1.0),   1.0, 1e-14);
    EXPECT_NEAR(hermite_h(1, 1.0),   2.0, 1e-14);
    EXPECT_NEAR(hermite_h(2, 1.0),   2.0, 1e-14);
    EXPECT_NEAR(hermite_h(3, 1.0),  -4.0, 1e-14);
    EXPECT_NEAR(hermite_h(4, 1.0), -20.0, 1e-14);
}

TEST(NumericalSpecialExt, HermiteH_AtTwo) {
    // H_n(2): H_0=1, H_1=4, H_2=14, H_3=40, H_4=76
    EXPECT_NEAR(hermite_h(0, 2.0),  1.0,  1e-14);
    EXPECT_NEAR(hermite_h(1, 2.0),  4.0,  1e-14);
    EXPECT_NEAR(hermite_h(2, 2.0), 14.0,  1e-14);
    EXPECT_NEAR(hermite_h(3, 2.0), 40.0,  1e-13);
    EXPECT_NEAR(hermite_h(4, 2.0), 76.0,  1e-13);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hermite polynomials – probabilists' He_n(x): He_0=1, He_1=x, He_2=x²-1, …
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, HermiteHe_Reference) {
    // He_n(0): 1 for n=0, 0 for odd, (-1)^(n/2) (n-1)!! for even
    EXPECT_NEAR(hermite_he(0, 0.0),  1.0,  1e-14);
    EXPECT_NEAR(hermite_he(1, 0.0),  0.0,  1e-14);
    EXPECT_NEAR(hermite_he(2, 0.0), -1.0,  1e-14);
    EXPECT_NEAR(hermite_he(3, 0.0),  0.0,  1e-14);
    // He_n(2): He_0=1, He_1=2, He_2=3, He_3=2
    EXPECT_NEAR(hermite_he(0, 2.0),  1.0,  1e-14);
    EXPECT_NEAR(hermite_he(1, 2.0),  2.0,  1e-14);
    EXPECT_NEAR(hermite_he(2, 2.0),  3.0,  1e-14);
    EXPECT_NEAR(hermite_he(3, 2.0),  2.0,  1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chebyshev T polynomials  T_n(x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, ChebyshevT_AtEndpoints) {
    // T_n(1) = 1 for all n
    EXPECT_NEAR(chebyshev_t(0,  1.0),  1.0, 1e-14);
    EXPECT_NEAR(chebyshev_t(1,  1.0),  1.0, 1e-14);
    EXPECT_NEAR(chebyshev_t(2,  1.0),  1.0, 1e-14);
    EXPECT_NEAR(chebyshev_t(3,  1.0),  1.0, 1e-14);
    // T_n(-1) = (-1)^n
    EXPECT_NEAR(chebyshev_t(0, -1.0),  1.0, 1e-14);
    EXPECT_NEAR(chebyshev_t(1, -1.0), -1.0, 1e-14);
    EXPECT_NEAR(chebyshev_t(2, -1.0),  1.0, 1e-14);
    EXPECT_NEAR(chebyshev_t(3, -1.0), -1.0, 1e-14);
}

TEST(NumericalSpecialExt, ChebyshevT_AtHalf) {
    // x = 0.5 = cos(π/3), so T_n(0.5) = cos(n π/3)
    EXPECT_NEAR(chebyshev_t(0, 0.5),  1.0,  1e-14);
    EXPECT_NEAR(chebyshev_t(1, 0.5),  0.5,  1e-14);
    EXPECT_NEAR(chebyshev_t(2, 0.5), -0.5,  1e-14);
    EXPECT_NEAR(chebyshev_t(3, 0.5), -1.0,  1e-14);
    EXPECT_NEAR(chebyshev_t(4, 0.5), -0.5,  1e-14);
    // T_6(cos(π/3)) = cos(2π) = 1
    EXPECT_NEAR(chebyshev_t(6, 0.5),  1.0,  1e-13);
}

TEST(NumericalSpecialExt, ChebyshevT_CosIdentity) {
    // T_n(cos θ) = cos(nθ)  (DLMF 18.5.1)
    const double theta = M_PI / 5.0;
    const double x = std::cos(theta);
    for (int n = 0; n <= 5; ++n) {
        EXPECT_NEAR(chebyshev_t(n, x), std::cos(n * theta), 1e-12)
            << "T_" << n << "(cos(π/5))";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chebyshev U polynomials  U_n(x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, ChebyshevU_Reference) {
    // U_n(cos θ) = sin((n+1)θ)/sin θ; at x = 0.5 = cos(π/3)
    EXPECT_NEAR(chebyshev_u(0, 0.5),  1.0,  1e-14);
    EXPECT_NEAR(chebyshev_u(1, 0.5),  1.0,  1e-14);  // U_1 = 2x = 1
    // sin(2π/3)/sin(π/3) = 1;  sin(3π/3)/sin(π/3) = 0
    EXPECT_NEAR(chebyshev_u(2, 0.5),  0.0,  1e-14);
    EXPECT_NEAR(chebyshev_u(3, 0.5), -1.0,  1e-13);
    // U_n(1) = n + 1
    EXPECT_NEAR(chebyshev_u(0, 1.0), 1.0, 1e-14);
    EXPECT_NEAR(chebyshev_u(1, 1.0), 2.0, 1e-14);
    EXPECT_NEAR(chebyshev_u(2, 1.0), 3.0, 1e-14);
    EXPECT_NEAR(chebyshev_u(3, 1.0), 4.0, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fresnel integrals  (DLMF convention: C(x) = ∫₀ˣ cos(πt²/2) dt)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, FresnelC_Reference) {
    EXPECT_NEAR(fresnel_c(0.0),  0.0,        1e-15);
    // Series-verified (10 terms): C(0.5) ≈ 0.492344
    EXPECT_NEAR(fresnel_c(0.5),  0.492344,   1e-5);
    // Series-verified: C(1.0) ≈ 0.779893
    EXPECT_NEAR(fresnel_c(1.0),  0.779893,   1e-5);
    // Odd symmetry: C(-x) = -C(x)
    EXPECT_NEAR(fresnel_c(-1.0), -fresnel_c(1.0), 1e-14);
}

TEST(NumericalSpecialExt, FresnelS_Reference) {
    EXPECT_NEAR(fresnel_s(0.0),  0.0,        1e-15);
    // Series-verified: S(0.5) ≈ 0.064730
    EXPECT_NEAR(fresnel_s(0.5),  0.064730,   1e-5);
    // Series-verified: S(1.0) ≈ 0.438259
    EXPECT_NEAR(fresnel_s(1.0),  0.438259,   1e-5);
    // Odd symmetry: S(-x) = -S(x)
    EXPECT_NEAR(fresnel_s(-1.0), -fresnel_s(1.0), 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Error function – extended reference
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, Erf_OddSymmetry) {
    EXPECT_NEAR(ms::erf(-0.5), -ms::erf(0.5), 1e-14);
    EXPECT_NEAR(ms::erf(-1.0), -ms::erf(1.0), 1e-14);
    EXPECT_NEAR(ms::erf(-2.0), -ms::erf(2.0), 1e-14);
}

TEST(NumericalSpecialExt, Erfc_ComplementRelation) {
    // ms::erf(x) + ms::erfc(x) = 1 for all x  (DLMF 7.2.2)
    const double xs[] = {0.0, 0.5, 1.0, 2.0, 3.0};
    for (double x : xs) {
        EXPECT_NEAR(ms::erf(x) + ms::erfc(x), 1.0, 1e-14) << "at x=" << x;
    }
}

TEST(NumericalSpecialExt, Erf_AdditionalValues) {
    // Abramowitz & Stegun Table 7.1
    EXPECT_NEAR(ms::erf(0.25),  0.276326390, 1e-8);
    EXPECT_NEAR(ms::erf(0.5),   0.520499878, 1e-8);
    EXPECT_NEAR(ms::erf(1.5),   0.966105099, 1e-6);
    EXPECT_NEAR(ms::erf(3.0),   0.999977910, 1e-8);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma function  Γ(x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, GammaFunc_PositiveIntegers) {
    // Γ(n) = (n-1)!  for positive integers n
    EXPECT_NEAR(gamma_func(1.0),   1.0,   1e-12);
    EXPECT_NEAR(gamma_func(2.0),   1.0,   1e-12);
    EXPECT_NEAR(gamma_func(3.0),   2.0,   1e-12);
    EXPECT_NEAR(gamma_func(4.0),   6.0,   1e-12);
    EXPECT_NEAR(gamma_func(5.0),  24.0,   1e-12);
    EXPECT_NEAR(gamma_func(6.0), 120.0,   1e-10);
    EXPECT_NEAR(gamma_func(7.0), 720.0,   1e-9);
}

TEST(NumericalSpecialExt, GammaFunc_HalfIntegers) {
    // Γ(1/2) = √π, Γ(n+1/2) = (2n-1)!! √π / 2^n
    const double sp = std::sqrt(M_PI);
    EXPECT_NEAR(gamma_func(0.5),   sp,           1e-12);
    EXPECT_NEAR(gamma_func(1.5),   0.5 * sp,     1e-12);
    EXPECT_NEAR(gamma_func(2.5),   0.75 * sp,    1e-12);
    EXPECT_NEAR(gamma_func(3.5),   1.875 * sp,   1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Beta function  B(a, b) = Γ(a)Γ(b)/Γ(a+b)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, BetaFunc_Reference) {
    EXPECT_NEAR(beta_func(1.0, 1.0),  1.0,         1e-12);
    // B(2,2) = 1/6
    EXPECT_NEAR(beta_func(2.0, 2.0),  1.0 / 6.0,   1e-12);
    // B(1/2, 1/2) = π  (DLMF 5.12.1)
    EXPECT_NEAR(beta_func(0.5, 0.5),  M_PI,         1e-10);
    // B(1, 2) = 1/2
    EXPECT_NEAR(beta_func(1.0, 2.0),  0.5,          1e-12);
    // B(2, 3) = 1/12
    EXPECT_NEAR(beta_func(2.0, 3.0),  1.0 / 12.0,  1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical Bessel functions j_n (first kind)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, SphericalBesselJ_Reference) {
    // j_0(x) = sin(x)/x  (DLMF 10.47.3)
    EXPECT_NEAR(spherical_jn(0, 0.0), 1.0,                          1e-14);
    EXPECT_NEAR(spherical_jn(0, 1.0), std::sin(1.0),                1e-12);
    EXPECT_NEAR(spherical_jn(0, 2.0), std::sin(2.0) / 2.0,         1e-12);
    // j_1(x) = sin(x)/x² - cos(x)/x  (DLMF 10.47.3)
    EXPECT_NEAR(spherical_jn(1, 0.0), 0.0,                          1e-14);
    EXPECT_NEAR(spherical_jn(1, 1.0),
                std::sin(1.0) - std::cos(1.0),                      1e-12);
    EXPECT_NEAR(spherical_jn(1, 2.0),
                std::sin(2.0) / 4.0 - std::cos(2.0) / 2.0,         1e-12);
    // j_2(0) = 0
    EXPECT_NEAR(spherical_jn(2, 0.0), 0.0,                          1e-14);
}

TEST(NumericalSpecialExt, SphericalBesselY_Reference) {
    // y_0(x) = -cos(x)/x  (DLMF 10.47.4)
    EXPECT_NEAR(spherical_yn(0, 1.0), -std::cos(1.0),               1e-12);
    EXPECT_NEAR(spherical_yn(0, 2.0), -std::cos(2.0) / 2.0,         1e-12);
    // y_1(x) = -cos(x)/x² - sin(x)/x  (DLMF 10.47.4)
    EXPECT_NEAR(spherical_yn(1, 1.0),
                -std::cos(1.0) - std::sin(1.0),                     1e-7);
    EXPECT_NEAR(spherical_yn(1, 2.0),
                -std::cos(2.0) / 4.0 - std::sin(2.0) / 2.0,        1e-7);
}

// ─────────────────────────────────────────────────────────────────────────────
// Laguerre polynomials  L_n(x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, LaguerreL_AtZero) {
    // L_n(0) = 1 for all n  (DLMF 18.6.1)
    EXPECT_NEAR(laguerre_l(0, 0.0), 1.0, 1e-14);
    EXPECT_NEAR(laguerre_l(1, 0.0), 1.0, 1e-14);
    EXPECT_NEAR(laguerre_l(2, 0.0), 1.0, 1e-14);
    EXPECT_NEAR(laguerre_l(3, 0.0), 1.0, 1e-14);
    EXPECT_NEAR(laguerre_l(4, 0.0), 1.0, 1e-14);
}

TEST(NumericalSpecialExt, LaguerreL_AtOneAndTwo) {
    // Exact values from L_0=1, L_1=1-x, L_2=(x²-4x+2)/2, L_3=(-x³+9x²-18x+6)/6
    EXPECT_NEAR(laguerre_l(0, 1.0),  1.0,        1e-14);
    EXPECT_NEAR(laguerre_l(1, 1.0),  0.0,        1e-14);
    EXPECT_NEAR(laguerre_l(2, 1.0), -0.5,        1e-14);
    EXPECT_NEAR(laguerre_l(3, 1.0), -2.0 / 3.0, 1e-14);
    EXPECT_NEAR(laguerre_l(0, 2.0),  1.0,        1e-14);
    EXPECT_NEAR(laguerre_l(1, 2.0), -1.0,        1e-14);
    EXPECT_NEAR(laguerre_l(2, 2.0), -1.0,        1e-14);
    EXPECT_NEAR(laguerre_l(3, 2.0), -1.0 / 3.0, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gegenbauer  C_n^λ(x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, GegenbauerC_Reference) {
    // C_n^{1/2} = P_n  (Legendre)  (DLMF 18.7.1)
    EXPECT_NEAR(gegenbauer_c(0, 0.5, 0.5),  1.0,     1e-14);
    EXPECT_NEAR(gegenbauer_c(1, 0.5, 0.5),  0.5,     1e-14);
    EXPECT_NEAR(gegenbauer_c(2, 0.5, 0.5), -0.125,   1e-14);
    // C_n^1 = U_n  (Chebyshev second kind)  (DLMF 18.7.2)
    EXPECT_NEAR(gegenbauer_c(0, 1.0, 0.5),  1.0,     1e-14);
    EXPECT_NEAR(gegenbauer_c(1, 1.0, 0.5),  1.0,     1e-14);  // U_1(0.5)=1
    EXPECT_NEAR(gegenbauer_c(2, 1.0, 0.5),  0.0,     1e-13);  // U_2(0.5)=0
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobi elliptic functions  sn, cn, dn
// ─────────────────────────────────────────────────────────────────────────────

TEST(NumericalSpecialExt, JacobiElliptic_AtZeroArgument) {
    // sn(0,k)=0, cn(0,k)=1, dn(0,k)=1 for all k  (DLMF 22.2.2)
    const double ks[] = {0.0, 0.5, 0.9};
    for (double k : ks) {
        EXPECT_NEAR(jacobi_sn(0.0, k), 0.0, 1e-14) << "sn(0," << k << ")";
        EXPECT_NEAR(jacobi_cn(0.0, k), 1.0, 1e-14) << "cn(0," << k << ")";
        EXPECT_NEAR(jacobi_dn(0.0, k), 1.0, 1e-14) << "dn(0," << k << ")";
    }
}

TEST(NumericalSpecialExt, JacobiElliptic_PythagoreanIdentities) {
    // sn²(u,k) + cn²(u,k) = 1  (DLMF 22.6.1)
    // dn²(u,k) + k² sn²(u,k) = 1  (DLMF 22.6.1)
    const double k = 0.7;
    const double u = 1.2;
    const double sn = jacobi_sn(u, k);
    const double cn = jacobi_cn(u, k);
    const double dn = jacobi_dn(u, k);
    EXPECT_NEAR(sn * sn + cn * cn,        1.0, 1e-12);
    EXPECT_NEAR(dn * dn + k * k * sn * sn, 1.0, 1e-12);
}

TEST(NumericalSpecialExt, JacobiElliptic_AtHalfPeriod) {
    // At u = K(k): sn = 1, cn = 0, dn = sqrt(1-k²)  (DLMF 22.5.1)
    const double k   = 0.5;
    const double K   = ellip_k(k);
    const double kp2 = 1.0 - k * k;  // k'^2
    EXPECT_NEAR(jacobi_sn(K, k), 1.0,              1e-10);
    EXPECT_NEAR(jacobi_cn(K, k), 0.0,              1e-10);
    EXPECT_NEAR(jacobi_dn(K, k), std::sqrt(kp2),   1e-10);
}

