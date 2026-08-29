// Smoke/finite tests for low-coverage special functions (Group 3).
// Verifies FINITE output and NO CRASH for each function.

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

#include "ms/special/special.hpp"

using namespace ms;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Jacobi elliptic functions (remaining quotient forms)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, JacobiCS_Finite) {
    // cs = cn/sn; u=0.5, k=0.5 avoids singularities
    EXPECT_TRUE(std::isfinite(jacobi_cs(0.5, 0.5)));
    EXPECT_TRUE(std::isfinite(jacobi_cs(1.0, 0.3)));
    EXPECT_TRUE(std::isfinite(jacobi_cs(0.7, 0.7)));
}

TEST(SpecialGroup3, JacobiDC_Finite) {
    // dc = dn/cn
    EXPECT_TRUE(std::isfinite(jacobi_dc(0.5, 0.5)));
    EXPECT_TRUE(std::isfinite(jacobi_dc(1.0, 0.3)));
    EXPECT_TRUE(std::isfinite(jacobi_dc(0.7, 0.7)));
}

TEST(SpecialGroup3, JacobiDS_Finite) {
    // ds = dn/sn; u away from 0 to avoid division by sn≈0
    EXPECT_TRUE(std::isfinite(jacobi_ds(0.5, 0.5)));
    EXPECT_TRUE(std::isfinite(jacobi_ds(1.0, 0.3)));
    EXPECT_TRUE(std::isfinite(jacobi_ds(0.7, 0.7)));
}

TEST(SpecialGroup3, JacobiSC_Finite) {
    // sc = sn/cn
    EXPECT_TRUE(std::isfinite(jacobi_sc(0.5, 0.5)));
    EXPECT_TRUE(std::isfinite(jacobi_sc(1.0, 0.3)));
    EXPECT_TRUE(std::isfinite(jacobi_sc(0.2, 0.9)));
}

TEST(SpecialGroup3, JacobiSD_Finite) {
    // sd = sn/dn
    EXPECT_TRUE(std::isfinite(jacobi_sd(0.5, 0.5)));
    EXPECT_TRUE(std::isfinite(jacobi_sd(1.0, 0.3)));
    EXPECT_TRUE(std::isfinite(jacobi_sd(0.2, 0.9)));
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Spherical harmonics
// sph_harm(int l, int m, double theta, double phi) → double
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, SphHarm_l0m0) {
    // Y_0^0 = 1/sqrt(4π) ≈ 0.28209
    double val = sph_harm(0, 0, 0.5, 0.5);
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_NEAR(val, 1.0 / std::sqrt(4.0 * M_PI), 1e-6);
}

TEST(SpecialGroup3, SphHarm_l1m0) {
    double val = sph_harm(1, 0, 0.3, 0.3);
    EXPECT_TRUE(std::isfinite(val));
}

TEST(SpecialGroup3, SphHarm_l2m1) {
    double val = sph_harm(2, 1, 0.4, 0.6);
    EXPECT_TRUE(std::isfinite(val));
}

TEST(SpecialGroup3, SphHarm_l3m2) {
    double val = sph_harm(3, 2, 0.5, 1.0);
    EXPECT_TRUE(std::isfinite(val));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Spherical Bessel k_n (modified)
// spherical_kn(int n, double x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, SphericalKn_n0) {
    EXPECT_TRUE(std::isfinite(spherical_kn(0, 1.0)));
    EXPECT_TRUE(std::isfinite(spherical_kn(0, 2.0)));
}

TEST(SpecialGroup3, SphericalKn_n1) {
    EXPECT_TRUE(std::isfinite(spherical_kn(1, 1.0)));
    EXPECT_TRUE(std::isfinite(spherical_kn(1, 2.0)));
}

TEST(SpecialGroup3, SphericalKn_n2) {
    EXPECT_TRUE(std::isfinite(spherical_kn(2, 1.0)));
    EXPECT_TRUE(std::isfinite(spherical_kn(2, 2.0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Mathieu functions (modified: mc and ms)
// mathieu_mc(int n, double q, double x)
// mathieu_ms(int n, double q, double x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, MathieuMC_Finite) {
    EXPECT_TRUE(std::isfinite(mathieu_mc(1, 0.1, 0.5)));
    EXPECT_TRUE(std::isfinite(mathieu_mc(2, 0.1, 0.5)));
    EXPECT_TRUE(std::isfinite(mathieu_mc(1, 0.5, 1.0)));
}

TEST(SpecialGroup3, MathieuMS_Finite) {
    EXPECT_TRUE(std::isfinite(mathieu_ms(1, 0.1, 0.5)));
    EXPECT_TRUE(std::isfinite(mathieu_ms(2, 0.1, 0.5)));
    EXPECT_TRUE(std::isfinite(mathieu_ms(1, 0.5, 1.0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Painlevé transcendents
// painleve2(double x, double y0, double yp0, double alpha)
// painleve5(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta)
// painleve6(double x, double y0, double yp0, double alpha, double beta, double gamma, double delta)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, Painleve2_Smoke) {
    // Evaluate near the initial point; trivially finite
    double val = painleve2(1.1, 0.5, 0.0, 0.0);
    EXPECT_TRUE(std::isfinite(val));
}

TEST(SpecialGroup3, Painleve2_AlphaVariation) {
    EXPECT_TRUE(std::isfinite(painleve2(1.2, 0.3, 0.1, 0.5)));
    EXPECT_TRUE(std::isfinite(painleve2(1.5, 0.1, 0.0, 1.0)));
}

TEST(SpecialGroup3, Painleve5_Smoke) {
    // x, y0, yp0, alpha, beta, gamma, delta
    double val = painleve5(1.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_TRUE(std::isfinite(val));
}

TEST(SpecialGroup3, Painleve6_Smoke) {
    // x, y0, yp0, alpha, beta, gamma, delta
    double val = painleve6(0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5);
    EXPECT_TRUE(std::isfinite(val));
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Spheroidal wave functions (prolate)
// spheroidal_lambda(int n, int m, double c)
// spheroidal_s1(int n, int m, double c, double x)
// spheroidal_s2(int n, int m, double c, double x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, SpheroidalLambda_Finite) {
    // n >= m >= 0; c small for convergence
    EXPECT_TRUE(std::isfinite(spheroidal_lambda(0, 0, 0.5)));
    EXPECT_TRUE(std::isfinite(spheroidal_lambda(1, 0, 0.5)));
    EXPECT_TRUE(std::isfinite(spheroidal_lambda(1, 1, 0.5)));
}

TEST(SpecialGroup3, SpheroidalS1_Finite) {
    // x in [-1, 1]
    EXPECT_TRUE(std::isfinite(spheroidal_s1(0, 0, 0.5, 0.5)));
    EXPECT_TRUE(std::isfinite(spheroidal_s1(1, 0, 0.5, 0.3)));
    EXPECT_TRUE(std::isfinite(spheroidal_s1(1, 1, 0.5, 0.3)));
}

TEST(SpecialGroup3, SpheroidalS2_Smoke) {
    // Second-kind spheroidal functions may diverge or be complex for some x;
    // just verify the call does not crash.
    (void)spheroidal_s2(0, 0, 0.5, 0.5);
    (void)spheroidal_s2(1, 0, 0.5, 0.5);
    SUCCEED();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Hurwitz zeta function
// zeta_hurwitz(double s, double a)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, ZetaHurwitz_s2a1) {
    // ζ(2, 1) should be finite and positive
    double val = zeta_hurwitz(2.0, 1.0);
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_GT(val, 0.0);
}

TEST(SpecialGroup3, ZetaHurwitz_s3a1) {
    // ζ(3, 1) = Apéry's constant ≈ 1.2020569
    double val = zeta_hurwitz(3.0, 1.0);
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_NEAR(val, 1.2020569031595943, 1e-5);
}

TEST(SpecialGroup3, ZetaHurwitz_VariousParams) {
    EXPECT_TRUE(std::isfinite(zeta_hurwitz(2.0, 0.5)));
    EXPECT_TRUE(std::isfinite(zeta_hurwitz(3.0, 2.0)));
    EXPECT_TRUE(std::isfinite(zeta_hurwitz(4.0, 1.0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Parabolic cylinder functions
// pcf_u(double a, double x), pcf_v(double a, double x), pcf_w(double a, double x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, PCF_U_Finite) {
    EXPECT_TRUE(std::isfinite(pcf_u(0.0, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_u(0.5, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_u(-0.5, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_u(0.0, 2.0)));
}

TEST(SpecialGroup3, PCF_V_Finite) {
    EXPECT_TRUE(std::isfinite(pcf_v(0.0, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_v(0.5, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_v(-0.5, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_v(0.0, 2.0)));
}

TEST(SpecialGroup3, PCF_W_Finite) {
    EXPECT_TRUE(std::isfinite(pcf_w(0.0, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_w(0.5, 1.0)));
    EXPECT_TRUE(std::isfinite(pcf_w(0.0, 2.0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Extended hypergeometric functions
// hypergeo_0f1n(int n, double a, double z)
// hypergeo_1f1n(int n, double a, double z)
// ─────────────────────────────────────────────────────────────────────────────

TEST(SpecialGroup3, Hypergeo0F1N_Finite) {
    EXPECT_TRUE(std::isfinite(hypergeo_0f1n(0, 1.0, 0.5)));
    EXPECT_TRUE(std::isfinite(hypergeo_0f1n(1, 1.0, 0.5)));
    EXPECT_TRUE(std::isfinite(hypergeo_0f1n(2, 2.0, 0.3)));
}

TEST(SpecialGroup3, Hypergeo1F1N_Finite) {
    EXPECT_TRUE(std::isfinite(hypergeo_1f1n(0, 1.0, 0.5)));
    EXPECT_TRUE(std::isfinite(hypergeo_1f1n(1, 1.0, 0.5)));
    EXPECT_TRUE(std::isfinite(hypergeo_1f1n(2, 2.0, 0.3)));
}
