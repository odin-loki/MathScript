#include <cmath>
#include <gtest/gtest.h>

#include "ms/special/special.hpp"

using namespace ms;

namespace {

void expect_finite(double value) {
    EXPECT_TRUE(std::isfinite(value));
}

} // namespace

TEST(SpecialEdgeTest, error_functions_at_extremes) {
    EXPECT_NEAR(ms::erf(0.0), 0.0, 1e-15);
    expect_finite(ms::erf(-3.5));
    expect_finite(ms::erf(4.0));
    EXPECT_NEAR(ms::erfc(0.0), 1.0, 1e-15);
    expect_finite(ms::erfc(6.0));
    expect_finite(erfi(0.0));
    expect_finite(erfi(-0.25));
    expect_finite(erfcx(8.0));
    expect_finite(dawson(0.0));
    expect_finite(dawsonx(-1.0));
    expect_finite(fresnel_c(0.0));
    expect_finite(fresnel_s(-2.0));
}

TEST(SpecialEdgeTest, gamma_beta_digamma_edges) {
    expect_finite(gamma_func(1.0));
    expect_finite(gamma_func(0.5));
    expect_finite(log_gamma(100.0));
    expect_finite(log_gamma(0.25));
    expect_finite(beta_func(0.2, 0.3));
    expect_finite(beta_func(5.0, 2.0));
    expect_finite(digamma(1.0));
    expect_finite(digamma(0.1));
    EXPECT_NEAR(bernoulli_number(0), 1.0, 1e-12);
    expect_finite(bernoulli_number(6));
    expect_finite(euler_number(0));
    expect_finite(euler_number(6));
}

TEST(SpecialEdgeTest, bessel_orders_and_zeros) {
    expect_finite(bessel_j0(0.0));
    expect_finite(bessel_j1(0.0));
    expect_finite(bessel_y0(0.1));
    expect_finite(bessel_y1(0.1));
    for (int nu = 0; nu <= 3; ++nu) {
        expect_finite(bessel_j(nu, 0.05 + 0.2 * nu));
        expect_finite(bessel_i(nu, 0.1));
        expect_finite(bessel_k(nu, 1.0 + nu));
        expect_finite(bessel_y(nu, 0.5 + nu));
    }
    expect_finite(bessel_zero_jnu(0, 1));
    expect_finite(bessel_zero_jnu(1, 2));
    expect_finite(bessel_zero_ynu(0, 1));
    expect_finite(spherical_jn(0, 0.0));
    expect_finite(spherical_yn(2, 0.5));
    expect_finite(spherical_in(1, 0.3));
    expect_finite(spherical_kn(1, 1.2));
}

TEST(SpecialEdgeTest, orthogonal_polynomial_degrees) {
    for (int n = 0; n <= 8; ++n) {
        expect_finite(legendre_p(n, -0.9 + 0.1 * n));
        expect_finite(legendre_q(n, 0.2));
        expect_finite(hermite_h(n, -0.5));
        expect_finite(laguerre_l(n, 0.4));
        expect_finite(chebyshev_t(n, 0.95));
        expect_finite(chebyshev_u(n, -0.95));
    }
    expect_finite(legendre_pn(3, 2, 0.2));
    expect_finite(hermite_he(4, -0.3));
    expect_finite(laguerre_la(3, 0.5, 0.2));
    expect_finite(laguerre_ln(2, 1, 0.6));
    expect_finite(chebyshev_tn(2, 2, 0.4));
    expect_finite(chebyshev_un(2, 2, 0.4));
    expect_finite(chebyshev_v(3, -0.2));
    expect_finite(chebyshev_w(3, 0.2));
    expect_finite(gegenbauer_c(4, 0.25, 0.3));
    expect_finite(jacobi_p(4, -0.1, 0.2, 0.1));
    expect_finite(sph_harm(3, 1, 1.0, 0.5));
}

TEST(SpecialEdgeTest, chebyshev_tn_un_with_positive_k) {
    for (int k : {1, 3, 5}) {
        EXPECT_NEAR(chebyshev_tn(3, k, 0.25), chebyshev_t(3, 0.25), 1e-12);
        EXPECT_NEAR(chebyshev_un(3, k, 0.25), chebyshev_u(3, 0.25), 1e-12);
    }
}

TEST(SpecialEdgeTest, polynomial_domain_branches) {
    EXPECT_DOUBLE_EQ(laguerre_la(-1, 0.5, 0.2), 0.0);
    EXPECT_DOUBLE_EQ(hermite_he(-2, 0.3), 0.0);
    EXPECT_TRUE(std::isnan(chebyshev_v(2, 1.5)));
    EXPECT_TRUE(std::isnan(chebyshev_w(2, -1.5)));
    EXPECT_TRUE(std::isnan(gegenbauer_c(2, 0.5, 1.5)));
    EXPECT_TRUE(std::isnan(sph_harm(1, 2, 0.5, 0.3)));
}

TEST(SpecialEdgeTest, elliptic_and_theta_variants) {
    const double k_small = 0.1;
    const double k_mid = 0.7;
    expect_finite(ellip_k(k_small));
    expect_finite(ellip_e(k_mid));
    expect_finite(ellip_pi(0.1, k_mid));
    expect_finite(ellip_f(0.2, k_mid));
    expect_finite(ellip_e_inc(0.2, k_mid));
    expect_finite(ellip_d(k_mid));
    for (double u : {0.0, 0.15, 0.8}) {
        expect_finite(jacobi_sn(u, k_mid));
        expect_finite(jacobi_cn(u, k_mid));
        expect_finite(jacobi_dn(u, k_mid));
        expect_finite(jacobi_am(u, k_mid));
    }
    expect_finite(theta1(0.2, 0.3));
    expect_finite(theta2(-0.1, 0.25));
    expect_finite(theta3(0.0, 0.2));
    expect_finite(theta4(0.1, 0.15));
    expect_finite(theta1_prime(0.1, 0.2));
    expect_finite(jacobi_theta(3, 0.2, 0.4));
    expect_finite(weierstrass_p(0.3, 1.0, 0.1));
    expect_finite(weierstrass_pprime(0.3, 1.0, 0.1));
    expect_finite(weierstrass_zeta(0.3, 1.0, 0.1));
    expect_finite(weierstrass_sigma(0.3, 1.0, 0.1));
}

TEST(SpecialEdgeTest, theta_series_branches) {
    const double z = 0.35;
    const double q = 0.18;
    expect_finite(theta1(z, q));
    expect_finite(theta2(z, q));
    expect_finite(theta3(z, q));
    expect_finite(theta4(z, q));
    expect_finite(theta1_prime(z, q));
    EXPECT_TRUE(std::isnan(theta3(z, 1.0)));
    EXPECT_TRUE(std::isnan(jacobi_theta(0, z, 0.5)));
    EXPECT_TRUE(std::isnan(jacobi_theta(5, z, 0.5)));
    EXPECT_TRUE(std::isnan(jacobi_theta(2, z, 0.0)));
    for (int kind : {1, 2, 3, 4}) {
        expect_finite(jacobi_theta(kind, z, 0.25));
    }
}

TEST(SpecialEdgeTest, zeta_polylog_and_clausen) {
    expect_finite(zeta(2.0));
    expect_finite(zeta(0.5));
    expect_finite(zeta_hurwitz(2.0, 0.75));
    expect_finite(lerch_phi(0.3, 2.0, 0.5));
    expect_finite(eta_dirichlet(2.0));
    expect_finite(beta_dirichlet(1.5));
    for (int n = 1; n <= 4; ++n) {
        expect_finite(polylog(n, 0.3));
        expect_finite(polylog(n, -0.2));
    }
    expect_finite(clausen(0.0));
    expect_finite(clausen(1.5));
}

TEST(SpecialEdgeTest, airy_kelvin_struve_anger) {
    expect_finite(airy_ai(0.0));
    expect_finite(airy_bi(0.0));
    expect_finite(airy_aip(-0.5));
    expect_finite(airy_bip(1.0));
    for (int nu = 0; nu <= 2; ++nu) {
        expect_finite(kelvin_ber(nu, 0.2));
        expect_finite(kelvin_bei(nu, 0.2));
        expect_finite(kelvin_ker(nu, 0.5));
        expect_finite(kelvin_kei(nu, 0.5));
        expect_finite(struve_h(nu, 0.4));
        expect_finite(struve_l(nu, 0.4));
        expect_finite(struve_k(nu, 0.4));
        expect_finite(struve_hn(nu, 0.4));
        expect_finite(struve_yn(nu, 0.4));
        expect_finite(anger_j(nu, 0.4));
        expect_finite(weber_e(nu, 0.4));
    }
}

TEST(SpecialEdgeTest, hypergeo_whittaker_meijer) {
    expect_finite(hypergeo_0f1(1.5, 0.2));
    expect_finite(hypergeo_1f1(0.5, 0.3));
    expect_finite(hypergeo_2f1(0.2, 0.3, 1.5, 0.1));
    expect_finite(kummer_m(0.2, 1.5, 0.1));
    expect_finite(tricomi_u(0.5, 1.5, 0.3));
    expect_finite(whittaker_m(0.0, 0.5, 0.2));
    expect_finite(whittaker_w(0.0, 0.5, 1.0));
    expect_finite(meijer_g(0.5, 1.0, 0.2));
    expect_finite(fox_h(0.5, 1.0, 0.2));
    expect_finite(hypergeo_0f1n(2, 1.5, 0.2));
    expect_finite(hypergeo_1f1n(1, 1.0, 0.3));
}

TEST(SpecialEdgeTest, mathieu_spheroidal_pcf) {
    for (int n = 1; n <= 3; ++n) {
        expect_finite(mathieu_a(n, 0.1 * n));
        expect_finite(mathieu_b(n, 0.1 * n));
        expect_finite(mathieu_ce(n, 0.2, 0.3));
        expect_finite(mathieu_se(n, 0.2, 0.3));
        expect_finite(mathieu_mc(n, 0.2, 0.3));
        expect_finite(mathieu_ms(n, 0.2, 0.3));
    }
    expect_finite(spheroidal_lambda(3, 1, 2.0));
    expect_finite(spheroidal_s1(2, 1, 3.0, 0.2));
    expect_finite(spheroidal_s2(2, 1, 4.0, 0.2));
    expect_finite(pcf_u(-0.5, 0.3));
    expect_finite(pcf_v(0.5, 0.3));
    expect_finite(pcf_w(0.0, 0.3));
}

TEST(SpecialEdgeTest, heun_and_painleve_edges) {
    expect_finite(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0));
    expect_finite(heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 0.1));
    expect_finite(heun_d(0.1, 0.2, 0.3, 0.4, 0.2));
    expect_finite(heun_b(0.1, 0.2, 0.3, 0.4, 0.15));
    expect_finite(heun_t(0.1, 0.2, 0.3, 0.4, 0.25));
    expect_finite(painleve1(0.0, 0.0, 0.0));
    expect_finite(painleve2(0.25, 0.0, 0.0, 0.0));
    expect_finite(painleve3(0.3, 0.1, 0.0, 0.1, 0.2));
    expect_finite(painleve4(0.3, 0.1, 0.0, 0.1, 0.2));
    expect_finite(painleve5(0.2, 0.1, 0.0, 0.01, 0.02, 0.03, 0.04));
    expect_finite(painleve6(1.5, 0.1, 0.0, 0.05, 0.1, 0.2, 0.3));
}

TEST(SpecialEdgeTest, bessel_negative_argument_parity) {
    EXPECT_NEAR(bessel_j(2, -1.5), bessel_j(2, 1.5), 1e-12);
    EXPECT_NEAR(bessel_j(3, -1.5), -bessel_j(3, 1.5), 1e-12);
    EXPECT_NEAR(bessel_i(2, -0.8), bessel_i(2, 0.8), 1e-12);
    EXPECT_NEAR(bessel_i(3, -0.8), -bessel_i(3, 0.8), 1e-12);
    expect_finite(bessel_y(3, 1.2));
    EXPECT_TRUE(std::isnan(bessel_y(1, -0.5)));
    EXPECT_TRUE(std::isnan(bessel_k(1, -0.5)));
    EXPECT_TRUE(std::isinf(bessel_y0(-0.1)));
}

TEST(SpecialEdgeTest, airy_large_argument_and_digamma_guard) {
    expect_finite(airy_ai(6.0));
    expect_finite(airy_bi(6.0));
    expect_finite(airy_aip(6.0));
    expect_finite(airy_bip(6.0));
    EXPECT_TRUE(std::isnan(digamma(-0.5)));
    EXPECT_DOUBLE_EQ(bernoulli_number(20), 0.0);
    EXPECT_DOUBLE_EQ(euler_number(20), 0.0);
}

TEST(SpecialEdgeTest, elliptic_phi_zero_and_invalid_modulus) {
    EXPECT_DOUBLE_EQ(ellip_f(0.0, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(ellip_e_inc(0.0, 0.5), 0.0);
    EXPECT_TRUE(std::isnan(ellip_k(1.0)));
    EXPECT_TRUE(std::isnan(ellip_e(1.1)));
    EXPECT_TRUE(std::isnan(ellip_d(0.0)));
}

TEST(SpecialEdgeTest, jacobi_ratios_and_kelvin_branches) {
    const double u = 0.35;
    const double k = 0.55;
    expect_finite(jacobi_nd(u, k));
    expect_finite(jacobi_nc(u, k));
    expect_finite(jacobi_dc(u, k));
    expect_finite(jacobi_cs(u, k));
    expect_finite(jacobi_ns(u, k));
    expect_finite(jacobi_cd(u, k));
    expect_finite(jacobi_ds(u, k));
    EXPECT_DOUBLE_EQ(kelvin_bei(2, 0.4), 0.0);
    EXPECT_TRUE(std::isnan(kelvin_ber(-1, 0.4)));
    EXPECT_TRUE(std::isnan(spherical_jn(-1, 0.5)));
    EXPECT_TRUE(std::isnan(anger_j(-1, 0.5)));
}

TEST(SpecialEdgeTest, hypergeo_invalid_and_whittaker_fallback) {
    EXPECT_TRUE(std::isnan(hypergeo_2f1(0.2, 0.3, 1.5, 1.2)));
    expect_finite(tricomi_u(0.2, 2.5, 0.4));
    expect_finite(whittaker_w(0.25, 0.35, 2.5));
    expect_finite(whittaker_m(-0.1, 0.4, 0.0));
    EXPECT_DOUBLE_EQ(struve_h(1, 0.0), 0.0);
}

TEST(SpecialEdgeTest, mathieu_q_one_origin) {
    expect_finite(mathieu_mc(1, 1.0, 0.0));
    expect_finite(mathieu_ms(1, 1.0, 0.0));
    expect_finite(mathieu_ce(2, 1.0, 0.15));
    expect_finite(mathieu_se(2, 1.0, 0.15));
}

TEST(SpecialEdgeTest, bessel_domain_guards_and_spherical_branches) {
    EXPECT_TRUE(std::isinf(bessel_y0(0.0)));
    EXPECT_TRUE(std::isinf(bessel_y1(-0.1)));
    EXPECT_DOUBLE_EQ(bessel_j(2, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(bessel_j(0, 0.0), 1.0);
    EXPECT_TRUE(std::isnan(bessel_i(-1, 0.5)));
    EXPECT_TRUE(std::isnan(spherical_jn(1, -0.8)));
    EXPECT_TRUE(std::isnan(spherical_in(1, -0.6)));
    EXPECT_DOUBLE_EQ(spherical_in(0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(spherical_in(2, 0.0), 0.0);
    EXPECT_NEAR(spherical_yn(0, 0.5), -std::cos(0.5) / 0.5, 1e-9);
    EXPECT_TRUE(std::isnan(spherical_yn(1, 0.0)));
    EXPECT_TRUE(std::isnan(bessel_zero_jnu(-1, 1)));
    EXPECT_TRUE(std::isnan(bessel_zero_ynu(0, 0)));
}

TEST(SpecialEdgeTest, kelvin_higher_order_and_struve_guards) {
    expect_finite(kelvin_ber(2, 0.4));
    expect_finite(kelvin_bei(2, 0.4));
    expect_finite(kelvin_ker(2, 0.5));
    expect_finite(kelvin_kei(2, 0.5));
    EXPECT_TRUE(std::isnan(weber_e(-1, 0.4)));
    EXPECT_TRUE(std::isnan(struve_h(-1, 0.4)));
}

TEST(SpecialEdgeTest, orthogonal_polynomial_invalid_degrees) {
    EXPECT_DOUBLE_EQ(hermite_h(-1, 0.2), 0.0);
    EXPECT_DOUBLE_EQ(laguerre_l(-1, 0.3), 0.0);
    EXPECT_DOUBLE_EQ(laguerre_ln(2, -1, 0.3), 0.0);
    EXPECT_TRUE(std::isnan(legendre_q(2, 1.0)));
    EXPECT_TRUE(std::isnan(legendre_q(2, -1.0)));
    EXPECT_DOUBLE_EQ(legendre_pn(2, 3, 0.2), 0.0);
}

TEST(SpecialEdgeTest, hypergeo_whittaker_and_meijer_fallbacks) {
    EXPECT_DOUBLE_EQ(meijer_g(0.5, 1.0, -0.2), 0.0);
    EXPECT_TRUE(std::isnan(tricomi_u(0.5, 1.5, -0.1)));
    expect_finite(tricomi_u(0.2, 0.5, 2.5));
    expect_finite(whittaker_w(0.25, 0.35, 4.0));
    expect_finite(whittaker_m(-0.2, 0.4, 0.0));
    EXPECT_DOUBLE_EQ(jacobi_p(0, 0.1, 0.2, 0.3), 1.0);
}

TEST(SpecialEdgeTest, chebyshev_vw_singular_denominators) {
    EXPECT_TRUE(std::isnan(chebyshev_v(1, 1.0)));
    expect_finite(chebyshev_w(1, -0.99));
}

TEST(SpecialEdgeTest, weierstrass_and_zeta_singularities) {
    EXPECT_TRUE(std::isinf(weierstrass_p(0.0, 1.0, 0.1)));
    EXPECT_TRUE(std::isnan(weierstrass_pprime(0.0, 1.0, 0.1)));
    EXPECT_TRUE(std::isinf(weierstrass_zeta(0.0, 1.0, 0.1)));
    EXPECT_TRUE(std::isinf(zeta(1.0)));
    EXPECT_TRUE(std::isnan(zeta_hurwitz(0.5, 0.0)));
    EXPECT_TRUE(std::isnan(lerch_phi(1.1, 2.0, 0.5)));
}

TEST(SpecialEdgeTest, heun_clamp_and_painleve_short_integration) {
    expect_finite(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 1e-7));
    expect_finite(heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 1e-7));
    expect_finite(heun_d(0.1, 0.2, 0.3, 0.4, 1e-7));
    expect_finite(painleve1(0.0, 1.0, 0.0));
}

TEST(SpecialEdgeTest, higher_order_bessel_and_domain_guards) {
    expect_finite(bessel_y(2, 1.5));
    expect_finite(bessel_y(4, 2.0));
    EXPECT_DOUBLE_EQ(bessel_j(2, 0.0), 0.0);
    EXPECT_TRUE(std::isnan(bessel_k(0, -0.1)));
    EXPECT_TRUE(std::isnan(chebyshev_t(2, 1.5)));
    EXPECT_TRUE(std::isnan(chebyshev_u(2, -1.5)));
    expect_finite(chebyshev_w(1, -0.999));
    EXPECT_TRUE(std::isnan(spherical_in(-1, 0.5)));
    EXPECT_TRUE(std::isnan(spherical_kn(-1, 0.5)));
    EXPECT_TRUE(std::isnan(jacobi_p(2, 0.1, 0.2, 1.5)));
}

TEST(SpecialEdgeTest, zeta_dirichlet_and_mathieu_guards) {
    EXPECT_TRUE(std::isnan(zeta(-0.5)));
    EXPECT_TRUE(std::isnan(eta_dirichlet(-0.5)));
    EXPECT_TRUE(std::isnan(beta_dirichlet(-0.1)));
    EXPECT_TRUE(std::isnan(polylog(2, 1.2)));
    EXPECT_TRUE(std::isnan(mathieu_a(-1, 0.1)));
    EXPECT_TRUE(std::isnan(mathieu_b(-1, 0.1)));
    expect_finite(mathieu_a(0, 0.2));
    EXPECT_TRUE(std::isnan(spheroidal_lambda(2, 3, 1.0)));
    EXPECT_TRUE(std::isnan(spheroidal_s1(2, 3, 1.0, 0.5)));
    EXPECT_TRUE(std::isnan(spheroidal_s2(2, 3, 1.0, 0.5)));
}

TEST(SpecialEdgeTest, elliptic_whittaker_and_kelvin_branches) {
    EXPECT_TRUE(std::isnan(ellip_pi(0.1, 1.0)));
    EXPECT_TRUE(std::isnan(ellip_f(0.2, 1.1)));
    EXPECT_TRUE(std::isnan(ellip_e_inc(0.2, 1.1)));
    EXPECT_TRUE(std::isnan(whittaker_m(0.5, 0.3, -0.1)));
    EXPECT_TRUE(std::isnan(whittaker_w(0.5, 0.3, 0.0)));
    expect_finite(kelvin_bei(0, 0.3));
    expect_finite(kelvin_kei(0, 0.5));
    expect_finite(bessel_zero_ynu(0, 1));
    expect_finite(bessel_zero_ynu(0, 2));
    expect_finite(pcf_u(0.0, 0.0));
    expect_finite(tricomi_u(0.5, 1.5, 3.0));
}

TEST(SpecialEdgeTest, heun_painleve_early_returns) {
    EXPECT_DOUBLE_EQ(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 1e-7), 1.0);
    EXPECT_DOUBLE_EQ(heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 1e-7), 1.0);
    EXPECT_DOUBLE_EQ(heun_d(0.1, 0.2, 0.3, 0.4, 1e-7), 1.0);
    EXPECT_DOUBLE_EQ(heun_b(0.1, 0.2, 0.3, 0.4, 1e-7), 1.0);
    EXPECT_DOUBLE_EQ(heun_t(0.1, 0.2, 0.3, 0.4, 1e-7), 1.0);
    EXPECT_DOUBLE_EQ(painleve1(-0.1, 2.0, 0.0), 2.0);
    EXPECT_DOUBLE_EQ(painleve2(-0.1, 1.5, 0.0, 0.1), 1.5);
    EXPECT_DOUBLE_EQ(painleve3(0.05, 1.0, 0.0, 0.1, 0.2), 1.0);
    EXPECT_DOUBLE_EQ(painleve4(0.05, 1.0, 0.0, 0.1, 0.2), 1.0);
}

TEST(SpecialEdgeTest, bessel_series_and_modified_branches) {
    EXPECT_DOUBLE_EQ(bessel_i(0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(bessel_j(2, 0.0), 0.0);
    EXPECT_NEAR(bessel_i(3, -0.8), -bessel_i(3, 0.8), 1e-12);
    expect_finite(bessel_y(3, 1.2));
    expect_finite(bessel_y(4, 2.0));
    expect_finite(bessel_zero_jnu(1, 1));
    expect_finite(bessel_zero_ynu(1, 1));
}

TEST(SpecialEdgeTest, kelvin_higher_order_and_legendre_guards) {
    EXPECT_DOUBLE_EQ(kelvin_bei(1, 0.4), 0.0);
    EXPECT_DOUBLE_EQ(kelvin_kei(1, 0.5), 0.0);
    expect_finite(kelvin_ker(1, 0.5));
    EXPECT_TRUE(std::isnan(legendre_p(-1, 0.2)));
    EXPECT_TRUE(std::isnan(legendre_p(2, 1.5)));
    expect_finite(legendre_p(2, 0.2));
}

TEST(SpecialEdgeTest, recurrence_degree_one_polynomials) {
    EXPECT_NEAR(hermite_he(1, 0.3), 0.3, 1e-12);
    EXPECT_NEAR(laguerre_la(1, 0.5, 0.2), 1.5 - 0.2, 1e-12);
    EXPECT_NEAR(gegenbauer_c(1, 0.25, 0.3), 2.0 * 0.25 * 0.3, 1e-12);
    expect_finite(hermite_hf(2, 0.5));
    expect_finite(hermite_hn(2, 0.5));
}

TEST(SpecialEdgeTest, whittaker_tricomi_and_meijer_branches) {
    expect_finite(ellip_pi(0.1, 0.5));
    EXPECT_TRUE(std::isnan(jacobi_ns(0.0, 0.55)));
    expect_finite(jacobi_cd(0.35, 0.55));
    expect_finite(tricomi_u(0.5, -0.5, 50.0));
    expect_finite(whittaker_w(0.25, 0.35, 8.0));
    EXPECT_TRUE(std::isnan(whittaker_w(0.5, 0.25, 0.0)));
    EXPECT_TRUE(std::isnan(chebyshev_w(2, -1.5)));
}

TEST(SpecialEdgeTest, struve_aliases_and_bessel_h_variants) {
    expect_finite(struve_l(1, 0.4));
    expect_finite(struve_k(1, 0.4));
    expect_finite(struve_hn(1, 0.4));
    expect_finite(struve_yn(1, 0.4));
    expect_finite(bessel_h(2, 0.5));
    expect_finite(bessel_hy(2, 0.5));
    expect_finite(bessel_l(2, 0.5));
    expect_finite(bessel_lu(2, 0.5));
}

TEST(SpecialEdgeTest, heun_clamp_during_integration) {
    expect_finite(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9));
    expect_finite(heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 0.85));
    expect_finite(heun_b(0.1, 0.2, 0.3, 0.4, 0.75));
    expect_finite(painleve3(0.5, 0.1, 0.0, 0.1, 0.2));
    expect_finite(painleve4(0.5, 0.1, 0.0, 0.1, 0.2));
}

TEST(SpecialEdgeTest, degree_zero_polynomial_recurrences) {
    EXPECT_NEAR(hermite_he(0, 0.4), 1.0, 1e-12);
    EXPECT_NEAR(laguerre_la(0, 0.5, 0.2), 1.0, 1e-12);
    EXPECT_NEAR(gegenbauer_c(0, 0.25, 0.3), 1.0, 1e-12);
    EXPECT_NEAR(legendre_p(0, 0.2), 1.0, 1e-12);
}

TEST(SpecialEdgeTest, bessel_derivative_and_zero_paths) {
    expect_finite(bessel_zero_jnu(2, 3));
    expect_finite(bessel_zero_ynu(2, 3));
    EXPECT_NEAR(bessel_j(0, 0.0), 1.0, 1e-15);
    expect_finite(bessel_y(2, 1.5));
}

TEST(SpecialEdgeTest, hypergeo_jacobi_and_tricomi_fallbacks) {
    expect_finite(jacobi_p(3, 0.1, 0.2, 0.1));
    expect_finite(tricomi_u(0.2, 0.5, 2.5));
    expect_finite(whittaker_w(0.25, 0.35, 4.0));
    expect_finite(hypergeo_1f1n(1, 1.0, 0.3));
}
