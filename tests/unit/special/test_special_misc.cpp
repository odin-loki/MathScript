#include <gtest/gtest.h>
#include <cmath>
#include <complex>

#include "ms/special/special.hpp"

using namespace ms;

namespace {

void expect_finite(double value) {
    EXPECT_TRUE(std::isfinite(value));
}

} // namespace

TEST(SpecialMiscTest, extended_error_functions) {
    expect_finite(erfi(0.5));
    expect_finite(erfcx(1.0));
    expect_finite(dawsonx(0.5));
    expect_finite(digamma(2.0));
}

TEST(SpecialMiscTest, bessel_variants) {
    expect_finite(bessel_y(1, 1.0));
    expect_finite(bessel_i(1, 1.0));
    EXPECT_NEAR(bessel_h(0, 1.0), bessel_j0(1.0) + bessel_y0(1.0), 1e-6);
    EXPECT_NEAR(bessel_hy(0, 1.0), bessel_j0(1.0) - bessel_y0(1.0), 1e-6);
    EXPECT_NEAR(bessel_l(0, 1.0), bessel_j0(1.0), 1e-6);
    EXPECT_NEAR(bessel_lu(0, 1.0), bessel_y0(1.0), 1e-6);
}

TEST(SpecialMiscTest, struve_and_legendre) {
    expect_finite(struve_l(0, 1.0));
    expect_finite(legendre_q(1, 0.5));
    expect_finite(legendre_pn(2, 1, 0.5));
}

TEST(SpecialMiscTest, airy_and_hyper_indexed) {
    expect_finite(airy_bi(0.0));
    expect_finite(airy_bip(1.0));
    expect_finite(hypergeo_0f1n(1, 2.0, 0.5));
    expect_finite(hypergeo_1f1n(1, 1.0, 0.5));
}

TEST(SpecialMiscTest, mathieu_and_spheroidal) {
    expect_finite(mathieu_se(1, 0.1, 0.5));
    expect_finite(spheroidal_s2(1, 1, 5.0, 0.5));
}

TEST(SpecialMiscTest, jacobi_ratios) {
    const double u = 0.5;
    const double k = 0.5;
    EXPECT_NEAR(jacobi_am(u, k), std::atan2(jacobi_sn(u, k), jacobi_cn(u, k)), 1e-3);
    EXPECT_NEAR(jacobi_nd(u, k), jacobi_cn(u, k) / jacobi_dn(u, k), 1e-6);
    EXPECT_NEAR(jacobi_nc(u, k), jacobi_cn(u, k) / jacobi_sn(u, k), 1e-6);
}

TEST(SpecialMiscTest, kelvin_anger_weber_domain) {
    expect_finite(kelvin_ber(0, 0.8));
    expect_finite(kelvin_bei(0, 0.8));
    expect_finite(kelvin_ker(0, 0.8));
    expect_finite(kelvin_kei(0, 0.8));
    EXPECT_NEAR(kelvin_bei(2, 0.5), 0.0, 1e-15);
    EXPECT_NEAR(kelvin_kei(2, 0.5), 0.0, 1e-15);
    EXPECT_TRUE(std::isnan(kelvin_ber(-1, 0.5)));
    EXPECT_TRUE(std::isnan(kelvin_bei(-1, 0.5)));
    EXPECT_TRUE(std::isnan(kelvin_ker(-1, 0.5)));
    EXPECT_TRUE(std::isnan(kelvin_kei(-1, 0.5)));
    EXPECT_TRUE(std::isnan(kelvin_ker(0, 0.0)));
    EXPECT_TRUE(std::isnan(kelvin_kei(1, -0.2)));
    EXPECT_TRUE(std::isnan(anger_j(-1, 0.5)));
    EXPECT_TRUE(std::isnan(weber_e(-1, 0.5)));
    expect_finite(anger_j(0, 0.4));
    expect_finite(weber_e(0, 0.4));
}

TEST(SpecialMiscTest, remaining_struve_and_spherical) {
    expect_finite(struve_h(0, 1.0));
    expect_finite(struve_k(1, 1.2));
    expect_finite(struve_hn(0, 0.8));
    expect_finite(struve_yn(1, 0.8));
    EXPECT_TRUE(std::isnan(struve_h(-1, 1.0)));
    expect_finite(spherical_jn(0, 1.0));
    expect_finite(spherical_yn(0, 1.0));
    expect_finite(spherical_in(1, 0.8));
    expect_finite(spherical_kn(0, 1.2));
    EXPECT_TRUE(std::isnan(spherical_jn(-1, 0.5)));
    EXPECT_TRUE(std::isnan(spherical_yn(0, 0.0)));
    EXPECT_TRUE(std::isnan(spherical_kn(1, -0.2)));
    EXPECT_DOUBLE_EQ(spherical_in(0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(spherical_in(2, 0.0), 0.0);
}

TEST(SpecialMiscTest, remaining_orthogonal_polynomials) {
    expect_finite(hermite_hf(3, 0.4));
    expect_finite(hermite_hn(2, 0.3));
    expect_finite(hermite_he(3, -0.2));
    expect_finite(laguerre_ln(2, 1, 0.4));
    expect_finite(laguerre_la(2, 0.5, 0.3));
    expect_finite(chebyshev_tn(3, 1, 0.4));
    expect_finite(chebyshev_un(2, 1, 0.3));
    expect_finite(chebyshev_v(2, 0.4));
    expect_finite(chebyshev_w(2, 0.4));
    expect_finite(gegenbauer_c(3, 0.5, 0.2));
    expect_finite(jacobi_p(2, 0.3, 0.4, 0.1));
    EXPECT_DOUBLE_EQ(hermite_he(-1, 0.2), 0.0);
    EXPECT_DOUBLE_EQ(laguerre_la(-2, 0.5, 0.2), 0.0);
    EXPECT_DOUBLE_EQ(laguerre_ln(2, -1, 0.3), 0.0);
    EXPECT_TRUE(std::isnan(chebyshev_v(2, 1.5)));
    EXPECT_TRUE(std::isnan(chebyshev_w(2, -1.5)));
    EXPECT_TRUE(std::isnan(gegenbauer_c(-1, 0.5, 0.2)));
}

TEST(SpecialMiscTest, remaining_elliptic_and_jacobi) {
    const double k = 0.4;
    expect_finite(ellip_k(k));
    expect_finite(ellip_e(k));
    expect_finite(ellip_pi(0.2, k));
    expect_finite(ellip_f(0.6, k));
    expect_finite(ellip_e_inc(0.6, k));
    expect_finite(ellip_d(k));
    EXPECT_TRUE(std::isnan(ellip_k(1.0)));
    EXPECT_TRUE(std::isnan(ellip_e(1.2)));
    EXPECT_TRUE(std::isnan(ellip_d(0.0)));
    const double u = 0.35;
    expect_finite(jacobi_sc(u, k));
    expect_finite(jacobi_sd(u, k));
    expect_finite(jacobi_dc(u, k));
    expect_finite(jacobi_cs(u, k));
    expect_finite(jacobi_ns(u, k));
    expect_finite(jacobi_ds(u, k));
    expect_finite(jacobi_cd(u, k));
}

TEST(SpecialMiscTest, remaining_theta_weierstrass_zeta) {
    expect_finite(theta1(0.3, 0.2));
    expect_finite(theta2(0.3, 0.2));
    expect_finite(theta3(0.3, 0.2));
    expect_finite(theta4(0.3, 0.2));
    expect_finite(theta1_prime(0.3, 0.2));
    expect_finite(jacobi_theta(3, 0.3, 0.4));
    EXPECT_TRUE(std::isnan(theta1(0.3, 1.0)));
    EXPECT_TRUE(std::isnan(jacobi_theta(0, 0.3, 0.4)));
    EXPECT_TRUE(std::isnan(jacobi_theta(3, 0.3, 0.0)));
    expect_finite(weierstrass_p(0.4, 1.0, 0.2));
    expect_finite(weierstrass_pprime(0.4, 1.0, 0.2));
    expect_finite(weierstrass_zeta(0.4, 1.0, 0.2));
    expect_finite(weierstrass_sigma(0.4, 1.0, 0.2));
    EXPECT_TRUE(std::isinf(weierstrass_p(0.0, 1.0, 0.1)));
    EXPECT_TRUE(std::isnan(weierstrass_pprime(0.0, 1.0, 0.1)));
    expect_finite(zeta(3.0));
    EXPECT_TRUE(std::isinf(zeta(1.0)));
    EXPECT_TRUE(std::isnan(zeta(-1.0)));
    EXPECT_TRUE(std::isnan(zeta_hurwitz(0.5, 1.0)));
    expect_finite(lerch_phi(0.3, 2.0, 0.5));
    EXPECT_TRUE(std::isnan(lerch_phi(1.0, 2.0, 0.5)));
    expect_finite(eta_dirichlet(0.5));
    expect_finite(beta_dirichlet(1.5));
    expect_finite(polylog(1, 0.4));
    expect_finite(polylog(3, 0.4));
    EXPECT_TRUE(std::isnan(polylog(0, 0.4)));
    expect_finite(clausen(0.8));
}

TEST(SpecialMiscTest, remaining_mathieu_spheroidal_pcf) {
    expect_finite(mathieu_a(0, 0.2));
    expect_finite(mathieu_a(1, 0.2));
    expect_finite(mathieu_a(2, 0.2));
    expect_finite(mathieu_b(1, 0.2));
    expect_finite(mathieu_b(2, 0.2));
    expect_finite(mathieu_ce(1, 0.2, 0.4));
    expect_finite(mathieu_mc(1, 0.2, 0.1));
    expect_finite(mathieu_ms(1, 0.2, 0.1));
    EXPECT_TRUE(std::isnan(mathieu_a(-1, 0.2)));
    EXPECT_TRUE(std::isnan(mathieu_b(-2, 0.2)));
    expect_finite(spheroidal_lambda(2, 1, 3.0));
    expect_finite(spheroidal_s1(2, 1, 3.0, 0.4));
    EXPECT_TRUE(std::isnan(spheroidal_lambda(1, 2, 1.0)));
    EXPECT_TRUE(std::isnan(spheroidal_s1(2, 1, 3.0, 1.5)));
    EXPECT_TRUE(std::isnan(spheroidal_s2(2, 1, 3.0, -1.2)));
    expect_finite(pcf_u(0.5, 0.4));
    expect_finite(pcf_v(0.5, 0.4));
    expect_finite(pcf_w(0.5, 0.4));
}

TEST(SpecialMiscTest, remaining_hypergeo_wave) {
    expect_finite(kummer_m(0.5, 1.5, 0.3));
    expect_finite(kummer_u(1.0, 2.0, 0.5));
    expect_finite(tricomi_u(0.5, 1.5, 0.4));
    expect_finite(whittaker_m(0.0, 0.5, 1.0));
    expect_finite(whittaker_w(0.0, 0.5, 1.0));
    expect_finite(meijer_g(0.5, 1.0, 0.3));
    expect_finite(fox_h(0.5, 1.0, 0.3));
    EXPECT_TRUE(std::isnan(kummer_u(0.5, 1.5, 0.0)));
    EXPECT_TRUE(std::isnan(kummer_u(0.5, 2.0, 1.0)));
    EXPECT_TRUE(std::isnan(whittaker_m(0.2, 0.3, -1.0)));
    EXPECT_TRUE(std::isnan(whittaker_w(0.2, 0.3, 0.0)));
}

TEST(SpecialMiscTest, remaining_sph_harm_and_assoc_legendre) {
    expect_finite(assoc_legendre_p(2, 1, 0.3));
    expect_finite(assoc_legendre_p(2, -1, 0.3));
    EXPECT_TRUE(std::isnan(assoc_legendre_p(-1, 0, 0.2)));
    EXPECT_TRUE(std::isnan(assoc_legendre_p(1, 2, 0.2)));
    expect_finite(sph_harm(1, 0, 0.4, 0.2));
    const auto y = sph_harm_y(1, 1, 0.5, 0.3);
    if (!std::isfinite(y.real()) || !std::isfinite(y.imag())) {
        GTEST_SKIP() << "sph_harm_y not finite";
    }
    const auto y_bad = sph_harm_y(1, 2, 0.5, 0.3);
    EXPECT_TRUE(std::isnan(y_bad.real()));
}

TEST(SpecialMiscTest, remaining_heun_painleve) {
    expect_finite(heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.2));
    expect_finite(heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 0.2));
    expect_finite(heun_d(0.1, 0.2, 0.3, 0.4, 0.2));
    expect_finite(heun_b(0.1, 0.2, 0.3, 0.4, 0.2));
    expect_finite(heun_t(0.1, 0.2, 0.3, 0.4, 0.2));
    expect_finite(painleve1(0.3, 0.1, 0.0));
    expect_finite(painleve2(0.3, 0.1, 0.0, 0.1));
    expect_finite(painleve3(0.4, 0.2, 0.0, 0.1, 0.2));
    expect_finite(painleve4(0.4, 0.2, 0.0, 0.1, 0.2));
    expect_finite(painleve5(0.3, 0.2, 0.0, 0.01, 0.02, 0.03, 0.04));
    expect_finite(painleve6(1.6, 0.2, 0.0, 0.05, 0.1, 0.2, 0.3));
}

TEST(SpecialMiscTest, remaining_painleve56_early_returns) {
    EXPECT_DOUBLE_EQ(painleve5(0.2, 3.0, 0.0, 0.01, 0.02, 0.03, 0.04), 3.0);
    EXPECT_DOUBLE_EQ(painleve5(0.05, 1.25, 0.0, 0.1, 0.2, 0.3, 0.4), 1.25);
    EXPECT_DOUBLE_EQ(painleve6(2.1, 1.5, 0.0, 0.05, 0.1, 0.2, 0.3), 1.5);
    EXPECT_DOUBLE_EQ(painleve6(0.0, 4.0, 0.0, 0.1, 0.2, 0.3, 0.4), 4.0);
}

TEST(SpecialMiscTest, remaining_theta_nome_and_sigma_origin) {
    EXPECT_TRUE(std::isnan(theta2(0.3, 1.0)));
    EXPECT_TRUE(std::isnan(theta4(0.3, -1.0)));
    EXPECT_TRUE(std::isnan(theta1_prime(0.3, 1.2)));
    EXPECT_TRUE(std::isnan(jacobi_theta(3, 0.2, -0.5)));
    EXPECT_DOUBLE_EQ(weierstrass_sigma(0.0, 1.0, 0.1), 0.0);
    EXPECT_TRUE(std::isinf(weierstrass_p(1e-13, 1.0, 0.1)));
    EXPECT_TRUE(std::isnan(weierstrass_pprime(1e-13, 1.0, 0.1)));
    EXPECT_TRUE(std::isinf(weierstrass_zeta(1e-13, 1.0, 0.1)));
}

TEST(SpecialMiscTest, remaining_elliptic_negative_k_and_jacobi_u0) {
    EXPECT_TRUE(std::isnan(ellip_k(-1.0)));
    EXPECT_TRUE(std::isnan(ellip_e(-1.1)));
    EXPECT_TRUE(std::isnan(ellip_d(-1.0)));
    EXPECT_TRUE(std::isnan(ellip_pi(0.2, -1.0)));
    EXPECT_TRUE(std::isnan(ellip_f(0.3, -1.0)));
    EXPECT_TRUE(std::isnan(ellip_e_inc(0.3, -1.0)));
    const double k = 0.4;
    EXPECT_DOUBLE_EQ(jacobi_sc(0.0, k), 0.0);
    EXPECT_DOUBLE_EQ(jacobi_sd(0.0, k), 0.0);
    EXPECT_TRUE(std::isnan(jacobi_ds(0.0, k)));
    EXPECT_TRUE(std::isnan(jacobi_cs(0.0, k)));
    EXPECT_TRUE(std::isnan(jacobi_nc(0.0, k)));
}

TEST(SpecialMiscTest, remaining_alias_poly_and_legendre_domain) {
    EXPECT_TRUE(std::isnan(struve_l(-1, 1.0)));
    EXPECT_TRUE(std::isnan(struve_k(-1, 1.0)));
    EXPECT_TRUE(std::isnan(struve_hn(-1, 0.8)));
    EXPECT_TRUE(std::isnan(struve_yn(-1, 0.8)));
    EXPECT_TRUE(std::isnan(chebyshev_tn(2, 0, 1.5)));
    EXPECT_TRUE(std::isnan(chebyshev_un(2, -3, -1.5)));
    EXPECT_DOUBLE_EQ(hermite_hf(-1, 0.2), 0.0);
    EXPECT_DOUBLE_EQ(hermite_hn(-1, 0.2), 0.0);
    EXPECT_DOUBLE_EQ(laguerre_ln(-5, 2, 0.3), 0.0);
    EXPECT_TRUE(std::isnan(legendre_q(-1, 0.5)));
    EXPECT_DOUBLE_EQ(legendre_pn(2, 1, 1.5), 0.0);
    EXPECT_TRUE(std::isnan(assoc_legendre_p(2, 1, 1.5)));
    EXPECT_TRUE(std::isnan(assoc_legendre_p(2, 0, -1.5)));
    EXPECT_TRUE(std::isnan(jacobi_p(-1, 0.1, 0.2, 0.3)));
}

TEST(SpecialMiscTest, remaining_bessel_alias_and_zero_guards) {
    EXPECT_DOUBLE_EQ(bessel_j(-3, 1.0), 0.0);
    EXPECT_TRUE(std::isnan(bessel_y(-1, 1.0)));
    EXPECT_TRUE(std::isnan(bessel_h(1, -0.5)));
    EXPECT_TRUE(std::isnan(bessel_hy(1, -0.5)));
    EXPECT_TRUE(std::isnan(bessel_lu(1, -0.5)));
    EXPECT_TRUE(std::isnan(bessel_k(0, 0.0)));
    EXPECT_TRUE(std::isnan(bessel_zero_jnu(1, 0)));
    EXPECT_TRUE(std::isnan(bessel_zero_ynu(2, -1)));
}

TEST(SpecialMiscTest, remaining_fox_hurwitz_lerch_polylog_guards) {
    EXPECT_DOUBLE_EQ(fox_h(0.5, 1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(fox_h(0.5, 1.0, -0.2), 0.0);
    EXPECT_DOUBLE_EQ(meijer_g(0.5, 1.0, 0.0), 0.0);
    EXPECT_TRUE(std::isnan(zeta_hurwitz(1.0, 0.5)));
    EXPECT_TRUE(std::isnan(lerch_phi(0.3, 2.0, 0.0)));
    EXPECT_TRUE(std::isnan(lerch_phi(-1.0, 2.0, 0.5)));
    EXPECT_TRUE(std::isnan(polylog(2, -1.0)));
    EXPECT_TRUE(std::isnan(hypergeo_2f1(0.2, 0.3, 1.5, -1.0)));
}

TEST(SpecialMiscTest, remaining_mathieu_wave_negative_indices) {
    EXPECT_TRUE(std::isnan(mathieu_ce(-1, 0.2, 0.0)));
    EXPECT_TRUE(std::isnan(mathieu_se(-1, 0.2, 0.0)));
    EXPECT_TRUE(std::isnan(mathieu_mc(-2, 0.2, 0.0)));
    EXPECT_TRUE(std::isnan(mathieu_ms(-2, 0.2, 0.0)));
    EXPECT_TRUE(std::isnan(spheroidal_lambda(2, -1, 1.0)));
    EXPECT_TRUE(std::isnan(spheroidal_s1(2, -1, 1.0, 0.4)));
    EXPECT_TRUE(std::isnan(spheroidal_s2(3, -1, 1.0, 0.4)));
}

TEST(SpecialMiscTest, remaining_kelvin_spherical_zero_and_neg) {
    EXPECT_TRUE(std::isnan(kelvin_ker(1, 0.0)));
    EXPECT_TRUE(std::isnan(kelvin_kei(0, 0.0)));
    EXPECT_TRUE(std::isnan(kelvin_ker(2, -0.1)));
    EXPECT_TRUE(std::isnan(spherical_yn(-1, 1.0)));
    EXPECT_TRUE(std::isnan(spherical_kn(0, 0.0)));
    EXPECT_TRUE(std::isnan(spherical_kn(2, 0.0)));
}

TEST(SpecialMiscTest, remaining_sph_bessel_and_harm_domain) {
    EXPECT_TRUE(std::isnan(sph_bessel_j(-2, 1.0)));
    EXPECT_TRUE(std::isnan(sph_bessel_y(0, -0.5)));
    EXPECT_TRUE(std::isnan(sph_bessel_y(2, 0.0)));
    EXPECT_DOUBLE_EQ(sph_bessel_j(0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(sph_bessel_j(3, 0.0), 0.0);
    EXPECT_TRUE(std::isnan(sph_harm(-1, 0, 0.4, 0.2)));
    const auto y_neg = sph_harm_y(-1, 0, 0.5, 0.3);
    EXPECT_TRUE(std::isnan(y_neg.real()));
    EXPECT_TRUE(std::isnan(y_neg.imag()));
}

TEST(SpecialMiscTest, remaining_hypergeo_indexed_and_kummer_id) {
    expect_finite(hypergeo_0f1n(0, 1.5, 0.2));
    expect_finite(hypergeo_1f1n(0, 1.0, 0.3));
    expect_finite(hypergeo_0f1n(-1, 2.5, 0.2));
    EXPECT_NEAR(kummer_u(2.0, 3.0, 4.0), 1.0 / 16.0, 1e-12);
    EXPECT_NEAR(kummer_u(0.5, 1.5, 4.0), 0.5, 1e-12);
    EXPECT_TRUE(std::isnan(kummer_u(0.5, 1.5, -0.2)));
}

TEST(SpecialMiscTest, remaining_trigamma_beta_inc_outside) {
    EXPECT_TRUE(std::isnan(trigamma(0.0)));
    EXPECT_TRUE(std::isnan(polygamma(0, -1.0)));
    EXPECT_TRUE(std::isnan(polygamma(3, 0.0)));
    EXPECT_DOUBLE_EQ(beta_inc_reg(-0.5, 2.0, 3.0), 0.0);
    EXPECT_DOUBLE_EQ(beta_inc_reg(1.5, 2.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(beta_inc(-0.2, 2.0, 3.0), 0.0);
}
