#include <gtest/gtest.h>
#include <cmath>

#include "ms/special/special.hpp"

using namespace ms;

namespace {

void expect_finite(double value) {
    EXPECT_TRUE(std::isfinite(value));
}

} // namespace

TEST(SpecialSmokeTest, struve_and_chebyshev_variants) {
    expect_finite(struve_k(0, 1.0));
    expect_finite(struve_hn(1, 1.0));
    expect_finite(struve_yn(0, 1.0));
    expect_finite(chebyshev_tn(2, 1, 0.5));
    expect_finite(chebyshev_un(2, 1, 0.5));
    expect_finite(hermite_hf(2, 0.5));
    expect_finite(hermite_hn(2, 0.5));
    expect_finite(laguerre_ln(2, 1, 0.5));
}

TEST(SpecialSmokeTest, jacobi_elliptic_ratios) {
    const double u = 0.4;
    const double k = 0.6;
    expect_finite(jacobi_dc(u, k));
    expect_finite(jacobi_cs(u, k));
    expect_finite(jacobi_ds(u, k));
    expect_finite(jacobi_cd(u, k));
    expect_finite(ellip_d(k));
}

TEST(SpecialSmokeTest, whittaker_and_tricomi) {
    expect_finite(tricomi_u(0.5, 1.5, 0.3));
    expect_finite(whittaker_w(0.0, 0.5, 1.0));
    expect_finite(meijer_g(0.5, 1.0, 0.2));
    expect_finite(fox_h(0.5, 1.0, 0.2));
}

TEST(SpecialSmokeTest, mathieu_and_spheroidal_more) {
    expect_finite(mathieu_a(2, 0.2));
    expect_finite(mathieu_b(2, 0.2));
    expect_finite(spheroidal_lambda(2, 1, 3.0));
    expect_finite(pcf_u(0.5, 1.0));
    expect_finite(pcf_v(0.5, 1.0));
}

TEST(SpecialSmokeTest, error_and_gamma_variants) {
    expect_finite(erfi(0.5));
    expect_finite(erfcx(0.5));
    expect_finite(dawson(0.5));
    expect_finite(dawsonx(0.5));
    expect_finite(fresnel_c(1.0));
    expect_finite(fresnel_s(1.0));
    expect_finite(log_gamma(2.5));
    expect_finite(digamma(2.5));
    expect_finite(bernoulli_number(4));
    expect_finite(euler_number(4));
}

TEST(SpecialSmokeTest, airy_kelvin_anger_weber) {
    expect_finite(airy_ai(0.5));
    expect_finite(airy_bi(0.5));
    expect_finite(airy_aip(0.5));
    expect_finite(airy_bip(0.5));
    expect_finite(kelvin_ber(1, 0.5));
    expect_finite(kelvin_bei(1, 0.5));
    expect_finite(kelvin_ker(1, 0.5));
    expect_finite(kelvin_kei(1, 0.5));
    expect_finite(anger_j(1, 0.5));
    expect_finite(weber_e(1, 0.5));
    expect_finite(struve_h(1, 0.5));
    expect_finite(struve_l(1, 0.5));
}

TEST(SpecialSmokeTest, orthogonal_and_elliptic_more) {
    expect_finite(hermite_he(2, 0.5));
    expect_finite(laguerre_la(2, 0.5, 0.3));
    expect_finite(chebyshev_v(2, 0.5));
    expect_finite(chebyshev_w(2, 0.5));
    expect_finite(gegenbauer_c(2, 0.5, 0.3));
    expect_finite(jacobi_p(2, 0.2, 0.3, 0.4));
    expect_finite(sph_harm(2, 1, 0.5, 0.3));
    expect_finite(ellip_pi(0.2, 0.5));
    expect_finite(ellip_f(0.5, 0.6));
    expect_finite(ellip_e_inc(0.5, 0.6));
    expect_finite(jacobi_sc(0.4, 0.6));
    expect_finite(jacobi_sd(0.4, 0.6));
    expect_finite(jacobi_nd(0.4, 0.6));
    expect_finite(jacobi_nc(0.4, 0.6));
}

TEST(SpecialSmokeTest, bessel_and_hypergeo_variants) {
    expect_finite(bessel_y(1, 0.5));
    expect_finite(bessel_y0(0.5));
    expect_finite(bessel_y1(0.5));
    expect_finite(bessel_l(1, 0.5));
    expect_finite(bessel_lu(1, 0.5));
    expect_finite(bessel_h(1, 0.5));
    expect_finite(bessel_hy(1, 0.5));
    expect_finite(hypergeo_0f1n(2, 1.5, 0.2));
    expect_finite(hypergeo_1f1n(1, 1.0, 0.3));
    expect_finite(kummer_m(0.5, 1.5, 0.2));
    expect_finite(legendre_q(2, 0.3));
    expect_finite(legendre_pn(2, 1, 0.3));
}

TEST(SpecialSmokeTest, elliptic_jacobi_full_set) {
    const double u = 0.35;
    const double k = 0.55;
    expect_finite(jacobi_sn(u, k));
    expect_finite(jacobi_cn(u, k));
    expect_finite(jacobi_dn(u, k));
    expect_finite(jacobi_am(u, k));
    expect_finite(jacobi_ns(u, k));
    expect_finite(jacobi_ds(u, k));
    expect_finite(jacobi_cd(u, k));
    expect_finite(ellip_k(k));
    expect_finite(ellip_e(k));
}

TEST(SpecialSmokeTest, zeta_theta_and_wave_more) {
    expect_finite(theta2(0.4, 0.3));
    expect_finite(theta3(0.4, 0.3));
    expect_finite(theta4(0.4, 0.3));
    expect_finite(jacobi_theta(2, 0.4, 0.2));
    expect_finite(lerch_phi(0.4, 2.0, 0.5));
    expect_finite(eta_dirichlet(3.0));
    expect_finite(beta_dirichlet(2.5));
    expect_finite(polylog(3, 0.4));
    expect_finite(clausen(0.7));
    expect_finite(mathieu_se(1, 0.2, 0.4));
    expect_finite(spheroidal_s2(1, 1, 4.0, 0.4));
}
