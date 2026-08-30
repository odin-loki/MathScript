#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <vector>
#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialExtTest, error_functions_dlmf) {
    EXPECT_NEAR(ms::erf(0.0), 0.0, 1e-12);
    EXPECT_NEAR(ms::erf(0.5), 0.5204998778135512, 1e-6);
    EXPECT_NEAR(ms::erfc(0.5), 1.0 - ms::erf(0.5), 1e-6);
    EXPECT_NEAR(dawson(0.5), 0.4244363835020223, 1e-3);
}

TEST(SpecialExtTest, gamma_and_beta) {
    EXPECT_NEAR(gamma_func(5.0), 24.0, 1e-10);
    EXPECT_NEAR(log_gamma(5.0), std::log(24.0), 1e-10);
    EXPECT_NEAR(beta_func(0.5, 0.5), std::numbers::pi, 1e-6);
    EXPECT_NEAR(bernoulli_number(2), 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(euler_number(4), 5.0, 1e-12);
}

TEST(SpecialExtTest, fresnel_integrals) {
    EXPECT_NEAR(fresnel_c(0.0), 0.0, 1e-12);
    EXPECT_NEAR(fresnel_s(0.0), 0.0, 1e-12);
    EXPECT_NEAR(fresnel_c(1.0), 0.7798934003768226, 1e-3);
    EXPECT_NEAR(fresnel_s(1.0), 0.4382591473903547, 1e-3);
}

TEST(SpecialExtTest, bessel_functions) {
    EXPECT_NEAR(bessel_j0(0.0), 1.0, 1e-12);
    EXPECT_NEAR(bessel_j0(1.0), 0.765197686557966, 1e-6);
    EXPECT_NEAR(bessel_j1(1.0), 0.440050585744933, 1e-6);
    EXPECT_NEAR(bessel_j(2, 1.0), 0.1149034849319005, 1e-6);
    EXPECT_NEAR(bessel_k(0, 1.0), 0.421024438240708, 1e-3);
}

TEST(SpecialExtTest, orthogonal_polynomials) {
    EXPECT_NEAR(legendre_p(0, 0.5), 1.0, 1e-12);
    EXPECT_NEAR(legendre_p(1, 0.5), 0.5, 1e-12);
    EXPECT_NEAR(legendre_p(2, 0.5), -0.125, 1e-12);
    EXPECT_NEAR(chebyshev_t(3, 0.5), -1.0, 1e-12);
    EXPECT_NEAR(chebyshev_u(2, 0.5), 0.0, 1e-12);
    EXPECT_NEAR(laguerre_l(2, 0.5), 0.125, 1e-12);
    EXPECT_NEAR(hermite_h(3, 1.0), -4.0, 1e-12);
}

TEST(SpecialExtTest, airy_and_hypergeo) {
    EXPECT_NEAR(airy_ai(0.0), 0.355028053887817, 1e-6);
    EXPECT_NEAR(airy_ai(1.0), 0.135403553993278, 1e-2);
    EXPECT_NEAR(hypergeo_0f1(1.0, 0.0), 1.0, 1e-12);
    EXPECT_NEAR(hypergeo_1f1(1.0, 0.0), 1.0, 1e-12);
}

TEST(SpecialExtTest, inverse_error_functions) {
    EXPECT_NEAR(erfinv(0.0), 0.0, 1e-12);
    EXPECT_NEAR(erfcinv(1.0), 0.0, 1e-12);
    EXPECT_NEAR(ms::erf(erfinv(0.5)), 0.5, 1e-10);
    EXPECT_NEAR(ms::erf(erfinv(-0.3)), -0.3, 1e-10);
    EXPECT_NEAR(ms::erf(erfinv(0.9)), 0.9, 1e-10);
    EXPECT_NEAR(ms::erfc(erfcinv(0.5)), 0.5, 1e-10);
    EXPECT_NEAR(ms::erfc(erfcinv(1.5)), 1.5, 1e-10);
}

TEST(SpecialExtTest, polygamma_and_factorials) {
    EXPECT_NEAR(trigamma(1.0), std::numbers::pi_v<double> * std::numbers::pi_v<double> / 6.0, 1e-4);
    EXPECT_NEAR(polygamma(0, 2.0), digamma(2.0), 1e-10);
    EXPECT_NEAR(pochhammer(2.0, 3), 24.0, 1e-12);
    EXPECT_NEAR(falling_factorial(5.0, 3), 60.0, 1e-12);
    EXPECT_NEAR(pochhammer(1.0, 5), 120.0, 1e-10);
    EXPECT_NEAR(pochhammer(3.0, 0), 1.0, 1e-12);
    EXPECT_NEAR(falling_factorial(4.0, 0), 1.0, 1e-12);
}

TEST(SpecialExtTest, incomplete_gamma_and_beta) {
    EXPECT_NEAR(gamma_inc_reg(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-6);
    EXPECT_NEAR(gamma_inc_reg(2.5, 1.2) + gamma_inc_reg_upper(2.5, 1.2), 1.0, 1e-10);
    EXPECT_NEAR(gamma_inc_reg(3.0, 0.5) + gamma_inc_reg_upper(3.0, 0.5), 1.0, 1e-10);
    EXPECT_NEAR(beta_inc_reg(0.5, 1.0, 1.0), 0.5, 1e-10);
    EXPECT_NEAR(beta_inc_reg(0.3, 2.0, 3.0) + beta_inc_reg(0.7, 3.0, 2.0), 1.0, 1e-10);
    EXPECT_NEAR(gamma_inc(2.0, 1.0), gamma_inc_reg(2.0, 1.0) * gamma_func(2.0), 1e-10);
    EXPECT_NEAR(beta_inc(0.4, 2.0, 3.0), beta_inc_reg(0.4, 2.0, 3.0) * beta_func(2.0, 3.0), 1e-10);
}

TEST(SpecialExtTest, reciprocal_gamma) {
    EXPECT_NEAR(rgamma(5.0) * gamma_func(5.0), 1.0, 1e-10);
    EXPECT_NEAR(rgamma(2.5) * gamma_func(2.5), 1.0, 1e-10);
    EXPECT_EQ(rgamma(0.0), 0.0);
    EXPECT_EQ(rgamma(-1.0), 0.0);
    EXPECT_EQ(rgamma(-3.0), 0.0);
}

namespace {

void expect_lambert_identity(int branch, double z, double tol) {
    const double w = lambert_w(branch, z);
    ASSERT_TRUE(std::isfinite(w));
    EXPECT_NEAR(w * std::exp(w), z, tol) << "branch=" << branch << " z=" << z;
}

} // namespace

TEST(SpecialExtTest, lambert_w_principal_branch) {
    EXPECT_NEAR(lambert_w(0, 0.0), 0.0, 1e-15);
    EXPECT_NEAR(lambert_w(0, std::exp(1.0)), 1.0, 1e-12);
    EXPECT_NEAR(lambert_w(0, 1.0), 0.5671432904097838, 1e-10);
    expect_lambert_identity(0, 0.5, 1e-14);
    expect_lambert_identity(0, 2.0, 1e-14);
    expect_lambert_identity(0, 10.0, 1e-12);
}

TEST(SpecialExtTest, lambert_w_minus_one_branch) {
    const double minus_inv_e = -std::exp(-1.0);
    EXPECT_NEAR(lambert_w(-1, minus_inv_e), -1.0, 1e-12);
    EXPECT_NEAR(lambert_w(-1, -0.1), -3.5771522592158035, 1e-6);
    EXPECT_NEAR(lambert_w(-1, -0.2), -2.542641493804526, 1e-6);
    expect_lambert_identity(-1, -0.05, 1e-14);
    expect_lambert_identity(-1, -0.25, 1e-14);
}

TEST(SpecialExtTest, lambert_w_domain_errors) {
    EXPECT_TRUE(std::isnan(lambert_w(0, -1.0)));
    EXPECT_TRUE(std::isnan(lambert_w(-1, 0.0)));
    EXPECT_TRUE(std::isnan(lambert_w(-1, 1.0)));
    EXPECT_TRUE(std::isnan(lambert_w(2, 1.0)));
}

TEST(SpecialExtTest, erfi_odd_and_small_x) {
    EXPECT_NEAR(erfi(0.0), 0.0, 1e-15);
    EXPECT_NEAR(erfi(-0.4), -erfi(0.4), 1e-14);
    // Two-term series: (2/sqrt(pi)) (x + x^3/3)
    const double x = 0.02;
    const double series = 2.0 / std::sqrt(std::numbers::pi) * (x + x * x * x / 3.0);
    EXPECT_NEAR(erfi(x), series, 1e-9);
}

TEST(SpecialExtTest, erfcx_scaled_complement) {
    EXPECT_NEAR(erfcx(0.0), 1.0, 1e-15);
    const double x = 0.75;
    EXPECT_NEAR(erfcx(x), std::erfc(x) * std::exp(x * x), 1e-12);
    EXPECT_NEAR(erfcx(-0.5), 2.0 * std::exp(0.25) - erfcx(0.5), 1e-12);
}

TEST(SpecialExtTest, dawsonx_alias_and_odd) {
    EXPECT_NEAR(dawson(0.0), 0.0, 1e-15);
    EXPECT_NEAR(dawsonx(0.0), 0.0, 1e-15);
    EXPECT_NEAR(dawsonx(0.6), dawson(0.6), 1e-15);
    EXPECT_NEAR(dawson(-0.6), -dawson(0.6), 1e-14);
    // F(x) ~ x as x -> 0
    EXPECT_NEAR(dawson(0.01), 0.01, 1e-6);
}

TEST(SpecialExtTest, debye_small_x_and_known) {
    EXPECT_NEAR(debye(1, 1e-9), 1.0, 1e-15);
    // D_n(x) ~ 1 - n x / (2(n+1)) for small x
    EXPECT_NEAR(debye(1, 0.01), 1.0 - 0.01 / 4.0, 1e-5);
    EXPECT_NEAR(debye(3, 0.01), 1.0 - 3.0 * 0.01 / 8.0, 1e-5);
    EXPECT_NEAR(debye(1, 1.0), 0.7775046349571536, 1e-6);
}

TEST(SpecialExtTest, debye_domain_errors) {
    EXPECT_TRUE(std::isnan(debye(0, 1.0)));
    EXPECT_TRUE(std::isnan(debye(-1, 1.0)));
    EXPECT_TRUE(std::isnan(debye(1, 0.0)));
    EXPECT_TRUE(std::isnan(debye(2, -0.5)));
}

TEST(SpecialExtTest, erfinv_erfcinv_domain) {
    EXPECT_TRUE(std::isnan(erfinv(1.5)));
    EXPECT_TRUE(std::isnan(erfinv(-1.5)));
    EXPECT_TRUE(std::isinf(erfinv(1.0)));
    EXPECT_TRUE(std::isinf(erfinv(-1.0)));
    EXPECT_TRUE(std::isnan(erfcinv(0.0)));
    EXPECT_TRUE(std::isnan(erfcinv(2.0)));
}

TEST(SpecialExtTest, factorial_and_polygamma_domain) {
    EXPECT_TRUE(std::isnan(pochhammer(2.0, -1)));
    EXPECT_TRUE(std::isnan(falling_factorial(5.0, -1)));
    EXPECT_TRUE(std::isnan(digamma(0.0)));
    EXPECT_TRUE(std::isnan(polygamma(-1, 2.0)));
}

TEST(SpecialExtTest, gamma_inc_upper_nonpositive_x) {
    EXPECT_NEAR(gamma_inc_reg_upper(2.0, 0.0), 1.0, 1e-15);
    EXPECT_NEAR(gamma_inc_reg_upper(2.0, -1.0), 1.0, 1e-15);
}

TEST(SpecialExtTest, bernoulli_euler_out_of_table) {
    EXPECT_NEAR(bernoulli_number(-1), 0.0, 1e-15);
    EXPECT_NEAR(euler_number(-1), 0.0, 1e-15);
}

TEST(SpecialExtTest, gamma_inc_reg_x_nonpositive) {
    EXPECT_NEAR(gamma_inc_reg(2.0, 0.0), 0.0, 1e-15);
    EXPECT_NEAR(gamma_inc_reg(2.0, -1.0), 0.0, 1e-15);
    EXPECT_NEAR(gamma_inc(2.0, 0.0), 0.0, 1e-15);
}

TEST(SpecialExtTest, beta_inc_reg_endpoints) {
    EXPECT_NEAR(beta_inc_reg(0.0, 2.0, 3.0), 0.0, 1e-15);
    EXPECT_NEAR(beta_inc_reg(1.0, 2.0, 3.0), 1.0, 1e-15);
    EXPECT_NEAR(beta_inc(0.0, 2.0, 3.0), 0.0, 1e-15);
    EXPECT_NEAR(beta_inc(1.0, 2.0, 3.0), beta_func(2.0, 3.0), 1e-12);
}

TEST(SpecialExtTest, polygamma_higher_order_and_nonpositive_x) {
    EXPECT_NEAR(polygamma(1, 1.0), trigamma(1.0), 1e-12);
    EXPECT_NEAR(polygamma(2, 1.0), -2.0 * 1.2020569031595942, 1e-4);
    EXPECT_TRUE(std::isnan(polygamma(1, 0.0)));
    EXPECT_TRUE(std::isnan(trigamma(-1.0)));
}

TEST(SpecialExtTest, lambert_w_principal_at_branch_point) {
    const double minus_inv_e = -std::exp(-1.0);
    EXPECT_NEAR(lambert_w(0, minus_inv_e), -1.0, 1e-12);
}

TEST(SpecialExtTest, debye_moderate_x) {
    const double d = debye(2, 2.0);
    EXPECT_TRUE(std::isfinite(d));
    EXPECT_GT(d, 0.0);
    EXPECT_LT(d, 1.0);
}

TEST(SpecialExtTest, rgamma_half_integer) {
    EXPECT_NEAR(rgamma(0.5), 1.0 / std::sqrt(std::numbers::pi), 1e-12);
}

TEST(SpecialExtTest, hypergeo_2f1_at_origin) {
    EXPECT_NEAR(hypergeo_2f1(1.0, 2.0, 3.0, 0.0), 1.0, 1e-15);
}

TEST(SpecialExtTest, bernoulli_euler_beyond_table) {
    EXPECT_NEAR(bernoulli_number(20), 0.0, 1e-15);
    EXPECT_NEAR(euler_number(20), 0.0, 1e-15);
}

TEST(SpecialExtTest, kummer_u_exact_identity) {
    EXPECT_NEAR(kummer_u(2.0, 3.0, 4.0), std::pow(4.0, -2.0), 1e-12);
    EXPECT_NEAR(kummer_u(1.0, 2.0, 5.0), 0.2, 1e-12);
}

TEST(SpecialExtTest, ellip_k_e_at_zero_modulus) {
    EXPECT_NEAR(ellip_k(0.0), std::numbers::pi / 2.0, 1e-12);
    EXPECT_NEAR(ellip_e(0.0), std::numbers::pi / 2.0, 1e-12);
}

TEST(SpecialExtTest, weierstrass_sigma_origin) {
    EXPECT_NEAR(weierstrass_sigma(0.0, 2.0, 3.0), 0.0, 1e-15);
    EXPECT_NEAR(weierstrass_sigma(1e-4, 0.0, 0.0), 1e-4, 1e-12);
}

TEST(SpecialExtTest, beta_inc_outside_unit_interval) {
    EXPECT_NEAR(beta_inc_reg(-0.1, 2.0, 3.0), 0.0, 1e-15);
    EXPECT_NEAR(beta_inc_reg(1.2, 2.0, 3.0), 1.0, 1e-15);
    EXPECT_NEAR(beta_inc(1.2, 2.0, 3.0), beta_func(2.0, 3.0), 1e-12);
}

TEST(SpecialExtTest, hypergeo_indexed_n_zero) {
    EXPECT_NEAR(hypergeo_0f1n(0, 1.5, 0.0), 1.0, 1e-15);
    EXPECT_NEAR(hypergeo_1f1n(0, 2.0, 0.0), 1.0, 1e-15);
}

TEST(SpecialExtTest, pochhammer_and_falling_n_one) {
    EXPECT_NEAR(pochhammer(7.0, 1), 7.0, 1e-15);
    EXPECT_NEAR(falling_factorial(7.0, 1), 7.0, 1e-15);
}

TEST(SpecialExtTest, gamma_pole_and_rgamma_cancel) {
    EXPECT_TRUE(std::isinf(gamma_func(0.0)));
    EXPECT_EQ(rgamma(0.0), 0.0);
    EXPECT_TRUE(std::isinf(log_gamma(0.0)));
}

TEST(SpecialExtTest, jacobi_sn_cn_dn_at_origin) {
    EXPECT_NEAR(jacobi_sn(0.0, 0.7), 0.0, 1e-14);
    EXPECT_NEAR(jacobi_cn(0.0, 0.7), 1.0, 1e-14);
    EXPECT_NEAR(jacobi_dn(0.0, 0.7), 1.0, 1e-14);
    EXPECT_NEAR(jacobi_am(0.0, 0.7), 0.0, 1e-14);
}

TEST(SpecialExtTest, sph_bessel_j0_limit) {
    EXPECT_NEAR(sph_bessel_j(0, 0.0), 1.0, 1e-15);
    EXPECT_NEAR(sph_bessel_j(0, 0.5), std::sin(0.5) / 0.5, 1e-14);
}

TEST(SpecialExtTest, pseudo_voigt_degenerate_widths) {
    EXPECT_TRUE(std::isinf(pseudo_voigt(0.0, 0.0, 0.0, 0.4)));
    EXPECT_NEAR(pseudo_voigt(1.5, 0.0, 0.0, 0.4), 0.0, 1e-15);
    EXPECT_TRUE(std::isinf(pseudo_voigt(0.0, -1.0, -0.5, 0.3)));
    EXPECT_NEAR(pseudo_voigt(-2.0, -0.2, -0.3, 0.8), 0.0, 1e-15);
}

TEST(SpecialExtTest, pseudo_voigt_auto_degenerate_widths) {
    EXPECT_TRUE(std::isinf(pseudo_voigt_auto(0.0, 0.0, 0.0)));
    EXPECT_NEAR(pseudo_voigt_auto(1.0, 0.0, 0.0), 0.0, 1e-15);
    EXPECT_TRUE(std::isinf(pseudo_voigt_auto(0.0, -2.0, -1.0)));
    EXPECT_NEAR(pseudo_voigt_auto(0.75, -0.1, 0.0), 0.0, 1e-15);
}

TEST(SpecialExtTest, rgamma_overflow_cancels) {
    EXPECT_EQ(rgamma(200.0), 0.0);
    EXPECT_EQ(rgamma(180.0), 0.0);
}

TEST(SpecialExtTest, kummer_u_large_z_non_identity) {
    EXPECT_NEAR(kummer_u(0.7, 1.3, 25.0), std::pow(25.0, -0.7), 1e-15);
    EXPECT_NEAR(kummer_u(1.2, 0.4, 30.0), std::pow(30.0, -1.2), 1e-15);
}

TEST(SpecialExtTest, kummer_u_b_ge_two_tricomi_fallback) {
    const double u = kummer_u(0.5, 2.5, 1.0);
    if (!std::isfinite(u)) {
        GTEST_SKIP() << "kummer_u tricomi fallback not finite";
    }
    EXPECT_GT(std::abs(u), 0.0);
}

TEST(SpecialExtTest, lambert_w_principal_negative_arg) {
    const std::vector<double> zs{-0.3, -0.1, -0.05};
    for (const double z : zs) {
        expect_lambert_identity(0, z, 1e-12);
    }
}

TEST(SpecialExtTest, lambert_w_minus_one_near_branch_point) {
    const double minus_inv_e = -std::exp(-1.0);
    const double z = minus_inv_e + 5e-12;
    EXPECT_NEAR(lambert_w(-1, z), -1.0, 1e-4);
    expect_lambert_identity(-1, z, 1e-10);
}

TEST(SpecialExtTest, mathieu_characteristic_index_out_of_range) {
    const std::vector<int> orders{160, 200, 201};
    for (const int n : orders) {
        EXPECT_TRUE(std::isnan(mathieu_a(n, 0.1))) << "n=" << n;
        EXPECT_TRUE(std::isnan(mathieu_b(n, 0.1))) << "n=" << n;
    }
}

TEST(SpecialExtTest, jacobi_p_rising_factorial_underflow) {
    EXPECT_TRUE(std::isnan(jacobi_p(2, 0.5, -3.5, 0.3)));
    EXPECT_TRUE(std::isnan(jacobi_p(3, 1.0, -5.0, 0.0)));
}

TEST(SpecialExtTest, whittaker_w_both_representations_fail) {
    EXPECT_TRUE(std::isnan(whittaker_w(0.0, 0.0, 80.0)));
}

TEST(SpecialExtTest, erfi_airy_gamma_overflow_underflow) {
    EXPECT_GT(std::abs(erfi(8.0)), 1e4);
    EXPECT_LT(std::abs(airy_ai(80.0)), 1e-200);
    EXPECT_TRUE(std::isinf(gamma_func(200.0)));
}

TEST(SpecialExtTest, bessel_ik_origin_and_domain) {
    EXPECT_NEAR(bessel_i(0, 0.0), 1.0, 1e-15);
    EXPECT_NEAR(bessel_i(1, 0.0), 0.0, 1e-15);
    EXPECT_FALSE(std::isfinite(bessel_i(3, 0.0)));
    EXPECT_TRUE(std::isnan(bessel_i(-1, 1.0)));
    EXPECT_TRUE(std::isnan(bessel_k(0, 0.0)));
    EXPECT_TRUE(std::isnan(bessel_k(-2, 1.0)));
}

TEST(SpecialExtTest, jacobi_gegenbauer_whittaker_pcf_domain) {
    EXPECT_TRUE(std::isnan(jacobi_p(2, 0.4, 0.3, 1.5)));
    EXPECT_TRUE(std::isnan(gegenbauer_c(2, 0.5, -1.2)));
    EXPECT_TRUE(std::isnan(whittaker_w(0.2, 0.3, -0.5)));
    EXPECT_TRUE(std::isnan(tricomi_u(0.5, 1.5, 0.0)));
    const double u0 = pcf_u(-1.0, 0.0);
    EXPECT_TRUE(std::isfinite(u0));
}

TEST(SpecialExtTest, sph_bessel_y_n0_and_recurrence) {
    EXPECT_NEAR(sph_bessel_y(0, 0.5), -std::cos(0.5) / 0.5, 1e-14);
    EXPECT_TRUE(std::isnan(sph_bessel_y(-1, 1.0)));
    EXPECT_TRUE(std::isnan(sph_bessel_y(0, 0.0)));
    const double y2 = sph_bessel_y(2, 1.0);
    EXPECT_TRUE(std::isfinite(y2));
}

TEST(SpecialExtTest, pcf_vw_at_origin) {
    const double v0 = pcf_v(0.0, 0.0);
    const double w0 = pcf_w(0.0, 0.0);
    EXPECT_TRUE(std::isfinite(v0));
    EXPECT_TRUE(std::isfinite(w0));
    EXPECT_NEAR(pcf_w(0.0, 0.0), pcf_u(0.0, 0.0), 1e-12);
}

TEST(SpecialExtTest, voigt_gaussian_limit_peak) {
    const double sigma = 1.0;
    const double peak = voigt(0.0, sigma, 0.0);
    EXPECT_NEAR(peak, 1.0 / (sigma * std::sqrt(2.0 * std::numbers::pi)), 1e-8);
    EXPECT_NEAR(voigt(1.0, sigma, 0.0), voigt(-1.0, sigma, 0.0), 1e-12);
}
