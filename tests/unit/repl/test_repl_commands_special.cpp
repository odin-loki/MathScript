#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, special_erfinv) {
    Interpreter interp;
    expect_contains(interp, "help", "special_erfinv(x)");

    const double ref = ms::erfinv(0.5);
    expect_ok(interp, "y = special_erfinv(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_erfinv(0.5)", std::to_string(ref));
}

TEST(ReplCommandsTest, special_erfcinv) {
    Interpreter interp;
    expect_contains(interp, "help", "special_erfcinv(x)");

    // erfcinv(x) == erfinv(1-x) (DLMF 7.17 relationship between erf^-1 and erfc^-1).
    const double ref = ms::erfinv(1.0 - 0.3);
    expect_ok(interp, "y = special_erfcinv(0.3)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_erfcinv(0.3)", std::to_string(ref));
}

TEST(ReplCommandsTest, special_log_gamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_log_gamma(x)");

    // log_gamma(5) == ln(4!) == ln(24).
    const double ref = ms::log_gamma(5.0);
    EXPECT_NEAR(ref, std::log(24.0), 1e-9);
    expect_ok(interp, "y = special_log_gamma(5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_log_gamma(5)", std::to_string(ref));
}

TEST(ReplCommandsTest, special_digamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_digamma(x)");

    // digamma(1) == -EulerGamma.
    const double ref = ms::digamma(1.0);
    EXPECT_NEAR(ref, -0.5772156649015329, 1e-6);
    expect_ok(interp, "y = special_digamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_digamma(1)", std::to_string(ref));
}

TEST(ReplCommandsTest, special_trigamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_trigamma(x)");

    // trigamma(1) == pi^2/6 (implementation uses a series approximation, hence the looser tolerance).
    const double ref = ms::trigamma(1.0);
    constexpr double kPi = 3.14159265358979323846;
    EXPECT_NEAR(ref, kPi * kPi / 6.0, 1e-5);
    expect_ok(interp, "y = special_trigamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_trigamma(1)", std::to_string(ref));
}

TEST(ReplCommandsTest, special_polygamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_polygamma(n,x)");

    // polygamma(1, x) is the trigamma function.
    const double ref = ms::polygamma(1, 1.0);
    EXPECT_NEAR(ref, ms::trigamma(1.0), 1e-9);
    expect_ok(interp, "y = special_polygamma(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_polygamma(1, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, special_gamma_inc_reg) {
    Interpreter interp;
    expect_contains(interp, "help", "special_gamma_inc_reg(a,x)");

    // P(1, x) == 1 - exp(-x); use x = ln(2) so the expected value is 0.5.
    // 0.69314718055994530942 is ln(2) to full double precision (std::to_string would
    // truncate to 6 decimals and reintroduce error when re-parsed by the REPL).
    const std::string x_str = "0.69314718055994530942";
    const double x = std::log(2.0);
    const double ref = ms::gamma_inc_reg(1.0, x);
    EXPECT_NEAR(ref, 0.5, 1e-6);
    expect_ok(interp, "y = special_gamma_inc_reg(1, " + x_str + ")");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_gamma_inc_reg(1, " + x_str + ")", std::to_string(ref));
}

TEST(ReplCommandsTest, special_gamma_inc_reg_upper) {
    Interpreter interp;
    expect_contains(interp, "help", "special_gamma_inc_reg_upper(a,x)");

    // Q(1, x) == exp(-x); use x = ln(2) so the expected value is 0.5.
    const std::string x_str = "0.69314718055994530942";
    const double x = std::log(2.0);
    const double ref = ms::gamma_inc_reg_upper(1.0, x);
    EXPECT_NEAR(ref, 0.5, 1e-6);
    expect_ok(interp, "y = special_gamma_inc_reg_upper(1, " + x_str + ")");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_gamma_inc_reg_upper(1, " + x_str + ")",
                    std::to_string(ref));
}

TEST(ReplCommandsTest, special_beta_inc_reg) {
    Interpreter interp;
    expect_contains(interp, "help", "special_beta_inc_reg(x,a,b)");

    // I_x(1,1) == x since Beta(1,1) is the uniform distribution on [0,1].
    const double ref = ms::beta_inc_reg(0.3, 1.0, 1.0);
    EXPECT_NEAR(ref, 0.3, 1e-6);
    expect_ok(interp, "y = special_beta_inc_reg(0.3, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_beta_inc_reg(0.3, 1, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, special_erfi_gamma_inc) {
    Interpreter interp;
    expect_contains(interp, "help", "erfi(x)");
    expect_contains(interp, "help", "erfcx(x)");
    expect_contains(interp, "help", "dawson(x)");
    expect_contains(interp, "help", "special_rgamma(x)");
    expect_contains(interp, "help", "special_pochhammer(a,n)");
    expect_contains(interp, "help", "special_falling_factorial(a,n)");
    expect_contains(interp, "help", "special_gamma_inc(a,x)");
    expect_contains(interp, "help", "special_beta_inc(x,a,b)");
    expect_contains(interp, "help", "beta(a,b)");

    const double erfi_ref = ms::erfi(0.5);
    expect_ok(interp, "y = erfi(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), erfi_ref, 1e-9);
    expect_contains(interp, "erfi(0.5)", std::to_string(erfi_ref));

    const double erfcx_ref = ms::erfcx(0.0);
    EXPECT_NEAR(erfcx_ref, 1.0, 1e-12);
    expect_ok(interp, "y = erfcx(0)");
    EXPECT_NEAR(interp.state().scalars.at("y"), erfcx_ref, 1e-9);

    const double dawson_ref = ms::dawson(0.0);
    EXPECT_NEAR(dawson_ref, 0.0, 1e-12);
    expect_ok(interp, "y = dawson(0)");
    EXPECT_NEAR(interp.state().scalars.at("y"), dawson_ref, 1e-9);

    const double rgamma_ref = ms::rgamma(1.0);
    EXPECT_NEAR(rgamma_ref, 1.0, 1e-12);
    expect_ok(interp, "y = special_rgamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), rgamma_ref, 1e-9);

    const double poch_ref = ms::pochhammer(2.5, 3);
    expect_ok(interp, "y = special_pochhammer(2.5, 3)");
    EXPECT_NEAR(interp.state().scalars.at("y"), poch_ref, 1e-9);

    const double fall_ref = ms::falling_factorial(5.0, 2);
    expect_ok(interp, "y = special_falling_factorial(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("y"), fall_ref, 1e-9);

    const std::string x_str = "0.69314718055994530942";
    const double x = std::log(2.0);
    const double gamma_inc_ref = ms::gamma_inc(1.0, x);
    EXPECT_NEAR(gamma_inc_ref, 0.5, 1e-6);
    expect_ok(interp, "y = special_gamma_inc(1, " + x_str + ")");
    EXPECT_NEAR(interp.state().scalars.at("y"), gamma_inc_ref, 1e-9);
    expect_contains(interp, "special_gamma_inc(1, " + x_str + ")", std::to_string(gamma_inc_ref));

    const double beta_inc_ref = ms::beta_inc(0.3, 1.0, 1.0);
    EXPECT_NEAR(beta_inc_ref, 0.3, 1e-6);
    expect_ok(interp, "y = special_beta_inc(0.3, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), beta_inc_ref, 1e-9);

    const double beta_ref = ms::beta_func(2.0, 3.0);
    expect_ok(interp, "y = beta(2, 3)");
    EXPECT_NEAR(interp.state().scalars.at("y"), beta_ref, 1e-9);
    expect_contains(interp, "beta(2, 3)", std::to_string(beta_ref));
}

TEST(ReplCommandsTest, special_voigt_airy) {
    Interpreter interp;
    expect_contains(interp, "help", "special_voigt(x,sigma,gamma)");
    expect_contains(interp, "help", "special_pseudo_voigt_auto(x,sigma,gamma)");
    expect_contains(interp, "help", "special_airy_ai(x)");

    constexpr double kPi = 3.14159265358979323846;
    const double voigt_ref = ms::voigt(0.0, 1.0, 0.0);
    EXPECT_NEAR(voigt_ref, 1.0 / std::sqrt(2.0 * kPi), 1e-9);
    expect_ok(interp, "y = special_voigt(0,1,0)");
    EXPECT_NEAR(interp.state().scalars.at("y"), voigt_ref, 1e-9);
    expect_contains(interp, "special_voigt(0,1,0)", std::to_string(voigt_ref));

    const double airy_ref = ms::airy_ai(0.0);
    EXPECT_NEAR(airy_ref, 0.355028053887817, 1e-9);
    expect_ok(interp, "z = special_airy_ai(0)");
    EXPECT_NEAR(interp.state().scalars.at("z"), airy_ref, 1e-9);
    expect_contains(interp, "special_airy_ai(0)", std::to_string(airy_ref));
}

TEST(ReplCommandsTest, special_bessel_lambert_kummer) {
    Interpreter interp;
    expect_contains(interp, "help", "bessel_y(nu,x)");
    expect_contains(interp, "help", "bessel_i(nu,x)");
    expect_contains(interp, "help", "lambert_w(branch,z)");
    expect_contains(interp, "help", "kummer_u(a,b,z)");
    expect_contains(interp, "help", "special_airy_bi(x)");

    const double y_ref = ms::bessel_y(0, 1.0);
    EXPECT_NEAR(y_ref, 0.088256964, 1e-7);
    expect_ok(interp, "y = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), y_ref, 1e-9);
    expect_contains(interp, "bessel_y(0, 1)", std::to_string(y_ref));
    expect_ok(interp, "ys = special_bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ys"), y_ref, 1e-9);

    const double i_ref = ms::bessel_i(0, 1.0);
    EXPECT_NEAR(i_ref, 1.2660658777520084, 1e-6);
    expect_ok(interp, "i = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("i"), i_ref, 1e-9);
    expect_ok(interp, "is = special_bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("is"), i_ref, 1e-9);

    const double w_ref = ms::lambert_w(0, 1.0);
    EXPECT_NEAR(w_ref, 0.5671432904097838, 1e-10);
    expect_ok(interp, "w = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("w"), w_ref, 1e-9);
    expect_ok(interp, "ws = special_lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ws"), w_ref, 1e-9);

    const double u_ref = ms::kummer_u(1.0, 2.0, 0.5);
    EXPECT_NEAR(u_ref, 2.0, 1e-12);
    expect_ok(interp, "u = kummer_u(1, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("u"), u_ref, 1e-9);
    expect_contains(interp, "kummer_u(1, 2, 0.5)", std::to_string(u_ref));

    const double bi_ref = ms::airy_bi(0.0);
    EXPECT_NEAR(bi_ref, 0.6149266274460007, 1e-6);
    expect_ok(interp, "bi = special_airy_bi(0)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), bi_ref, 1e-9);
    expect_contains(interp, "special_airy_bi(0)", std::to_string(bi_ref));
}

TEST(ReplCommandsTest, special_orthog_bessel) {
    Interpreter interp;
    expect_contains(interp, "help", "bessel_k(nu,x)");
    expect_contains(interp, "help", "chebyshev_t(n,x)");
    expect_contains(interp, "help", "chebyshev_u(n,x)");
    expect_contains(interp, "help", "hermite_h(n,x)");
    expect_contains(interp, "help", "laguerre_l(n,x)");
    expect_contains(interp, "help", "sph_bessel_j(n,x)");
    expect_contains(interp, "help", "sph_bessel_y(n,x)");
    expect_contains(interp, "help", "assoc_legendre_p(l,m,x)");
    expect_contains(interp, "help", "gegenbauer_c(n,lambda,x)");

    const double k_ref = ms::bessel_k(0, 1.0);
    EXPECT_NEAR(k_ref, 0.421024438240708, 1e-6);
    expect_ok(interp, "k = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("k"), k_ref, 1e-9);
    expect_contains(interp, "bessel_k(0, 1)", std::to_string(k_ref));
    expect_ok(interp, "ks = special_bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ks"), k_ref, 1e-9);

    const double t_ref = ms::chebyshev_t(3, 0.5);
    EXPECT_NEAR(t_ref, -1.0, 1e-12);
    expect_ok(interp, "t = chebyshev_t(3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("t"), t_ref, 1e-9);
    expect_contains(interp, "chebyshev_t(3, 0.5)", std::to_string(t_ref));

    const double u_ref = ms::chebyshev_u(2, 0.5);
    EXPECT_NEAR(u_ref, 0.0, 1e-12);
    expect_ok(interp, "u = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("u"), u_ref, 1e-9);

    const double h_ref = ms::hermite_h(3, 1.0);
    EXPECT_NEAR(h_ref, -4.0, 1e-12);
    expect_ok(interp, "h = hermite_h(3, 1)");
    EXPECT_NEAR(interp.state().scalars.at("h"), h_ref, 1e-9);

    const double l_ref = ms::laguerre_l(2, 0.5);
    EXPECT_NEAR(l_ref, 0.125, 1e-12);
    expect_ok(interp, "l = laguerre_l(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("l"), l_ref, 1e-9);

    const double j_ref = ms::sph_bessel_j(2, 1.0);
    EXPECT_NEAR(j_ref, 0.062035052011373916, 1e-9);
    expect_ok(interp, "j = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("j"), j_ref, 1e-9);

    const double y_ref = ms::sph_bessel_y(1, 1.0);
    expect_ok(interp, "y = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), y_ref, 1e-9);

    const double p_ref = ms::assoc_legendre_p(2, 1, 0.5);
    expect_ok(interp, "p = assoc_legendre_p(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("p"), p_ref, 1e-9);
    expect_contains(interp, "assoc_legendre_p(2, 1, 0.5)", std::to_string(p_ref));

    const double c_ref = ms::gegenbauer_c(2, 1.0, 0.5);
    EXPECT_NEAR(c_ref, 0.0, 1e-12);
    expect_ok(interp, "c = gegenbauer_c(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("c"), c_ref, 1e-9);
}

TEST(ReplCommandsTest, special_legendre_q_sph_harm) {
    Interpreter interp;
    expect_contains(interp, "help", "legendre_q(n,x)");
    expect_contains(interp, "help", "hermite_he(n,x)");
    expect_contains(interp, "help", "laguerre_la(n,a,x)");
    expect_contains(interp, "help", "chebyshev_v(n,x)");
    expect_contains(interp, "help", "chebyshev_w(n,x)");
    expect_contains(interp, "help", "sph_harm(l,m,theta,phi)");

    const double q_ref = ms::legendre_q(2, 0.3);
    EXPECT_TRUE(std::isfinite(q_ref));
    expect_ok(interp, "q = legendre_q(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("q"), q_ref, 1e-9);
    expect_contains(interp, "legendre_q(2, 0.3)", std::to_string(q_ref));

    const double he_ref = ms::hermite_he(2, 0.5);
    EXPECT_NEAR(he_ref, -0.75, 1e-12);
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), he_ref, 1e-9);

    const double la_ref = ms::laguerre_la(2, 1.0, 0.5);
    EXPECT_NEAR(la_ref, 1.625, 1e-6);
    expect_ok(interp, "la = laguerre_la(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("la"), la_ref, 1e-9);
    expect_contains(interp, "laguerre_la(2, 1, 0.5)", std::to_string(la_ref));

    const double v_ref = ms::chebyshev_v(2, 0.5);
    EXPECT_NEAR(v_ref, 1.0, 1e-6);
    expect_ok(interp, "v = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("v"), v_ref, 1e-9);

    const double w_ref = ms::chebyshev_w(2, 0.5);
    EXPECT_NEAR(w_ref, 0.5773502691896257, 1e-6);
    expect_ok(interp, "w = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("w"), w_ref, 1e-9);

    const std::complex<double> y_ref = ms::sph_harm_y(1, 1, 0.5, 1.0);
    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("Y")(0, 0), y_ref.real(), 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("Y")(0, 1), y_ref.imag(), 1e-6);

    expect_ok(interp, "sph_harm(1, 1, 0.5, 1)");
}

TEST(ReplCommandsTest, special_elliptic_jacobi_theta) {
    Interpreter interp;
    expect_contains(interp, "help", "ellip_e(k)");
    expect_contains(interp, "help", "ellip_pi(n,k)");
    expect_contains(interp, "help", "ellip_f(phi,k)");
    expect_contains(interp, "help", "ellip_e_inc(phi,k)");
    expect_contains(interp, "help", "jacobi_cn(u,k)");
    expect_contains(interp, "help", "jacobi_dn(u,k)");
    expect_contains(interp, "help", "jacobi_am(u,k)");
    expect_contains(interp, "help", "theta1(z,q)");
    expect_contains(interp, "help", "theta2(z,q)");
    expect_contains(interp, "help", "theta3(z,q)");
    expect_contains(interp, "help", "theta4(z,q)");

    const double k = 0.5;
    const double u = 0.5;
    const double z = 0.5;
    const double q = 0.3;

    const double ee_ref = ms::ellip_e(k);
    EXPECT_NEAR(ee_ref, 1.4674622093394272, 1e-6);
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ee_ref, 1e-9);
    expect_contains(interp, "ellip_e(0.5)", "\n");

    const double pi_ref = ms::ellip_pi(0.5, k);
    EXPECT_NEAR(pi_ref, 2.4136715042011945, 1e-3);
    expect_ok(interp, "epi = ellip_pi(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("epi"), pi_ref, 1e-3);
    expect_contains(interp, "ellip_pi(0.5, 0.5)", "\n");

    const double f_ref = ms::ellip_f(0.3, k);
    EXPECT_NEAR(f_ref, 0.30111597966406606, 1e-3);
    expect_ok(interp, "ef = ellip_f(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), f_ref, 1e-3);

    const double einc_ref = ms::ellip_e_inc(0.3, k);
    EXPECT_NEAR(einc_ref, 0.2988914110164986, 1e-3);
    expect_ok(interp, "einc = ellip_e_inc(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("einc"), einc_ref, 1e-3);

    const double cn_ref = ms::jacobi_cn(u, k);
    EXPECT_NEAR(cn_ref, 0.8799410229637583, 1e-3);
    expect_ok(interp, "cn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cn"), cn_ref, 1e-3);
    expect_contains(interp, "jacobi_cn(0.5, 0.5)", "\n");

    const double dn_ref = ms::jacobi_dn(u, k);
    EXPECT_NEAR(dn_ref, 0.9713773988381788, 1e-3);
    expect_ok(interp, "dn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dn"), dn_ref, 1e-3);

    const double am_ref = ms::jacobi_am(u, k);
    EXPECT_NEAR(am_ref, std::atan2(ms::jacobi_sn(u, k), cn_ref), 1e-3);
    expect_ok(interp, "am = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("am"), am_ref, 1e-3);

    const double t1_ref = ms::theta1(z, q);
    EXPECT_NEAR(t1_ref, 0.5773940463248446, 1e-3);
    expect_ok(interp, "t1 = theta1(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t1"), t1_ref, 1e-3);
    expect_contains(interp, "theta1(0.5, 0.3)", "\n");

    const double t2_ref = ms::theta2(z, q);
    EXPECT_NEAR(t2_ref, 1.3075255735032947, 1e-3);
    expect_ok(interp, "t2 = theta2(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), t2_ref, 1e-3);

    const double t3_ref = ms::theta3(z, q);
    EXPECT_NEAR(t3_ref, 1.317400827096804, 1e-3);
    expect_ok(interp, "t3 = theta3(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), t3_ref, 1e-3);
    expect_contains(interp, "theta3(0.5, 0.3)", "\n");

    const double t4_ref = ms::theta4(z, q);
    EXPECT_NEAR(t4_ref, 0.6691160041441827, 1e-3);
    expect_ok(interp, "t4 = theta4(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), t4_ref, 1e-3);
}

TEST(ReplCommandsTest, special_bessel_struve_kelvin_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "spherical_in(n,x)");
    expect_contains(interp, "help", "spherical_kn(n,x)");
    expect_contains(interp, "help", "struve_l(nu,x)");
    expect_contains(interp, "help", "struve_k(nu,x)");
    expect_contains(interp, "help", "anger_j(nu,x)");
    expect_contains(interp, "help", "weber_e(nu,x)");
    expect_contains(interp, "help", "kelvin_bei(nu,x)");
    expect_contains(interp, "help", "kelvin_ker(nu,x)");
    expect_contains(interp, "help", "kelvin_kei(nu,x)");
    expect_contains(interp, "help", "bessel_zero_ynu(nu,n)");

    const double in_ref = ms::spherical_in(0, 1.0);
    EXPECT_NEAR(in_ref, 1.1752011936438014, 1e-6);
    expect_ok(interp, "in0 = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("in0"), in_ref, 1e-9);
    expect_contains(interp, "spherical_in(0, 1)", std::to_string(in_ref));

    const double kn_ref = ms::spherical_kn(0, 1.0);
    EXPECT_NEAR(kn_ref, 0.5778636748954609, 1e-3);
    expect_ok(interp, "kn0 = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kn0"), kn_ref, 1e-3);

    const double sl_ref = ms::struve_l(0, 1.0);
    EXPECT_NEAR(sl_ref, 0.5686566270482879, 1e-6);
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), sl_ref, 1e-9);

    const double sk_ref = ms::struve_k(0, 1.0);
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), sk_ref, 1e-9);

    const double aj_ref = ms::anger_j(1, 1.0);
    EXPECT_NEAR(aj_ref, 0.440050585744933, 1e-6);
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), aj_ref, 1e-9);

    const double we_ref = ms::weber_e(0, 1.0);
    EXPECT_NEAR(we_ref, -0.5686566270482879, 1e-6);
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), we_ref, 1e-9);

    const double bei_ref = ms::kelvin_bei(0, 1.0);
    EXPECT_NEAR(bei_ref, 0.24956604003665972, 1e-6);
    expect_ok(interp, "bei = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bei"), bei_ref, 1e-9);
    expect_contains(interp, "kelvin_bei(0, 1)", std::to_string(bei_ref));

    const double ker_ref = ms::kelvin_ker(0, 1.0);
    EXPECT_NEAR(ker_ref, 0.28670620872831604, 1e-6);
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ker_ref, 1e-9);

    const double kei_ref = ms::kelvin_kei(0, 1.0);
    EXPECT_NEAR(kei_ref, -0.49499463651872, 1e-6);
    expect_ok(interp, "kei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kei"), kei_ref, 1e-9);

    const double yz_ref = ms::bessel_zero_ynu(0, 1);
    EXPECT_NEAR(yz_ref, 0.893576974377206, 1e-3);
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), yz_ref, 1e-3);
    expect_contains(interp, "bessel_zero_ynu(0, 1)", std::to_string(yz_ref));
}

TEST(ReplCommandsTest, special_zeta_airy_orthog_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "zeta_hurwitz(s,a)");
    expect_contains(interp, "help", "lerch_phi(z,s,a)");
    expect_contains(interp, "help", "beta_dirichlet(s)");
    expect_contains(interp, "help", "bernoulli_number(n)");
    expect_contains(interp, "help", "euler_number(n)");
    expect_contains(interp, "help", "airy_aip(x)");
    expect_contains(interp, "help", "airy_bip(x)");
    expect_contains(interp, "help", "legendre_pn(n,m,x)");
    expect_contains(interp, "help", "hermite_hf(n,x)");
    expect_contains(interp, "help", "laguerre_ln(n,k,x)");
    expect_contains(interp, "help", "chebyshev_tn(n,k,x)");
    expect_contains(interp, "help", "chebyshev_un(n,k,x)");

    const double zeta_ref = ms::zeta_hurwitz(2.0, 0.3);
    EXPECT_NEAR(zeta_ref, 12.245364546107732, 1e-3);
    expect_ok(interp, "z = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("z"), zeta_ref, 1e-3);
    expect_contains(interp, "zeta_hurwitz(2, 0.3)", std::to_string(zeta_ref));

    const double lerch_ref = ms::lerch_phi(0.5, 2.0, 0.3);
    EXPECT_NEAR(lerch_ref, 11.47083462974499, 1e-3);
    expect_ok(interp, "lp = lerch_phi(0.5, 2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), lerch_ref, 1e-3);

    const double beta_ref = ms::beta_dirichlet(2.0);
    EXPECT_NEAR(beta_ref, 0.915965594127219, 1e-3);
    expect_ok(interp, "bd = beta_dirichlet(2)");
    EXPECT_NEAR(interp.state().scalars.at("bd"), beta_ref, 1e-3);

    const double bern_ref = ms::bernoulli_number(2);
    EXPECT_NEAR(bern_ref, 1.0 / 6.0, 1e-12);
    expect_ok(interp, "b = bernoulli_number(2)");
    EXPECT_NEAR(interp.state().scalars.at("b"), bern_ref, 1e-9);

    const double euler_ref = ms::euler_number(4);
    EXPECT_NEAR(euler_ref, 5.0, 1e-12);
    expect_ok(interp, "e = euler_number(4)");
    EXPECT_NEAR(interp.state().scalars.at("e"), euler_ref, 1e-9);

    const double aip_ref = ms::airy_aip(0.0);
    EXPECT_LT(aip_ref, 0.0);
    expect_ok(interp, "aip = airy_aip(0)");
    EXPECT_NEAR(interp.state().scalars.at("aip"), aip_ref, 1e-3);

    const double bip_ref = ms::airy_bip(0.0);
    EXPECT_GT(bip_ref, 0.0);
    expect_ok(interp, "bip = airy_bip(0)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), bip_ref, 1e-3);

    const double pn_ref = ms::legendre_pn(2, 0, 0.5);
    EXPECT_NEAR(pn_ref, -0.125, 1e-12);
    expect_ok(interp, "pn = legendre_pn(2, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("pn"), pn_ref, 1e-9);

    const double hf_ref = ms::hermite_hf(2, 0.5);
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), hf_ref, 1e-9);

    const double ln_ref = ms::laguerre_ln(2, 0, 0.5);
    EXPECT_NEAR(ln_ref, 0.125, 1e-12);
    expect_ok(interp, "ln = laguerre_ln(2, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ln"), ln_ref, 1e-9);

    const double tn_ref = ms::chebyshev_tn(3, 0, 0.5);
    EXPECT_NEAR(tn_ref, -1.0, 1e-12);
    expect_ok(interp, "tn = chebyshev_tn(3, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("tn"), tn_ref, 1e-9);

    const double un_ref = ms::chebyshev_un(2, 0, 0.5);
    EXPECT_NEAR(un_ref, ms::chebyshev_u(2, 0.5), 1e-12);
    expect_ok(interp, "un = chebyshev_un(2, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("un"), un_ref, 1e-9);
}

TEST(ReplCommandsTest, special_voigt_weierstrass) {
    Interpreter interp;
    expect_contains(interp, "help", "special_pseudo_voigt(x,sigma,gamma,eta)");
    expect_contains(interp, "help", "weierstrass_p(z,g2,g3)");
    expect_contains(interp, "help", "weierstrass_pprime(z,g2,g3)");

    const double pv_ref = ms::pseudo_voigt(0.0, 1.0, 0.5, 0.4);
    expect_ok(interp, "pv = special_pseudo_voigt(0, 1, 0.5, 0.4)");
    EXPECT_NEAR(interp.state().scalars.at("pv"), pv_ref, 1e-9);
    expect_contains(interp, "special_pseudo_voigt(0, 1, 0.5, 0.4)", std::to_string(pv_ref));

    const double wp_ref = ms::weierstrass_p(0.5, 1.0, 0.0);
    EXPECT_NEAR(wp_ref, 4.012516276465522, 1e-3);
    expect_ok(interp, "wp = weierstrass_p(0.5, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("wp"), wp_ref, 1e-3);
    expect_contains(interp, "weierstrass_p(0.5, 1, 0)", "\n");

    const double wpp_ref = ms::weierstrass_pprime(0.5, 1.0, 0.0);
    EXPECT_NEAR(wpp_ref, -15.9498046875, 1e-2);
    expect_ok(interp, "wpp = weierstrass_pprime(0.5, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("wpp"), wpp_ref, 1e-2);
    expect_contains(interp, "weierstrass_pprime(0.5, 1, 0)", "\n");
}

TEST(ReplCommandsTest, special_jacobi_struve) {
    Interpreter interp;
    expect_contains(interp, "help", "jacobi_sc(u,k)");
    expect_contains(interp, "help", "jacobi_sd(u,k)");
    expect_contains(interp, "help", "jacobi_nc(u,k)");
    expect_contains(interp, "help", "jacobi_dc(u,k)");
    expect_contains(interp, "help", "struve_hn(nu,x)");
    expect_contains(interp, "help", "struve_yn(nu,x)");

    const double u = 0.5;
    const double k = 0.5;
    const double sn_ref = ms::jacobi_sn(u, k);
    const double cn_ref = ms::jacobi_cn(u, k);
    const double dn_ref = ms::jacobi_dn(u, k);

    const double sc_ref = ms::jacobi_sc(u, k);
    EXPECT_NEAR(sc_ref, sn_ref / cn_ref, 1e-6);
    expect_ok(interp, "sc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("sc"), sc_ref, 1e-3);
    expect_contains(interp, "jacobi_sc(0.5, 0.5)", "\n");

    const double sd_ref = ms::jacobi_sd(u, k);
    EXPECT_NEAR(sd_ref, sn_ref / dn_ref, 1e-6);
    expect_ok(interp, "sd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("sd"), sd_ref, 1e-3);

    const double nc_ref = ms::jacobi_nc(u, k);
    EXPECT_NEAR(nc_ref, cn_ref / sn_ref, 1e-6);
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), nc_ref, 1e-3);
    expect_contains(interp, "jacobi_nc(0.5, 0.5)", "\n");

    const double dc_ref = ms::jacobi_dc(u, k);
    EXPECT_NEAR(dc_ref, dn_ref / cn_ref, 1e-6);
    expect_ok(interp, "dc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dc"), dc_ref, 1e-3);

    const double hn_ref = ms::struve_hn(1, 1.0);
    EXPECT_TRUE(std::isfinite(hn_ref));
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), hn_ref, 1e-6);
    expect_contains(interp, "struve_hn(1, 1)", std::to_string(hn_ref));

    const double yn_ref = ms::struve_yn(1, 1.0);
    EXPECT_TRUE(std::isfinite(yn_ref));
    expect_ok(interp, "yn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yn"), yn_ref, 1e-6);
    expect_contains(interp, "struve_yn(1, 1)", std::to_string(yn_ref));
}

TEST(ReplCommandsTest, special_jacobi_extra) {
    Interpreter interp;
    expect_contains(interp, "help", "jacobi_nd(u,k)");
    expect_contains(interp, "help", "jacobi_ds(u,k)");

    const double u = 0.5;
    const double k = 0.5;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(u, k), 1e-9);
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(u, k), 1e-9);
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(u, k), 1e-9);
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(u, k), 1e-9);
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(u, k), 1e-9);
}

TEST(ReplCommandsTest, special_theta) {
    Interpreter interp;
    expect_contains(interp, "help", "theta1_prime(z,q)");
    expect_contains(interp, "help", "jacobi_theta(n,z,tau)");

    expect_ok(interp, "tp = theta1_prime(0.2, 0.3)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("tp")));
    expect_ok(interp, "jt = jacobi_theta(1, 0.2, 0.5)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jt")));
}

TEST(ReplCommandsTest, special_crypto_cellai) {
    Interpreter interp;
    expect_ok(interp, "spherical_yn(0, 1.0)");
    expect_ok(interp, "bessel_h(0, 1.0)");
    expect_ok(interp, "bessel_hy(1, 0.5)");
    expect_ok(interp, "bessel_l(1, 0.5)");
    expect_ok(interp, "bessel_lu(1, 0.5)");
    expect_ok(interp, "hermite_hn(2, 0.5)");
    expect_contains(interp, "crypto_to_hex(4142)", "4142");

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_bessel) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_time_evolve_psi");

    expect_ok(interp, "H = [1, 0; 0, 2]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "pt = quantum_time_evolve_psi(H, psi, 0.05)");
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 2u);

    expect_ok(interp, "bessel_j(0, 1.0)");
    expect_ok(interp, "bessel_j1(1.0)");
    expect_ok(interp, "bessel_y0(1.0)");
    expect_ok(interp, "bessel_y1(1.0)");
    expect_ok(interp, "bessel_zero_jnu(0, 1)");
}

TEST(ReplCommandsTest, ode_signal_special) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "sh = struve_h(1, 0.25)");
    expect_ok(interp, "js = jacobi_sn(0.5, 0.5)");
    EXPECT_TRUE(interp.state().scalars.count("sh") > 0);
    EXPECT_TRUE(interp.state().scalars.count("js") > 0);
}

TEST(ReplCommandsTest, ode_special) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-9);
}

TEST(ReplCommandsTest, signal_ode_special_2) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "lq = legendre_q(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(2, 0.5), 1e-9);
}

TEST(ReplCommandsTest, special_scalar) {
    Interpreter interp;

    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-9);

    expect_ok(interp, "sj = spherical_jn(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(0, 1.0), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_2) {
    Interpreter interp;

    expect_ok(interp, "ll = laguerre_l(1, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ll"), ms::laguerre_l(1, 0.3), 1e-9);

    expect_ok(interp, "ct = chebyshev_t(2, 0.4)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.4), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_3) {
    Interpreter interp;

    expect_ok(interp, "cu = chebyshev_u(1, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(1, 0.2), 1e-9);

    expect_ok(interp, "he = hermite_he(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.3), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_4) {
    Interpreter interp;

    expect_ok(interp, "cv = chebyshev_v(1, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(1, 0.3), 1e-9);

    expect_ok(interp, "cw = chebyshev_w(2, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.2), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_5) {
    Interpreter interp;

    expect_ok(interp, "lp = legendre_pn(1, 0, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_pn(1, 0, 0.3), 1e-9);

    expect_ok(interp, "ap = assoc_legendre_p(1, 0, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ap"), ms::assoc_legendre_p(1, 0, 0.3), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_9) {
    Interpreter interp;

    expect_ok(interp, "tn = chebyshev_tn(3, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("tn"), ms::chebyshev_tn(3, 0, 0.5), 1e-9);

    expect_ok(interp, "un = chebyshev_un(2, 1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("un"), ms::chebyshev_un(2, 1, 0.25), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_11) {
    Interpreter interp;

    expect_ok(interp, "ph = special_pochhammer(2.5, 3)");
    EXPECT_NEAR(interp.state().scalars.at("ph"), ms::pochhammer(2.5, 3), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_12) {
    Interpreter interp;

    expect_ok(interp, "ff = special_falling_factorial(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("ff"), ms::falling_factorial(5.0, 2), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_13) {
    Interpreter interp;

    expect_ok(interp, "pg = special_polygamma(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pg"), ms::polygamma(1, 1.0), 1e-9);
}

TEST(ReplCommandsTest, bessel_scalar) {
    Interpreter interp;

    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-9);

    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-9);

    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-9);

    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-9);

    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-9);

    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-9);
}

TEST(ReplCommandsTest, spherical_bessel_zero_scalar) {
    Interpreter interp;

    expect_ok(interp, "in0 = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("in0"), ms::spherical_in(0, 1.0), 1e-9);

    expect_ok(interp, "kn0 = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kn0"), ms::spherical_kn(0, 1.0), 1e-9);

    expect_ok(interp, "yn0 = spherical_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yn0"), ms::spherical_yn(0, 1.0), 1e-9);

    expect_ok(interp, "jz = bessel_zero_jnu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("jz"), ms::bessel_zero_jnu(0, 1), 1e-9);

    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-9);
}

TEST(ReplCommandsTest, bessel_scalar_2) {
    Interpreter interp;

    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-9);

    expect_ok(interp, "bhy = bessel_hy(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bhy"), ms::bessel_hy(1, 0.5), 1e-9);

    expect_ok(interp, "bl = bessel_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(1, 0.5), 1e-9);

    expect_ok(interp, "blu = bessel_lu(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(1, 0.5), 1e-9);

    expect_ok(interp, "hh = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_hn(2, 0.5), 1e-9);
}

TEST(ReplCommandsTest, struve_scalar) {
    Interpreter interp;

    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-9);

    expect_ok(interp, "sh = struve_h(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.5), 1e-9);

    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1.0), 1e-9);

    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-9);

    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-9);
}

TEST(ReplCommandsTest, lambert_w_scalar) {
    Interpreter interp;

    const double w_ref = ms::lambert_w(0, 1.0);
    expect_ok(interp, "w = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("w"), w_ref, 1e-9);

    expect_ok(interp, "ws = special_lambert_w(-1, -0.2)");
    EXPECT_NEAR(interp.state().scalars.at("ws"), ms::lambert_w(-1, -0.2), 1e-9);
}

TEST(ReplCommandsTest, hermite_h_scalar) {
    Interpreter interp;

    expect_ok(interp, "h = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar) {
    Interpreter interp;

    expect_ok(interp, "t = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("t"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar) {
    Interpreter interp;

    expect_ok(interp, "j = bessel_j(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("j"), ms::bessel_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar) {
    Interpreter interp;

    expect_ok(interp, "y = bessel_y(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ms::bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar) {
    Interpreter interp;

    expect_ok(interp, "i = bessel_i(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("i"), ms::bessel_i(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar) {
    Interpreter interp;

    expect_ok(interp, "k = bessel_k(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("k"), ms::bessel_k(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar) {
    Interpreter interp;

    expect_ok(interp, "u = chebyshev_u(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("u"), ms::chebyshev_u(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar) {
    Interpreter interp;

    expect_ok(interp, "v = chebyshev_v(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("v"), ms::chebyshev_v(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar) {
    Interpreter interp;

    expect_ok(interp, "w = chebyshev_w(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("w"), ms::chebyshev_w(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar) {
    Interpreter interp;
    expect_ok(interp, "h = bessel_h(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::bessel_h(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar) {
    Interpreter interp;
    expect_ok(interp, "lu = bessel_lu(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lu"), ms::bessel_lu(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_q_scalar) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_tn_scalar) {
    Interpreter interp;
    expect_ok(interp, "tn = chebyshev_tn(2, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("tn"), ms::chebyshev_tn(2, 0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_un_scalar) {
    Interpreter interp;
    expect_ok(interp, "un = chebyshev_un(2, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("un"), ms::chebyshev_un(2, 0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_jnu_scalar) {
    Interpreter interp;
    expect_ok(interp, "z = bessel_zero_jnu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("z"), ms::bessel_zero_jnu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar) {
    Interpreter interp;
    expect_ok(interp, "z = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("z"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar) {
    Interpreter interp;
    expect_ok(interp, "cn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar) {
    Interpreter interp;
    expect_ok(interp, "dn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar) {
    Interpreter interp;
    expect_ok(interp, "am = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("am"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar) {
    Interpreter interp;
    expect_ok(interp, "sc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("sc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar) {
    Interpreter interp;
    expect_ok(interp, "sd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("sd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar) {
    Interpreter interp;
    expect_ok(interp, "dc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar) {
    Interpreter interp;
    expect_ok(interp, "cs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar) {
    Interpreter interp;
    expect_ok(interp, "ds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cn(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cn(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sn(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sn(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dn(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dn(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cn(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cn(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dn(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dn(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_2) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_2) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_3) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_4) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_5) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_6) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_7) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_8) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_9) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_10) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_11) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_12) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_13) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_14) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_15) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_16) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_17) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_18) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_19) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_20) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_21) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_22) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_23) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_24) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_25) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_26) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_27) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_28) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_29) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_30) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_v_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_w_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_h_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_u_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, chebyshev_t_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_h_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_i_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_k_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sn_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cn_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dn_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_am_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sc_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_sd_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nc_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_dc_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_nd_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cd_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_cs_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ds_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, jacobi_ns_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j0_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_j1_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj1"), ms::bessel_j1(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y0_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_y1_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "by1 = bessel_y1(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by1"), ms::bessel_y1(0.5), 1e-8);
}

TEST(ReplCommandsTest, legendre_p_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, bessel_zero_ynu_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_l_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_h_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_hy_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_l_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}

TEST(ReplCommandsTest, bessel_lu_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}

TEST(ReplCommandsTest, poly_factor_sph_harm_31) {
    Interpreter interp;

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, legendre_q_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, special_pseudo_voigt_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "special_pseudo_voigt(0, 1, 0.5, 0.4)");
    expect_error_contains(interp, "special_pseudo_voigt(0, 1, 0.5, missing)",
                          "special_pseudo_voigt");
}

TEST(ReplCommandsTest, clausen_noassign) {
    Interpreter interp;
    expect_ok(interp, "clausen(1)");
}

TEST(ReplCommandsTest, eta_dirichlet_noassign) {
    Interpreter interp;
    expect_ok(interp, "eta_dirichlet(2)");
}

TEST(ReplCommandsTest, erf_noassign) {
    Interpreter interp;
    expect_ok(interp, "erf(0.5)");
    expect_error_contains(interp, "erf(missing)", "expected numeric");
}

TEST(ReplCommandsTest, erfc_noassign) {
    Interpreter interp;
    expect_ok(interp, "erfc(0.5)");
}

TEST(ReplCommandsTest, erfi_noassign) {
    Interpreter interp;
    expect_ok(interp, "erfi(0.5)");
}

TEST(ReplCommandsTest, erfcx_noassign) {
    Interpreter interp;
    expect_ok(interp, "erfcx(0)");
}

TEST(ReplCommandsTest, dawson_noassign) {
    Interpreter interp;
    expect_ok(interp, "dawson(0.5)");
}

TEST(ReplCommandsTest, dawsonx_noassign) {
    Interpreter interp;
    expect_ok(interp, "dawsonx(0.5)");
}

TEST(ReplCommandsTest, gamma_noassign) {
    Interpreter interp;
    expect_ok(interp, "gamma(5)");
}

TEST(ReplCommandsTest, bessel_j0_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_j0(1)");
}

TEST(ReplCommandsTest, bessel_j1_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_j1(1)");
}

TEST(ReplCommandsTest, bessel_y0_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_y0(1)");
}

TEST(ReplCommandsTest, bessel_y1_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_y1(1)");
}

TEST(ReplCommandsTest, fresnel_c_noassign) {
    Interpreter interp;
    expect_ok(interp, "fresnel_c(1)");
}

TEST(ReplCommandsTest, fresnel_s_noassign) {
    Interpreter interp;
    expect_ok(interp, "fresnel_s(1)");
}

TEST(ReplCommandsTest, ellip_k_noassign) {
    Interpreter interp;
    expect_ok(interp, "ellip_k(0.5)");
}

TEST(ReplCommandsTest, ellip_e_noassign) {
    Interpreter interp;
    expect_ok(interp, "ellip_e(0.5)");
}

TEST(ReplCommandsTest, zeta_noassign) {
    Interpreter interp;
    expect_ok(interp, "zeta(2)");
}

TEST(ReplCommandsTest, sph_harm_noassign) {
    Interpreter interp;
    expect_contains(interp, "sph_harm(1, 1, 0.5, 1)", "Y =");
    expect_error_contains(interp, "sph_harm(1.5, 1, 0.5, 1)", "integer l and m");
}

TEST(ReplCommandsTest, hypergeo_2f1_noassign) {
    Interpreter interp;
    expect_ok(interp, "hypergeo_2f1(1, 1, 2, 0.5)");
    expect_error_contains(interp, "hypergeo_2f1(1, b, 2, 0.5)", "numeric");
}

TEST(ReplCommandsTest, jacobi_p_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_p(2, 0.5, 0.5, 0.3)");
    expect_error_contains(interp, "jacobi_p(1, a, b, 0.3)", "numeric");
}

TEST(ReplCommandsTest, spheroidal_s1_noassign) {
    Interpreter interp;
    expect_ok(interp, "spheroidal_s1(1, 0, 0.1, 0.5)");
    expect_error_contains(interp, "spheroidal_s1(1, 0, 0.1, missing)", "spheroidal_s1");
}

TEST(ReplCommandsTest, spheroidal_s2_noassign) {
    Interpreter interp;
    expect_ok(interp, "spheroidal_s2(1, 0, 0.1, 0.5)");
    expect_error_contains(interp, "spheroidal_s2(1, 0, 0.1, missing)", "spheroidal_s2");
}

TEST(ReplCommandsTest, painleve2_noassign) {
    Interpreter interp;
    expect_ok(interp, "painleve2(0.5, 0.0, -0.5, 0.0)");
    expect_error_contains(interp, "painleve2(0.5, 0.0, -0.5, missing)", "painleve2");
}

TEST(ReplCommandsTest, painleve3_noassign) {
    Interpreter interp;
    expect_ok(interp, "painleve3(0.5, 0.5, -0.1, 0.5, 0.3)");
    expect_error_contains(interp, "painleve3(0.5, 0.5, -0.1, 0.5, missing)", "painleve3");
}

TEST(ReplCommandsTest, painleve4_noassign) {
    Interpreter interp;
    expect_ok(interp, "painleve4(0.5, 0.8, -0.05, 0.2, 0.4)");
    expect_error_contains(interp, "painleve4(0.5, 0.8, -0.05, 0.2, missing)", "painleve4");
}

TEST(ReplCommandsTest, heun_g_noassign) {
    Interpreter interp;
    expect_ok(interp, "heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0)");
    expect_error_contains(interp, "heun_g(a, 1, 2, 3, 4, 5, 6)", "heun_g");
}

TEST(ReplCommandsTest, painleve5_noassign) {
    Interpreter interp;
    expect_ok(interp, "painleve5(0.5, 0.5, -0.05, 0.01, 0.02, 0.03, 0.04)");
    expect_error_contains(interp, "painleve5(0.5, 0.5, -0.05, 0.01, 0.02, 0.03, missing)",
                          "painleve5");
}

TEST(ReplCommandsTest, painleve6_noassign) {
    Interpreter interp;
    expect_ok(interp, "painleve6(2.5, 0.5, -0.05, 0.1, 0.2, 0.3, 0.4)");
    expect_error_contains(interp, "painleve6(2.5, 0.5, -0.05, 0.1, 0.2, 0.3, missing)",
                          "painleve6");
}

TEST(ReplCommandsTest, heun_c_noassign) {
    Interpreter interp;
    expect_ok(interp, "heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 0.2)");
    expect_error_contains(interp, "heun_c(a, 0.2, 0.3, 0.4, 0.5, 0.2)", "heun_c");
}

TEST(ReplCommandsTest, mathieu_se_noassign) {
    Interpreter interp;
    expect_ok(interp, "mathieu_se(1, 0.1, 0.5)");
    expect_error_contains(interp, "mathieu_se(1, 0.1, missing)", "mathieu_se");
}

TEST(ReplCommandsTest, mathieu_mc_noassign) {
    Interpreter interp;
    expect_ok(interp, "mathieu_mc(1, 0.2, 0.3)");
    expect_error_contains(interp, "mathieu_mc(1, 0.2, missing)", "mathieu_mc");
}

TEST(ReplCommandsTest, mathieu_ms_noassign) {
    Interpreter interp;
    expect_ok(interp, "mathieu_ms(1, 0.2, 0.3)");
    expect_error_contains(interp, "mathieu_ms(1, 0.2, missing)", "mathieu_ms");
}

TEST(ReplCommandsTest, spheroidal_lambda_noassign) {
    Interpreter interp;
    expect_ok(interp, "spheroidal_lambda(1, 0, 0.1)");
    expect_error_contains(interp, "spheroidal_lambda(1, 0, missing)", "spheroidal_lambda");
}

TEST(ReplCommandsTest, legendre_pn_noassign) {
    Interpreter interp;
    expect_ok(interp, "legendre_pn(2, 0, 0.5)");
    expect_error_contains(interp, "legendre_pn(2, 0, missing)", "legendre_pn");
}

TEST(ReplCommandsTest, lerch_phi_noassign) {
    Interpreter interp;
    expect_ok(interp, "lerch_phi(0.5, 2, 0.3)");
    expect_error_contains(interp, "lerch_phi(0.5, 2, missing)", "lerch_phi");
}

TEST(ReplCommandsTest, laguerre_ln_noassign) {
    Interpreter interp;
    expect_ok(interp, "laguerre_ln(2, 0, 0.5)");
    expect_error_contains(interp, "laguerre_ln(2, 0, missing)", "laguerre_ln");
}

TEST(ReplCommandsTest, chebyshev_tn_noassign) {
    Interpreter interp;
    expect_ok(interp, "chebyshev_tn(3, 0, 0.5)");
    expect_error_contains(interp, "chebyshev_tn(3, 0, missing)", "chebyshev_tn");
}

TEST(ReplCommandsTest, chebyshev_un_noassign) {
    Interpreter interp;
    expect_ok(interp, "chebyshev_un(2, 0, 0.5)");
    expect_error_contains(interp, "chebyshev_un(2, 0, missing)", "chebyshev_un");
}

TEST(ReplCommandsTest, gegenbauer_c_noassign) {
    Interpreter interp;
    expect_ok(interp, "gegenbauer_c(2, 1, 0.5)");
    expect_error_contains(interp, "gegenbauer_c(2, 1, missing)", "gegenbauer_c");
}

TEST(ReplCommandsTest, jacobi_theta_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_theta(1, 0.2, 0.5)");
    expect_error_contains(interp, "jacobi_theta(1, 0.2, missing)", "jacobi_theta");
}

TEST(ReplCommandsTest, special_beta_inc_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_beta_inc(0.3, 1, 1)");
    expect_error_contains(interp, "special_beta_inc(0.3, 1, missing)", "special_beta_inc");
}

TEST(ReplCommandsTest, special_pseudo_voigt_auto_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_pseudo_voigt_auto(0, 1, 0.5)");
    expect_error_contains(interp, "special_pseudo_voigt_auto(0, 1, missing)",
                          "special_pseudo_voigt_auto");
}

TEST(ReplCommandsTest, weierstrass_sigma_noassign) {
    Interpreter interp;
    expect_ok(interp, "weierstrass_sigma(0.5, 1, 0)");
    expect_error_contains(interp, "weierstrass_sigma(0.5, 1, missing)", "weierstrass_sigma");
}

TEST(ReplCommandsTest, gamma_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "gamma_cdf(1, 2, 1)");
    expect_error_contains(interp, "gamma_cdf(1, 2, missing)", "gamma_cdf");
}

TEST(ReplCommandsTest, beta_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "beta_pdf(0.5, 1, 1)");
    expect_error_contains(interp, "beta_pdf(0.5, 1, missing)", "beta_pdf");
}

TEST(ReplCommandsTest, beta_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "beta_cdf(0.5, 1, 1)");
    expect_error_contains(interp, "beta_cdf(0.5, 1, missing)", "beta_cdf");
}

TEST(ReplCommandsTest, f_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "f_pdf(1, 5, 5)");
    expect_error_contains(interp, "f_pdf(1, 5, missing)", "f_pdf");
}

TEST(ReplCommandsTest, f_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "f_cdf(1, 5, 5)");
    expect_error_contains(interp, "f_cdf(1, 5, missing)", "f_cdf");
}

TEST(ReplCommandsTest, special_bessel_y_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_bessel_y(0, 1)");
    expect_error_contains(interp, "special_bessel_y(0, missing)", "special_bessel_y");
}

TEST(ReplCommandsTest, bessel_i_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_i(1, 1)");
    expect_error_contains(interp, "bessel_i(1, missing)", "bessel_i");
}

TEST(ReplCommandsTest, special_bessel_i_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_bessel_i(0, 1)");
    expect_error_contains(interp, "special_bessel_i(0, missing)", "special_bessel_i");
}

TEST(ReplCommandsTest, special_bessel_k_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_bessel_k(1, 1)");
    expect_error_contains(interp, "special_bessel_k(1, missing)", "special_bessel_k");
}

TEST(ReplCommandsTest, chebyshev_u_noassign) {
    Interpreter interp;
    expect_ok(interp, "chebyshev_u(1, 0.5)");
    expect_error_contains(interp, "chebyshev_u(1, missing)", "chebyshev_u");
}

TEST(ReplCommandsTest, hermite_h_noassign) {
    Interpreter interp;
    expect_ok(interp, "hermite_h(2, 0.5)");
    expect_error_contains(interp, "hermite_h(2, missing)", "hermite_h");
}

TEST(ReplCommandsTest, hermite_hf_noassign) {
    Interpreter interp;
    expect_ok(interp, "hermite_hf(2, 0.5)");
    expect_error_contains(interp, "hermite_hf(2, missing)", "hermite_hf");
}

TEST(ReplCommandsTest, laguerre_l_noassign) {
    Interpreter interp;
    expect_ok(interp, "laguerre_l(1, 0.5)");
    expect_error_contains(interp, "laguerre_l(1, missing)", "laguerre_l");
}

TEST(ReplCommandsTest, hermite_he_noassign) {
    Interpreter interp;
    expect_ok(interp, "hermite_he(2, 0.5)");
    expect_error_contains(interp, "hermite_he(2, missing)", "hermite_he");
}

TEST(ReplCommandsTest, chebyshev_v_noassign) {
    Interpreter interp;
    expect_ok(interp, "chebyshev_v(1, 0.5)");
    expect_error_contains(interp, "chebyshev_v(1, missing)", "chebyshev_v");
}

TEST(ReplCommandsTest, chebyshev_w_noassign) {
    Interpreter interp;
    expect_ok(interp, "chebyshev_w(1, 0.5)");
    expect_error_contains(interp, "chebyshev_w(1, missing)", "chebyshev_w");
}

TEST(ReplCommandsTest, mathieu_a_noassign) {
    Interpreter interp;
    expect_ok(interp, "mathieu_a(1, 0.1)");
    expect_error_contains(interp, "mathieu_a(1, missing)", "mathieu_a");
}

TEST(ReplCommandsTest, mathieu_b_noassign) {
    Interpreter interp;
    expect_ok(interp, "mathieu_b(1, 0)");
    expect_error_contains(interp, "mathieu_b(1, missing)", "mathieu_b");
}

TEST(ReplCommandsTest, pcf_u_noassign) {
    Interpreter interp;
    expect_ok(interp, "pcf_u(0.5, 1)");
    expect_error_contains(interp, "pcf_u(0.5, missing)", "pcf_u");
}

TEST(ReplCommandsTest, pcf_v_noassign) {
    Interpreter interp;
    expect_ok(interp, "pcf_v(0.5, 1)");
    expect_error_contains(interp, "pcf_v(0.5, missing)", "pcf_v");
}

TEST(ReplCommandsTest, pcf_w_noassign) {
    Interpreter interp;
    expect_ok(interp, "pcf_w(0.5, 1)");
    expect_error_contains(interp, "pcf_w(0.5, missing)", "pcf_w");
}

TEST(ReplCommandsTest, sph_bessel_j_noassign) {
    Interpreter interp;
    expect_ok(interp, "sph_bessel_j(2, 1)");
    expect_error_contains(interp, "sph_bessel_j(2, missing)", "sph_bessel_j");
}

TEST(ReplCommandsTest, sph_bessel_y_noassign) {
    Interpreter interp;
    expect_ok(interp, "sph_bessel_y(1, 1)");
    expect_error_contains(interp, "sph_bessel_y(1, missing)", "sph_bessel_y");
}

TEST(ReplCommandsTest, lambert_w_noassign) {
    Interpreter interp;
    expect_ok(interp, "lambert_w(0, 1)");
    expect_error_contains(interp, "lambert_w(0, missing)", "lambert_w");
}

TEST(ReplCommandsTest, special_lambert_w_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_lambert_w(-1, -0.2)");
    expect_error_contains(interp, "special_lambert_w(-1, missing)", "special_lambert_w");
}

TEST(ReplCommandsTest, hypergeo_1f1_noassign) {
    Interpreter interp;
    expect_ok(interp, "hypergeo_1f1(1, 0)");
    expect_error_contains(interp, "hypergeo_1f1(1, missing)", "hypergeo_1f1");
}

TEST(ReplCommandsTest, special_pochhammer_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_pochhammer(2.5, 3)");
    expect_error_contains(interp, "special_pochhammer(2.5, missing)", "special_pochhammer");
}

TEST(ReplCommandsTest, special_falling_factorial_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_falling_factorial(5, 2)");
    expect_error_contains(interp, "special_falling_factorial(5, missing)",
                          "special_falling_factorial");
}

TEST(ReplCommandsTest, spherical_kn_noassign) {
    Interpreter interp;
    expect_ok(interp, "spherical_kn(1, 1)");
    expect_error_contains(interp, "spherical_kn(1, missing)", "spherical_kn");
}

TEST(ReplCommandsTest, struve_l_noassign) {
    Interpreter interp;
    expect_ok(interp, "struve_l(1, 1)");
    expect_error_contains(interp, "struve_l(1, missing)", "struve_l");
}

TEST(ReplCommandsTest, struve_k_noassign) {
    Interpreter interp;
    expect_ok(interp, "struve_k(0, 1)");
    expect_error_contains(interp, "struve_k(0, missing)", "struve_k");
}

TEST(ReplCommandsTest, anger_j_noassign) {
    Interpreter interp;
    expect_ok(interp, "anger_j(1, 1)");
    expect_error_contains(interp, "anger_j(1, missing)", "anger_j");
}

TEST(ReplCommandsTest, weber_e_noassign) {
    Interpreter interp;
    expect_ok(interp, "weber_e(0, 1)");
    expect_error_contains(interp, "weber_e(0, missing)", "weber_e");
}

TEST(ReplCommandsTest, kelvin_ker_noassign) {
    Interpreter interp;
    expect_ok(interp, "kelvin_ker(0, 1)");
    expect_error_contains(interp, "kelvin_ker(0, missing)", "kelvin_ker");
}

TEST(ReplCommandsTest, kelvin_kei_noassign) {
    Interpreter interp;
    expect_ok(interp, "kelvin_kei(0, 1)");
    expect_error_contains(interp, "kelvin_kei(0, missing)", "kelvin_kei");
}

TEST(ReplCommandsTest, jacobi_dn_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_dn(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_dn(0.5, missing)", "jacobi_dn");
}

TEST(ReplCommandsTest, jacobi_am_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_am(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_am(0.5, missing)", "jacobi_am");
}

TEST(ReplCommandsTest, jacobi_sd_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_sd(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_sd(0.5, missing)", "jacobi_sd");
}

TEST(ReplCommandsTest, jacobi_dc_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_dc(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_dc(0.5, missing)", "jacobi_dc");
}

TEST(ReplCommandsTest, jacobi_nd_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_nd(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_nd(0.5, missing)", "jacobi_nd");
}

TEST(ReplCommandsTest, jacobi_cd_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_cd(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_cd(0.5, missing)", "jacobi_cd");
}

TEST(ReplCommandsTest, jacobi_cs_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_cs(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_cs(0.5, missing)", "jacobi_cs");
}

TEST(ReplCommandsTest, jacobi_ns_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_ns(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_ns(0.5, missing)", "jacobi_ns");
}

TEST(ReplCommandsTest, jacobi_ds_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_ds(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_ds(0.5, missing)", "jacobi_ds");
}

TEST(ReplCommandsTest, ellip_f_noassign) {
    Interpreter interp;
    expect_ok(interp, "ellip_f(0.3, 0.5)");
    expect_error_contains(interp, "ellip_f(0.3, missing)", "ellip_f");
}

TEST(ReplCommandsTest, ellip_e_inc_noassign) {
    Interpreter interp;
    expect_ok(interp, "ellip_e_inc(0.3, 0.5)");
    expect_error_contains(interp, "ellip_e_inc(0.3, missing)", "ellip_e_inc");
}

TEST(ReplCommandsTest, theta1_prime_noassign) {
    Interpreter interp;
    expect_ok(interp, "theta1_prime(0.2, 0.3)");
    expect_error_contains(interp, "theta1_prime(0.2, missing)", "theta1_prime");
}

TEST(ReplCommandsTest, theta2_noassign) {
    Interpreter interp;
    expect_ok(interp, "theta2(0.5, 0.3)");
    expect_error_contains(interp, "theta2(0.5, missing)", "theta2");
}

TEST(ReplCommandsTest, theta4_noassign) {
    Interpreter interp;
    expect_ok(interp, "theta4(0.5, 0.3)");
    expect_error_contains(interp, "theta4(0.5, missing)", "theta4");
}

TEST(ReplCommandsTest, debye_noassign) {
    Interpreter interp;
    expect_ok(interp, "debye(1, 1)");
    expect_error_contains(interp, "debye(1, missing)", "debye");
}

TEST(ReplCommandsTest, special_airy_aip_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_airy_aip(0)");
    expect_error_contains(interp, "special_airy_aip(missing)", "special_airy_aip");
}

TEST(ReplCommandsTest, special_airy_bip_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_airy_bip(0)");
    expect_error_contains(interp, "special_airy_bip(missing)", "special_airy_bip");
}

TEST(ReplCommandsTest, airy_aip_noassign) {
    Interpreter interp;
    expect_ok(interp, "airy_aip(0)");
    expect_error_contains(interp, "airy_aip(missing)", "airy_aip");
}

TEST(ReplCommandsTest, airy_bip_noassign) {
    Interpreter interp;
    expect_ok(interp, "airy_bip(0)");
    expect_error_contains(interp, "airy_bip(missing)", "airy_bip");
}

TEST(ReplCommandsTest, special_rgamma_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_rgamma(1)");
    expect_error_contains(interp, "special_rgamma(missing)", "special_rgamma");
}

TEST(ReplCommandsTest, bernoulli_number_noassign) {
    Interpreter interp;
    expect_ok(interp, "bernoulli_number(2)");
    expect_error_contains(interp, "bernoulli_number(missing)", "bernoulli_number");
}

TEST(ReplCommandsTest, euler_number_noassign) {
    Interpreter interp;
    expect_ok(interp, "euler_number(4)");
    expect_error_contains(interp, "euler_number(missing)", "euler_number");
}

TEST(ReplCommandsTest, special_erfinv_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_erfinv(0.5)");
    expect_error_contains(interp, "special_erfinv(missing)", "expected numeric argument");
}

TEST(ReplCommandsTest, special_erfcinv_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_erfcinv(0.3)");
    expect_error_contains(interp, "special_erfcinv(missing)", "expected numeric argument");
}

TEST(ReplCommandsTest, special_log_gamma_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_log_gamma(5)");
    expect_error_contains(interp, "special_log_gamma(missing)", "expected numeric argument");
}

TEST(ReplCommandsTest, special_digamma_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_digamma(1)");
    expect_error_contains(interp, "special_digamma(missing)", "expected numeric argument");
}

TEST(ReplCommandsTest, special_trigamma_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_trigamma(1)");
    expect_error_contains(interp, "special_trigamma(missing)", "expected numeric argument");
}

TEST(ReplCommandsTest, special_airy_ai_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_airy_ai(0)");
    expect_error_contains(interp, "special_airy_ai(missing)", "expected numeric argument");
}

TEST(ReplCommandsTest, special_airy_bi_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_airy_bi(0)");
    expect_error_contains(interp, "special_airy_bi(missing)", "expected numeric argument");
}

TEST(ReplCommandsTest, special_gamma_inc_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_gamma_inc(1, 1)");
    expect_error_contains(interp, "special_gamma_inc(1, missing)", "expected special_gamma_inc(a,x)");
}

TEST(ReplCommandsTest, special_polygamma_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_polygamma(1, 1)");
    expect_error_contains(interp, "special_polygamma(1, missing)", "expected special_polygamma(n,x)");
}

TEST(ReplCommandsTest, special_gamma_inc_reg_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_gamma_inc_reg(1, 1)");
    expect_error_contains(interp, "special_gamma_inc_reg(1, missing)",
                          "expected special_gamma_inc_reg(a,x)");
}

TEST(ReplCommandsTest, special_gamma_inc_reg_upper_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_gamma_inc_reg_upper(1, 1)");
    expect_error_contains(interp, "special_gamma_inc_reg_upper(1, missing)",
                          "expected special_gamma_inc_reg_upper(a,x)");
}

TEST(ReplCommandsTest, special_voigt_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_voigt(0, 1, 0)");
    expect_error_contains(interp, "special_voigt(0, 1, missing)",
                          "expected special_voigt(x,sigma,gamma)");
}

TEST(ReplCommandsTest, special_beta_inc_reg_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_beta_inc_reg(0.3, 1, 1)");
    expect_error_contains(interp, "special_beta_inc_reg(0.3, 1, missing)",
                          "expected special_beta_inc_reg(x,a,b)");
}

TEST(ReplCommandsTest, spherical_in_noassign) {
    Interpreter interp;
    expect_ok(interp, "spherical_in(0, 1)");
    expect_error_contains(interp, "spherical_in(0, missing)", "expected spherical_in(n,x)");
}

TEST(ReplCommandsTest, kelvin_bei_noassign) {
    Interpreter interp;
    expect_ok(interp, "kelvin_bei(0, 1)");
    expect_error_contains(interp, "kelvin_bei(0, missing)", "expected kelvin_bei(nu,x)");
}

TEST(ReplCommandsTest, spherical_jn_noassign) {
    Interpreter interp;
    expect_ok(interp, "spherical_jn(0, 1)");
    expect_error_contains(interp, "spherical_jn(0, missing)", "expected spherical_jn(n,x)");
}

TEST(ReplCommandsTest, spherical_yn_noassign) {
    Interpreter interp;
    expect_ok(interp, "spherical_yn(0, 1)");
    expect_error_contains(interp, "spherical_yn(0, missing)", "expected spherical_yn(n,x)");
}

TEST(ReplCommandsTest, bessel_h_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_h(0, 1)");
    expect_error_contains(interp, "bessel_h(0, missing)", "expected bessel_h(nu,x)");
}

TEST(ReplCommandsTest, bessel_hy_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_hy(1, 0.5)");
    expect_error_contains(interp, "bessel_hy(1, missing)", "expected bessel_hy(nu,x)");
}

TEST(ReplCommandsTest, bessel_l_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_l(1, 0.5)");
    expect_error_contains(interp, "bessel_l(1, missing)", "expected bessel_l(nu,x)");
}

TEST(ReplCommandsTest, bessel_lu_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_lu(1, 0.5)");
    expect_error_contains(interp, "bessel_lu(1, missing)", "expected bessel_lu(nu,x)");
}

TEST(ReplCommandsTest, hermite_hn_noassign) {
    Interpreter interp;
    expect_ok(interp, "hermite_hn(2, 0.5)");
    expect_error_contains(interp, "hermite_hn(2, missing)", "expected hermite_hn(n,x)");
}

TEST(ReplCommandsTest, bessel_y_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_y(0, 1)");
    expect_error_contains(interp, "bessel_y(0, missing)", "expected bessel_y(nu,x)");
}

TEST(ReplCommandsTest, bessel_k_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_k(0, 1)");
    expect_error_contains(interp, "bessel_k(0, missing)", "expected bessel_k(nu,x)");
}

TEST(ReplCommandsTest, legendre_q_noassign) {
    Interpreter interp;
    expect_ok(interp, "legendre_q(2, 0.3)");
    expect_error_contains(interp, "legendre_q(2, missing)", "expected legendre_q(n,x)");
}

TEST(ReplCommandsTest, kelvin_ber_noassign) {
    Interpreter interp;
    expect_ok(interp, "kelvin_ber(0, 1)");
    expect_error_contains(interp, "kelvin_ber(0, missing)", "expected kelvin_ber(nu,x)");
}

TEST(ReplCommandsTest, bessel_zero_jnu_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_zero_jnu(0, 1)");
    expect_error_contains(interp, "bessel_zero_jnu(0, missing)", "expected bessel_zero_jnu(nu,n)");
}

TEST(ReplCommandsTest, struve_hn_noassign) {
    Interpreter interp;
    expect_ok(interp, "struve_hn(1, 1)");
    expect_error_contains(interp, "struve_hn(1, missing)", "expected struve_hn(nu,x)");
}

TEST(ReplCommandsTest, struve_yn_noassign) {
    Interpreter interp;
    expect_ok(interp, "struve_yn(1, 1)");
    expect_error_contains(interp, "struve_yn(1, missing)", "expected struve_yn(nu,x)");
}

TEST(ReplCommandsTest, bessel_zero_ynu_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_zero_ynu(0, 1)");
    expect_error_contains(interp, "bessel_zero_ynu(0, missing)", "expected bessel_zero_ynu(nu,n)");
}

TEST(ReplCommandsTest, jacobi_nc_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_nc(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_nc(0.5, missing)", "expected jacobi_nc(u,k)");
}

TEST(ReplCommandsTest, theta3_noassign) {
    Interpreter interp;
    expect_ok(interp, "theta3(0.5, 0.3)");
    expect_error_contains(interp, "theta3(0.5, missing)", "expected theta3(z,q)");
}

TEST(ReplCommandsTest, polylog_noassign) {
    Interpreter interp;
    expect_ok(interp, "polylog(2, 0.5)");
    expect_error_contains(interp, "polylog(2, missing)", "expected polylog(n,z)");
}

TEST(ReplCommandsTest, legendre_p_noassign) {
    Interpreter interp;
    expect_ok(interp, "legendre_p(2, 0.5)");
    expect_error_contains(interp, "legendre_p(2, missing)", "expected legendre_p(n,x)");
}

TEST(ReplCommandsTest, kummer_m_noassign) {
    Interpreter interp;
    expect_ok(interp, "kummer_m(1, 2, 0.5)");
    expect_error_contains(interp, "kummer_m(1, 2, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, kummer_u_noassign) {
    Interpreter interp;
    expect_ok(interp, "kummer_u(1, 2, 0.5)");
    expect_error_contains(interp, "kummer_u(1, 2, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, whittaker_m_noassign) {
    Interpreter interp;
    expect_ok(interp, "whittaker_m(0, 0.5, 1)");
    expect_error_contains(interp, "whittaker_m(0, 0.5, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, whittaker_w_noassign) {
    Interpreter interp;
    expect_ok(interp, "whittaker_w(0, 0.5, 1)");
    expect_error_contains(interp, "whittaker_w(0, 0.5, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, tricomi_u_noassign) {
    Interpreter interp;
    expect_ok(interp, "tricomi_u(1, 2, 0.5)");
    expect_error_contains(interp, "tricomi_u(1, 2, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, meijer_g_noassign) {
    Interpreter interp;
    expect_ok(interp, "meijer_g(1, 2, 0.5)");
    expect_error_contains(interp, "meijer_g(1, 2, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, fox_h_noassign) {
    Interpreter interp;
    expect_ok(interp, "fox_h(1, 2, 0.5)");
    expect_error_contains(interp, "fox_h(1, 2, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, hypergeo_0f1n_noassign) {
    Interpreter interp;
    expect_ok(interp, "hypergeo_0f1n(2, 1.5, 0.2)");
    expect_error_contains(interp, "hypergeo_0f1n(2, 1.5, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, hypergeo_1f1n_noassign) {
    Interpreter interp;
    expect_ok(interp, "hypergeo_1f1n(1, 1, 0.3)");
    expect_error_contains(interp, "hypergeo_1f1n(1, 1, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, mathieu_ce_noassign) {
    Interpreter interp;
    expect_ok(interp, "mathieu_ce(1, 0.1, 0.5)");
    expect_error_contains(interp, "mathieu_ce(1, 0.1, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, painleve1_noassign) {
    Interpreter interp;
    expect_ok(interp, "painleve1(0.5, 0, 0)");
    expect_error_contains(interp, "painleve1(0.5, 0, missing)", "expected numeric arguments");
}

TEST(ReplCommandsTest, assoc_legendre_p_noassign) {
    Interpreter interp;
    expect_ok(interp, "assoc_legendre_p(2, 1, 0.5)");
    expect_error_contains(interp, "assoc_legendre_p(2, 1, missing)", "expected assoc_legendre_p(l,m,x)");
}

TEST(ReplCommandsTest, laguerre_la_noassign) {
    Interpreter interp;
    expect_ok(interp, "laguerre_la(2, 1, 0.5)");
    expect_error_contains(interp, "laguerre_la(2, 1, missing)", "expected laguerre_la(n,a,x)");
}

TEST(ReplCommandsTest, special_pseudo_voigt_noassign) {
    Interpreter interp;
    expect_ok(interp, "special_pseudo_voigt(0, 1, 0.5, 0.4)");
    expect_error_contains(interp, "special_pseudo_voigt(0, 1, 0.5, missing)",
                          "expected special_pseudo_voigt(x,sigma,gamma,eta)");
}

TEST(ReplCommandsTest, ellip_d_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "x = ellip_d(0.5)");
    ASSERT_GT(interp.state().scalars.count("x"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("x"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, erf_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = erf(0)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, special_rgamma_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = special_rgamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, special_erfinv_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = special_erfinv(0)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, bernoulli_number_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = bernoulli_number(2)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 1.0 / 6.0, 1e-12);
}

TEST(ReplCommandsTest, euler_number_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = euler_number(0)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, special_digamma_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = special_digamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("s"), ms::digamma(1.0), 1e-8);
}

TEST(ReplCommandsTest, special_trigamma_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = special_trigamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("s"), ms::trigamma(1.0), 1e-8);
}

TEST(ReplCommandsTest, special_log_gamma_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = special_log_gamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 0.0, 1e-8);
}

TEST(ReplCommandsTest, special_airy_ai_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = special_airy_ai(0)");
    EXPECT_NEAR(interp.state().scalars.at("s"), ms::airy_ai(0.0), 1e-8);
}
