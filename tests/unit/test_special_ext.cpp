#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
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
