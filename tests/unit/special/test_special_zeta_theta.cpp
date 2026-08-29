#include <gtest/gtest.h>
#include <cmath>
#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialThetaTest, jacobi_theta_functions) {
    const double q = 0.3;
    const double z = 0.5;
    EXPECT_NEAR(theta1(z, q), 0.5773940463248446, 1e-3);
    EXPECT_NEAR(theta2(z, q), 1.3075255735032947, 1e-3);
    EXPECT_NEAR(theta3(z, q), 1.317400827096804, 1e-3);
    EXPECT_NEAR(theta4(z, q), 0.6691160041441827, 1e-3);
    EXPECT_NEAR(theta1_prime(z, q), 1.2663806017958978, 1e-3);
    EXPECT_NEAR(jacobi_theta(3, z, 0.2), 1.5020847332591976, 1e-3);
}

TEST(SpecialWeierstrassTest, short_series) {
    const double z = 0.5;
    const double g2 = 1.0;
    const double g3 = 0.0;
    EXPECT_NEAR(weierstrass_p(z, g2, g3), 4.012516276465522, 1e-3);
    EXPECT_NEAR(weierstrass_pprime(z, g2, g3), -15.9498046875, 1e-2);
    EXPECT_NEAR(weierstrass_zeta(z, g2, g3), 2.082291666666667, 1e-3);
    EXPECT_NEAR(weierstrass_sigma(z, g2, g3), 0.4998694525824653, 1e-3);
}

TEST(SpecialZetaTest, zeta_and_related) {
    EXPECT_NEAR(zeta(2.0), 1.6449340668482264, 1e-9);
    EXPECT_NEAR(zeta(0.5), -1.4603545088095868, 5e-3);
    EXPECT_NEAR(zeta_hurwitz(2.0, 0.3), 12.245364546107732, 1e-3);
    EXPECT_NEAR(eta_dirichlet(2.0), 0.8224670334241132, 1e-6);
    EXPECT_NEAR(beta_dirichlet(2.0), 0.915965594127219, 1e-3);
    EXPECT_NEAR(polylog(2, 0.5), 0.5822405264650125, 1e-3);
    EXPECT_NEAR(clausen(0.5), 0.8483118777036792, 1e-3);
    EXPECT_NEAR(lerch_phi(0.5, 2.0, 0.3), 11.47083462974499, 1e-3);
}

TEST(SpecialMathieuTest, characteristic_and_periodic) {
    const double q = 0.1;
    EXPECT_NEAR(mathieu_a(1, q), 1.0987343129634084, 1e-3);
    EXPECT_NEAR(mathieu_b(1, q), 0.8987655569943626, 1e-3);
    EXPECT_NEAR(mathieu_ce(1, q, 0.5), 0.8765747185555349, 7e-2);
}

TEST(SpecialMathieuTest, modified_functions) {
    const double q = 1.0;
    EXPECT_NEAR(mathieu_mc(1, q, 0.0), 0.6877267249062208, 1e-3);
    EXPECT_NEAR(mathieu_ms(1, q, 0.0), 0.40462549819476684, 1e-3);
}

TEST(SpecialWaveTest, spheroidal_and_parabolic) {
    EXPECT_NEAR(spheroidal_lambda(1, 1, 5.0), -7.493388284110646, 3e-2);
    EXPECT_NEAR(spheroidal_s1(1, 1, 5.0, 0.5), 0.03747174337125646, 5e-2);
    EXPECT_NEAR(pcf_w(0.5, 1.0), pcf_u(0.5, 1.0) * std::cos(0.5 * std::acos(-1.0)) -
                                    pcf_v(0.5, 1.0) * std::sin(0.5 * std::acos(-1.0)),
                1e-6);
}
